"""Stage 1 — Haiku Retrieval Agent.

Runs an agentic loop: the model issues tool calls (web_search, fetch_page),
receives results, and iterates until it produces a final SchoolProfile JSON
or the turn budget is exhausted.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
import httpx

from grad_agent.config import Config
from grad_agent.events import EventCallback, ToolCalled, TurnProgress
from grad_agent.llm.vllm import LocalVLLMClient
from grad_agent.models import SchoolProfile
from grad_agent.pipeline.prompts import (
    LOCAL_RETRIEVAL_PROTOCOL,
    RETRIEVAL_SYSTEM,
    retrieval_turn_status,
    retrieval_user_prompt,
)
from grad_agent.pipeline.tools import TOOL_DEFINITIONS, dispatch_tool
from grad_agent.reporting.stats import StageStats, add_usage, timed
from grad_agent.reporting.trajectory import TrajectoryLogger
from grad_agent.util.json import extract_json_object
from grad_agent.util.log import get_school_logger
from grad_agent.util.retry import api_create_with_retry

logger = logging.getLogger(__name__)


_extract_json_from_text = extract_json_object


async def run_retrieval(
    school_name: str,
    program_name: str,
    config: Config,
    client: anthropic.AsyncAnthropic,
    http: httpx.AsyncClient,
    context_text: str = "",
    on_event: EventCallback | None = None,
    traj: TrajectoryLogger | None = None,
    local_client: LocalVLLMClient | None = None,
) -> tuple[SchoolProfile, StageStats]:
    """Run the Haiku retrieval agent for a single school.

    Args:
        school_name: Name of the school.
        program_name: Name of the graduate program.
        config: Pipeline configuration.
        client: Anthropic async client.
        http: Async HTTP client for tool execution.
        context_text: Optional applicant context (from input/context.md) used to
            focus the search on relevant subfields and advisor types.
        on_event: Optional progress callback fired on each turn and tool call.
        traj: Optional trajectory logger for recording API interactions and results.

    Returns:
        A tuple of (SchoolProfile, StageStats). If the turn budget is exhausted
        without a valid profile, returns a minimal stub profile with a warning note.
    """
    school_label = f"{school_name} — {program_name}"
    if config.uses_local_retrieval:
        return await run_local_profile_loop(
            school_name=school_name,
            program_name=program_name,
            config=config,
            http=http,
            initial_prompt=retrieval_user_prompt(
                school_name, program_name, context_text, config.max_retrieval_turns,
            ),
            stage="retrieval",
            max_turns=config.max_retrieval_turns,
            on_event=on_event,
            traj=traj,
            local_client=local_client,
        )

    log = get_school_logger(__name__, school_label)
    stats = StageStats(stage="retrieval", model=config.retrieval_model)

    if traj:
        traj.log_stage_start("retrieval")

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": retrieval_user_prompt(
            school_name, program_name, context_text, config.max_retrieval_turns,
        )},
    ]

    with timed() as elapsed:
        for turn in range(1, config.max_retrieval_turns + 1):
            log.info("Turn %d/%d", turn, config.max_retrieval_turns)
            if on_event:
                on_event(
                    TurnProgress(
                        school=school_label,
                        turn=turn,
                        max_turns=config.max_retrieval_turns,
                    )
                )

            response = await api_create_with_retry(
                lambda: client.messages.create(
                    model=config.haiku_model,
                    max_tokens=4096,
                    system=RETRIEVAL_SYSTEM,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            )

            stats.api_calls += 1
            add_usage(stats, response.usage)

            if traj:
                traj.log_api_response("retrieval", turn, config.retrieval_model, response)

            if response.stop_reason == "tool_use":
                tool_results: list[dict[str, Any]] = []
                assistant_content: list[dict[str, Any]] = []

                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                        stats.tool_calls += 1
                        log.info("Tool call: %s(%s)", block.name, _summarize_args(block.input))
                        if on_event:
                            on_event(ToolCalled(school=school_label, tool_name=block.name))

                        result = await dispatch_tool(
                            block.name, block.input, config, http, school_label,
                        )
                        if traj:
                            traj.log_tool_result("retrieval", turn, block.name, block.input, result)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({
                    "role": "user",
                    "content": [
                        *tool_results,
                        {
                            "type": "text",
                            "text": retrieval_turn_status(turn, config.max_retrieval_turns),
                        },
                    ],
                })

            elif response.stop_reason == "end_turn":
                text_parts = [
                    block.text for block in response.content if block.type == "text"
                ]
                full_text = "\n".join(text_parts)

                parsed = _extract_json_from_text(full_text)
                if parsed is not None:
                    # Override these fields in case the model omitted or misspelled them.
                    parsed["school_name"] = school_name
                    parsed["program_name"] = program_name
                    try:
                        profile = SchoolProfile.model_validate(parsed)
                        stats.elapsed_seconds = elapsed[0]
                        log.info(
                            "Profile complete — %d sources, %d research areas, %d advisors",
                            len(profile.sources),
                            len(profile.research_areas),
                            len(profile.advisor_candidates),
                        )
                        if traj:
                            traj.log_profile(profile)
                            traj.log_stage_end("retrieval", elapsed[0])
                        return profile, stats
                    except Exception as exc:
                        log.warning("Profile validation failed: %s", exc)
                        # Ask the model to fix its output
                        messages.append({"role": "assistant", "content": full_text})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Your JSON output had a validation error: {exc}\n"
                                f"Please fix the JSON and output it again. "
                                f"Output ONLY valid JSON."
                            ),
                        })
                else:
                    log.warning("No JSON found in model output (turn %d)", turn)
                    messages.append({"role": "assistant", "content": full_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "I could not parse a JSON object from your response. "
                            "Please output the complete SchoolProfile as a single "
                            "JSON object — nothing else."
                        ),
                    })
            else:
                log.warning("Unexpected stop reason: %s", response.stop_reason)
                break

    stats.elapsed_seconds = elapsed[0]
    if traj:
        traj.log_stage_end("retrieval", elapsed[0])

    # If we get here, we exhausted the turn budget without a clean profile.
    # Make one last attempt to parse whatever we have.
    log.warning("Turn budget exhausted — attempting to salvage partial profile")
    return SchoolProfile(
        school_name=school_name,
        program_name=program_name,
        notes=(
            "WARNING: Retrieval agent exhausted turn budget without producing "
            "a complete profile."
        ),
    ), stats


def _summarize_args(args: dict[str, Any]) -> str:
    """Compact representation of tool arguments for logging."""
    if "query" in args:
        q = args["query"]
        return f"query={q!r}" if len(q) < 80 else f"query={q[:77]!r}..."
    if "url" in args:
        return f"url={args['url']!r}"
    return str(args)


async def run_local_profile_loop(
    *,
    school_name: str,
    program_name: str,
    config: Config,
    http: httpx.AsyncClient,
    initial_prompt: str,
    stage: str,
    max_turns: int,
    on_event: EventCallback | None = None,
    traj: TrajectoryLogger | None = None,
    local_client: LocalVLLMClient | None = None,
) -> tuple[SchoolProfile, StageStats]:
    """Run retrieval using a local OpenAI-compatible model and JSON tool commands."""
    school_label = f"{school_name} — {program_name}"
    log = get_school_logger(__name__, school_label)
    stats = StageStats(stage=stage, model=config.retrieval_model)
    client = local_client or LocalVLLMClient.from_config(config)

    if traj:
        traj.log_stage_start(stage)

    messages = [
        {"role": "system", "content": f"{RETRIEVAL_SYSTEM}\n\n{LOCAL_RETRIEVAL_PROTOCOL}"},
        {"role": "user", "content": initial_prompt},
    ]

    with timed() as elapsed:
        for turn in range(1, max_turns + 1):
            log.info("%s turn %d/%d", stage, turn, max_turns)
            if on_event:
                on_event(TurnProgress(school=school_label, turn=turn, max_turns=max_turns))

            response = await client.create(
                http,
                model=config.retrieval_model,
                messages=messages,
                max_tokens=4096,
            )
            stats.api_calls += 1
            add_usage(stats, response.usage)
            if traj:
                traj.log_api_response(stage, turn, config.retrieval_model, response)

            full_text = response.text.strip()
            parsed = _extract_json_from_text(full_text)
            if parsed is None:
                log.warning("No JSON found in local model output (turn %d)", turn)
                messages.append({"role": "assistant", "content": full_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Output exactly one JSON object: either a tool command "
                        "or the complete SchoolProfile. Do not include prose."
                    ),
                })
                continue

            tool_name = parsed.get("tool")
            if isinstance(tool_name, str):
                args = parsed.get("args")
                if not isinstance(args, dict):
                    args = {}
                stats.tool_calls += 1
                log.info("Tool call: %s(%s)", tool_name, _summarize_args(args))
                if on_event:
                    on_event(ToolCalled(school=school_label, tool_name=tool_name))
                result = await dispatch_tool(tool_name, args, config, http, school_label)
                if traj:
                    traj.log_tool_result(stage, turn, tool_name, args, result)
                messages.append({"role": "assistant", "content": full_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name}:\n\n{result}\n\n"
                        f"{retrieval_turn_status(turn, max_turns)}"
                    ),
                })
                continue

            parsed["school_name"] = school_name
            parsed["program_name"] = program_name
            try:
                profile = SchoolProfile.model_validate(parsed)
                stats.elapsed_seconds = elapsed[0]
                log.info(
                    "Profile complete — %d sources, %d research areas, %d advisors",
                    len(profile.sources),
                    len(profile.research_areas),
                    len(profile.advisor_candidates),
                )
                if traj:
                    traj.log_profile(profile)
                    traj.log_stage_end(stage, elapsed[0])
                return profile, stats
            except Exception as exc:
                log.warning("Profile validation failed: %s", exc)
                messages.append({"role": "assistant", "content": full_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your SchoolProfile JSON had a validation error: {exc}\n"
                        f"Fix it and output ONLY the complete SchoolProfile JSON."
                    ),
                })

    stats.elapsed_seconds = elapsed[0]
    if traj:
        traj.log_stage_end(stage, elapsed[0])
    log.warning("Turn budget exhausted — returning stub profile")
    return SchoolProfile(
        school_name=school_name,
        program_name=program_name,
        notes=(
            "WARNING: Retrieval agent exhausted turn budget without producing "
            "a complete profile."
        ),
    ), stats
