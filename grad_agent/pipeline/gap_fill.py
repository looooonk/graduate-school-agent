"""Targeted gap-fill retrieval stage."""

from __future__ import annotations

from typing import Any

import anthropic
import httpx

from grad_agent.config import Config
from grad_agent.events import EventCallback, ToolCalled, TurnProgress
from grad_agent.llm.vllm import LocalVLLMClient
from grad_agent.models import JudgeReport, SchoolProfile
from grad_agent.pipeline.local_retrieval import (
    run_local_parallel_profile_loop,
    run_local_profile_loop,
)
from grad_agent.pipeline.prompts import RETRIEVAL_SYSTEM, retrieval_turn_status
from grad_agent.pipeline.tools import TOOL_DEFINITIONS, dispatch_tool
from grad_agent.reporting.stats import StageStats, add_usage, timed
from grad_agent.reporting.trajectory import TrajectoryLogger
from grad_agent.util.json import extract_json_object
from grad_agent.util.log import get_school_logger
from grad_agent.util.retry import api_create_with_retry


async def run_gap_fill(
    profile: SchoolProfile,
    judge_report: JudgeReport,
    config: Config,
    client: anthropic.AsyncAnthropic,
    http: httpx.AsyncClient,
    on_event: EventCallback | None = None,
    traj: TrajectoryLogger | None = None,
    context_text: str = "",
    local_client: LocalVLLMClient | None = None,
) -> tuple[SchoolProfile, StageStats]:
    """Run targeted retrieval using the judge's suggested queries."""
    school_label = f"{profile.school_name} — {profile.program_name}"
    log = get_school_logger(__name__, school_label)
    stats = StageStats(stage="gap_fill", model=config.retrieval_model)
    initial_prompt = gap_fill_prompt(profile, judge_report, config, context_text)

    if config.uses_local_retrieval:
        if config.local_retrieval_parallel_agents > 1:
            return await run_local_parallel_profile_loop(
                school_name=profile.school_name,
                program_name=profile.program_name,
                config=config,
                http=http,
                initial_prompt=initial_prompt,
                stage="gap_fill",
                max_turns=config.gap_fill_max_turns,
                on_event=on_event,
                traj=traj,
                local_client=local_client,
            )
        return await run_local_profile_loop(
            school_name=profile.school_name,
            program_name=profile.program_name,
            config=config,
            http=http,
            initial_prompt=initial_prompt,
            stage="gap_fill",
            max_turns=config.gap_fill_max_turns,
            on_event=on_event,
            traj=traj,
            local_client=local_client,
        )

    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_prompt}]
    with timed() as elapsed:
        for turn in range(1, config.gap_fill_max_turns + 1):
            log.info("Gap-fill turn %d/%d", turn, config.gap_fill_max_turns)
            if on_event:
                on_event(
                    TurnProgress(
                        school=school_label,
                        turn=turn,
                        max_turns=config.gap_fill_max_turns,
                        stage="gap_fill",
                    )
                )

            response = await api_create_with_retry(
                lambda: client.messages.create(
                    model=config.retrieval_model,
                    max_tokens=4096,
                    system=RETRIEVAL_SYSTEM,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            )
            stats.api_calls += 1
            add_usage(stats, response.usage)
            if traj:
                traj.log_api_response("gap_fill", turn, config.retrieval_model, response)

            if response.stop_reason == "tool_use":
                tool_results = []
                assistant_content = []
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
                        if on_event:
                            on_event(ToolCalled(
                                school=school_label,
                                tool_name=block.name,
                                stage="gap_fill",
                            ))
                        result = await dispatch_tool(
                            block.name, block.input, config, http, school_label,
                        )
                        if traj:
                            traj.log_tool_result("gap_fill", turn, block.name, block.input, result)
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
                            "text": retrieval_turn_status(turn, config.gap_fill_max_turns),
                        },
                    ],
                })
                continue

            if response.stop_reason == "end_turn":
                full_text = "\n".join(
                    block.text for block in response.content if block.type == "text"
                )
                parsed = extract_json_object(full_text)
                if parsed:
                    parsed["school_name"] = profile.school_name
                    parsed["program_name"] = profile.program_name
                    try:
                        updated = SchoolProfile.model_validate(parsed)
                        stats.elapsed_seconds = elapsed[0]
                        log.info("Gap-fill produced updated profile")
                        if traj:
                            traj.log_profile(updated)
                            traj.log_stage_end("gap_fill", elapsed[0])
                        return updated, stats
                    except Exception as exc:
                        log.warning("Gap-fill validation failed: %s", exc)
                break

    stats.elapsed_seconds = elapsed[0]
    if traj:
        traj.log_stage_end("gap_fill", elapsed[0])
    return profile, stats


def gap_fill_prompt(
    profile: SchoolProfile,
    judge_report: JudgeReport,
    config: Config,
    context_text: str = "",
) -> str:
    existing_json = profile.model_dump_json(indent=2)
    context_section = (
        f"## Applicant Context\n\n{context_text.strip()}\n\n"
        if context_text.strip()
        else ""
    )
    flags = "\n".join(
        f"- {flag.field}: {flag.reason}" for flag in judge_report.flagged_fields
    ) or "- No specific flagged fields were provided."
    suggested = "\n".join(f"- {q}" for q in judge_report.suggested_queries)

    return (
        f"{context_section}"
        f"Here is an existing SchoolProfile that was rated as insufficient:\n\n"
        f"```json\n{existing_json}\n```\n\n"
        f"The quality judge flagged these gaps:\n"
        f"{flags}\n\n"
        f"Suggested targeted queries:\n"
        f"{suggested}\n\n"
        f"Please run the most relevant suggested searches first, then any "
        f"other narrow searches needed for the same flagged fields. Then output "
        f"an UPDATED complete SchoolProfile JSON incorporating both the existing "
        f"data and new findings. "
        f"Preserve existing sourced facts unless new official evidence corrects them. "
        f"You have a budget of **{config.gap_fill_max_turns} turns** total."
    )
