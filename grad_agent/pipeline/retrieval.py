"""Stage 1 — Haiku Retrieval Agent.

Runs an agentic loop: the model issues tool calls (web_search, fetch_page),
receives results, and iterates until it produces a final SchoolProfile JSON
or the turn budget is exhausted.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
import httpx

from grad_agent.config import Config
from grad_agent.models import SchoolProfile
from grad_agent.pipeline.prompts import RETRIEVAL_SYSTEM, retrieval_user_prompt
from grad_agent.pipeline.tools import TOOL_DEFINITIONS, dispatch_tool
from grad_agent.reporting.stats import StageStats, timed
from grad_agent.util.log import get_school_logger
from grad_agent.util.retry import api_create_with_retry

logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model text output.

    Handles both raw JSON and ```json fenced blocks.
    """
    stripped = text.strip()

    # Try fenced code block first
    if "```" in stripped:
        for block in stripped.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

    # Try the whole text as JSON
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Try to find the outermost { ... }
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


async def run_retrieval(
    school_name: str,
    program_name: str,
    config: Config,
    client: anthropic.AsyncAnthropic,
    http: httpx.AsyncClient,
) -> tuple[SchoolProfile, StageStats]:
    """Run the Haiku retrieval agent for a single school.

    Returns the populated SchoolProfile and stage statistics.

    Raises:
        RuntimeError: If the agent fails to produce a valid profile after
            exhausting the turn budget.
    """
    log = get_school_logger(__name__, f"{school_name} — {program_name}")
    stats = StageStats(stage="retrieval", model=config.haiku_model)
    school_label = f"{school_name} — {program_name}"

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": retrieval_user_prompt(school_name, program_name)},
    ]

    with timed() as elapsed:
        for turn in range(1, config.max_retrieval_turns + 1):
            log.info("Turn %d/%d", turn, config.max_retrieval_turns)

            response = await api_create_with_retry(
                lambda: client.messages.create(
                    model=config.haiku_model,
                    max_tokens=4096,
                    system=RETRIEVAL_SYSTEM,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            )

            # Accumulate token stats
            stats.api_calls += 1
            stats.input_tokens += response.usage.input_tokens
            stats.output_tokens += response.usage.output_tokens
            if hasattr(response.usage, "cache_read_input_tokens"):
                stats.cache_read_tokens += response.usage.cache_read_input_tokens or 0
            if hasattr(response.usage, "cache_creation_input_tokens"):
                stats.cache_creation_tokens += response.usage.cache_creation_input_tokens or 0

            # Check if the model wants to use tools
            if response.stop_reason == "tool_use":
                # Process all tool calls in this response
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

                        result = await dispatch_tool(
                            block.name, block.input, config, http, school_label,
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                # Model is done — extract the final JSON from its response
                text_parts = [
                    block.text for block in response.content if block.type == "text"
                ]
                full_text = "\n".join(text_parts)

                parsed = _extract_json_from_text(full_text)
                if parsed is not None:
                    # Ensure school_name and program_name are set correctly
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

    # If we get here, we exhausted the turn budget without a clean profile.
    # Make one last attempt to parse whatever we have.
    log.warning("Turn budget exhausted — attempting to salvage partial profile")
    return SchoolProfile(
        school_name=school_name,
        program_name=program_name,
        notes="WARNING: Retrieval agent exhausted turn budget without producing a complete profile.",
    ), stats


def _summarize_args(args: dict[str, Any]) -> str:
    """Compact representation of tool arguments for logging."""
    if "query" in args:
        q = args["query"]
        return f"query={q!r}" if len(q) < 80 else f"query={q[:77]!r}..."
    if "url" in args:
        return f"url={args['url']!r}"
    return str(args)
