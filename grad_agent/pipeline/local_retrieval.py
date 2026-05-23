"""Local OpenAI-compatible retrieval loops."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from grad_agent.config import Config
from grad_agent.events import EventCallback, TurnProgress
from grad_agent.llm.vllm import LocalVLLMClient
from grad_agent.models import SchoolProfile
from grad_agent.pipeline.prompts import (
    LOCAL_RETRIEVAL_PROTOCOL,
    RETRIEVAL_SYSTEM,
    retrieval_turn_status,
)
from grad_agent.pipeline.tool_loop import ToolCommand, event_stage_parts, run_tool_commands
from grad_agent.reporting.stats import StageStats, add_usage, timed
from grad_agent.reporting.trajectory import TrajectoryLogger
from grad_agent.util.json import extract_json_object
from grad_agent.util.log import get_school_logger


async def run_local_parallel_profile_loop(
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
    school_label = f"{school_name} — {program_name}"
    log = get_school_logger(__name__, school_label)
    client = local_client or LocalVLLMClient.from_config(config)
    agent_prompts = _parallel_local_prompts(initial_prompt, config.local_retrieval_parallel_agents)
    aggregate = StageStats(stage=stage, model=config.retrieval_model)

    with timed() as elapsed:
        results = await asyncio.gather(
            *(
                run_local_profile_loop(
                    school_name=school_name,
                    program_name=program_name,
                    config=config,
                    http=http,
                    initial_prompt=prompt,
                    stage=f"{stage}:{label}",
                    max_turns=max_turns,
                    on_event=on_event,
                    traj=traj,
                    local_client=client,
                )
                for label, prompt in agent_prompts
            ),
            return_exceptions=True,
        )

    profiles: list[SchoolProfile] = []
    for item in results:
        if isinstance(item, Exception):
            log.warning("Parallel %s worker failed: %s", stage, item)
            continue
        profile, stats = item
        _accumulate_stats(aggregate, stats)
        if _profile_has_evidence(profile):
            profiles.append(profile)

    aggregate.elapsed_seconds = elapsed[0]
    if profiles:
        profile = _merge_profiles(school_name, program_name, profiles)
        log.info("Merged %d parallel %s profiles", len(profiles), stage)
        if traj:
            traj.log_profile(profile)
        return profile, aggregate

    log.warning("Parallel %s workers produced no usable profile", stage)
    return SchoolProfile(
        school_name=school_name,
        program_name=program_name,
        notes=(
            "WARNING: Parallel retrieval workers exhausted their turn budgets "
            "without producing a usable profile."
        ),
    ), aggregate


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
                base_stage, worker = event_stage_parts(stage)
                on_event(TurnProgress(
                    school=school_label,
                    turn=turn,
                    max_turns=max_turns,
                    stage=base_stage,
                    worker=worker,
                ))

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
            parsed = extract_json_object(full_text)
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
            commands = _local_tool_commands(
                parsed, config.local_retrieval_max_parallel_tool_calls
            )
            if tool_name or commands:
                if not commands:
                    log.warning("Invalid local tool command on turn %d", turn)
                    messages.append({"role": "assistant", "content": full_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your tool command was invalid. Output a JSON object "
                            "with tool and args, or tools as a list of commands."
                        ),
                    })
                    continue
                results = await run_tool_commands(
                    commands, config, http, school_label, log, stats, on_event, traj, stage, turn,
                )
                result_text = "\n\n".join(
                    f"Tool result for {command.name}:\n\n{result}"
                    for command, result in results
                )
                messages.append({"role": "assistant", "content": full_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"{result_text}\n\n"
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


def _local_tool_commands(parsed: dict[str, Any], max_count: int) -> list[ToolCommand]:
    tool_name = parsed.get("tool")
    if isinstance(tool_name, str):
        args = parsed.get("args")
        return [ToolCommand(tool_name, args if isinstance(args, dict) else {})]

    raw_tools = parsed.get("tools") or parsed.get("tool_calls") or parsed.get("commands")
    if not isinstance(raw_tools, list):
        return []

    commands: list[ToolCommand] = []
    for item in raw_tools[:max_count]:
        if not isinstance(item, dict):
            continue
        name = item.get("tool") or item.get("name")
        if not isinstance(name, str):
            continue
        args = item.get("args") or item.get("input") or {}
        commands.append(ToolCommand(name, args if isinstance(args, dict) else {}))
    return commands


def _parallel_local_prompts(initial_prompt: str, count: int) -> list[tuple[str, str]]:
    focuses = [
        ("full", "Build the strongest complete profile you can."),
        (
            "admissions",
            "Prioritize official admissions facts: deadlines, fee, GRE, GPA, SOP, "
            "recommendations, requirements, and essay prompts.",
        ),
        (
            "faculty",
            "Prioritize department research areas, labs, and advisor candidates "
            "that match the applicant context.",
        ),
        (
            "applicants",
            "Prioritize applicant reports, GradCafe or Reddit signals, acceptance "
            "patterns, and competitiveness evidence.",
        ),
    ]
    prompts: list[tuple[str, str]] = []
    for idx in range(count):
        label, focus = focuses[idx] if idx < len(focuses) else (f"full{idx + 1}", focuses[0][1])
        prompts.append((
            label,
            (
                f"{initial_prompt}\n\n"
                f"Parallel retrieval focus: {focus} Search independently. "
                f"Return a complete SchoolProfile JSON with any missing fields left "
                f"empty rather than guessed."
            ),
        ))
    return prompts


def _accumulate_stats(target: StageStats, source: StageStats) -> None:
    target.input_tokens += source.input_tokens
    target.output_tokens += source.output_tokens
    target.cache_read_tokens += source.cache_read_tokens
    target.cache_creation_tokens += source.cache_creation_tokens
    target.api_calls += source.api_calls
    target.tool_calls += source.tool_calls


def _profile_has_evidence(profile: SchoolProfile) -> bool:
    return any(
        [
            profile.deadline,
            profile.application_fee,
            profile.requirements.model_dump(exclude_none=True, exclude_defaults=True),
            profile.essay_prompts,
            profile.research_areas,
            profile.advisor_candidates,
            profile.applicant_reports.model_dump(exclude_none=True, exclude_defaults=True),
            profile.sources,
        ]
    )


def _merge_profiles(
    school_name: str,
    program_name: str,
    profiles: list[SchoolProfile],
) -> SchoolProfile:
    merged = SchoolProfile(school_name=school_name, program_name=program_name)
    merged.deadline = _first_text(profile.deadline for profile in profiles)
    merged.application_fee = _first_text(profile.application_fee for profile in profiles)
    merged.requirements.gre_required = _first_value(
        profile.requirements.gre_required for profile in profiles
    )
    merged.requirements.gre_policy = _first_value(
        profile.requirements.gre_policy for profile in profiles
    )
    merged.requirements.gpa_minimum = _first_text(
        profile.requirements.gpa_minimum for profile in profiles
    )
    merged.requirements.statement_of_purpose = _first_value(
        profile.requirements.statement_of_purpose for profile in profiles
    )
    merged.requirements.recommendations = _first_value(
        profile.requirements.recommendations for profile in profiles
    )
    merged.requirements.other = _unique(
        item for profile in profiles for item in profile.requirements.other
    )
    merged.essay_prompts = _unique(
        item for profile in profiles for item in profile.essay_prompts
    )
    merged.research_areas = _unique(
        item for profile in profiles for item in profile.research_areas
    )
    merged.advisor_candidates = _unique(
        item for profile in profiles for item in profile.advisor_candidates
    )
    merged.applicant_reports.typical_gpa = _first_text(
        profile.applicant_reports.typical_gpa for profile in profiles
    )
    merged.applicant_reports.typical_gre = _first_text(
        profile.applicant_reports.typical_gre for profile in profiles
    )
    merged.applicant_reports.acceptance_signals = _first_text(
        profile.applicant_reports.acceptance_signals for profile in profiles
    )
    merged.sources = _unique(item for profile in profiles for item in profile.sources)
    merged.notes = "\n".join(
        _unique(profile.notes for profile in profiles if profile.notes)
    ) or None
    return merged


def _first_text(values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_value(values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
