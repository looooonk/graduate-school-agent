"""Concrete retrieval backend implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import httpx

from grad_agent.config import Config
from grad_agent.events import EventCallback, TurnProgress
from grad_agent.llm.vllm import LocalVLLMClient
from grad_agent.models import SchoolProfile
from grad_agent.pipeline.local_retrieval import (
    run_local_parallel_profile_loop,
    run_local_profile_loop,
)
from grad_agent.pipeline.prompts import RETRIEVAL_SYSTEM, retrieval_turn_status
from grad_agent.pipeline.tool_loop import ToolCommand, run_tool_commands
from grad_agent.pipeline.tools import TOOL_DEFINITIONS
from grad_agent.reporting.stats import StageStats, add_usage, timed
from grad_agent.reporting.trajectory import TrajectoryLogger
from grad_agent.retrieval_registry import RETRIEVAL_BACKENDS
from grad_agent.util.json import extract_json_object
from grad_agent.util.log import get_school_logger
from grad_agent.util.retry import api_create_with_retry


@dataclass(frozen=True)
class RetrievalRequest:
    school_name: str
    program_name: str
    initial_prompt: str
    stage: str
    max_turns: int
    config: Config
    anthropic_client: anthropic.AsyncAnthropic
    http: httpx.AsyncClient
    on_event: EventCallback | None = None
    traj: TrajectoryLogger | None = None
    local_client: LocalVLLMClient | None = None
    fallback_profile: SchoolProfile | None = None

    @property
    def school_label(self) -> str:
        return f"{self.school_name} — {self.program_name}"


class RetrievalBackend(Protocol):
    async def run(self, request: RetrievalRequest) -> tuple[SchoolProfile, StageStats]:
        ...


class LocalOpenAICompatibleBackend:
    async def run(self, request: RetrievalRequest) -> tuple[SchoolProfile, StageStats]:
        kwargs = {
            "school_name": request.school_name,
            "program_name": request.program_name,
            "config": request.config,
            "http": request.http,
            "initial_prompt": request.initial_prompt,
            "stage": request.stage,
            "max_turns": request.max_turns,
            "on_event": request.on_event,
            "traj": request.traj,
            "local_client": request.local_client,
        }
        if request.config.local_retrieval_parallel_agents > 1:
            return await run_local_parallel_profile_loop(**kwargs)
        return await run_local_profile_loop(**kwargs)


class AnthropicToolUseBackend:
    async def run(self, request: RetrievalRequest) -> tuple[SchoolProfile, StageStats]:
        config = request.config
        log = get_school_logger(__name__, request.school_label)
        stats = StageStats(stage=request.stage, model=config.retrieval_model)

        if request.traj:
            request.traj.log_stage_start(request.stage)

        messages: list[dict[str, Any]] = [{"role": "user", "content": request.initial_prompt}]

        with timed() as elapsed:
            for turn in range(1, request.max_turns + 1):
                log.info("%s turn %d/%d", request.stage, turn, request.max_turns)
                if request.on_event:
                    request.on_event(
                        TurnProgress(
                            school=request.school_label,
                            turn=turn,
                            max_turns=request.max_turns,
                            stage=request.stage,
                        )
                    )

                response = await api_create_with_retry(
                    lambda: request.anthropic_client.messages.create(
                        model=config.retrieval_model,
                        max_tokens=4096,
                        system=RETRIEVAL_SYSTEM,
                        tools=TOOL_DEFINITIONS,
                        messages=messages,
                    )
                )

                stats.api_calls += 1
                add_usage(stats, response.usage)

                if request.traj:
                    request.traj.log_api_response(
                        request.stage,
                        turn,
                        config.retrieval_model,
                        response,
                    )

                if response.stop_reason == "tool_use":
                    assistant_content, commands = _anthropic_tool_commands(response.content)
                    results = await run_tool_commands(
                        commands,
                        config,
                        request.http,
                        request.school_label,
                        log,
                        stats,
                        request.on_event,
                        request.traj,
                        request.stage,
                        turn,
                    )
                    tool_results: list[dict[str, Any]] = []
                    for command, result in results:
                        if command.tool_use_id is None:
                            continue
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": command.tool_use_id,
                            "content": result,
                        })

                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({
                        "role": "user",
                        "content": [
                            *tool_results,
                            {
                                "type": "text",
                                "text": retrieval_turn_status(turn, request.max_turns),
                            },
                        ],
                    })
                    continue

                if response.stop_reason == "end_turn":
                    parsed = _profile_json_from_response(response.content)
                    if parsed is not None:
                        parsed["school_name"] = request.school_name
                        parsed["program_name"] = request.program_name
                        try:
                            profile = SchoolProfile.model_validate(parsed)
                            stats.elapsed_seconds = elapsed[0]
                            log.info(
                                "Profile complete — %d sources, %d research areas, %d advisors",
                                len(profile.sources),
                                len(profile.research_areas),
                                len(profile.advisor_candidates),
                            )
                            if request.traj:
                                request.traj.log_profile(profile)
                                request.traj.log_stage_end(request.stage, elapsed[0])
                            return profile, stats
                        except Exception as exc:
                            log.warning("Profile validation failed: %s", exc)
                            messages.append({
                                "role": "assistant",
                                "content": _text_from_blocks(response.content),
                            })
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Your SchoolProfile JSON had a validation error: {exc}\n"
                                    "Fix it and output ONLY the complete SchoolProfile JSON."
                                ),
                            })
                            continue

                    log.warning("No JSON found in model output (turn %d)", turn)
                    messages.append({
                        "role": "assistant",
                        "content": _text_from_blocks(response.content),
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "I could not parse a JSON object from your response. "
                            "Please output the complete SchoolProfile as a single JSON object."
                        ),
                    })
                    continue

                log.warning("Unexpected stop reason: %s", response.stop_reason)
                break

        stats.elapsed_seconds = elapsed[0]
        if request.traj:
            request.traj.log_stage_end(request.stage, elapsed[0])
        if request.fallback_profile is not None:
            log.warning("Turn budget exhausted — returning fallback profile")
            return request.fallback_profile, stats
        log.warning("Turn budget exhausted — returning stub profile")
        return SchoolProfile(
            school_name=request.school_name,
            program_name=request.program_name,
            notes=(
                "WARNING: Retrieval agent exhausted turn budget without producing "
                "a complete profile."
            ),
        ), stats


_BACKEND_IMPLEMENTATIONS: dict[str, RetrievalBackend] = {
    "local_qwen_vllm": LocalOpenAICompatibleBackend(),
    "anthropic_haiku": AnthropicToolUseBackend(),
}


def get_retrieval_backend(backend_id: str) -> RetrievalBackend:
    try:
        return _BACKEND_IMPLEMENTATIONS[backend_id]
    except KeyError as exc:
        allowed = ", ".join(sorted(RETRIEVAL_BACKENDS))
        raise ValueError(
            f"Unsupported retrieval backend {backend_id!r}. Expected one of: {allowed}"
        ) from exc


def _anthropic_tool_commands(blocks: list[Any]) -> tuple[list[dict[str, Any]], list[ToolCommand]]:
    assistant_content: list[dict[str, Any]] = []
    commands: list[ToolCommand] = []
    for block in blocks:
        if block.type == "text":
            assistant_content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            assistant_content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
            args = block.input if isinstance(block.input, dict) else {}
            commands.append(ToolCommand(block.name, args, block.id))
    return assistant_content, commands


def _profile_json_from_response(blocks: list[Any]) -> dict[str, Any] | None:
    return extract_json_object(_text_from_blocks(blocks))


def _text_from_blocks(blocks: list[Any]) -> str:
    return "\n".join(block.text for block in blocks if block.type == "text")
