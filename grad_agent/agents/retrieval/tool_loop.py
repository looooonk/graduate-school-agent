"""Shared helpers for retrieval tool loops."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from grad_agent.agents.retrieval.tools import dispatch_tool
from grad_agent.config import Config
from grad_agent.events import EventCallback, ToolCalled
from grad_agent.reporting.stats import StageStats
from grad_agent.reporting.trajectory import TrajectoryLogger


@dataclass(frozen=True)
class ToolCommand:
    name: str
    args: dict[str, Any]
    tool_use_id: str | None = None


def summarize_args(args: dict[str, Any]) -> str:
    """Compact representation of tool arguments for logging."""
    if "query" in args:
        q = args["query"]
        return f"query={q!r}" if len(q) < 80 else f"query={q[:77]!r}..."
    if "url" in args:
        return f"url={args['url']!r}"
    return str(args)


async def run_tool_commands(
    commands: list[ToolCommand],
    config: Config,
    http: httpx.AsyncClient,
    school_label: str,
    log: logging.LoggerAdapter,
    stats: StageStats,
    on_event: EventCallback | None,
    traj: TrajectoryLogger | None,
    stage: str,
    turn: int,
) -> list[tuple[ToolCommand, str]]:
    if not commands:
        return []
    for command in commands:
        stats.tool_calls += 1
        log.info("Tool call: %s(%s)", command.name, summarize_args(command.args))
        if on_event:
            base_stage, worker = event_stage_parts(stage)
            on_event(
                ToolCalled(
                    school=school_label,
                    tool_name=command.name,
                    stage=base_stage,
                    worker=worker,
                    batch_size=len(commands),
                )
            )

    raw_results = await asyncio.gather(
        *(
            dispatch_tool(command.name, command.args, config, http, school_label)
            for command in commands
        ),
        return_exceptions=True,
    )
    results: list[tuple[ToolCommand, str]] = []
    for command, raw_result in zip(commands, raw_results, strict=True):
        result = f"Tool failed: {raw_result}" if isinstance(raw_result, Exception) else raw_result
        if traj:
            traj.log_tool_result(stage, turn, command.name, command.args, result)
        results.append((command, result))
    return results


def event_stage_parts(stage: str) -> tuple[str, str]:
    if ":" not in stage:
        return stage, ""
    base, worker = stage.split(":", 1)
    return base, worker
