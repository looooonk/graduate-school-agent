from __future__ import annotations

import io
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from grad_agent.config import Config
from grad_agent.events import (
    PipelineEvent,
    SchoolDone,
    SchoolStarted,
    StageStarted,
    ToolCalled,
    TurnProgress,
)
from grad_agent.tui import PipelineTUI, TUIRunSettings, _Renderable

_LOG_MESSAGES: tuple[tuple[str, str], ...] = (
    ("INFO", "Fetched admissions page and program requirements"),
    ("INFO", "Fetched faculty directory and research group pages"),
    ("INFO", "Parsed applicant outcome notes"),
    ("INFO", "Merged worker notes into a school profile"),
    ("WARNING", "Deadline looked incomplete; queued targeted gap-fill search"),
    ("ERROR", "Transient fetch failure; retry will continue"),
)

_TOOL_NAMES = ("web_search", "fetch_page")
_WORKERS = ("full", "admissions", "faculty", "applicants")


@dataclass(frozen=True)
class DemoStep:
    event: PipelineEvent
    log: tuple[str, str, str] | None = None
    delay: float = 0.0


@dataclass
class _SchoolSim:
    label: str
    idx: int
    target_turns: int
    gap_target_turns: int
    turn: int = 0
    gap_turn: int = 0
    stage: str = "retrieval"
    elapsed: float = 0.0


def load_preview_config(config_path: str | Path | None = None) -> Config:
    return Config.load(config_path)


def load_configured_schools(config: Config) -> list[tuple[str, str]]:
    data = _load_schools_json(Path(config.schools_path))
    schools: list[tuple[str, str]] = []
    for entry in data:
        school = entry.get("school") if isinstance(entry, dict) else None
        program = entry.get("program") if isinstance(entry, dict) else None
        if not school or not program:
            raise ValueError(f"Invalid school entry in {config.schools_path}: {entry}")
        schools.append((school, program))
    if not schools:
        raise ValueError(f"No schools found in {config.schools_path}")
    return schools


def generate_demo_steps(
    schools: list[tuple[str, str]],
    config: Config,
    *,
    seed: int = 7,
    frame_delay: float = 0.35,
) -> tuple[DemoStep, ...]:
    rng = random.Random(seed)
    labels = [f"{school} - {program}" for school, program in schools]
    max_parallel = min(len(labels), max(1, config.max_schools_parallel))
    workers = _workers_for_config(config)
    pending = list(enumerate(labels, start=1))
    active: list[_SchoolSim] = []
    steps: list[DemoStep] = []

    def start_next() -> None:
        idx, label = pending.pop(0)
        state = _SchoolSim(
            label=label,
            idx=idx,
            target_turns=_target_turns(rng, config.max_retrieval_turns),
            gap_target_turns=_target_turns(rng, config.gap_fill_max_turns),
        )
        active.append(state)
        steps.extend(
            (
                DemoStep(
                    SchoolStarted(school=label, idx=idx, total=len(labels)),
                    _log(rng, label, include_warning=False),
                    _latency(rng, frame_delay),
                ),
                DemoStep(
                    StageStarted(school=label, stage="retrieval"),
                    _log(rng, label, include_warning=False),
                    _latency(rng, frame_delay),
                ),
            )
        )

    while pending and len(active) < max_parallel:
        start_next()

    while active:
        state = rng.choice(active)
        if state.stage == "retrieval":
            steps.extend(_retrieval_steps(state, config, rng, workers, frame_delay))
            if state.turn >= state.target_turns:
                state.stage = "judge+fit"
                steps.append(
                    DemoStep(
                        StageStarted(school=state.label, stage="judge+fit"),
                        _log(rng, state.label, include_warning=False),
                        _latency(rng, frame_delay),
                    )
                )
        elif state.stage == "judge+fit":
            if config.retry_gap_fill and (state.idx % 3 == 0 or rng.random() < 0.25):
                state.stage = "gap_fill"
                steps.append(
                    DemoStep(
                        StageStarted(school=state.label, stage="gap_fill"),
                        ("WARNING", state.label, _LOG_MESSAGES[4][1]),
                        _latency(rng, frame_delay),
                    )
                )
            else:
                _finish_school(active, pending, steps, state, config, rng, frame_delay, start_next)
        elif state.stage == "gap_fill":
            state.gap_turn += 1
            steps.append(
                DemoStep(
                    TurnProgress(
                        school=state.label,
                        turn=state.gap_turn,
                        max_turns=config.gap_fill_max_turns,
                        stage="gap_fill",
                    ),
                    _log(rng, state.label, include_warning=True),
                    _latency(rng, frame_delay),
                )
            )
            if rng.random() < 0.65:
                steps.append(_tool_step(state.label, "gap_fill", "main", config, rng, frame_delay))
            if state.gap_turn >= state.gap_target_turns:
                _finish_school(active, pending, steps, state, config, rng, frame_delay, start_next)

    return tuple(steps)


def build_demo_renderable(
    config: Config,
    schools: list[tuple[str, str]],
    *,
    seed: int = 7,
    max_steps: int = 80,
) -> _Renderable:
    renderable = _Renderable(
        total=len(schools),
        settings=TUIRunSettings.from_config(config),
    )
    active_elapsed: dict[str, float] = {}
    active_labels: set[str] = set()
    for step in generate_demo_steps(schools, config, seed=seed)[:max_steps]:
        renderable.on_event(step.event)
        if step.log:
            renderable.log_buffer.append(step.log)
        label = getattr(step.event, "school", "")
        if isinstance(step.event, SchoolStarted):
            active_labels.add(label)
            active_elapsed[label] = 0.0
        elif isinstance(step.event, SchoolDone):
            active_labels.discard(label)
        for active_label in active_labels:
            active_elapsed[active_label] = active_elapsed.get(active_label, 0.0) + step.delay

    now = time.monotonic()
    for label, elapsed in active_elapsed.items():
        if label in renderable.rows and label in active_labels:
            renderable.rows[label]._start = now - elapsed
    return renderable


def render_demo_snapshot(
    *,
    config: Config | None = None,
    schools: list[tuple[str, str]] | None = None,
    config_path: str | Path | None = None,
    width: int = 120,
    seed: int = 7,
    max_steps: int = 80,
) -> str:
    config = config or load_preview_config(config_path)
    schools = schools or load_configured_schools(config)
    console = Console(width=width, record=True, color_system=None, file=io.StringIO())
    console.print(build_demo_renderable(config, schools, seed=seed, max_steps=max_steps))
    return console.export_text()


def run_demo_tui(
    *,
    config_path: str | Path | None = None,
    seed: int = 7,
    frame_delay: float = 0.35,
    hold_seconds: float = 5.0,
) -> None:
    config = load_preview_config(config_path)
    schools = load_configured_schools(config)
    tui = PipelineTUI(total=len(schools), settings=TUIRunSettings.from_config(config))
    tui.start()
    try:
        for step in generate_demo_steps(
            schools,
            config,
            seed=seed,
            frame_delay=frame_delay,
        ):
            tui.on_event(step.event)
            if step.log:
                tui._renderable.log_buffer.append(step.log)
            time.sleep(step.delay)
        time.sleep(hold_seconds)
    finally:
        tui.stop()


def _load_schools_json(path: Path) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"schools file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"schools JSON must be a list: {path}")
    return data


def _target_turns(rng: random.Random, max_turns: int) -> int:
    high = max(1, min(max_turns, 8))
    low = min(3, high)
    return rng.randint(low, high)


def _retrieval_steps(
    state: _SchoolSim,
    config: Config,
    rng: random.Random,
    workers: tuple[str, ...],
    frame_delay: float,
) -> list[DemoStep]:
    state.turn += 1
    worker = rng.choice(workers)
    steps = [
        DemoStep(
            TurnProgress(
                school=state.label,
                turn=state.turn,
                max_turns=config.max_retrieval_turns,
                stage="retrieval",
                worker=worker,
            ),
            _log(rng, state.label, include_warning=True),
            _latency(rng, frame_delay),
        )
    ]
    if rng.random() < 0.85:
        steps.append(_tool_step(state.label, "retrieval", worker, config, rng, frame_delay))
    return steps


def _tool_step(
    label: str,
    stage: str,
    worker: str,
    config: Config,
    rng: random.Random,
    frame_delay: float,
) -> DemoStep:
    return DemoStep(
        ToolCalled(
            school=label,
            tool_name=rng.choice(_TOOL_NAMES),
            stage=stage,
            worker=worker,
            batch_size=rng.randint(1, max(1, config.local_retrieval_max_parallel_tool_calls)),
        ),
        _log(rng, label, include_warning=True),
        _latency(rng, frame_delay),
    )


def _finish_school(
    active: list[_SchoolSim],
    pending: list[tuple[int, str]],
    steps: list[DemoStep],
    state: _SchoolSim,
    config: Config,
    rng: random.Random,
    frame_delay: float,
    start_next: Callable[[], None],
) -> None:
    active.remove(state)
    state.elapsed += rng.uniform(20.0, 240.0)
    steps.append(
        DemoStep(
            SchoolDone(
                school=state.label,
                success=rng.random() > 0.05,
                elapsed=state.elapsed,
                cost=rng.uniform(0.01, 0.09),
            ),
            _log(rng, state.label, include_warning=True),
            _latency(rng, frame_delay),
        )
    )
    if pending and len(active) < max(1, config.max_schools_parallel):
        start_next()


def _workers_for_config(config: Config) -> tuple[str, ...]:
    if not config.uses_local_retrieval:
        return ("main",)
    count = max(1, min(config.local_retrieval_parallel_agents, len(_WORKERS)))
    return _WORKERS[:count]


def _log(
    rng: random.Random,
    label: str,
    *,
    include_warning: bool,
) -> tuple[str, str, str]:
    choices = _LOG_MESSAGES if include_warning else _LOG_MESSAGES[:4]
    level, message = rng.choice(choices)
    return level, label, message


def _latency(rng: random.Random, frame_delay: float) -> float:
    return max(0.0, frame_delay * rng.uniform(0.35, 1.8))
