"""Rich-based live TUI for pipeline progress display.

Renders three stacked panels that refresh at ~4 fps:
  - header: overall progress bar, cost, elapsed, and run topology
  - school table: one row per school with stage, worker turns, tools, time
  - log tail: the last 12 log records from the pipeline
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from grad_agent.events import (
    EventCallback,
    PipelineEvent,
    SchoolDone,
    SchoolStarted,
    StageStarted,
    ToolCalled,
    TurnProgress,
)

_LEVEL_STYLES: dict[str, str] = {
    "DEBUG":    "dim",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold red",
}

_STAGE_LABELS: dict[str, tuple[str, str]] = {
    "queued":    ("dim",      "queued"),
    "retrieval": ("cyan",     "retrieval"),
    "judge+fit": ("yellow",   "judge + fit"),
    "gap_fill":  ("magenta",  "gap-fill"),
    "done":      ("green",    "done ✓"),
    "failed":    ("bold red", "failed ✗"),
}

_WORKER_SHORT_LABELS: dict[str, str] = {
    "full":       "full",
    "admissions": "adm",
    "faculty":    "fac",
    "applicants": "app",
}


@dataclass(frozen=True)
class TUIRunSettings:
    retrieval_backend: str = ""
    retrieval_model: str = ""
    local_model_count: int = 0
    local_parallel_agents: int = 0
    local_max_parallel_tool_calls: int = 0
    local_endpoint_count: int = 0
    max_schools_parallel: int = 0
    max_sonnet_parallel: int = 0

    @classmethod
    def from_config(cls, config: Any) -> TUIRunSettings:
        return cls(
            retrieval_backend=getattr(config, "retrieval_backend", ""),
            retrieval_model=getattr(config, "retrieval_model", ""),
            local_model_count=getattr(config, "local_retrieval_model_count", 0),
            local_parallel_agents=getattr(config, "local_retrieval_parallel_agents", 0),
            local_max_parallel_tool_calls=getattr(
                config, "local_retrieval_max_parallel_tool_calls", 0
            ),
            local_endpoint_count=len(getattr(config, "local_retrieval_base_urls", ()) or ()),
            max_schools_parallel=getattr(config, "max_schools_parallel", 0),
            max_sonnet_parallel=getattr(config, "max_sonnet_parallel", 0),
        )

    def summary(self) -> str:
        parts: list[str] = []
        if self.retrieval_backend:
            model = f"/{self.retrieval_model}" if self.retrieval_model else ""
            parts.append(f"retrieval {self.retrieval_backend}{model}")
        if self.retrieval_backend == "local_qwen_vllm":
            parts.append(
                "local "
                f"{self.local_model_count} models, "
                f"{self.local_parallel_agents} agents/school, "
                f"{self.local_max_parallel_tool_calls} tools/turn, "
                f"{self.local_endpoint_count} endpoints"
            )
        concurrency: list[str] = []
        if self.max_schools_parallel:
            concurrency.append(f"{self.max_schools_parallel} schools")
        if self.max_sonnet_parallel:
            concurrency.append(f"{self.max_sonnet_parallel} Sonnet")
        if concurrency:
            parts.append("parallel " + ", ".join(concurrency))
        return " | ".join(parts)


@dataclass
class _WorkerProgress:
    turn: int = 0
    max_turns: int = 0
    tool_calls: int = 0
    stage: str = ""


@dataclass
class _SchoolRow:
    label: str
    idx: int
    stage: str = "queued"
    turn: int = 0
    max_turns: int = 0
    tool_calls: int = 0
    max_tool_batch: int = 1
    tool_names: Counter[str] = field(default_factory=Counter)
    workers: dict[str, _WorkerProgress] = field(default_factory=dict)
    done: bool = False
    success: bool = True
    elapsed: float = 0.0
    cost: float = 0.0
    _start: float = field(default_factory=time.monotonic)

    def current_elapsed(self) -> float:
        """Return wall-clock seconds since the school started, or the final elapsed time if done."""
        return self.elapsed if self.done else time.monotonic() - self._start


class _TUILogHandler(logging.Handler):
    """Routes log records into a bounded deque for the TUI log panel."""

    def __init__(self, buffer: deque[tuple[str, str, str]]) -> None:
        """
        Args:
            buffer: Shared deque to append (levelname, school, message) tuples into.
        """
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            school = getattr(record, "school", "[global]")
            self._buffer.append((record.levelname, school, record.getMessage()))
        except Exception:
            self.handleError(record)


class _Renderable:
    """Mutable object whose ``__rich__`` method is called on every Live refresh."""

    def __init__(self, total: int, settings: TUIRunSettings | None = None) -> None:
        """
        Args:
            total: Total number of schools in this run; used for the progress bar.
        """
        self.total = total
        self.settings = settings or TUIRunSettings()
        self.done_count = 0
        self.rows: dict[str, _SchoolRow] = {}
        self.log_buffer: deque[tuple[str, str, str]] = deque(maxlen=12)
        self._run_start = time.monotonic()

    def __rich__(self) -> Group:
        return Group(
            self._render_header(),
            self._render_table(),
            self._render_log(),
        )

    def _render_header(self) -> Panel:
        n, t = self.done_count, self.total
        bar_width = 34
        filled = int(bar_width * n / t) if t else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        total_cost = sum(r.cost for r in self.rows.values())
        elapsed = time.monotonic() - self._run_start

        txt = Text()
        txt.append(bar + "  ", style="green")
        txt.append(f"{n} / {t} schools", style="bold white")
        txt.append(f"    ${total_cost:.4f}    {elapsed:.0f}s elapsed", style="dim")
        settings_summary = self.settings.summary()
        if settings_summary:
            txt.append("\n" + settings_summary, style="dim")
        return Panel(txt, title="[bold]Graduate School Research Agent[/bold]", expand=True)

    def _render_table(self) -> Table:
        tbl = Table(
            expand=True,
            show_header=True,
            header_style="bold dim",
            box=None,
            padding=(0, 1),
        )
        tbl.add_column("School",  ratio=5, overflow="ellipsis", no_wrap=True)
        tbl.add_column("Stage",   ratio=2)
        tbl.add_column("Turns",   ratio=3, overflow="fold")
        tbl.add_column("Tools",   ratio=2, overflow="ellipsis")
        tbl.add_column("Time",    justify="right", ratio=1)
        tbl.add_column("Cost",    justify="right", ratio=1)

        for row in sorted(self.rows.values(), key=lambda r: r.idx):
            style, stage_label = _STAGE_LABELS.get(row.stage, ("", row.stage))
            turn_str = _format_worker_turns(row)
            tool_str = _format_tools(row)
            cost_str = f"${row.cost:.3f}" if row.done else "…"
            tbl.add_row(
                row.label,
                Text(stage_label, style=style),
                turn_str,
                tool_str,
                f"{row.current_elapsed():.0f}s",
                cost_str,
            )

        return tbl

    def _render_log(self) -> Panel:
        txt = Text(overflow="fold")
        if not self.log_buffer:
            txt.append("Waiting for events…", style="dim")
        else:
            for level, school, msg in self.log_buffer:
                txt.append(f"{school:<36}", style="dim")
                txt.append(f" {msg}\n", style=_LEVEL_STYLES.get(level, ""))
        return Panel(txt, title="Log", expand=True)

    def on_event(self, event: PipelineEvent) -> None:
        """Update internal row state in response to a pipeline event."""
        if isinstance(event, SchoolStarted):
            self.rows[event.school] = _SchoolRow(label=event.school, idx=event.idx)
        elif isinstance(event, StageStarted):
            if event.school in self.rows:
                self.rows[event.school].stage = event.stage
        elif isinstance(event, TurnProgress):
            if event.school in self.rows:
                r = self.rows[event.school]
                r.turn = event.turn
                r.max_turns = event.max_turns
                if event.stage:
                    r.stage = event.stage
                worker = _worker_key(event.worker)
                r.workers[worker] = _WorkerProgress(
                    turn=event.turn,
                    max_turns=event.max_turns,
                    tool_calls=r.workers.get(worker, _WorkerProgress()).tool_calls,
                    stage=event.stage,
                )
        elif isinstance(event, ToolCalled):
            if event.school in self.rows:
                r = self.rows[event.school]
                if event.stage:
                    r.stage = event.stage
                worker = _worker_key(event.worker)
                r.tool_calls += 1
                r.max_tool_batch = max(r.max_tool_batch, event.batch_size)
                r.tool_names[event.tool_name] += 1
                progress = r.workers.setdefault(worker, _WorkerProgress())
                progress.tool_calls += 1
                if event.stage:
                    progress.stage = event.stage
        elif isinstance(event, SchoolDone):
            if event.school in self.rows:
                r = self.rows[event.school]
                r.done = True
                r.success = event.success
                r.stage = "done" if event.success else "failed"
                r.elapsed = event.elapsed
                r.cost = event.cost
            self.done_count += 1


def _worker_key(worker: str) -> str:
    return worker or "main"


def _format_worker_turns(row: _SchoolRow) -> str:
    if not row.workers:
        return f"{row.turn}/{row.max_turns}" if row.max_turns else "—"

    workers = {
        name: worker
        for name, worker in row.workers.items()
        if worker.stage == row.stage
    } or row.workers

    if set(workers) == {"main"}:
        worker = workers["main"]
        return f"{worker.turn}/{worker.max_turns}" if worker.max_turns else "—"

    parts: list[str] = []
    for name, worker in sorted(workers.items(), key=lambda item: _worker_sort_key(item[0])):
        label = _WORKER_SHORT_LABELS.get(name, name[:4])
        value = f"{worker.turn}/{worker.max_turns}" if worker.max_turns else "—"
        parts.append(f"{label} {value}")
    return ", ".join(parts)


def _worker_sort_key(name: str) -> tuple[int, str]:
    order = ["full", "admissions", "faculty", "applicants", "main"]
    try:
        return order.index(name), name
    except ValueError:
        return len(order), name


def _format_tools(row: _SchoolRow) -> str:
    if not row.tool_calls:
        return "—"
    pieces = [str(row.tool_calls)]
    if row.max_tool_batch > 1:
        pieces.append(f"batch {row.max_tool_batch}")
    if row.tool_names:
        names = ", ".join(
            f"{name.replace('_', ' ')} {count}"
            for name, count in row.tool_names.most_common(2)
        )
        pieces.append(names)
    return " | ".join(pieces)


class PipelineTUI:
    """Manages the Rich live display and log capture for a pipeline run.

    Replaces the root logging handler so all pipeline log records appear in
    the TUI log panel instead of stderr.

    Usage::

        tui = PipelineTUI(total=len(schools))
        tui.start()
        try:
            await run_all_schools(..., on_event=tui.on_event)
        finally:
            tui.stop()
    """

    def __init__(
        self,
        total: int,
        config: Any | None = None,
        settings: TUIRunSettings | None = None,
    ) -> None:
        """
        Args:
            total: Total number of schools to be processed; drives the progress bar.
            config: Optional pipeline config shown in the run header.
            settings: Optional explicit run settings, mainly for demos and tests.
        """
        if config is not None and settings is None:
            settings = TUIRunSettings.from_config(config)
        self._renderable = _Renderable(total=total, settings=settings)
        self._live = Live(
            self._renderable,
            refresh_per_second=4,
            redirect_stderr=False,
            transient=False,
        )
        self._log_handler = _TUILogHandler(self._renderable.log_buffer)

    def start(self) -> None:
        """Install the log handler and start the live display."""
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(self._log_handler)
        self._live.start(refresh=True)

    def stop(self) -> None:
        """Stop the live display, leaving the final state visible on screen."""
        self._live.stop()

    @property
    def on_event(self) -> EventCallback:
        return self._renderable.on_event
