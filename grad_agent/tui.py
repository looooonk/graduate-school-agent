"""Rich-based live TUI for pipeline progress display.

Renders three stacked panels that refresh at ~4 fps:
  - header: overall progress bar (M / N schools, cost, elapsed)
  - school table: one row per school with stage, turn, tool-call count, time
  - log tail: the last 12 log records from the pipeline
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

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


@dataclass
class _SchoolRow:
    label: str
    idx: int
    stage: str = "queued"
    turn: int = 0
    max_turns: int = 0
    tool_calls: int = 0
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

    def __init__(self, total: int) -> None:
        """
        Args:
            total: Total number of schools in this run; used for the progress bar.
        """
        self.total = total
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
        tbl.add_column("Turn",    justify="right", ratio=1)
        tbl.add_column("Tools",   justify="right", ratio=1)
        tbl.add_column("Time",    justify="right", ratio=1)
        tbl.add_column("Cost",    justify="right", ratio=1)

        for row in sorted(self.rows.values(), key=lambda r: r.idx):
            style, stage_label = _STAGE_LABELS.get(row.stage, ("", row.stage))
            turn_str = f"{row.turn}/{row.max_turns}" if row.max_turns else "—"
            tool_str = str(row.tool_calls) if row.tool_calls else "—"
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
        elif isinstance(event, ToolCalled):
            if event.school in self.rows:
                self.rows[event.school].tool_calls += 1
        elif isinstance(event, SchoolDone):
            if event.school in self.rows:
                r = self.rows[event.school]
                r.done = True
                r.success = event.success
                r.stage = "done" if event.success else "failed"
                r.elapsed = event.elapsed
                r.cost = event.cost
            self.done_count += 1


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

    def __init__(self, total: int) -> None:
        """
        Args:
            total: Total number of schools to be processed; drives the progress bar.
        """
        self._renderable = _Renderable(total=total)
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
