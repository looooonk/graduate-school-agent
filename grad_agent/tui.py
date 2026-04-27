"""Rich-based live TUI for pipeline progress display.

Renders three stacked panels that refresh at ~4 fps:
  - header: overall progress bar (M / N schools, cost, elapsed)
  - school table: one row per school with stage, turn, tool-call count, time
  - log tail: the last 12 log records from the pipeline
"""

from __future__ import annotations

import io
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
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

_DEMO_SCHOOLS: tuple[tuple[str, str], ...] = (
    ("Stanford University", "MS Computer Science"),
    ("MIT", "PhD Electrical Engineering and Computer Science"),
    ("Carnegie Mellon University", "PhD Machine Learning"),
    ("University of Washington", "MS Human Centered Design and Engineering"),
)

_DEMO_LOGS: tuple[tuple[str, str, str], ...] = (
    (
        "INFO",
        "Stanford University - MS Computer Science",
        "Fetched admissions requirements and application fee",
    ),
    (
        "INFO",
        "Stanford University - MS Computer Science",
        "Judged profile as complete with high source confidence",
    ),
    (
        "WARNING",
        "MIT - PhD Electrical Engineering and Computer Science",
        "Deadline looked stale; queued targeted gap-fill search",
    ),
    (
        "INFO",
        "Carnegie Mellon University - PhD Machine Learning",
        "Fetched faculty directory and research group pages",
    ),
    (
        "ERROR",
        "University of Washington - MS Human Centered Design and Engineering",
        "Transient fetch failure; retry will continue",
    ),
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


def demo_tui_events() -> tuple[PipelineEvent, ...]:
    """Return fake pipeline events for a no-token TUI preview."""
    labels = [f"{school} - {program}" for school, program in _DEMO_SCHOOLS]
    return (
        SchoolStarted(school=labels[0], idx=1, total=len(labels)),
        StageStarted(school=labels[0], stage="retrieval"),
        TurnProgress(school=labels[0], turn=8, max_turns=25),
        ToolCalled(school=labels[0], tool_name="web_search"),
        ToolCalled(school=labels[0], tool_name="fetch_page"),
        ToolCalled(school=labels[0], tool_name="fetch_page"),
        StageStarted(school=labels[0], stage="judge+fit"),
        SchoolDone(school=labels[0], success=True, elapsed=193.0, cost=0.0362),
        SchoolStarted(school=labels[1], idx=2, total=len(labels)),
        StageStarted(school=labels[1], stage="retrieval"),
        TurnProgress(school=labels[1], turn=25, max_turns=25),
        ToolCalled(school=labels[1], tool_name="web_search"),
        ToolCalled(school=labels[1], tool_name="fetch_page"),
        StageStarted(school=labels[1], stage="gap_fill"),
        TurnProgress(school=labels[1], turn=3, max_turns=5),
        SchoolStarted(school=labels[2], idx=3, total=len(labels)),
        StageStarted(school=labels[2], stage="retrieval"),
        TurnProgress(school=labels[2], turn=14, max_turns=25),
        ToolCalled(school=labels[2], tool_name="web_search"),
        ToolCalled(school=labels[2], tool_name="fetch_page"),
        ToolCalled(school=labels[2], tool_name="fetch_page"),
        ToolCalled(school=labels[2], tool_name="fetch_page"),
        SchoolStarted(school=labels[3], idx=4, total=len(labels)),
    )


def build_demo_renderable() -> _Renderable:
    """Build a filled TUI state for visual checks and snapshot tests."""
    renderable = _Renderable(total=len(_DEMO_SCHOOLS))
    for event in demo_tui_events():
        renderable.on_event(event)

    now = time.monotonic()
    elapsed_by_label = {
        f"{_DEMO_SCHOOLS[1][0]} - {_DEMO_SCHOOLS[1][1]}": 242.0,
        f"{_DEMO_SCHOOLS[2][0]} - {_DEMO_SCHOOLS[2][1]}": 74.0,
        f"{_DEMO_SCHOOLS[3][0]} - {_DEMO_SCHOOLS[3][1]}": 0.0,
    }
    for label, elapsed in elapsed_by_label.items():
        if label in renderable.rows:
            renderable.rows[label]._start = now - elapsed

    renderable.log_buffer.extend(_DEMO_LOGS)
    return renderable


def render_demo_snapshot(width: int = 120) -> str:
    """Render the fake TUI state as plain text for regression tests."""
    console = Console(width=width, record=True, color_system=None, file=io.StringIO())
    console.print(build_demo_renderable())
    return console.export_text()


def run_demo_tui(frame_delay: float = 0.35, hold_seconds: float = 5.0) -> None:
    """Play fake events through the live TUI without API calls."""
    tui = PipelineTUI(total=len(_DEMO_SCHOOLS))
    tui.start()
    try:
        for idx, event in enumerate(demo_tui_events()):
            tui.on_event(event)
            if idx < len(_DEMO_LOGS):
                tui._renderable.log_buffer.append(_DEMO_LOGS[idx])
            time.sleep(frame_delay)
        for log_record in _DEMO_LOGS[len(demo_tui_events()):]:
            tui._renderable.log_buffer.append(log_record)
            time.sleep(frame_delay)
        time.sleep(hold_seconds)
    finally:
        tui.stop()


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
