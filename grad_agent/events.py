"""Pipeline event types for TUI progress tracking.

Emitted by the pipeline stages and consumed by the TUI (or ignored when
running without a TTY).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union


@dataclass
class SchoolStarted:
    """Emitted before a school's pipeline begins."""

    school: str
    idx: int    # 1-based position in the run
    total: int


@dataclass
class StageStarted:
    """Emitted at the start of each pipeline stage."""

    school: str
    stage: str  # "retrieval" | "judge+fit" | "gap_fill"


@dataclass
class TurnProgress:
    """Emitted at the start of each retrieval agent turn."""

    school: str
    turn: int
    max_turns: int


@dataclass
class ToolCalled:
    """Emitted when the retrieval agent issues a tool call."""

    school: str
    tool_name: str


@dataclass
class SchoolDone:
    """Emitted after a school's full pipeline completes."""

    school: str
    success: bool
    elapsed: float
    cost: float


PipelineEvent = Union[SchoolStarted, StageStarted, TurnProgress, ToolCalled, SchoolDone]
EventCallback = Callable[[PipelineEvent], None]
