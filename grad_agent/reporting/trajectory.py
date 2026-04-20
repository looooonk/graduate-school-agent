"""Per-school trajectory logger.

Writes a JSONL file capturing the full agent trajectory for one school.
Each line is a self-contained JSON object with an ISO-8601 timestamp and
a ``type`` discriminator.  Event types:

  stage_start     — a pipeline stage began
  stage_end       — a pipeline stage completed (includes elapsed seconds)
  api_response    — model response received (content, token counts, stop_reason)
  tool_result     — tool executed (name, input, full result string)
  profile         — SchoolProfile produced by retrieval (or gap-fill)
  judge_report    — JudgeReport from the judge stage
  fit_assessment  — FitAssessment from the fit stage
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from grad_agent.models import FitAssessment, JudgeReport, SchoolProfile


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _serialize_content(content: list[Any]) -> list[dict[str, Any]]:
    """Convert Anthropic response content blocks to plain dicts for JSONL serialization.

    Args:
        content: List of Anthropic SDK content block objects (TextBlock, ToolUseBlock, etc.).

    Returns:
        List of plain dicts with a ``type`` key; unknown block types are reduced to
        ``{"type": <type_str>}`` so the record is never lost.
    """
    out: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            out.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            out.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        else:
            out.append({"type": block.type})
    return out


class TrajectoryLogger:
    """JSONL trajectory writer for a single school's pipeline run.

    Designed to be used as a context manager::

        with TrajectoryLogger(path) as traj:
            await run_school(..., traj=traj)

    The parent directory is created automatically on ``__enter__``.
    Flushing is synchronous and per-line so partial runs are still readable.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: Any = None

    def __enter__(self) -> TrajectoryLogger:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _write(self, obj: dict[str, Any]) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._file.flush()

    # ── Stage lifecycle ───────────────────────────────────────────────────────

    def log_stage_start(self, stage: str) -> None:
        """Record the beginning of a pipeline stage."""
        self._write({"ts": _now(), "type": "stage_start", "stage": stage})

    def log_stage_end(self, stage: str, elapsed_s: float) -> None:
        """Record the end of a pipeline stage with wall-clock elapsed time."""
        self._write({
            "ts": _now(),
            "type": "stage_end",
            "stage": stage,
            "elapsed_s": round(elapsed_s, 3),
        })

    # ── API interaction ───────────────────────────────────────────────────────

    def log_api_response(
        self,
        stage: str,
        turn: int,
        model: str,
        response: Any,
    ) -> None:
        """Record a model API response (content, token counts, stop reason)."""
        usage = response.usage
        self._write({
            "ts": _now(),
            "type": "api_response",
            "stage": stage,
            "turn": turn,
            "model": model,
            "stop_reason": response.stop_reason,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "content": _serialize_content(response.content),
        })

    def log_tool_result(
        self,
        stage: str,
        turn: int,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
    ) -> None:
        """Record a tool call and its full result string."""
        self._write({
            "ts": _now(),
            "type": "tool_result",
            "stage": stage,
            "turn": turn,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "result": result,
        })

    # ── Structured outputs ────────────────────────────────────────────────────

    def log_profile(self, profile: SchoolProfile) -> None:
        """Record the final SchoolProfile produced by retrieval or gap-fill."""
        self._write({"ts": _now(), "type": "profile", "data": profile.model_dump()})

    def log_judge_report(self, report: JudgeReport) -> None:
        """Record the JudgeReport from the judge stage."""
        self._write({"ts": _now(), "type": "judge_report", "data": report.model_dump()})

    def log_fit_assessment(self, assessment: FitAssessment) -> None:
        """Record the FitAssessment from the fit stage."""
        self._write({"ts": _now(), "type": "fit_assessment", "data": assessment.model_dump()})
