"""Lightweight statistics collection for pipeline runs.

Tracks per-school and aggregate token usage, costs, timing, and stage outcomes.
Thread-safe via a lock since schools may run concurrently.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

# Approximate costs per million tokens (USD) as of 2025-04.
_COST_PER_M: dict[str, tuple[float, float]] = {
    # (input, output) per 1M tokens
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}

_DEFAULT_COST = (3.00, 15.00)  # fallback


@dataclass
class StageStats:
    """Token and timing stats for a single stage invocation."""

    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    api_calls: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0

    @property
    def estimated_cost_usd(self) -> float:
        input_rate, output_rate = _COST_PER_M.get(self.model, _DEFAULT_COST)
        effective_input = max(0, self.input_tokens - self.cache_read_tokens)
        cache_cost = self.cache_read_tokens * (input_rate * 0.1)  # 90% discount
        return (
            effective_input * input_rate + self.output_tokens * output_rate + cache_cost
        ) / 1_000_000


def add_usage(stats: StageStats, usage: Any) -> None:
    """Accumulate Anthropic token usage into stage stats."""
    stats.input_tokens += usage.input_tokens
    stats.output_tokens += usage.output_tokens
    stats.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
    stats.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0


@dataclass
class SchoolStats:
    """Aggregated stats for a single school's full pipeline run."""

    school: str
    stages: list[StageStats] = field(default_factory=list)
    success: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.stages)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.stages)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.estimated_cost_usd for s in self.stages)


class StatsCollector:
    """Process-level statistics aggregator across all schools in a run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._schools: list[SchoolStats] = []

    def add_school(self, stats: SchoolStats) -> None:
        """Append a completed school's stats. Thread-safe."""
        with self._lock:
            self._schools.append(stats)

    @property
    def schools(self) -> list[SchoolStats]:
        """Snapshot of all recorded school stats. Thread-safe."""
        with self._lock:
            return list(self._schools)

    @property
    def total_cost_usd(self) -> float:
        """Sum of estimated USD cost across all schools."""
        return sum(s.total_cost_usd for s in self.schools)

    @property
    def total_input_tokens(self) -> int:
        """Sum of input tokens consumed across all schools."""
        return sum(s.total_input_tokens for s in self.schools)

    @property
    def total_output_tokens(self) -> int:
        """Sum of output tokens produced across all schools."""
        return sum(s.total_output_tokens for s in self.schools)

    def summary(self) -> str:
        """Human-readable summary of all schools."""
        lines = ["=" * 70, "Pipeline Statistics", "=" * 70]
        for s in self.schools:
            status = "OK" if s.success else f"FAILED: {s.error}"
            lines.append(
                f"  {s.school:<40} {status:<20} "
                f"${s.total_cost_usd:.4f}  {s.elapsed_seconds:.1f}s"
            )
            for st in s.stages:
                lines.append(
                    f"    {st.stage:<20} in={st.input_tokens:>8} out={st.output_tokens:>6} "
                    f"calls={st.api_calls} tools={st.tool_calls} "
                    f"${st.estimated_cost_usd:.4f} {st.elapsed_seconds:.1f}s"
                )
        lines.append("-" * 70)
        lines.append(
            f"  Total: {len(self.schools)} schools | "
            f"${self.total_cost_usd:.4f} | "
            f"{self.total_input_tokens} in / {self.total_output_tokens} out tokens"
        )
        lines.append("=" * 70)
        return "\n".join(lines)


@contextmanager
def timed() -> Generator[list[float], None, None]:
    """Context manager that stores elapsed wall-clock seconds in a single-element list.

    Usage:
        with timed() as t:
            do_work()
        elapsed = t[0]
    """
    container: list[float] = [0.0]
    start = time.monotonic()
    try:
        yield container
    finally:
        container[0] = time.monotonic() - start
