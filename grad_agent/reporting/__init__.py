"""Reporting: Markdown output and statistics collection."""

from grad_agent.reporting.markdown import render_school_markdown, render_summary_table
from grad_agent.reporting.stats import SchoolStats, StageStats, StatsCollector

__all__ = [
    "render_school_markdown",
    "render_summary_table",
    "StageStats",
    "SchoolStats",
    "StatsCollector",
]
