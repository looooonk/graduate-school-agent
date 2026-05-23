"""Reporting: Markdown/PDF output and statistics collection."""

from grad_agent.reporting.markdown import render_school_markdown, render_summary_table
from grad_agent.reporting.pdf import render_markdown_pdf, render_markdown_tree
from grad_agent.reporting.stats import SchoolStats, StageStats, StatsCollector

__all__ = [
    "render_markdown_pdf",
    "render_markdown_tree",
    "render_school_markdown",
    "render_summary_table",
    "StageStats",
    "SchoolStats",
    "StatsCollector",
]
