"""Per-school pipeline runner.

Orchestrates Stage 1 → 2 → 3 for a single school with fault tolerance,
and provides the top-level parallel launcher for multiple schools.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import anthropic
import httpx

from grad_agent.config import Config
from grad_agent.models import (
    FitAssessment,
    JudgeReport,
    QualityRating,
    SchoolProfile,
    SchoolResult,
)
from grad_agent.pipeline.fit import run_fit_assessment
from grad_agent.pipeline.judge import run_judge
from grad_agent.pipeline.retrieval import run_retrieval
from grad_agent.util.retry import api_create_with_retry
from grad_agent.reporting.markdown import render_school_markdown, render_summary_table
from grad_agent.reporting.stats import SchoolStats, StageStats, StatsCollector, timed
from grad_agent.util.log import get_school_logger

logger = logging.getLogger(__name__)


async def run_school(
    school_name: str,
    program_name: str,
    cv_text: str,
    config: Config,
    client: anthropic.AsyncAnthropic,
    http: httpx.AsyncClient,
) -> tuple[SchoolResult, SchoolStats]:
    """Run the full 3-stage pipeline for a single school.

    Never raises — all errors are captured in SchoolResult.error and SchoolStats.
    """
    school_label = f"{school_name} — {program_name}"
    log = get_school_logger(__name__, school_label)
    school_stats = SchoolStats(school=school_label)

    log.info("Starting pipeline")

    with timed() as total_elapsed:
        # --- Stage 1: Retrieval ---
        try:
            profile, retrieval_stats = await run_retrieval(
                school_name, program_name, config, client, http,
            )
            school_stats.stages.append(retrieval_stats)
            log.info("Retrieval complete")
        except Exception as exc:
            log.error("Retrieval failed: %s", exc)
            school_stats.elapsed_seconds = total_elapsed[0]
            school_stats.error = f"Retrieval failed: {exc}"
            return SchoolResult(
                profile=SchoolProfile(school_name=school_name, program_name=program_name),
                error=str(exc),
            ), school_stats

        # --- Stage 2 & 3: Judge and Fit run concurrently ---
        judge_report: JudgeReport | None = None
        judge_stats: StageStats | None = None
        fit_assessment: FitAssessment | None = None
        fit_stats: StageStats | None = None

        async def _run_judge() -> None:
            nonlocal judge_report, judge_stats
            try:
                judge_report, judge_stats = await run_judge(profile, config, client)
            except Exception as exc:
                log.error("Judge failed: %s", exc)
                judge_stats = StageStats(stage="judge", model=config.sonnet_model)

        async def _run_fit() -> None:
            nonlocal fit_assessment, fit_stats
            try:
                fit_assessment, fit_stats = await run_fit_assessment(
                    cv_text, profile, config, client,
                )
            except Exception as exc:
                log.error("Fit assessment failed: %s", exc)
                fit_stats = StageStats(stage="fit", model=config.sonnet_model)

        await asyncio.gather(_run_judge(), _run_fit())

        if judge_stats:
            school_stats.stages.append(judge_stats)
        if fit_stats:
            school_stats.stages.append(fit_stats)

        # --- Optional gap-fill: re-run retrieval if judge says "insufficient" ---
        if (
            config.retry_gap_fill
            and judge_report is not None
            and judge_report.overall_quality == QualityRating.INSUFFICIENT
            and judge_report.suggested_queries
        ):
            log.info(
                "Judge rated profile as insufficient — running targeted gap-fill (%d queries)",
                len(judge_report.suggested_queries),
            )
            try:
                profile, gap_stats = await _run_gap_fill(
                    profile, judge_report, config, client, http,
                )
                school_stats.stages.append(gap_stats)

                # Re-run judge on updated profile
                judge_report, judge_stats2 = await run_judge(profile, config, client)
                school_stats.stages.append(judge_stats2)
                log.info("Post-gap-fill judge verdict: %s", judge_report.overall_quality.value)

                # Re-run fit on updated profile
                fit_assessment, fit_stats2 = await run_fit_assessment(
                    cv_text, profile, config, client,
                )
                school_stats.stages.append(fit_stats2)
            except Exception as exc:
                log.warning("Gap-fill pass failed: %s", exc)

    school_stats.elapsed_seconds = total_elapsed[0]
    school_stats.success = True
    log.info("Pipeline complete (%.1fs, $%.4f)", school_stats.elapsed_seconds, school_stats.total_cost_usd)

    return SchoolResult(
        profile=profile,
        judge=judge_report,
        fit=fit_assessment,
    ), school_stats


async def _run_gap_fill(
    profile: SchoolProfile,
    judge_report: JudgeReport,
    config: Config,
    client: anthropic.AsyncAnthropic,
    http: httpx.AsyncClient,
) -> tuple[SchoolProfile, StageStats]:
    """Targeted gap-fill: re-run retrieval with judge's suggested queries as guidance."""
    from grad_agent.pipeline.prompts import RETRIEVAL_SYSTEM
    from grad_agent.pipeline.retrieval import _extract_json_from_text
    from grad_agent.pipeline.tools import TOOL_DEFINITIONS, dispatch_tool

    school_label = f"{profile.school_name} — {profile.program_name}"
    log = get_school_logger(__name__, school_label)
    stats = StageStats(stage="gap_fill", model=config.haiku_model)

    existing_json = profile.model_dump_json(indent=2)
    suggested = "\n".join(f"- {q}" for q in judge_report.suggested_queries)

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Here is an existing SchoolProfile that was rated as insufficient:\n\n"
                f"```json\n{existing_json}\n```\n\n"
                f"The quality judge flagged these gaps and suggested these queries:\n{suggested}\n\n"
                f"Please run these suggested searches (and any others you think are needed) "
                f"to fill in the missing information, then output an UPDATED complete "
                f"SchoolProfile JSON incorporating both the existing data and new findings."
            ),
        },
    ]

    with timed() as elapsed:
        for turn in range(1, config.gap_fill_max_turns + 1):
            log.info("Gap-fill turn %d/%d", turn, config.gap_fill_max_turns)

            response = await api_create_with_retry(
                lambda: client.messages.create(
                    model=config.haiku_model,
                    max_tokens=4096,
                    system=RETRIEVAL_SYSTEM,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            )
            stats.api_calls += 1
            stats.input_tokens += response.usage.input_tokens
            stats.output_tokens += response.usage.output_tokens

            if response.stop_reason == "tool_use":
                tool_results = []
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                        stats.tool_calls += 1
                        result = await dispatch_tool(
                            block.name, block.input, config, http, school_label,
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                text_parts = [b.text for b in response.content if b.type == "text"]
                full_text = "\n".join(text_parts)
                parsed = _extract_json_from_text(full_text)
                if parsed:
                    parsed["school_name"] = profile.school_name
                    parsed["program_name"] = profile.program_name
                    try:
                        updated = SchoolProfile.model_validate(parsed)
                        stats.elapsed_seconds = elapsed[0]
                        log.info("Gap-fill produced updated profile")
                        return updated, stats
                    except Exception as exc:
                        log.warning("Gap-fill validation failed: %s", exc)
                break

    stats.elapsed_seconds = elapsed[0]
    return profile, stats


async def run_all_schools(
    schools: list[tuple[str, str]],
    cv_text: str,
    config: Config,
) -> StatsCollector:
    """Launch pipelines for all schools with bounded concurrency.

    Args:
        schools: List of (school_name, program_name) tuples.
        cv_text: The applicant's CV text.
        config: Pipeline configuration.

    Returns:
        StatsCollector with results for all schools.
    """
    collector = StatsCollector()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(config.max_schools_parallel)
    client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async with httpx.AsyncClient() as http:
        all_results: list[tuple[SchoolResult, SchoolStats]] = []

        async def _process_school(school_name: str, program_name: str) -> None:
            async with semaphore:
                result, stats = await run_school(
                    school_name, program_name, cv_text, config, client, http,
                )
                collector.add_school(stats)
                all_results.append((result, stats))

                # Write individual school markdown
                md = render_school_markdown(result.profile, result.judge, result.fit)
                safe_name = _safe_filename(school_name, program_name)
                path = output_dir / f"{safe_name}_profile.md"
                path.write_text(md, encoding="utf-8")
                logger.info("Wrote %s", path)

        tasks = [
            _process_school(school, program)
            for school, program in schools
        ]
        await asyncio.gather(*tasks)

    # Write summary table
    summary_data = [
        (r.profile, r.fit) for r, _ in all_results
    ]
    summary_md = render_summary_table(summary_data)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    logger.info("Wrote %s", summary_path)

    return collector


def _safe_filename(school: str, program: str) -> str:
    """Convert school + program to a filesystem-safe slug."""
    combined = f"{school}_{program}".lower()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in combined)
    # Collapse consecutive underscores
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")
