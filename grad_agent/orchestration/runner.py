"""Per-school orchestration for the research agent."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import anthropic
import httpx

from grad_agent.agents.fit.confidence import calibrate_fit_confidence
from grad_agent.agents.fit.service import run_fit_assessment
from grad_agent.agents.judge.service import run_judge
from grad_agent.agents.retrieval.gap_fill import run_gap_fill
from grad_agent.agents.retrieval.service import run_retrieval
from grad_agent.config import Config
from grad_agent.events import (
    EventCallback,
    SchoolDone,
    SchoolStarted,
    StageStarted,
)
from grad_agent.llm.vllm import LocalVLLMClient
from grad_agent.models import (
    FitAssessment,
    JudgeReport,
    QualityRating,
    SchoolProfile,
    SchoolResult,
)
from grad_agent.reporting.markdown import render_school_markdown, render_summary_table
from grad_agent.reporting.paths import safe_filename
from grad_agent.reporting.pdf import ReportDirs, ensure_report_dirs, write_markdown_report
from grad_agent.reporting.stats import SchoolStats, StageStats, StatsCollector, timed
from grad_agent.reporting.trajectory import TrajectoryLogger
from grad_agent.util.log import get_school_logger

logger = logging.getLogger(__name__)


async def run_school(
    school_name: str,
    program_name: str,
    cv_text: str,
    config: Config,
    client: anthropic.AsyncAnthropic,
    http: httpx.AsyncClient,
    context_text: str = "",
    on_event: EventCallback | None = None,
    traj: TrajectoryLogger | None = None,
    local_client: LocalVLLMClient | None = None,
    sonnet_semaphore: asyncio.Semaphore | None = None,
) -> tuple[SchoolResult, SchoolStats]:
    """Run the full 3-stage pipeline for a single school.

    Args:
        school_name: Name of the school.
        program_name: Name of the graduate program.
        cv_text: The applicant's CV text.
        config: Pipeline configuration.
        client: Anthropic async client.
        http: Async HTTP client.
        context_text: Optional applicant context from input/context.md.
        on_event: Optional callback invoked with progress events for the TUI.
        traj: Optional trajectory logger; records every API call and tool result.

    Returns:
        A tuple of (SchoolResult, SchoolStats). Never raises; all errors are
        captured in SchoolResult.error and SchoolStats.error.
    """
    school_label = f"{school_name} — {program_name}"
    log = get_school_logger(__name__, school_label)
    school_stats = SchoolStats(school=school_label)

    log.info("Starting pipeline")

    with timed() as total_elapsed:
        try:
            if on_event:
                on_event(StageStarted(school=school_label, stage="retrieval"))
            profile, retrieval_stats = await run_retrieval(
                school_name,
                program_name,
                config,
                client,
                http,
                context_text,
                on_event,
                traj,
                local_client,
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

        judge_report: JudgeReport | None = None
        judge_stats: StageStats | None = None
        fit_assessment: FitAssessment | None = None
        fit_stats: StageStats | None = None
        stage_errors: list[str] = []

        async def _run_judge() -> None:
            nonlocal judge_report, judge_stats
            try:
                async with _maybe_semaphore(sonnet_semaphore):
                    judge_report, judge_stats = await run_judge(
                        profile, config, client, context_text, traj, http,
                    )
            except Exception as exc:
                log.error("Judge failed: %s", exc)
                judge_stats = StageStats(stage="judge", model=config.judge_model)
                stage_errors.append(f"Judge failed: {exc}")

        async def _run_fit() -> None:
            nonlocal fit_assessment, fit_stats
            try:
                async with _maybe_semaphore(sonnet_semaphore):
                    fit_assessment, fit_stats = await run_fit_assessment(
                        cv_text, profile, config, client, context_text, traj,
                    )
            except Exception as exc:
                log.error("Fit assessment failed: %s", exc)
                fit_stats = StageStats(stage="fit", model=config.sonnet_model)
                stage_errors.append(f"Fit assessment failed: {exc}")

        if on_event:
            on_event(StageStarted(school=school_label, stage="judge+fit"))
        await asyncio.gather(_run_judge(), _run_fit())

        if judge_stats:
            school_stats.stages.append(judge_stats)
        if fit_stats:
            school_stats.stages.append(fit_stats)

        if (
            config.retry_gap_fill
            and judge_report is not None
            and judge_report.overall_quality == QualityRating.INSUFFICIENT
            and judge_report.suggested_queries
        ):
            log.info(
                "Judge rated profile as insufficient; running targeted gap-fill (%d queries)",
                len(judge_report.suggested_queries),
            )
            try:
                if on_event:
                    on_event(StageStarted(school=school_label, stage="gap_fill"))
                profile, gap_stats = await run_gap_fill(
                    profile,
                    judge_report,
                    config,
                    client,
                    http,
                    on_event,
                    traj,
                    context_text=context_text,
                    local_client=local_client,
                )
                school_stats.stages.append(gap_stats)

                post_judge: JudgeReport | None = None
                post_judge_stats: StageStats | None = None
                post_fit: FitAssessment | None = None
                post_fit_stats: StageStats | None = None
                post_errors: list[str] = []

                async def _run_post_judge() -> None:
                    nonlocal post_judge, post_judge_stats
                    try:
                        async with _maybe_semaphore(sonnet_semaphore):
                            post_judge, post_judge_stats = await run_judge(
                                profile, config, client, context_text, traj, http,
                            )
                    except Exception as exc:
                        log.error("Post-gap-fill judge failed: %s", exc)
                        post_judge_stats = StageStats(stage="judge", model=config.judge_model)
                        post_errors.append(f"Post-gap-fill judge failed: {exc}")

                async def _run_post_fit() -> None:
                    nonlocal post_fit, post_fit_stats
                    try:
                        async with _maybe_semaphore(sonnet_semaphore):
                            post_fit, post_fit_stats = await run_fit_assessment(
                                cv_text, profile, config, client, context_text, traj,
                            )
                    except Exception as exc:
                        log.error("Post-gap-fill fit assessment failed: %s", exc)
                        post_fit_stats = StageStats(stage="fit", model=config.sonnet_model)
                        post_errors.append(f"Post-gap-fill fit assessment failed: {exc}")

                await asyncio.gather(_run_post_judge(), _run_post_fit())
                stage_errors.extend(post_errors)
                if post_judge is not None:
                    judge_report = post_judge
                if post_fit is not None:
                    fit_assessment = post_fit
                if post_judge_stats:
                    school_stats.stages.append(post_judge_stats)
                if post_fit_stats:
                    school_stats.stages.append(post_fit_stats)
                if judge_report:
                    log.info("Post-gap-fill judge verdict: %s", judge_report.overall_quality.value)
            except Exception as exc:
                log.warning("Gap-fill pass failed: %s", exc)
                stage_errors.append(f"Gap-fill failed: {exc}")

        fit_assessment = calibrate_fit_confidence(fit_assessment, judge_report)

    school_stats.elapsed_seconds = total_elapsed[0]
    school_stats.success = not stage_errors
    school_stats.error = "; ".join(stage_errors) or None
    log.info(
        "Pipeline complete (%.1fs, $%.4f)",
        school_stats.elapsed_seconds,
        school_stats.total_cost_usd,
    )

    return SchoolResult(
        profile=profile,
        judge=judge_report,
        fit=fit_assessment,
        error=school_stats.error,
    ), school_stats


async def run_all_schools(
    schools: list[tuple[str, str]],
    cv_text: str,
    config: Config,
    context_text: str = "",
    on_event: EventCallback | None = None,
) -> StatsCollector:
    """Launch pipelines for all schools with bounded concurrency.

    Args:
        schools: List of (school_name, program_name) tuples.
        cv_text: The applicant's CV text.
        config: Pipeline configuration.
        context_text: Optional applicant context from input/context.md, injected
            into every stage (retrieval, judge, fit) for all schools.
        on_event: Optional callback invoked with progress events for each school.

    Returns:
        StatsCollector with aggregated results for all schools.
    """
    collector = StatsCollector()
    output_dirs = ensure_report_dirs(Path(config.output_dir))

    # Build the run-scoped log directory once, or None if logging is disabled.
    run_log_dir: Path | None = None
    if config.logs_dir:
        run_ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        run_log_dir = Path(config.logs_dir) / run_ts
        run_log_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Trajectory logs: %s", run_log_dir)

    client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
    local_client = LocalVLLMClient.from_config(config) if config.uses_local_retrieval else None
    sonnet_semaphore = asyncio.Semaphore(config.max_sonnet_parallel)

    async with httpx.AsyncClient() as http:
        total = len(schools)
        all_results: list[tuple[SchoolResult, SchoolStats] | None] = [None] * total
        semaphore = asyncio.Semaphore(config.max_schools_parallel)

        async def run_one(idx: int, school_name: str, program_name: str) -> None:
            async with semaphore:
                await _run_one_school(
                    idx,
                    total,
                    school_name,
                    program_name,
                    cv_text,
                    config,
                    client,
                    http,
                    context_text,
                    output_dirs,
                    run_log_dir,
                    collector,
                    all_results,
                    on_event,
                    local_client,
                    sonnet_semaphore,
                )

        await asyncio.gather(
            *(run_one(idx, school_name, program_name)
              for idx, (school_name, program_name) in enumerate(schools, start=1))
        )

    # Write summary table
    completed_results = [item for item in all_results if item is not None]
    summary_data = [(result.profile, result.fit) for result, _ in completed_results]
    summary_md = render_summary_table(summary_data)
    summary_path = output_dirs.markdown_dir / "summary.md"
    write_markdown_report(summary_path, summary_md, output_dirs)
    logger.info("Wrote %s", summary_path)

    return collector


async def _run_one_school(
    idx: int,
    total: int,
    school_name: str,
    program_name: str,
    cv_text: str,
    config: Config,
    client: anthropic.AsyncAnthropic,
    http: httpx.AsyncClient,
    context_text: str,
    output_dirs: ReportDirs,
    run_log_dir: Path | None,
    collector: StatsCollector,
    all_results: list[tuple[SchoolResult, SchoolStats] | None],
    on_event: EventCallback | None,
    local_client: LocalVLLMClient | None,
    sonnet_semaphore: asyncio.Semaphore | None,
) -> None:
    school_label = f"{school_name} — {program_name}"
    if on_event:
        on_event(SchoolStarted(school=school_label, idx=idx, total=total))

    traj_ctx = (
        TrajectoryLogger(run_log_dir / f"{safe_filename(school_name, program_name)}.jsonl")
        if run_log_dir else nullcontext()
    )
    with traj_ctx as traj:
        result, stats = await run_school(
            school_name, program_name, cv_text, config, client, http,
            context_text, on_event, traj, local_client, sonnet_semaphore,
        )
    if on_event:
        on_event(SchoolDone(
            school=school_label,
            success=stats.success,
            elapsed=stats.elapsed_seconds,
            cost=stats.total_cost_usd,
        ))
    collector.add_school(stats)
    all_results[idx - 1] = (result, stats)

    md = render_school_markdown(result.profile, result.judge, result.fit)
    safe_name = safe_filename(school_name, program_name)
    path = output_dirs.markdown_dir / f"{safe_name}_profile.md"
    await asyncio.to_thread(write_markdown_report, path, md, output_dirs)
    logger.info("Wrote %s", path)


def _safe_filename(school: str, program: str) -> str:
    return safe_filename(school, program)


@asynccontextmanager
async def _maybe_semaphore(
    semaphore: asyncio.Semaphore | None,
) -> AsyncIterator[None]:
    if semaphore is None:
        yield
        return
    async with semaphore:
        yield
