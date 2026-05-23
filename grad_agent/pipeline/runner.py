"""Per-school pipeline runner.

Orchestrates Stage 1 → 2 → 3 for a single school with fault tolerance,
and provides the top-level parallel launcher for multiple schools.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import anthropic
import httpx

from grad_agent.config import Config
from grad_agent.events import (
    EventCallback,
    SchoolDone,
    SchoolStarted,
    StageStarted,
    ToolCalled,
    TurnProgress,
)
from grad_agent.llm.vllm import LocalVLLMClient
from grad_agent.models import (
    ConfidenceLevel,
    FitAssessment,
    JudgeReport,
    QualityRating,
    SchoolProfile,
    SchoolResult,
)
from grad_agent.pipeline.fit import run_fit_assessment
from grad_agent.pipeline.judge import run_judge
from grad_agent.pipeline.retrieval import (
    run_local_parallel_profile_loop,
    run_local_profile_loop,
    run_retrieval,
)
from grad_agent.reporting.markdown import render_school_markdown, render_summary_table
from grad_agent.reporting.pdf import ReportDirs, ensure_report_dirs, write_markdown_report
from grad_agent.reporting.stats import SchoolStats, StageStats, StatsCollector, add_usage, timed
from grad_agent.reporting.trajectory import TrajectoryLogger
from grad_agent.util.json import extract_json_object
from grad_agent.util.log import get_school_logger
from grad_agent.util.retry import api_create_with_retry

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
        A tuple of (SchoolResult, SchoolStats). Never raises — all errors are
        captured in SchoolResult.error and SchoolStats.error.
    """
    school_label = f"{school_name} — {program_name}"
    log = get_school_logger(__name__, school_label)
    school_stats = SchoolStats(school=school_label)

    log.info("Starting pipeline")

    with timed() as total_elapsed:
        # --- Stage 1: Retrieval ---
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

        # --- Stage 2 & 3: Judge and Fit run concurrently ---
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
                        profile, config, client, context_text, traj,
                    )
            except Exception as exc:
                log.error("Judge failed: %s", exc)
                judge_stats = StageStats(stage="judge", model=config.sonnet_model)
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
                if on_event:
                    on_event(StageStarted(school=school_label, stage="gap_fill"))
                profile, gap_stats = await _run_gap_fill(
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
                                profile, config, client, context_text, traj,
                            )
                    except Exception as exc:
                        log.error("Post-gap-fill judge failed: %s", exc)
                        post_judge_stats = StageStats(stage="judge", model=config.sonnet_model)
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


async def _run_gap_fill(
    profile: SchoolProfile,
    judge_report: JudgeReport,
    config: Config,
    client: anthropic.AsyncAnthropic,
    http: httpx.AsyncClient,
    on_event: EventCallback | None = None,
    traj: TrajectoryLogger | None = None,
    context_text: str = "",
    local_client: LocalVLLMClient | None = None,
) -> tuple[SchoolProfile, StageStats]:
    """Re-run a targeted retrieval pass using the judge's suggested queries.

    Args:
        profile: The existing profile rated as insufficient.
        judge_report: The judge's assessment, including suggested search queries.
        config: Pipeline configuration.
        client: Anthropic async client.
        http: Async HTTP client for tool execution.
        on_event: Optional progress callback.
        traj: Optional trajectory logger.
        context_text: Optional applicant context used to prioritise the missing
            facts that matter most for fit.

    Returns:
        A tuple of (updated SchoolProfile, StageStats). Returns the original
        profile unchanged if the model fails to produce a valid updated one.
    """
    from grad_agent.pipeline.prompts import RETRIEVAL_SYSTEM, retrieval_turn_status
    from grad_agent.pipeline.tools import TOOL_DEFINITIONS, dispatch_tool

    school_label = f"{profile.school_name} — {profile.program_name}"
    log = get_school_logger(__name__, school_label)
    stats = StageStats(stage="gap_fill", model=config.retrieval_model)

    existing_json = profile.model_dump_json(indent=2)
    context_section = (
        f"## Applicant Context\n\n{context_text.strip()}\n\n"
        if context_text.strip()
        else ""
    )
    flags = "\n".join(
        f"- {flag.field}: {flag.reason}" for flag in judge_report.flagged_fields
    ) or "- No specific flagged fields were provided."
    suggested = "\n".join(f"- {q}" for q in judge_report.suggested_queries)

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"{context_section}"
                f"Here is an existing SchoolProfile that was rated as insufficient:\n\n"
                f"```json\n{existing_json}\n```\n\n"
                f"The quality judge flagged these gaps:\n"
                f"{flags}\n\n"
                f"Suggested targeted queries:\n"
                f"{suggested}\n\n"
                f"Please run the most relevant suggested searches first, then any "
                f"other narrow searches needed for the same flagged fields. Then output "
                f"an UPDATED complete SchoolProfile JSON incorporating both the existing "
                f"data and new findings. "
                f"Preserve existing sourced facts unless new official evidence corrects them. "
                f"You have a budget of **{config.gap_fill_max_turns} turns** total."
            ),
        },
    ]

    if config.uses_local_retrieval:
        if config.local_retrieval_parallel_agents > 1:
            return await run_local_parallel_profile_loop(
                school_name=profile.school_name,
                program_name=profile.program_name,
                config=config,
                http=http,
                initial_prompt=messages[0]["content"],
                stage="gap_fill",
                max_turns=config.gap_fill_max_turns,
                on_event=on_event,
                traj=traj,
                local_client=local_client,
            )
        return await run_local_profile_loop(
            school_name=profile.school_name,
            program_name=profile.program_name,
            config=config,
            http=http,
            initial_prompt=messages[0]["content"],
            stage="gap_fill",
            max_turns=config.gap_fill_max_turns,
            on_event=on_event,
            traj=traj,
            local_client=local_client,
        )

    with timed() as elapsed:
        for turn in range(1, config.gap_fill_max_turns + 1):
            log.info("Gap-fill turn %d/%d", turn, config.gap_fill_max_turns)
            if on_event:
                on_event(
                    TurnProgress(
                        school=school_label,
                        turn=turn,
                        max_turns=config.gap_fill_max_turns,
                        stage="gap_fill",
                    )
                )

            response = await api_create_with_retry(
                lambda: client.messages.create(
                    model=config.retrieval_model,
                    max_tokens=4096,
                    system=RETRIEVAL_SYSTEM,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            )
            stats.api_calls += 1
            add_usage(stats, response.usage)
            if traj:
                traj.log_api_response("gap_fill", turn, config.retrieval_model, response)

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
                        if on_event:
                            on_event(ToolCalled(
                                school=school_label,
                                tool_name=block.name,
                                stage="gap_fill",
                            ))
                        result = await dispatch_tool(
                            block.name, block.input, config, http, school_label,
                        )
                        if traj:
                            traj.log_tool_result("gap_fill", turn, block.name, block.input, result)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({
                    "role": "user",
                    "content": [
                        *tool_results,
                        {
                            "type": "text",
                            "text": retrieval_turn_status(turn, config.gap_fill_max_turns),
                        },
                    ],
                })

            elif response.stop_reason == "end_turn":
                text_parts = [b.text for b in response.content if b.type == "text"]
                full_text = "\n".join(text_parts)
                parsed = extract_json_object(full_text)
                if parsed:
                    parsed["school_name"] = profile.school_name
                    parsed["program_name"] = profile.program_name
                    try:
                        updated = SchoolProfile.model_validate(parsed)
                        stats.elapsed_seconds = elapsed[0]
                        log.info("Gap-fill produced updated profile")
                        if traj:
                            traj.log_profile(updated)
                            traj.log_stage_end("gap_fill", elapsed[0])
                        return updated, stats
                    except Exception as exc:
                        log.warning("Gap-fill validation failed: %s", exc)
                break

    stats.elapsed_seconds = elapsed[0]
    if traj:
        traj.log_stage_end("gap_fill", elapsed[0])
    return profile, stats


def calibrate_fit_confidence(
    fit: FitAssessment | None,
    judge: JudgeReport | None,
) -> FitAssessment | None:
    """Align fit confidence with the judge's data-quality verdict."""
    if fit is None or judge is None:
        return fit
    if judge.overall_quality == QualityRating.INSUFFICIENT:
        target = ConfidenceLevel.LOW
    elif (
        judge.overall_quality == QualityRating.PARTIAL
        and fit.confidence == ConfidenceLevel.HIGH
    ):
        target = ConfidenceLevel.MEDIUM
    else:
        return fit
    if fit.confidence == target:
        return fit
    return fit.model_copy(update={"confidence": target})


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
        logger.info("Trajectory logs → %s", run_log_dir)

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
        TrajectoryLogger(run_log_dir / f"{_safe_filename(school_name, program_name)}.jsonl")
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
    safe_name = _safe_filename(school_name, program_name)
    path = output_dirs.markdown_dir / f"{safe_name}_profile.md"
    await asyncio.to_thread(write_markdown_report, path, md, output_dirs)
    logger.info("Wrote %s", path)


def _safe_filename(school: str, program: str) -> str:
    """Convert school + program to a filesystem-safe slug."""
    combined = f"{school}_{program}".lower()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in combined)
    # Collapse consecutive underscores
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


@asynccontextmanager
async def _maybe_semaphore(
    semaphore: asyncio.Semaphore | None,
) -> AsyncIterator[None]:
    if semaphore is None:
        yield
        return
    async with semaphore:
        yield
