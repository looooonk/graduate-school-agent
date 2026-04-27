"""Stage 3 — Sonnet Fit Assessor.

Cross-references the applicant's CV against a SchoolProfile to produce
a structured fit assessment.
"""

from __future__ import annotations

import logging

import anthropic

from grad_agent.config import Config
from grad_agent.models import FitAssessment, SchoolProfile
from grad_agent.pipeline.prompts import FIT_SYSTEM, fit_user_prompt
from grad_agent.reporting.stats import StageStats, add_usage, timed
from grad_agent.reporting.trajectory import TrajectoryLogger
from grad_agent.util.json import extract_json_object
from grad_agent.util.log import get_school_logger
from grad_agent.util.retry import api_create_with_retry

logger = logging.getLogger(__name__)


async def run_fit_assessment(
    cv_text: str,
    profile: SchoolProfile,
    config: Config,
    client: anthropic.AsyncAnthropic,
    context_text: str = "",
    traj: TrajectoryLogger | None = None,
) -> tuple[FitAssessment, StageStats]:
    """Assess applicant fit against a school profile.

    Args:
        cv_text: The applicant's CV in plain text or Markdown.
        profile: The school profile to assess fit against.
        config: Pipeline configuration.
        client: Anthropic async client.
        context_text: Optional applicant context providing additional goals or
            preferences beyond the CV (e.g. target subfield, geographic constraints).
        traj: Optional trajectory logger for recording API interactions.

    Returns:
        A tuple of (FitAssessment, StageStats).

    Raises:
        RuntimeError: If the model fails to produce a valid FitAssessment.
    """
    school_label = f"{profile.school_name} — {profile.program_name}"
    log = get_school_logger(__name__, school_label)
    stats = StageStats(stage="fit", model=config.sonnet_model)

    if traj:
        traj.log_stage_start("fit")

    profile_json = profile.model_dump_json(indent=2)

    with timed() as elapsed:
        response = await api_create_with_retry(
            lambda: client.messages.create(
                model=config.sonnet_model,
                max_tokens=2048,
                system=FIT_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": fit_user_prompt(cv_text, profile_json, context_text),
                    }
                ],
            )
        )

        stats.api_calls += 1
        add_usage(stats, response.usage)
        if traj:
            traj.log_api_response("fit", 1, config.sonnet_model, response)

    stats.elapsed_seconds = elapsed[0]

    text_parts = [block.text for block in response.content if block.type == "text"]
    full_text = "\n".join(text_parts)

    parsed = extract_json_object(full_text)
    if parsed is None:
        log.error("Fit assessor produced no parseable JSON")
        raise RuntimeError(f"Fit assessor failed to produce valid JSON for {school_label}")

    try:
        assessment = FitAssessment.model_validate(parsed)
    except Exception as exc:
        log.error("Fit assessment validation failed: %s", exc)
        raise RuntimeError(
            f"Fit assessment validation failed for {school_label}: {exc}"
        ) from exc

    log.info(
        "Fit score: %.2f (confidence: %s)",
        assessment.overall_score,
        assessment.confidence.value,
    )
    if traj:
        traj.log_fit_assessment(assessment)
        traj.log_stage_end("fit", elapsed[0])
    return assessment, stats
