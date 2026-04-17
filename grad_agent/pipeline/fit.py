"""Stage 3 — Sonnet Fit Assessor.

Cross-references the applicant's CV against a SchoolProfile to produce
a structured fit assessment.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from grad_agent.config import Config
from grad_agent.models import FitAssessment, SchoolProfile
from grad_agent.pipeline.prompts import FIT_SYSTEM, fit_user_prompt
from grad_agent.reporting.stats import StageStats, timed
from grad_agent.util.log import get_school_logger
from grad_agent.util.retry import api_create_with_retry

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if "```" in stripped:
        for block in stripped.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


async def run_fit_assessment(
    cv_text: str,
    profile: SchoolProfile,
    config: Config,
    client: anthropic.AsyncAnthropic,
) -> tuple[FitAssessment, StageStats]:
    """Assess applicant fit against a school profile.

    Raises:
        RuntimeError: If the model fails to produce a valid FitAssessment.
    """
    school_label = f"{profile.school_name} — {profile.program_name}"
    log = get_school_logger(__name__, school_label)
    stats = StageStats(stage="fit", model=config.sonnet_model)

    profile_json = profile.model_dump_json(indent=2)

    with timed() as elapsed:
        response = await api_create_with_retry(
            lambda: client.messages.create(
                model=config.sonnet_model,
                max_tokens=2048,
                system=FIT_SYSTEM,
                messages=[{"role": "user", "content": fit_user_prompt(cv_text, profile_json)}],
            )
        )

        stats.api_calls += 1
        stats.input_tokens += response.usage.input_tokens
        stats.output_tokens += response.usage.output_tokens

    stats.elapsed_seconds = elapsed[0]

    text_parts = [block.text for block in response.content if block.type == "text"]
    full_text = "\n".join(text_parts)

    parsed = _extract_json(full_text)
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
    return assessment, stats
