"""Stage 2 — Sonnet Judge.

Single-pass quality and coverage assessment of a SchoolProfile.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from grad_agent.config import Config
from grad_agent.models import JudgeReport, SchoolProfile
from grad_agent.pipeline.prompts import JUDGE_SYSTEM, judge_user_prompt
from grad_agent.reporting.stats import StageStats, timed
from grad_agent.util.log import get_school_logger
from grad_agent.util.retry import api_create_with_retry

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from model text, handling fenced blocks."""
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


async def run_judge(
    profile: SchoolProfile,
    config: Config,
    client: anthropic.AsyncAnthropic,
    context_text: str = "",
) -> tuple[JudgeReport, StageStats]:
    """Evaluate a SchoolProfile and return a JudgeReport.

    Args:
        profile: The school profile to evaluate.
        config: Pipeline configuration.
        client: Anthropic async client.
        context_text: Optional applicant context used to prioritise gap detection
            for fields relevant to the applicant's subfield and goals.

    Raises:
        RuntimeError: If the model fails to produce a valid JudgeReport.
    """
    school_label = f"{profile.school_name} — {profile.program_name}"
    log = get_school_logger(__name__, school_label)
    stats = StageStats(stage="judge", model=config.sonnet_model)

    profile_json = profile.model_dump_json(indent=2)

    with timed() as elapsed:
        response = await api_create_with_retry(
            lambda: client.messages.create(
                model=config.sonnet_model,
                max_tokens=2048,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": judge_user_prompt(profile_json, context_text)}],
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
        log.error("Judge produced no parseable JSON")
        raise RuntimeError(f"Judge failed to produce valid JSON for {school_label}")

    try:
        report = JudgeReport.model_validate(parsed)
    except Exception as exc:
        log.error("Judge output validation failed: %s", exc)
        raise RuntimeError(f"Judge output validation failed for {school_label}: {exc}") from exc

    log.info(
        "Judge verdict: %s (%d flags, %d suggested queries)",
        report.overall_quality.value,
        len(report.flagged_fields),
        len(report.suggested_queries),
    )
    return report, stats
