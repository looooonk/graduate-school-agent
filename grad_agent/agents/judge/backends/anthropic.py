"""Anthropic Messages API judge backend."""

from __future__ import annotations

from grad_agent.agents.judge.prompts import JUDGE_SYSTEM, judge_user_prompt
from grad_agent.agents.judge.types import JudgeRequest
from grad_agent.models import JudgeReport
from grad_agent.reporting.stats import StageStats, add_usage, timed
from grad_agent.util.json import extract_json_object
from grad_agent.util.log import get_school_logger
from grad_agent.util.retry import api_create_with_retry


class AnthropicMessagesJudgeBackend:
    async def run(self, request: JudgeRequest) -> tuple[JudgeReport, StageStats]:
        config = request.config
        model = config.judge_model
        log = get_school_logger(__name__, request.school_label)
        stats = StageStats(stage="judge", model=model)

        if request.traj:
            request.traj.log_stage_start("judge")

        profile_json = request.profile.model_dump_json(indent=2)

        with timed() as elapsed:
            response = await api_create_with_retry(
                lambda: request.anthropic_client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=JUDGE_SYSTEM,
                    messages=[
                        {
                            "role": "user",
                            "content": judge_user_prompt(profile_json, request.context_text),
                        }
                    ],
                )
            )

            stats.api_calls += 1
            add_usage(stats, response.usage)
            if request.traj:
                request.traj.log_api_response("judge", 1, model, response)

        stats.elapsed_seconds = elapsed[0]
        report = _parse_judge_report(response.content, request.school_label, log)

        log.info(
            "Judge verdict: %s (%d flags, %d suggested queries)",
            report.overall_quality.value,
            len(report.flagged_fields),
            len(report.suggested_queries),
        )
        if request.traj:
            request.traj.log_judge_report(report)
            request.traj.log_stage_end("judge", elapsed[0])
        return report, stats


def _parse_judge_report(blocks: list[object], school_label: str, log: object) -> JudgeReport:
    full_text = "\n".join(
        block.text for block in blocks if getattr(block, "type", None) == "text"
    )
    parsed = extract_json_object(full_text)
    if parsed is None:
        log.error("Judge produced no parseable JSON")
        raise RuntimeError(f"Judge failed to produce valid JSON for {school_label}")

    try:
        return JudgeReport.model_validate(parsed)
    except Exception as exc:
        log.error("Judge output validation failed: %s", exc)
        raise RuntimeError(f"Judge output validation failed for {school_label}: {exc}") from exc
