"""OpenAI-compatible judge backend."""

from __future__ import annotations

from grad_agent.agents.judge.backends.anthropic import _parse_judge_report
from grad_agent.agents.judge.prompts import JUDGE_SYSTEM, judge_user_prompt
from grad_agent.agents.judge.types import JudgeRequest
from grad_agent.llm.vllm import OpenAICompatibleChatClient
from grad_agent.models import JudgeReport
from grad_agent.reporting.stats import StageStats, add_usage, timed
from grad_agent.util.log import get_school_logger


class OpenAICompatibleJudgeBackend:
    async def run(self, request: JudgeRequest) -> tuple[JudgeReport, StageStats]:
        if request.http is None:
            raise RuntimeError("OpenAI-compatible judge backend requires an HTTP client")

        config = request.config
        model = config.judge_model
        log = get_school_logger(__name__, request.school_label)
        stats = StageStats(stage="judge", model=model)
        client = (
            request.openai_compatible_client
            or OpenAICompatibleChatClient.from_judge_config(config)
        )

        if request.traj:
            request.traj.log_stage_start("judge")

        profile_json = request.profile.model_dump_json(indent=2)

        with timed() as elapsed:
            response = await client.create(
                request.http,
                model=model,
                max_tokens=2048,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": judge_user_prompt(profile_json, request.context_text),
                    },
                ],
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
