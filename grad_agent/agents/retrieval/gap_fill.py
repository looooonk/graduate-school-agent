"""Targeted gap-fill retrieval stage."""

from __future__ import annotations

import anthropic
import httpx

from grad_agent.agents.retrieval.backends import RetrievalRequest, get_retrieval_backend
from grad_agent.config import Config
from grad_agent.events import EventCallback
from grad_agent.llm.vllm import LocalVLLMClient
from grad_agent.models import JudgeReport, SchoolProfile
from grad_agent.reporting.stats import StageStats
from grad_agent.reporting.trajectory import TrajectoryLogger


async def run_gap_fill(
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
    """Run targeted retrieval using the judge's suggested queries."""
    backend = get_retrieval_backend(config.retrieval_backend)
    return await backend.run(
        RetrievalRequest(
            school_name=profile.school_name,
            program_name=profile.program_name,
            initial_prompt=gap_fill_prompt(profile, judge_report, config, context_text),
            stage="gap_fill",
            max_turns=config.gap_fill_max_turns,
            config=config,
            anthropic_client=client,
            http=http,
            on_event=on_event,
            traj=traj,
            openai_compatible_client=local_client,
            fallback_profile=profile,
        )
    )


def gap_fill_prompt(
    profile: SchoolProfile,
    judge_report: JudgeReport,
    config: Config,
    context_text: str = "",
) -> str:
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

    return (
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
    )
