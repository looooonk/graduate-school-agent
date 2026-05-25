"""Judge agent dispatch."""

from __future__ import annotations

import anthropic
import httpx

from grad_agent.agents.judge.backends import JudgeRequest, get_judge_backend
from grad_agent.config import Config
from grad_agent.llm.vllm import OpenAICompatibleChatClient
from grad_agent.models import JudgeReport, SchoolProfile
from grad_agent.reporting.stats import StageStats
from grad_agent.reporting.trajectory import TrajectoryLogger


async def run_judge(
    profile: SchoolProfile,
    config: Config,
    client: anthropic.AsyncAnthropic,
    context_text: str = "",
    traj: TrajectoryLogger | None = None,
    http: httpx.AsyncClient | None = None,
    openai_compatible_client: OpenAICompatibleChatClient | None = None,
) -> tuple[JudgeReport, StageStats]:
    """Evaluate a SchoolProfile and return a JudgeReport.

    Args:
        profile: The school profile to evaluate.
        config: Pipeline configuration.
        client: Anthropic async client.
        context_text: Optional applicant context used to prioritise gap detection
            for fields relevant to the applicant's subfield and goals.
        traj: Optional trajectory logger for recording API interactions.
        http: Optional HTTP client used by OpenAI-compatible judge backends.
        openai_compatible_client: Optional injected OpenAI-compatible client for tests.

    Returns:
        A tuple of (JudgeReport, StageStats).

    Raises:
        RuntimeError: If the model fails to produce a valid JudgeReport.
    """
    backend = get_judge_backend(config.judge_backend)
    return await backend.run(
        JudgeRequest(
            profile=profile,
            config=config,
            anthropic_client=client,
            context_text=context_text,
            traj=traj,
            http=http,
            openai_compatible_client=openai_compatible_client,
        )
    )
