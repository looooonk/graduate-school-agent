"""Stage 1 retrieval dispatch."""

from __future__ import annotations

import anthropic
import httpx

from grad_agent.config import Config
from grad_agent.events import EventCallback
from grad_agent.llm.vllm import LocalVLLMClient
from grad_agent.models import SchoolProfile
from grad_agent.pipeline.prompts import retrieval_user_prompt
from grad_agent.pipeline.retrieval_backends import RetrievalRequest, get_retrieval_backend
from grad_agent.reporting.stats import StageStats
from grad_agent.reporting.trajectory import TrajectoryLogger


async def run_retrieval(
    school_name: str,
    program_name: str,
    config: Config,
    client: anthropic.AsyncAnthropic,
    http: httpx.AsyncClient,
    context_text: str = "",
    on_event: EventCallback | None = None,
    traj: TrajectoryLogger | None = None,
    local_client: LocalVLLMClient | None = None,
) -> tuple[SchoolProfile, StageStats]:
    """Run the configured retrieval backend for a single school."""
    backend = get_retrieval_backend(config.retrieval_backend)
    return await backend.run(
        RetrievalRequest(
            school_name=school_name,
            program_name=program_name,
            initial_prompt=retrieval_user_prompt(
                school_name,
                program_name,
                context_text,
                config.max_retrieval_turns,
            ),
            stage="retrieval",
            max_turns=config.max_retrieval_turns,
            config=config,
            anthropic_client=client,
            http=http,
            on_event=on_event,
            traj=traj,
            local_client=local_client,
        )
    )
