"""OpenAI-compatible retrieval backends using JSON tool commands."""

from __future__ import annotations

from collections.abc import Callable

from grad_agent.config import Config
from grad_agent.llm.vllm import OpenAICompatibleChatClient
from grad_agent.models import SchoolProfile
from grad_agent.pipeline.local_retrieval import (
    run_local_parallel_profile_loop,
    run_local_profile_loop,
)
from grad_agent.pipeline.retrieval_backends.base import RetrievalRequest
from grad_agent.reporting.stats import StageStats


class OpenAICompatibleToolCommandBackend:
    def __init__(
        self,
        client_factory: Callable[[Config], OpenAICompatibleChatClient],
        *,
        allow_parallel_agents: bool,
    ) -> None:
        self._client_factory = client_factory
        self._allow_parallel_agents = allow_parallel_agents

    async def run(self, request: RetrievalRequest) -> tuple[SchoolProfile, StageStats]:
        client = request.openai_compatible_client or self._client_factory(request.config)
        kwargs = {
            "school_name": request.school_name,
            "program_name": request.program_name,
            "config": request.config,
            "http": request.http,
            "initial_prompt": request.initial_prompt,
            "stage": request.stage,
            "max_turns": request.max_turns,
            "on_event": request.on_event,
            "traj": request.traj,
            "local_client": client,
        }
        if self._allow_parallel_agents and request.config.local_retrieval_parallel_agents > 1:
            return await run_local_parallel_profile_loop(**kwargs)
        return await run_local_profile_loop(**kwargs)
