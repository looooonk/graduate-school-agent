"""Shared retrieval backend types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import anthropic
import httpx

from grad_agent.config import Config
from grad_agent.events import EventCallback
from grad_agent.llm.vllm import OpenAICompatibleChatClient
from grad_agent.models import SchoolProfile
from grad_agent.reporting.stats import StageStats
from grad_agent.reporting.trajectory import TrajectoryLogger


@dataclass(frozen=True)
class RetrievalRequest:
    school_name: str
    program_name: str
    initial_prompt: str
    stage: str
    max_turns: int
    config: Config
    anthropic_client: anthropic.AsyncAnthropic
    http: httpx.AsyncClient
    on_event: EventCallback | None = None
    traj: TrajectoryLogger | None = None
    openai_compatible_client: OpenAICompatibleChatClient | None = None
    fallback_profile: SchoolProfile | None = None

    @property
    def school_label(self) -> str:
        return f"{self.school_name} — {self.program_name}"


class RetrievalBackend(Protocol):
    async def run(self, request: RetrievalRequest) -> tuple[SchoolProfile, StageStats]:
        ...
