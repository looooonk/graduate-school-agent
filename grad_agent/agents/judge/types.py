"""Shared judge backend types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import anthropic
import httpx

from grad_agent.config import Config
from grad_agent.llm.vllm import OpenAICompatibleChatClient
from grad_agent.models import JudgeReport, SchoolProfile
from grad_agent.reporting.stats import StageStats
from grad_agent.reporting.trajectory import TrajectoryLogger


@dataclass(frozen=True)
class JudgeRequest:
    profile: SchoolProfile
    config: Config
    anthropic_client: anthropic.AsyncAnthropic
    context_text: str = ""
    traj: TrajectoryLogger | None = None
    http: httpx.AsyncClient | None = None
    openai_compatible_client: OpenAICompatibleChatClient | None = None

    @property
    def school_label(self) -> str:
        return f"{self.profile.school_name} — {self.profile.program_name}"


class JudgeBackend(Protocol):
    async def run(self, request: JudgeRequest) -> tuple[JudgeReport, StageStats]:
        ...
