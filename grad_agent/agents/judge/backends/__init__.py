"""Judge backend implementations."""

from __future__ import annotations

from grad_agent.agents.judge.backends.anthropic import AnthropicMessagesJudgeBackend
from grad_agent.agents.judge.backends.openai_compatible import OpenAICompatibleJudgeBackend
from grad_agent.agents.judge.registry import JUDGE_BACKENDS
from grad_agent.agents.judge.types import JudgeBackend, JudgeRequest

_BACKEND_IMPLEMENTATIONS: dict[str, JudgeBackend] = {
    "anthropic_sonnet": AnthropicMessagesJudgeBackend(),
    "anthropic_haiku": AnthropicMessagesJudgeBackend(),
    "openai_compatible": OpenAICompatibleJudgeBackend(),
}


def get_judge_backend(backend_id: str) -> JudgeBackend:
    try:
        return _BACKEND_IMPLEMENTATIONS[backend_id]
    except KeyError as exc:
        allowed = ", ".join(sorted(JUDGE_BACKENDS))
        raise ValueError(
            f"Unsupported judge backend {backend_id!r}. Expected one of: {allowed}"
        ) from exc


__all__ = ["JudgeBackend", "JudgeRequest", "get_judge_backend"]
