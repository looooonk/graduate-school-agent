"""Retrieval backend implementations."""

from __future__ import annotations

from grad_agent.llm.vllm import OpenAICompatibleChatClient
from grad_agent.pipeline.retrieval_backends.anthropic_tool_use import AnthropicToolUseBackend
from grad_agent.pipeline.retrieval_backends.base import RetrievalBackend, RetrievalRequest
from grad_agent.pipeline.retrieval_backends.openai_compatible import (
    OpenAICompatibleToolCommandBackend,
)
from grad_agent.retrieval_registry import RETRIEVAL_BACKENDS

_BACKEND_IMPLEMENTATIONS: dict[str, RetrievalBackend] = {
    "anthropic_haiku": AnthropicToolUseBackend(),
    "anthropic_sonnet": AnthropicToolUseBackend(),
    "local_qwen_vllm": OpenAICompatibleToolCommandBackend(
        OpenAICompatibleChatClient.from_local_config,
        allow_parallel_agents=True,
    ),
    "local_openai_compatible": OpenAICompatibleToolCommandBackend(
        OpenAICompatibleChatClient.from_local_config,
        allow_parallel_agents=True,
    ),
    "openai_compatible": OpenAICompatibleToolCommandBackend(
        OpenAICompatibleChatClient.from_api_config,
        allow_parallel_agents=False,
    ),
}


def get_retrieval_backend(backend_id: str) -> RetrievalBackend:
    try:
        return _BACKEND_IMPLEMENTATIONS[backend_id]
    except KeyError as exc:
        allowed = ", ".join(sorted(RETRIEVAL_BACKENDS))
        raise ValueError(
            f"Unsupported retrieval backend {backend_id!r}. Expected one of: {allowed}"
        ) from exc


__all__ = ["RetrievalBackend", "RetrievalRequest", "get_retrieval_backend"]
