"""Retrieval backend registry and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RetrievalBackendKind = Literal[
    "anthropic_tool_use",
    "local_openai_compatible",
    "openai_compatible_api",
]


@dataclass(frozen=True)
class RetrievalBackendSpec:
    id: str
    label: str
    kind: RetrievalBackendKind
    model_field: str
    description: str

    @property
    def is_local(self) -> bool:
        return self.kind == "local_openai_compatible"

    @property
    def is_openai_compatible_api(self) -> bool:
        return self.kind == "openai_compatible_api"


RETRIEVAL_BACKENDS: dict[str, RetrievalBackendSpec] = {
    "local_qwen_vllm": RetrievalBackendSpec(
        id="local_qwen_vllm",
        label="Local Qwen via vLLM",
        kind="local_openai_compatible",
        model_field="local_retrieval_model",
        description="Local OpenAI-compatible endpoint using JSON tool commands.",
    ),
    "local_openai_compatible": RetrievalBackendSpec(
        id="local_openai_compatible",
        label="Local OpenAI-compatible",
        kind="local_openai_compatible",
        model_field="local_retrieval_model",
        description="Generic local OpenAI-compatible endpoint using JSON tool commands.",
    ),
    "openai_compatible": RetrievalBackendSpec(
        id="openai_compatible",
        label="OpenAI-compatible API",
        kind="openai_compatible_api",
        model_field="openai_retrieval_model",
        description="Remote OpenAI-compatible chat completions using JSON tool commands.",
    ),
    "anthropic_haiku": RetrievalBackendSpec(
        id="anthropic_haiku",
        label="Anthropic Haiku",
        kind="anthropic_tool_use",
        model_field="haiku_model",
        description="Anthropic Messages API endpoint using native tool calls.",
    ),
    "anthropic_sonnet": RetrievalBackendSpec(
        id="anthropic_sonnet",
        label="Anthropic Sonnet",
        kind="anthropic_tool_use",
        model_field="sonnet_model",
        description="Anthropic Messages API endpoint using native tool calls.",
    ),
}


def retrieval_backend_ids() -> tuple[str, ...]:
    return tuple(sorted(RETRIEVAL_BACKENDS))


def get_retrieval_backend_spec(backend_id: str) -> RetrievalBackendSpec | None:
    return RETRIEVAL_BACKENDS.get(backend_id)
