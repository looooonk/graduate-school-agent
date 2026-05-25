"""Retrieval backend registry and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RetrievalBackendKind = Literal["api_tool_use", "local_openai_compatible"]


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


RETRIEVAL_BACKENDS: dict[str, RetrievalBackendSpec] = {
    "local_qwen_vllm": RetrievalBackendSpec(
        id="local_qwen_vllm",
        label="Local Qwen via vLLM",
        kind="local_openai_compatible",
        model_field="local_retrieval_model",
        description="Local OpenAI-compatible endpoint using JSON tool commands.",
    ),
    "anthropic_haiku": RetrievalBackendSpec(
        id="anthropic_haiku",
        label="Anthropic Haiku",
        kind="api_tool_use",
        model_field="haiku_model",
        description="Anthropic Messages API endpoint using native tool calls.",
    ),
}


def retrieval_backend_ids() -> tuple[str, ...]:
    return tuple(sorted(RETRIEVAL_BACKENDS))


def get_retrieval_backend_spec(backend_id: str) -> RetrievalBackendSpec | None:
    return RETRIEVAL_BACKENDS.get(backend_id)
