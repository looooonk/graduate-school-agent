"""Judge backend registry and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

JudgeBackendKind = Literal[
    "anthropic_messages",
    "openai_compatible_api",
]


@dataclass(frozen=True)
class JudgeBackendSpec:
    id: str
    label: str
    kind: JudgeBackendKind
    model_field: str
    description: str

    @property
    def is_openai_compatible_api(self) -> bool:
        return self.kind == "openai_compatible_api"


JUDGE_BACKENDS: dict[str, JudgeBackendSpec] = {
    "anthropic_sonnet": JudgeBackendSpec(
        id="anthropic_sonnet",
        label="Anthropic Sonnet",
        kind="anthropic_messages",
        model_field="sonnet_model",
        description="Anthropic Messages API judge using the configured Sonnet model.",
    ),
    "anthropic_haiku": JudgeBackendSpec(
        id="anthropic_haiku",
        label="Anthropic Haiku",
        kind="anthropic_messages",
        model_field="haiku_model",
        description="Anthropic Messages API judge using the configured Haiku model.",
    ),
    "openai_compatible": JudgeBackendSpec(
        id="openai_compatible",
        label="OpenAI-compatible API",
        kind="openai_compatible_api",
        model_field="openai_judge_model",
        description="Remote OpenAI-compatible chat completions judge.",
    ),
}


def judge_backend_ids() -> tuple[str, ...]:
    return tuple(sorted(JUDGE_BACKENDS))


def get_judge_backend_spec(backend_id: str) -> JudgeBackendSpec | None:
    return JUDGE_BACKENDS.get(backend_id)
