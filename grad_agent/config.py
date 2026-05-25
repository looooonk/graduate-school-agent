"""Configuration loaded from a YAML file (non-secret settings) and environment variables (API keys).

Usage:
    config = Config.load()                         # config.yaml + env
    config = Config.load("custom.yaml")            # custom path
    config = Config.load(overrides={"output_dir": "out2"})  # programmatic overrides
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dotenv
import yaml

from grad_agent.retrieval_registry import get_retrieval_backend_spec, retrieval_backend_ids

_DEFAULT_YAML_PATH = Path("config.yaml")
_DEFAULT_LOCAL_RETRIEVAL_BASE_URLS = (
    "http://127.0.0.1:8001/v1",
    "http://127.0.0.1:8002/v1",
)
_DEFAULT_OPENAI_RETRIEVAL_BASE_URLS = ("https://api.openai.com/v1",)


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration for the agent pipeline."""

    # --- API keys (env-only) ---
    anthropic_api_key: str
    brave_api_key: str

    # --- Model selection ---
    haiku_model: str
    sonnet_model: str
    local_retrieval_model: str
    retrieval_backend: str
    local_retrieval_model_count: int
    local_retrieval_base_urls: tuple[str, ...]
    local_retrieval_api_key: str
    local_retrieval_timeout: int

    # --- Retrieval agent ---
    max_retrieval_turns: int
    max_search_results: int
    max_page_chars: int
    local_retrieval_parallel_agents: int
    local_retrieval_max_parallel_tool_calls: int

    # --- Inputs ---
    cv_path: str
    context_path: str
    schools_path: str

    # --- Judge ---
    retry_gap_fill: bool
    gap_fill_max_turns: int

    # --- Concurrency ---
    max_schools_parallel: int
    max_sonnet_parallel: int

    # --- HTTP ---
    http_timeout: int
    http_retries: int

    # --- Output ---
    output_dir: str
    logs_dir: str  # set to "" to disable trajectory logging

    # --- Optional OpenAI-compatible API retrieval ---
    openai_retrieval_model: str = "gpt-4.1-mini"
    openai_retrieval_base_urls: tuple[str, ...] = _DEFAULT_OPENAI_RETRIEVAL_BASE_URLS
    openai_retrieval_api_key: str = ""
    openai_retrieval_timeout: int = 120

    @classmethod
    def load(
        cls,
        yaml_path: str | Path | None = None,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        """Load config from YAML file + environment variables.

        Args:
            yaml_path: Path to YAML config file. Defaults to ``config.yaml``
                in the current directory. If the file does not exist, built-in
                defaults are used.
            overrides: Key-value pairs that take precedence over both the YAML
                file and defaults. Keys match the Config field names.
        """
        dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True), override=True)
        path = Path(yaml_path) if yaml_path else _DEFAULT_YAML_PATH
        raw = _load_yaml(path)
        ov = overrides or {}
        local_retrieval_base_urls = _as_tuple(
            ov.get(
                "local_retrieval_base_urls",
                _get(raw, "retrieval.local_base_urls", _DEFAULT_LOCAL_RETRIEVAL_BASE_URLS),
            )
        )
        local_retrieval_model_count = ov.get(
            "local_retrieval_model_count",
            _get(raw, "retrieval.local_model_count", len(local_retrieval_base_urls)),
        )
        openai_retrieval_base_urls = _as_tuple(
            ov.get(
                "openai_retrieval_base_urls",
                _get(raw, "retrieval.openai_base_urls", _DEFAULT_OPENAI_RETRIEVAL_BASE_URLS),
            )
        )

        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            brave_api_key=os.environ.get("BRAVE_API_KEY", ""),
            haiku_model=ov.get(
                "haiku_model", _get(raw, "models.haiku", "claude-haiku-4-5-20251001")
            ),
            sonnet_model=ov.get("sonnet_model", _get(raw, "models.sonnet", "claude-sonnet-4-6")),
            local_retrieval_model=ov.get(
                "local_retrieval_model",
                _get(raw, "models.local_retrieval", "Qwen/Qwen3.6-35B-A3B-FP8"),
            ),
            openai_retrieval_model=ov.get(
                "openai_retrieval_model",
                _get(raw, "models.openai_retrieval", "gpt-4.1-mini"),
            ),
            retrieval_backend=ov.get(
                "retrieval_backend",
                _get(raw, "retrieval.backend", "local_qwen_vllm"),
            ),
            local_retrieval_model_count=local_retrieval_model_count,
            local_retrieval_base_urls=local_retrieval_base_urls,
            local_retrieval_api_key=ov.get(
                "local_retrieval_api_key",
                os.environ.get("VLLM_API_KEY", _get(raw, "retrieval.local_api_key", "")),
            ),
            local_retrieval_timeout=ov.get(
                "local_retrieval_timeout", _get(raw, "retrieval.local_timeout", 600)
            ),
            openai_retrieval_base_urls=openai_retrieval_base_urls,
            openai_retrieval_api_key=ov.get(
                "openai_retrieval_api_key",
                os.environ.get(
                    "OPENAI_COMPATIBLE_API_KEY",
                    os.environ.get(
                        "OPENAI_API_KEY",
                        _get(raw, "retrieval.openai_api_key", ""),
                    ),
                ),
            ),
            openai_retrieval_timeout=ov.get(
                "openai_retrieval_timeout",
                _get(raw, "retrieval.openai_timeout", 120),
            ),
            max_retrieval_turns=ov.get("max_retrieval_turns", _get(raw, "retrieval.max_turns", 15)),
            max_search_results=ov.get(
                "max_search_results", _get(raw, "retrieval.max_search_results", 5)
            ),
            max_page_chars=ov.get("max_page_chars", _get(raw, "retrieval.max_page_chars", 30_000)),
            local_retrieval_parallel_agents=ov.get(
                "local_retrieval_parallel_agents",
                _get(raw, "retrieval.local_parallel_agents", local_retrieval_model_count),
            ),
            local_retrieval_max_parallel_tool_calls=ov.get(
                "local_retrieval_max_parallel_tool_calls",
                _get(raw, "retrieval.local_max_parallel_tool_calls", 8),
            ),
            cv_path=ov.get("cv_path", _get(raw, "input.cv", "input/cv.md")),
            context_path=ov.get("context_path", _get(raw, "input.context", "input/context.md")),
            schools_path=ov.get("schools_path", _get(raw, "input.schools", "input/schools.json")),
            retry_gap_fill=ov.get("retry_gap_fill", _get(raw, "judge.retry_gap_fill", True)),
            gap_fill_max_turns=ov.get(
                "gap_fill_max_turns", _get(raw, "judge.gap_fill_max_turns", 5)
            ),
            max_schools_parallel=ov.get(
                "max_schools_parallel", _get(raw, "concurrency.max_schools_parallel", 8)
            ),
            max_sonnet_parallel=ov.get(
                "max_sonnet_parallel", _get(raw, "concurrency.max_sonnet_parallel", 8)
            ),
            http_timeout=ov.get("http_timeout", _get(raw, "http.timeout", 20)),
            http_retries=ov.get("http_retries", _get(raw, "http.retries", 2)),
            output_dir=ov.get("output_dir", _get(raw, "output.dir", "output")),
            logs_dir=ov.get("logs_dir", _get(raw, "logs.dir", "logs")),
        )

    def validate(self) -> list[str]:
        """Return a list of configuration errors. Empty list means valid."""
        errors: list[str] = []
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required (set in environment or .env)")
        if not self.brave_api_key:
            errors.append("BRAVE_API_KEY is required (set in environment or .env)")
        backend_spec = get_retrieval_backend_spec(self.retrieval_backend)
        if backend_spec is None:
            allowed = ", ".join(retrieval_backend_ids())
            errors.append(f"retrieval.backend must be one of: {allowed}")
        if backend_spec is None or backend_spec.is_local:
            errors.extend(
                _validate_positive_int(
                    "retrieval.local_model_count",
                    self.local_retrieval_model_count,
                )
            )
            if not self.local_retrieval_base_urls:
                errors.append("retrieval.local_base_urls must include at least one endpoint")
            elif _is_positive_int(self.local_retrieval_model_count) and (
                len(self.local_retrieval_base_urls) != self.local_retrieval_model_count
            ):
                errors.append(
                    "retrieval.local_model_count must match the number of "
                    "retrieval.local_base_urls endpoints"
                )
            errors.extend(
                _validate_positive_int(
                    "retrieval.local_parallel_agents",
                    self.local_retrieval_parallel_agents,
                )
            )
            errors.extend(
                _validate_positive_int(
                    "retrieval.local_max_parallel_tool_calls",
                    self.local_retrieval_max_parallel_tool_calls,
                )
            )
            errors.extend(
                _validate_positive_int("retrieval.local_timeout", self.local_retrieval_timeout)
            )
        if backend_spec and backend_spec.is_openai_compatible_api:
            if not isinstance(self.openai_retrieval_model, str) or (
                not self.openai_retrieval_model.strip()
            ):
                errors.append("models.openai_retrieval must be set")
            if not self.openai_retrieval_base_urls:
                errors.append("retrieval.openai_base_urls must include at least one endpoint")
            if not self.openai_retrieval_api_key:
                errors.append(
                    "OPENAI_API_KEY or OPENAI_COMPATIBLE_API_KEY is required "
                    "for retrieval.backend=openai_compatible"
                )
            errors.extend(
                _validate_positive_int("retrieval.openai_timeout", self.openai_retrieval_timeout)
            )
        errors.extend(_validate_positive_int("retrieval.max_turns", self.max_retrieval_turns))
        errors.extend(
            _validate_positive_int("retrieval.max_search_results", self.max_search_results)
        )
        errors.extend(
            _validate_positive_int("retrieval.max_page_chars", self.max_page_chars)
        )
        errors.extend(
            _validate_positive_int("judge.gap_fill_max_turns", self.gap_fill_max_turns)
        )
        errors.extend(
            _validate_positive_int("concurrency.max_schools_parallel", self.max_schools_parallel)
        )
        errors.extend(
            _validate_positive_int("concurrency.max_sonnet_parallel", self.max_sonnet_parallel)
        )
        errors.extend(_validate_positive_int("http.timeout", self.http_timeout))
        if (
            isinstance(self.http_retries, bool)
            or not isinstance(self.http_retries, int)
            or self.http_retries < 0
        ):
            errors.append("http.retries must be an integer >= 0")
        return errors

    @property
    def retrieval_model(self) -> str:
        """Return the concrete model used for retrieval and gap-fill."""
        backend_spec = get_retrieval_backend_spec(self.retrieval_backend)
        if backend_spec is None:
            return self.local_retrieval_model
        return str(getattr(self, backend_spec.model_field))

    @property
    def uses_local_retrieval(self) -> bool:
        """Whether retrieval should call local OpenAI-compatible endpoints."""
        backend_spec = get_retrieval_backend_spec(self.retrieval_backend)
        return bool(backend_spec and backend_spec.is_local)

    @property
    def local_retrieval_endpoints(self) -> tuple[str, ...]:
        """Return the configured local retrieval endpoints."""
        return self.local_retrieval_base_urls


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict if the file doesn't exist."""
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _get(data: dict[str, Any], dotted_key: str, default: Any) -> Any:
    """Traverse a nested dict using a dotted key path (e.g. ``models.haiku``)."""
    keys = dotted_key.split(".")
    current: Any = data
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return default
        current = current[k]
    return current


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def _validate_positive_int(name: str, value: Any) -> list[str]:
    if not _is_positive_int(value):
        return [f"{name} must be an integer >= 1"]
    return []


def _is_positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1
