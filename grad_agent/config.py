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

_DEFAULT_YAML_PATH = Path("config.yaml")
_DEFAULT_LOCAL_RETRIEVAL_BASE_URLS = (
    "http://127.0.0.1:8001/v1",
    "http://127.0.0.1:8002/v1",
    "http://127.0.0.1:8003/v1",
    "http://127.0.0.1:8004/v1",
)
_RETRIEVAL_BACKENDS = {"anthropic_haiku", "local_qwen_vllm"}


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
    local_retrieval_base_urls: tuple[str, ...]
    local_retrieval_api_key: str
    local_retrieval_timeout: int

    # --- Retrieval agent ---
    max_retrieval_turns: int
    max_search_results: int
    max_page_chars: int

    # --- Judge ---
    retry_gap_fill: bool
    gap_fill_max_turns: int

    # --- Concurrency ---
    max_schools_parallel: int

    # --- HTTP ---
    http_timeout: int
    http_retries: int

    # --- Output ---
    output_dir: str
    logs_dir: str  # set to "" to disable trajectory logging

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
            retrieval_backend=ov.get(
                "retrieval_backend",
                _get(raw, "retrieval.backend", "local_qwen_vllm"),
            ),
            local_retrieval_base_urls=_as_tuple(
                ov.get(
                    "local_retrieval_base_urls",
                    _get(raw, "retrieval.local_base_urls", _DEFAULT_LOCAL_RETRIEVAL_BASE_URLS),
                )
            ),
            local_retrieval_api_key=ov.get(
                "local_retrieval_api_key",
                os.environ.get("VLLM_API_KEY", _get(raw, "retrieval.local_api_key", "")),
            ),
            local_retrieval_timeout=ov.get(
                "local_retrieval_timeout", _get(raw, "retrieval.local_timeout", 600)
            ),
            max_retrieval_turns=ov.get("max_retrieval_turns", _get(raw, "retrieval.max_turns", 15)),
            max_search_results=ov.get(
                "max_search_results", _get(raw, "retrieval.max_search_results", 5)
            ),
            max_page_chars=ov.get("max_page_chars", _get(raw, "retrieval.max_page_chars", 30_000)),
            retry_gap_fill=ov.get("retry_gap_fill", _get(raw, "judge.retry_gap_fill", True)),
            gap_fill_max_turns=ov.get(
                "gap_fill_max_turns", _get(raw, "judge.gap_fill_max_turns", 5)
            ),
            max_schools_parallel=ov.get(
                "max_schools_parallel", _get(raw, "concurrency.max_schools_parallel", 3)
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
        if self.retrieval_backend not in _RETRIEVAL_BACKENDS:
            allowed = ", ".join(sorted(_RETRIEVAL_BACKENDS))
            errors.append(f"retrieval.backend must be one of: {allowed}")
        if self.retrieval_backend == "local_qwen_vllm" and not self.local_retrieval_base_urls:
            errors.append("retrieval.local_base_urls must include at least one endpoint")
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
        errors.extend(_validate_positive_int("http.timeout", self.http_timeout))
        errors.extend(
            _validate_positive_int("retrieval.local_timeout", self.local_retrieval_timeout)
        )
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
        if self.retrieval_backend == "local_qwen_vllm":
            return self.local_retrieval_model
        return self.haiku_model

    @property
    def uses_local_retrieval(self) -> bool:
        """Whether retrieval should call local OpenAI-compatible vLLM endpoints."""
        return self.retrieval_backend == "local_qwen_vllm"


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
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return [f"{name} must be an integer >= 1"]
    return []
