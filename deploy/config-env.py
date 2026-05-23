#!/usr/bin/env python3
from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_SYSTEM_PACKAGES = (
    "curl",
    "git",
    "build-essential",
    "tmux",
    "libcairo2",
    "libpango-1.0-0",
    "libpangoft2-1.0-0",
    "libgdk-pixbuf-2.0-0",
    "shared-mime-info",
)
DEFAULT_PIP_PACKAGES = ("vllm",)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: config-env.py CONFIG_PATH", file=sys.stderr)
        return 2

    config_path = Path(sys.argv[1]).expanduser().resolve()
    if not config_path.is_file():
        print(f"config file not found: {config_path}", file=sys.stderr)
        return 1

    config = _load_yaml(config_path)
    try:
        values = _deployment_values(config, config_path.parent)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for key, value in values.items():
        print(f"export {key}={shlex.quote(str(value))}")
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_simple_yaml(path)

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    records: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        records.append((len(line) - len(line.lstrip(" ")), line.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for idx, (indent, content) in enumerate(records):
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"unsupported YAML list placement in {path}")
            parent.append(_parse_scalar(content[2:].strip()))
            continue

        if ":" not in content or not isinstance(parent, dict):
            raise ValueError(f"unsupported YAML line in {path}: {content}")

        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue

        child: Any = [] if _next_child_is_list(records, idx, indent) else {}
        parent[key] = child
        stack.append((indent, child))

    return root


def _next_child_is_list(records: list[tuple[int, str]], idx: int, indent: int) -> bool:
    for next_indent, content in records[idx + 1 :]:
        if next_indent <= indent:
            return False
        return content.startswith("- ")
    return False


def _parse_scalar(value: str) -> Any:
    if value in {"''", '""'}:
        return ""
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def _deployment_values(config: dict[str, Any], config_dir: Path) -> dict[str, str | int]:
    retrieval = _dict_at(config, "retrieval")
    models = _dict_at(config, "models")
    deploy = _dict_at(config, "deploy")

    model_id = _as_str(models.get("local_retrieval", "Qwen/Qwen3.6-35B-A3B-FP8"))
    endpoints = _as_list(retrieval.get("local_base_urls", []))
    model_count = _as_int(
        retrieval.get("local_model_count", len(endpoints)), "retrieval.local_model_count"
    )
    if len(endpoints) != model_count:
        raise ValueError(
            "retrieval.local_model_count must match retrieval.local_base_urls for deployment"
        )

    ports = [_endpoint_port(endpoint) for endpoint in endpoints]
    log_dir = _resolve_path(config_dir, _as_str(deploy.get("log_dir", "logs/vllm")))
    vllm_args = deploy.get("vllm_args", ["--trust-remote-code"])
    system_packages = deploy.get("system_packages", list(DEFAULT_SYSTEM_PACKAGES))
    pip_packages = deploy.get("pip_packages", list(DEFAULT_PIP_PACKAGES))

    return {
        "DEPLOY_MODEL_ID": model_id,
        "DEPLOY_MODEL_COUNT": model_count,
        "DEPLOY_VLLM_ENDPOINTS": " ".join(endpoints),
        "DEPLOY_VLLM_PORTS": " ".join(str(port) for port in ports),
        "DEPLOY_VLLM_HOST": _as_str(deploy.get("host", "0.0.0.0")),
        "DEPLOY_VLLM_ARGS": " ".join(_as_list(vllm_args)),
        "DEPLOY_VLLM_LOG_DIR": str(log_dir),
        "DEPLOY_MICROMAMBA_ENV": _as_str(
            deploy.get("micromamba_env", "graduate-school-agent")
        ),
        "DEPLOY_PYTHON_VERSION": _as_str(deploy.get("python_version", "3.11")),
        "DEPLOY_SYSTEM_PACKAGES": " ".join(_as_list(system_packages)),
        "DEPLOY_PIP_PACKAGES": " ".join(_as_list(pip_packages)),
    }


def _dict_at(data: dict[str, Any], path: str) -> dict[str, Any]:
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _as_str(value: Any) -> str:
    return str(value)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _endpoint_port(endpoint: str) -> int:
    parsed = urlparse(endpoint)
    if parsed.port is None:
        raise ValueError(f"retrieval.local_base_urls endpoint must include a port: {endpoint}")
    return parsed.port


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
