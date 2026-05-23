"""Helpers for CLI input loading and override assembly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_CONTEXT_PATH = Path("input/context.md")


def load_schools(
    args: argparse.Namespace, schools_path: Path | None = None
) -> list[tuple[str, str]]:
    """Parse school specifications from CLI args."""
    if args.schools:
        data = _load_schools_json(args.schools)
        return _parse_schools(data)

    if args.school:
        if not args.program:
            _exit("Error: --program is required when using --school")
        return [(args.school, args.program)]

    if args.program:
        _exit("Error: --program requires --school")

    if schools_path is None:
        _exit("Error: --schools or --school is required")

    return _parse_schools(_load_schools_json(schools_path))


def _parse_schools(data: list[Any]) -> list[tuple[str, str]]:
    schools = []
    for entry in data:
        if not isinstance(entry, dict):
            _exit(f"Error: each entry in schools JSON must be an object. Got: {entry}")
        school = entry.get("school")
        program = entry.get("program")
        if not school or not program:
            _exit(
                "Error: each entry in schools JSON must have 'school' and "
                f"'program' keys. Got: {entry}"
            )
        schools.append((school, program))
    return schools


def read_context(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    if path != DEFAULT_CONTEXT_PATH:
        _exit(f"Error: context file not found: {path}")
    return ""


def read_required_text(path: Path, label: str) -> str:
    if not path.exists():
        _exit(f"Error: {label} file not found: {path}")
    return path.read_text(encoding="utf-8")


def config_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    _set_if_present(overrides, "max_retrieval_turns", args.max_turns)
    _set_if_present(overrides, "max_schools_parallel", args.max_parallel)
    _set_if_present(overrides, "retrieval_backend", args.retrieval_backend)
    _set_path_if_present(overrides, "cv_path", args.cv)
    _set_path_if_present(overrides, "context_path", args.context)
    _set_path_if_present(overrides, "schools_path", args.schools)
    if args.no_gap_fill:
        overrides["retry_gap_fill"] = False
    if args.output is not None:
        overrides["output_dir"] = str(args.output)
    return overrides


def _set_path_if_present(overrides: dict[str, object], key: str, value: Path | None) -> None:
    if value is not None:
        overrides[key] = str(value)


def _load_schools_json(path: Path) -> list[Any]:
    if not path.exists():
        _exit(f"Error: schools file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _exit(f"Error: schools file is not valid JSON: {exc}")
    if not isinstance(data, list):
        _exit("Error: schools JSON must be a list of objects")
    return data


def _set_if_present(overrides: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        overrides[key] = value


def _exit(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)
