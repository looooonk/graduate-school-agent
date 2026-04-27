"""JSON helpers for model outputs."""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from text or a fenced code block."""
    stripped = text.strip()
    if not stripped:
        return None

    for candidate in _json_candidates(stripped):
        parsed = _try_parse_object(candidate)
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for i, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    if "```" not in text:
        return candidates

    for block in text.split("```"):
        block = block.strip()
        if block.startswith("json"):
            block = block[4:].strip()
        if block:
            candidates.append(block)
    return candidates


def _try_parse_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
