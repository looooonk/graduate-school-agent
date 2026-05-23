"""Filesystem path helpers for report artifacts."""

from __future__ import annotations


def safe_filename(school: str, program: str) -> str:
    """Convert school + program to a filesystem-safe slug."""
    combined = f"{school}_{program}".lower()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in combined)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")
