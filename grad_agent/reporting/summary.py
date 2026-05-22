"""Rebuild summary tables from rendered school reports."""

from __future__ import annotations

import re
from pathlib import Path

from grad_agent.models import FitAssessment, Requirements, SchoolProfile
from grad_agent.reporting.markdown import render_summary_table

_TITLE_RE = re.compile(r"^#\s+(.+?)\s+(?:\u2014|-)\s+(.+?)\s*$", re.MULTILINE)
_DEADLINE_RE = re.compile(r"^\*\*Deadline\*\*:\s*(.*?)\s*$", re.MULTILINE)
_GRE_RE = re.compile(r"^-\s+\*\*GRE\*\*:\s*(.*?)\s*$", re.MULTILINE)
_SCORE_RE = re.compile(r"^-\s+\*\*Overall score\*\*:\s*([0-9]*\.?[0-9]+)", re.MULTILINE)
_CONFIDENCE_RE = re.compile(r"^-\s+\*\*Confidence\*\*:\s*(high|medium|low)\s*$", re.MULTILINE)


def rebuild_summary_from_profiles(input_path: Path, output_dir: Path | None = None) -> Path:
    """Read rendered profile Markdown and write a rebuilt summary table."""
    profiles = load_profile_summaries(input_path)
    target_dir = output_dir or (input_path.parent if input_path.is_file() else input_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "summary.md"
    summary_path.write_text(render_summary_table(profiles), encoding="utf-8")
    return summary_path


def load_profile_summaries(input_path: Path) -> list[tuple[SchoolProfile, FitAssessment | None]]:
    """Load summary fields from one profile file or a directory of profile files."""
    paths = _profile_paths(input_path)
    if not paths:
        raise ValueError(f"No profile Markdown files found in {input_path}")
    return [_parse_profile_summary(path) for path in paths]


def _profile_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    paths = sorted(input_path.glob("*_profile.md"))
    if paths:
        return paths
    return sorted(path for path in input_path.glob("*.md") if path.name != "summary.md")


def _parse_profile_summary(path: Path) -> tuple[SchoolProfile, FitAssessment | None]:
    markdown = path.read_text(encoding="utf-8")
    title = _match_required(_TITLE_RE, markdown, path, "title")
    school_name, program_name = title.group(1).strip(), title.group(2).strip()

    deadline_match = _DEADLINE_RE.search(markdown)
    deadline = _clean_optional(deadline_match.group(1)) if deadline_match else None

    gre_match = _GRE_RE.search(markdown)
    gre_policy = _clean_optional(gre_match.group(1)) if gre_match else None

    profile = SchoolProfile(
        school_name=school_name,
        program_name=program_name,
        deadline=deadline,
        requirements=Requirements(gre_policy=gre_policy),
    )
    return profile, _parse_fit_summary(markdown)


def _parse_fit_summary(markdown: str) -> FitAssessment | None:
    score_match = _SCORE_RE.search(markdown)
    confidence_match = _CONFIDENCE_RE.search(markdown)
    if not score_match or not confidence_match:
        return None
    return FitAssessment(
        overall_score=float(score_match.group(1)),
        confidence=confidence_match.group(1),
        research_alignment="",
        competitiveness="",
        gaps="",
    )


def _clean_optional(value: str) -> str | None:
    text = value.replace("[unverified]", "").strip()
    text = text.rstrip()
    if text in {"", "N/A", "*not found*"}:
        return None
    return text


def _match_required(pattern: re.Pattern[str], text: str, path: Path, field: str) -> re.Match[str]:
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Could not parse {field} from {path}")
    return match
