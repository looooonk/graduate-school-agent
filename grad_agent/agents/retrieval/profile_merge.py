"""Merge and evidence helpers for parallel retrieval workers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from grad_agent.models import SchoolProfile
from grad_agent.reporting.stats import StageStats


def accumulate_stats(target: StageStats, source: StageStats) -> None:
    target.input_tokens += source.input_tokens
    target.output_tokens += source.output_tokens
    target.cache_read_tokens += source.cache_read_tokens
    target.cache_creation_tokens += source.cache_creation_tokens
    target.api_calls += source.api_calls
    target.tool_calls += source.tool_calls


def profile_has_evidence(profile: SchoolProfile) -> bool:
    return any(
        [
            profile.deadline,
            profile.application_fee,
            profile.requirements.model_dump(exclude_none=True, exclude_defaults=True),
            profile.essay_prompts,
            profile.research_areas,
            profile.advisor_candidates,
            profile.applicant_reports.model_dump(exclude_none=True, exclude_defaults=True),
            profile.sources,
        ]
    )


def merge_profiles(
    school_name: str,
    program_name: str,
    profiles: list[SchoolProfile],
) -> SchoolProfile:
    merged = SchoolProfile(school_name=school_name, program_name=program_name)
    merged.deadline = _first_text(profile.deadline for profile in profiles)
    merged.application_fee = _first_text(profile.application_fee for profile in profiles)
    merged.requirements.gre_required = _first_value(
        profile.requirements.gre_required for profile in profiles
    )
    merged.requirements.gre_policy = _first_value(
        profile.requirements.gre_policy for profile in profiles
    )
    merged.requirements.gpa_minimum = _first_text(
        profile.requirements.gpa_minimum for profile in profiles
    )
    merged.requirements.statement_of_purpose = _first_value(
        profile.requirements.statement_of_purpose for profile in profiles
    )
    merged.requirements.recommendations = _first_value(
        profile.requirements.recommendations for profile in profiles
    )
    merged.requirements.other = _unique(
        item for profile in profiles for item in profile.requirements.other
    )
    merged.essay_prompts = _unique(
        item for profile in profiles for item in profile.essay_prompts
    )
    merged.research_areas = _unique(
        item for profile in profiles for item in profile.research_areas
    )
    merged.advisor_candidates = _unique(
        item for profile in profiles for item in profile.advisor_candidates
    )
    merged.applicant_reports.typical_gpa = _first_text(
        profile.applicant_reports.typical_gpa for profile in profiles
    )
    merged.applicant_reports.typical_gre = _first_text(
        profile.applicant_reports.typical_gre for profile in profiles
    )
    merged.applicant_reports.acceptance_signals = _first_text(
        profile.applicant_reports.acceptance_signals for profile in profiles
    )
    merged.sources = _unique(item for profile in profiles for item in profile.sources)
    merged.notes = "\n".join(
        _unique(profile.notes for profile in profiles if profile.notes)
    ) or None
    return merged


def _first_text(values: Iterable[Any]) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_value(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
