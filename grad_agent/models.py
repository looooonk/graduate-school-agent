"""Pydantic models for all structured data flowing through the pipeline.

Mirrors the schemas defined in DESIGN.md with minor ergonomic additions
(optional fields, default factories) so partially-populated profiles can
exist mid-retrieval without validation failures.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Stage 1 — SchoolProfile
# ---------------------------------------------------------------------------

class Requirements(BaseModel):
    """Formal application requirements.

    Fields that are commonly booleans (gre_required, statement_of_purpose) also
    accept descriptive strings (e.g. "Optional for 2025-2026"), because the
    model often produces more informative text than a bare yes/no.
    """

    gre_required: str | bool | None = None
    gpa_minimum: str | None = None
    statement_of_purpose: str | bool | None = None
    recommendations: str | int | None = None
    other: list[str] = Field(default_factory=list)

    @field_validator("other", mode="before")
    @classmethod
    def coerce_other(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v or []


class ApplicantReports(BaseModel):
    """Informal stats aggregated from GradCafe, Reddit, etc."""

    typical_gpa: str | None = None
    typical_gre: str | None = None
    acceptance_signals: str | None = None


class SchoolProfile(BaseModel):
    """Complete research profile for a single graduate program."""

    school_name: str
    program_name: str
    deadline: str | None = None
    application_fee: str | None = None
    requirements: Requirements = Field(default_factory=Requirements)
    essay_prompts: list[str] = Field(default_factory=list)
    research_areas: list[str] = Field(default_factory=list)
    advisor_candidates: list[str] = Field(default_factory=list)
    applicant_reports: ApplicantReports = Field(default_factory=ApplicantReports)
    sources: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("deadline", mode="before")
    @classmethod
    def coerce_deadline(cls, v: Any) -> str | None:
        """Accept a dict of deadline types (e.g. {'fall': 'March 1'}) → single string."""
        if isinstance(v, dict):
            return "; ".join(f"{k}: {val}" for k, val in v.items() if val)
        return v

    @field_validator("essay_prompts", "research_areas", mode="before")
    @classmethod
    def coerce_str_to_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v or []

    @field_validator("advisor_candidates", mode="before")
    @classmethod
    def coerce_advisor_candidates(cls, v: Any) -> list[str]:
        """Accept list[str] or list[dict] (with name/research/url keys)."""
        if not v:
            return []
        result: list[str] = []
        for item in v:
            if isinstance(item, dict):
                name = item.get("name", "")
                research = (
                    item.get("research")
                    or item.get("focus")
                    or item.get("research_focus", "")
                )
                url = item.get("url") or item.get("profile_url", "")
                entry = " — ".join(p for p in [name, research] if p)
                if url:
                    entry += f" ({url})"
                result.append(entry)
            else:
                result.append(str(item))
        return result

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, v: Any) -> list[str]:
        """Accept list[str] or a dict mapping category → URL."""
        if isinstance(v, dict):
            return [str(url) for url in v.values() if url]
        return v or []

    @field_validator("notes", mode="before")
    @classmethod
    def coerce_notes(cls, v: Any) -> str | None:
        if isinstance(v, list):
            return "\n".join(str(x) for x in v if x)
        return v


# ---------------------------------------------------------------------------
# Stage 2 — JudgeReport
# ---------------------------------------------------------------------------

class QualityRating(StrEnum):
    PASS = "pass"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class FlaggedField(BaseModel):
    field: str
    reason: str


class JudgeReport(BaseModel):
    """Quality / coverage assessment produced by the Sonnet judge."""

    overall_quality: QualityRating
    flagged_fields: list[FlaggedField] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)
    notes: str | None = None


# ---------------------------------------------------------------------------
# Stage 3 — FitAssessment
# ---------------------------------------------------------------------------

class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FitAssessment(BaseModel):
    """CV-aware fit assessment for a single school."""

    overall_score: float = Field(ge=0.0, le=1.0)
    research_alignment: str
    advisor_candidates: list[str] = Field(default_factory=list)
    competitiveness: str
    gaps: str
    confidence: ConfidenceLevel


# ---------------------------------------------------------------------------
# Pipeline composite
# ---------------------------------------------------------------------------

class SchoolResult(BaseModel):
    """Aggregated result for a single school after all stages complete."""

    profile: SchoolProfile
    judge: JudgeReport | None = None
    fit: FitAssessment | None = None
    error: str | None = None
