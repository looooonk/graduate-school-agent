"""Pydantic models for structured data shared across agents."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class GREPolicy(StrEnum):
    REQUIRED = "Required"
    CONSIDERED = "Considered"
    NOT_CONSIDERED = "Not Considered"


class Requirements(BaseModel):
    """Formal application requirements.

    Fields that are commonly booleans (gre_required, statement_of_purpose) also
    accept descriptive strings (e.g. "Optional for 2025-2026"), because the
    model often produces more informative text than a bare yes/no.
    """

    gre_required: str | bool | None = None
    gre_policy: GREPolicy | None = None
    gpa_minimum: str | None = None
    statement_of_purpose: str | bool | None = None
    recommendations: str | int | None = None
    other: list[str] = Field(default_factory=list)

    @field_validator("gre_policy", mode="before")
    @classmethod
    def coerce_gre_policy(cls, v: Any) -> GREPolicy | None:
        return _normalize_gre_policy(v)

    @field_validator("other", mode="before")
    @classmethod
    def coerce_other(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v or []

    @model_validator(mode="after")
    def infer_gre_policy(self) -> Requirements:
        if self.gre_policy is None:
            self.gre_policy = _infer_gre_policy(self.gre_required, self.other)
        return self


class ApplicantReports(BaseModel):
    """Informal stats aggregated from GradCafe, Reddit, etc."""

    typical_gpa: str | None = None
    typical_gre: str | None = None
    acceptance_signals: str | None = None


def _normalize_gre_policy(v: Any) -> GREPolicy | None:
    if v is None:
        return None
    if isinstance(v, GREPolicy):
        return v
    if isinstance(v, bool):
        return GREPolicy.REQUIRED if v else GREPolicy.NOT_CONSIDERED

    text = str(v).strip()
    if not text:
        return None

    lowered = text.lower().replace("_", " ").replace("-", " ")
    compact = " ".join(lowered.split())
    if compact in {"required", "require", "yes"}:
        return GREPolicy.REQUIRED
    if compact in {"considered", "optional", "recommended"}:
        return GREPolicy.CONSIDERED
    if compact in {"not considered", "not required", "waived", "no"}:
        return GREPolicy.NOT_CONSIDERED

    return _infer_gre_policy(text, [])


def _infer_gre_policy(gre_required: str | bool | None, other: list[str]) -> GREPolicy | None:
    gre_other = [
        item
        for item in other
        if "gre" in str(item).lower() or "graduate record" in str(item).lower()
    ]
    evidence = " ".join(
        str(part)
        for part in [gre_required, *gre_other]
        if part is not None and str(part).strip()
    ).lower()

    if any(
        phrase in evidence
        for phrase in [
            "not considered",
            "will not be considered",
            "will not be reviewed",
            "not be reviewed",
            "do not submit",
            "not accepted",
            "will not accept",
        ]
    ):
        return GREPolicy.NOT_CONSIDERED

    if any(
        phrase in evidence
        for phrase in [
            "optional",
            "considered",
            "recommended",
            "encouraged",
            "may submit",
            "may be submitted",
            "accepted",
            "reviewed",
        ]
    ):
        return GREPolicy.CONSIDERED

    if isinstance(gre_required, bool):
        return GREPolicy.REQUIRED if gre_required else GREPolicy.NOT_CONSIDERED

    if "not required" in evidence or "waived" in evidence:
        return GREPolicy.NOT_CONSIDERED

    if any(
        phrase in evidence
        for phrase in [
            "required",
            "must submit",
            "mandatory",
            "need to submit",
            "need submit",
        ]
    ):
        return GREPolicy.REQUIRED

    return None


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
        """Accept a dict of deadline types as a single string."""
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
        """Accept list[str] or a dict mapping category to URL."""
        if isinstance(v, dict):
            return [str(url) for url in v.values() if url]
        return v or []

    @field_validator("notes", mode="before")
    @classmethod
    def coerce_notes(cls, v: Any) -> str | None:
        if isinstance(v, list):
            return "\n".join(str(x) for x in v if x)
        return v


class QualityRating(StrEnum):
    PASS = "pass"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class FlaggedField(BaseModel):
    field: str
    reason: str


class JudgeReport(BaseModel):
    """Quality / coverage assessment produced by the judge stage."""

    overall_quality: QualityRating
    flagged_fields: list[FlaggedField] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)
    notes: str | None = None


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


class SchoolResult(BaseModel):
    """Aggregated result for a single school after all stages complete."""

    profile: SchoolProfile
    judge: JudgeReport | None = None
    fit: FitAssessment | None = None
    error: str | None = None
