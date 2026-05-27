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


class FitDimensionScore(BaseModel):
    """Evidence-backed 0-10 score for one fit dimension."""

    score: float = Field(ge=0.0, le=10.0)
    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def coerce_score(cls, v: Any) -> float:
        if isinstance(v, str):
            return float(v.strip().split("/", 1)[0])
        return v

    @field_validator("positive_evidence", "negative_evidence", mode="before")
    @classmethod
    def coerce_evidence(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v or []


class FitScoreBreakdown(BaseModel):
    """Structured inputs used to compute a deterministic overall fit score."""

    research_alignment: FitDimensionScore
    advisor_fit: FitDimensionScore
    applicant_competitiveness: FitDimensionScore
    program_structure_fit: FitDimensionScore
    constraint_fit: FitDimensionScore


FIT_SCORE_DIMENSIONS = (
    "research_alignment",
    "advisor_fit",
    "applicant_competitiveness",
    "program_structure_fit",
    "constraint_fit",
)

FIT_SCORE_WEIGHTS_DEFAULT = {
    "research_alignment": 0.35,
    "advisor_fit": 0.25,
    "applicant_competitiveness": 0.20,
    "program_structure_fit": 0.10,
    "constraint_fit": 0.10,
}

FIT_SCORE_CAPS = {
    "no_named_advisor": 0.65,
    "generic_area_match_only": 0.70,
    "degree_structure_mismatch": 0.65,
    "severe_constraint_mismatch": 0.70,
    "wrong_or_unverified_primary_advisors": 0.75,
    "insufficient_profile_evidence": 0.80,
    "no_competitiveness_evidence": 0.85,
}


def compute_overall_fit_score(
    breakdown: FitScoreBreakdown,
    score_caps: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    weights = weights or FIT_SCORE_WEIGHTS_DEFAULT
    total_weight = sum(weights[dim] for dim in FIT_SCORE_DIMENSIONS)
    raw = sum(
        getattr(breakdown, dim).score / 10.0 * weights[dim]
        for dim in FIT_SCORE_DIMENSIONS
    ) / total_weight
    cap = min(
        [1.0]
        + [
            FIT_SCORE_CAPS[cap]
            for cap in _normalize_fit_score_caps(score_caps)
            if cap in FIT_SCORE_CAPS
        ]
    )
    return round(min(raw, cap), 4)


def _normalize_fit_score_caps(score_caps: list[str] | None) -> list[str]:
    return [_normalize_fit_score_cap(cap) for cap in score_caps or []]


def _normalize_fit_score_cap(cap: Any) -> str:
    if isinstance(cap, dict):
        cap = cap.get("cap") or cap.get("name") or cap.get("type") or ""
    text = str(cap).strip().lower().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in text.split("_") if part)


class FitAssessment(BaseModel):
    """CV-aware fit assessment for a single school."""

    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    score_breakdown: FitScoreBreakdown | None = None
    score_caps: list[str] = Field(default_factory=list)
    scoring_notes: str | None = None
    research_alignment: str
    advisor_candidates: list[str] = Field(default_factory=list)
    competitiveness: str
    gaps: str
    confidence: ConfidenceLevel

    @field_validator("score_caps", mode="before")
    @classmethod
    def coerce_score_caps(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [_normalize_fit_score_cap(v)] if v.strip() else []
        return [_normalize_fit_score_cap(item) for item in v or []]

    @model_validator(mode="after")
    def score_from_breakdown(self) -> FitAssessment:
        if self.score_breakdown is not None:
            self.overall_score = compute_overall_fit_score(
                self.score_breakdown,
                self.score_caps,
            )
        elif "overall_score" not in self.model_fields_set:
            raise ValueError("FitAssessment requires score_breakdown or overall_score")
        return self


class SchoolResult(BaseModel):
    """Aggregated result for a single school after all stages complete."""

    profile: SchoolProfile
    judge: JudgeReport | None = None
    fit: FitAssessment | None = None
    error: str | None = None
