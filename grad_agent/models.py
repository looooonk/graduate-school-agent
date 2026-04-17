"""Pydantic models for all structured data flowing through the pipeline.

Mirrors the schemas defined in DESIGN.md with minor ergonomic additions
(optional fields, default factories) so partially-populated profiles can
exist mid-retrieval without validation failures.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage 1 — SchoolProfile
# ---------------------------------------------------------------------------

class Requirements(BaseModel):
    """Formal application requirements."""

    gre_required: Optional[bool] = None
    gpa_minimum: Optional[str] = None
    statement_of_purpose: Optional[bool] = None
    recommendations: Optional[int] = None
    other: list[str] = Field(default_factory=list)


class ApplicantReports(BaseModel):
    """Informal stats aggregated from GradCafe, Reddit, etc."""

    typical_gpa: Optional[str] = None
    typical_gre: Optional[str] = None
    acceptance_signals: Optional[str] = None


class SchoolProfile(BaseModel):
    """Complete research profile for a single graduate program."""

    school_name: str
    program_name: str
    deadline: Optional[str] = None
    application_fee: Optional[str] = None
    requirements: Requirements = Field(default_factory=Requirements)
    essay_prompts: list[str] = Field(default_factory=list)
    research_areas: list[str] = Field(default_factory=list)
    advisor_candidates: list[str] = Field(default_factory=list)
    applicant_reports: ApplicantReports = Field(default_factory=ApplicantReports)
    sources: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 2 — JudgeReport
# ---------------------------------------------------------------------------

class QualityRating(str, Enum):
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
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 3 — FitAssessment
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
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
    judge: Optional[JudgeReport] = None
    fit: Optional[FitAssessment] = None
    error: Optional[str] = None
