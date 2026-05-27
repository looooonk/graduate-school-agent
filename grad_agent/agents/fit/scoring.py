"""Deterministic fit-score composition."""

from __future__ import annotations

from grad_agent.models import (
    FIT_SCORE_WEIGHTS_DEFAULT,
    FitAssessment,
    compute_overall_fit_score,
)

PHD_FIT_SCORE_WEIGHTS = {
    "research_alignment": 0.35,
    "advisor_fit": 0.30,
    "applicant_competitiveness": 0.20,
    "program_structure_fit": 0.10,
    "constraint_fit": 0.05,
}

MS_FIT_SCORE_WEIGHTS = {
    "research_alignment": 0.30,
    "advisor_fit": 0.20,
    "applicant_competitiveness": 0.25,
    "program_structure_fit": 0.15,
    "constraint_fit": 0.10,
}


def apply_program_fit_score(
    assessment: FitAssessment,
    program_name: str,
) -> FitAssessment:
    if assessment.score_breakdown is None:
        return assessment
    return assessment.model_copy(
        update={
            "overall_score": compute_overall_fit_score(
                assessment.score_breakdown,
                assessment.score_caps,
                _fit_score_weights(program_name),
            )
        }
    )


def _fit_score_weights(program_name: str) -> dict[str, float]:
    name = program_name.lower()
    if "phd" in name or "doctor" in name:
        return PHD_FIT_SCORE_WEIGHTS
    if "ms" in name or "master" in name or "msc" in name:
        return MS_FIT_SCORE_WEIGHTS
    return FIT_SCORE_WEIGHTS_DEFAULT
