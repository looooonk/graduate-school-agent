"""Confidence calibration helpers for pipeline outputs."""

from __future__ import annotations

from grad_agent.models import ConfidenceLevel, FitAssessment, JudgeReport, QualityRating


def calibrate_fit_confidence(
    fit: FitAssessment | None,
    judge: JudgeReport | None,
) -> FitAssessment | None:
    """Align fit confidence with the judge's data-quality verdict."""
    if fit is None or judge is None:
        return fit
    if judge.overall_quality == QualityRating.INSUFFICIENT:
        target = ConfidenceLevel.LOW
    elif (
        judge.overall_quality == QualityRating.PARTIAL
        and fit.confidence == ConfidenceLevel.HIGH
    ):
        target = ConfidenceLevel.MEDIUM
    else:
        return fit
    if fit.confidence == target:
        return fit
    return fit.model_copy(update={"confidence": target})
