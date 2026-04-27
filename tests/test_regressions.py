from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grad_agent.models import (
    ConfidenceLevel,
    FitAssessment,
    JudgeReport,
    QualityRating,
    SchoolProfile,
)
from grad_agent.pipeline.runner import calibrate_fit_confidence
from grad_agent.reporting.markdown import render_summary_table
from grad_agent.reporting.trajectory import TrajectoryLogger
from grad_agent.util.json import extract_json_object
from grad_agent.util import retry


class JsonExtractionTests(unittest.TestCase):
    def test_extracts_fenced_json(self) -> None:
        text = 'Here:\n```json\n{"a": 1}\n```\nDone.'
        self.assertEqual(extract_json_object(text), {"a": 1})

    def test_extracts_first_embedded_json_object(self) -> None:
        text = 'prefix {"a": {"nested": true}} suffix {"b": 2}'
        self.assertEqual(extract_json_object(text), {"a": {"nested": True}})


class TrajectoryLoggerTests(unittest.TestCase):
    def test_serializes_enum_backed_models(self) -> None:
        report = JudgeReport(
            overall_quality=QualityRating.PARTIAL,
            notes="needs review",
        )
        assessment = FitAssessment(
            overall_score=0.7,
            research_alignment="good",
            competitiveness="strong",
            gaps="few",
            confidence=ConfidenceLevel.MEDIUM,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with TrajectoryLogger(path) as traj:
                traj.log_judge_report(report)
                traj.log_fit_assessment(assessment)

            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(lines[0]["data"]["overall_quality"], "partial")
        self.assertEqual(lines[1]["data"]["confidence"], "medium")


class MarkdownTests(unittest.TestCase):
    def test_summary_escapes_table_cells(self) -> None:
        profile = SchoolProfile(
            school_name="A | B",
            program_name="MS | CS",
            deadline="Dec | 1",
        )
        markdown = render_summary_table([(profile, None)])
        self.assertIn("A \\| B", markdown)
        self.assertIn("MS \\| CS", markdown)
        self.assertIn("Dec \\| 1", markdown)

    def test_summary_ranks_by_confidence_adjusted_score(self) -> None:
        high_raw_low_confidence = SchoolProfile(
            school_name="High Raw",
            program_name="PhD CS",
        )
        lower_raw_high_confidence = SchoolProfile(
            school_name="Supported",
            program_name="PhD CS",
        )
        markdown = render_summary_table([
            (
                high_raw_low_confidence,
                FitAssessment(
                    overall_score=0.8,
                    research_alignment="strong but thinly sourced",
                    competitiveness="unknown",
                    gaps="profile evidence is incomplete",
                    confidence=ConfidenceLevel.LOW,
                ),
            ),
            (
                lower_raw_high_confidence,
                FitAssessment(
                    overall_score=0.7,
                    research_alignment="well supported",
                    competitiveness="strong",
                    gaps="minor",
                    confidence=ConfidenceLevel.HIGH,
                ),
            ),
        ])

        self.assertLess(
            markdown.index("| 1 | Supported"),
            markdown.index("| 2 | High Raw"),
        )


class FitCalibrationTests(unittest.TestCase):
    def test_insufficient_judge_verdict_downgrades_fit_confidence(self) -> None:
        fit = FitAssessment(
            overall_score=0.75,
            research_alignment="good",
            competitiveness="strong",
            gaps="some",
            confidence=ConfidenceLevel.HIGH,
        )
        judge = JudgeReport(overall_quality=QualityRating.INSUFFICIENT)

        calibrated = calibrate_fit_confidence(fit, judge)

        self.assertIsNotNone(calibrated)
        assert calibrated is not None
        self.assertEqual(calibrated.confidence, ConfidenceLevel.LOW)

    def test_partial_judge_verdict_caps_high_confidence_at_medium(self) -> None:
        fit = FitAssessment(
            overall_score=0.75,
            research_alignment="good",
            competitiveness="strong",
            gaps="some",
            confidence=ConfidenceLevel.HIGH,
        )
        judge = JudgeReport(overall_quality=QualityRating.PARTIAL)

        calibrated = calibrate_fit_confidence(fit, judge)

        self.assertIsNotNone(calibrated)
        assert calibrated is not None
        self.assertEqual(calibrated.confidence, ConfidenceLevel.MEDIUM)


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limit_backoff_uses_one_point_five_multiplier(self) -> None:
        class FakeRateLimitError(Exception):
            response = None

        attempts = 0

        async def fn() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 4:
                raise FakeRateLimitError()
            return "ok"

        sleeps: list[float] = []

        async def fake_sleep(wait: float) -> None:
            sleeps.append(wait)

        with (
            patch.object(retry.anthropic, "RateLimitError", FakeRateLimitError),
            patch.object(retry.asyncio, "sleep", fake_sleep),
        ):
            result = await retry.api_create_with_retry(fn, max_retries=3)

        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [60.0, 90.0, 135.0])


if __name__ == "__main__":
    unittest.main()
