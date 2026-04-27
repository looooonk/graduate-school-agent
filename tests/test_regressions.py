from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from grad_agent.models import (
    ConfidenceLevel,
    FitAssessment,
    JudgeReport,
    QualityRating,
    SchoolProfile,
)
from grad_agent.reporting.markdown import render_summary_table
from grad_agent.reporting.trajectory import TrajectoryLogger
from grad_agent.util.json import extract_json_object


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


if __name__ == "__main__":
    unittest.main()
