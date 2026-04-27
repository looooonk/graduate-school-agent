from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from grad_agent.config import Config
from grad_agent.models import (
    ConfidenceLevel,
    FitAssessment,
    JudgeReport,
    QualityRating,
    SchoolProfile,
)
from grad_agent.pipeline.runner import _safe_filename, calibrate_fit_confidence
from grad_agent.reporting.markdown import render_school_markdown, render_summary_table
from grad_agent.reporting.stats import (
    SchoolStats,
    StageStats,
    StatsCollector,
    add_usage,
)


class ConfigTests(unittest.TestCase):
    def test_load_merges_yaml_env_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  haiku: yaml-haiku",
                        "  sonnet: yaml-sonnet",
                        "retrieval:",
                        "  max_turns: 9",
                        "  max_search_results: 7",
                        "output:",
                        "  dir: yaml-output",
                        "logs:",
                        "  dir: ''",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "ANTHROPIC_API_KEY": "anthropic-test-key",
                    "BRAVE_API_KEY": "brave-test-key",
                },
                clear=True,
            ), patch("grad_agent.config.dotenv.load_dotenv", return_value=True):
                config = Config.load(
                    config_path,
                    overrides={"max_retrieval_turns": 3, "output_dir": "override-output"},
                )

        self.assertEqual(config.anthropic_api_key, "anthropic-test-key")
        self.assertEqual(config.brave_api_key, "brave-test-key")
        self.assertEqual(config.haiku_model, "yaml-haiku")
        self.assertEqual(config.sonnet_model, "yaml-sonnet")
        self.assertEqual(config.max_retrieval_turns, 3)
        self.assertEqual(config.max_search_results, 7)
        self.assertEqual(config.output_dir, "override-output")
        self.assertEqual(config.logs_dir, "")

    def test_validate_reports_missing_keys_and_bad_numbers(self) -> None:
        config = Config(
            anthropic_api_key="",
            brave_api_key="",
            haiku_model="haiku",
            sonnet_model="sonnet",
            max_retrieval_turns=0,
            max_search_results=True,
            max_page_chars=10,
            retry_gap_fill=True,
            gap_fill_max_turns=1,
            max_schools_parallel=2,
            http_timeout=20,
            http_retries=-1,
            output_dir="output",
            logs_dir="logs",
        )

        errors = config.validate()

        self.assertIn("ANTHROPIC_API_KEY is required (set in environment or .env)", errors)
        self.assertIn("BRAVE_API_KEY is required (set in environment or .env)", errors)
        self.assertIn("retrieval.max_turns must be an integer >= 1", errors)
        self.assertIn("retrieval.max_search_results must be an integer >= 1", errors)
        self.assertIn("http.retries must be an integer >= 0", errors)


class ModelCoercionTests(unittest.TestCase):
    def test_school_profile_coerces_common_model_output_shapes(self) -> None:
        profile = SchoolProfile.model_validate(
            {
                "school_name": "Example",
                "program_name": "PhD CS",
                "deadline": {"fall": "December 1", "spring": ""},
                "requirements": {"other": "Portfolio optional"},
                "essay_prompts": "Describe your goals.",
                "research_areas": "",
                "advisor_candidates": [
                    {
                        "name": "Jane Smith",
                        "research_focus": "NLP",
                        "profile_url": "https://example.edu/jane",
                    },
                    "Bob Lee - systems",
                ],
                "sources": {"admissions": "https://example.edu/admissions"},
                "notes": ["Prior-cycle deadline.", "", "Verify fee."],
            }
        )

        self.assertEqual(profile.deadline, "fall: December 1")
        self.assertEqual(profile.requirements.other, ["Portfolio optional"])
        self.assertEqual(profile.essay_prompts, ["Describe your goals."])
        self.assertEqual(profile.research_areas, [])
        self.assertEqual(
            profile.advisor_candidates[0],
            "Jane Smith — NLP (https://example.edu/jane)",
        )
        self.assertEqual(profile.advisor_candidates[1], "Bob Lee - systems")
        self.assertEqual(profile.sources, ["https://example.edu/admissions"])
        self.assertEqual(profile.notes, "Prior-cycle deadline.\nVerify fee.")

    def test_fit_score_is_bounded(self) -> None:
        with self.assertRaises(ValidationError):
            FitAssessment(
                overall_score=1.2,
                research_alignment="too high",
                competitiveness="unknown",
                gaps="unknown",
                confidence=ConfidenceLevel.LOW,
            )


class MarkdownRenderingTests(unittest.TestCase):
    def test_school_markdown_marks_flagged_deadline_and_omits_empty_optional_sections(self) -> None:
        profile = SchoolProfile(
            school_name="Example University",
            program_name="MS CS",
            deadline="December 1",
            application_fee="$75",
            requirements={"gre_required": False, "statement_of_purpose": True},
            sources=["https://example.edu/mscs"],
        )
        judge = JudgeReport(
            overall_quality=QualityRating.PARTIAL,
            flagged_fields=[{"field": "deadline", "reason": "Prior-cycle deadline."}],
            notes="Usable but verify deadline.",
        )

        markdown = render_school_markdown(profile, judge=judge)

        self.assertIn("**Deadline**: December 1 [unverified]", markdown)
        self.assertIn("- **GRE required**: No", markdown)
        self.assertIn("- **Statement of Purpose**: Yes", markdown)
        self.assertIn("## Quality Assessment (partial)", markdown)
        self.assertIn("- **deadline**: Prior-cycle deadline.", markdown)
        self.assertNotIn("## Essay Prompts", markdown)
        self.assertNotIn("## Fit Summary", markdown)

    def test_summary_places_profiles_without_fit_last(self) -> None:
        no_fit = SchoolProfile(school_name="No Fit", program_name="MS")
        with_fit = SchoolProfile(school_name="With Fit", program_name="MS")

        markdown = render_summary_table(
            [
                (no_fit, None),
                (
                    with_fit,
                    FitAssessment(
                        overall_score=0.2,
                        research_alignment="some",
                        competitiveness="some",
                        gaps="many",
                        confidence=ConfidenceLevel.LOW,
                    ),
                ),
            ]
        )

        self.assertLess(markdown.index("| 1 | With Fit"), markdown.index("| 2 | No Fit"))
        self.assertIn("| 2 | No Fit | MS | N/A | N/A | N/A |", markdown)


class StatsTests(unittest.TestCase):
    def test_stage_cost_applies_cache_read_discount(self) -> None:
        stats = StageStats(
            stage="retrieval",
            model="claude-haiku-4-5-20251001",
            input_tokens=1_000_000,
            output_tokens=100_000,
            cache_read_tokens=250_000,
        )

        self.assertAlmostEqual(stats.estimated_cost_usd, 1.02)

    def test_add_usage_handles_optional_cache_fields(self) -> None:
        stats = StageStats(stage="judge", model="unknown")
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)

        add_usage(stats, usage)

        self.assertEqual(stats.input_tokens, 100)
        self.assertEqual(stats.output_tokens, 20)
        self.assertEqual(stats.cache_read_tokens, 0)
        self.assertEqual(stats.cache_creation_tokens, 0)

    def test_collector_returns_snapshot_and_aggregate_totals(self) -> None:
        collector = StatsCollector()
        school = SchoolStats(
            school="Example - MS",
            success=True,
            stages=[
                StageStats(
                    stage="retrieval",
                    model="unknown",
                    input_tokens=100,
                    output_tokens=20,
                )
            ],
        )
        collector.add_school(school)

        snapshot = collector.schools
        snapshot.clear()

        self.assertEqual(len(collector.schools), 1)
        self.assertEqual(collector.total_input_tokens, 100)
        self.assertEqual(collector.total_output_tokens, 20)
        self.assertIn("Example - MS", collector.summary())


class RunnerHelperTests(unittest.TestCase):
    def test_safe_filename_lowercases_replaces_punctuation_and_collapses_underscores(self) -> None:
        self.assertEqual(
            _safe_filename("A & B University", "MS: Computer Science"),
            "a_b_university_ms_computer_science",
        )

    def test_calibration_keeps_medium_confidence_for_partial_verdict(self) -> None:
        fit = FitAssessment(
            overall_score=0.6,
            research_alignment="ok",
            competitiveness="ok",
            gaps="some",
            confidence=ConfidenceLevel.MEDIUM,
        )
        judge = JudgeReport(overall_quality=QualityRating.PARTIAL)

        self.assertIs(calibrate_fit_confidence(fit, judge), fit)


if __name__ == "__main__":
    unittest.main()
