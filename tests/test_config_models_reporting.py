from __future__ import annotations

import os
import shlex
import subprocess
import sys
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
from grad_agent.reporting.pdf import pdf_path_for_markdown, report_dirs
from grad_agent.reporting.stats import (
    SchoolStats,
    StageStats,
    StatsCollector,
    add_usage,
)


def _parse_shell_exports(output: str) -> dict[str, str]:
    exports = {}
    for line in output.splitlines():
        parts = shlex.split(line)
        if len(parts) != 2 or parts[0] != "export":
            continue
        key, value = parts[1].split("=", 1)
        exports[key] = value
    return exports


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
                        "  local_retrieval: yaml-qwen",
                        "input:",
                        "  cv: yaml-cv.md",
                        "  context: yaml-context.md",
                        "  schools: yaml-schools.json",
                        "retrieval:",
                        "  backend: local_qwen_vllm",
                        "  local_model_count: 1",
                        "  local_base_urls:",
                        "    - http://127.0.0.1:9001/v1",
                        "  local_timeout: 120",
                        "  local_parallel_agents: 3",
                        "  local_max_parallel_tool_calls: 6",
                        "  max_turns: 9",
                        "  max_search_results: 7",
                        "concurrency:",
                        "  max_sonnet_parallel: 4",
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
                    overrides={
                        "max_retrieval_turns": 3,
                        "cv_path": "override-cv.md",
                        "output_dir": "override-output",
                    },
                )

        self.assertEqual(config.anthropic_api_key, "anthropic-test-key")
        self.assertEqual(config.brave_api_key, "brave-test-key")
        self.assertEqual(config.haiku_model, "yaml-haiku")
        self.assertEqual(config.sonnet_model, "yaml-sonnet")
        self.assertEqual(config.local_retrieval_model, "yaml-qwen")
        self.assertEqual(config.retrieval_backend, "local_qwen_vllm")
        self.assertEqual(config.local_retrieval_model_count, 1)
        self.assertEqual(config.local_retrieval_base_urls, ("http://127.0.0.1:9001/v1",))
        self.assertEqual(config.local_retrieval_timeout, 120)
        self.assertEqual(config.local_retrieval_parallel_agents, 3)
        self.assertEqual(config.local_retrieval_max_parallel_tool_calls, 6)
        self.assertEqual(config.retrieval_model, "yaml-qwen")
        self.assertEqual(config.max_retrieval_turns, 3)
        self.assertEqual(config.max_search_results, 7)
        self.assertEqual(config.max_sonnet_parallel, 4)
        self.assertEqual(config.cv_path, "override-cv.md")
        self.assertEqual(config.context_path, "yaml-context.md")
        self.assertEqual(config.schools_path, "yaml-schools.json")
        self.assertEqual(config.output_dir, "override-output")
        self.assertEqual(config.logs_dir, "")

    def test_validate_reports_missing_keys_and_bad_numbers(self) -> None:
        config = Config(
            anthropic_api_key="",
            brave_api_key="",
            haiku_model="haiku",
            sonnet_model="sonnet",
            local_retrieval_model="qwen",
            retrieval_backend="bad",
            local_retrieval_model_count=0,
            local_retrieval_base_urls=(),
            local_retrieval_api_key="",
            local_retrieval_timeout=0,
            max_retrieval_turns=0,
            max_search_results=True,
            max_page_chars=10,
            local_retrieval_parallel_agents=0,
            local_retrieval_max_parallel_tool_calls=0,
            cv_path="input/cv.md",
            context_path="input/context.md",
            schools_path="input/schools.json",
            retry_gap_fill=True,
            gap_fill_max_turns=1,
            max_schools_parallel=2,
            max_sonnet_parallel=0,
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
        self.assertIn("retrieval.backend must be one of: anthropic_haiku, local_qwen_vllm", errors)
        self.assertIn("retrieval.local_model_count must be an integer >= 1", errors)
        self.assertIn("retrieval.local_timeout must be an integer >= 1", errors)
        self.assertIn("retrieval.local_parallel_agents must be an integer >= 1", errors)
        self.assertIn(
            "retrieval.local_max_parallel_tool_calls must be an integer >= 1", errors
        )
        self.assertIn("concurrency.max_sonnet_parallel must be an integer >= 1", errors)
        self.assertIn("http.retries must be an integer >= 0", errors)

    def test_default_retrieval_backend_is_local_qwen(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "anthropic-test-key",
                "BRAVE_API_KEY": "brave-test-key",
            },
            clear=True,
        ), patch("grad_agent.config.dotenv.load_dotenv", return_value=True):
            config = Config.load(yaml_path=Path("/does/not/exist.yaml"))

        self.assertEqual(config.retrieval_backend, "local_qwen_vllm")
        self.assertEqual(config.retrieval_model, "Qwen/Qwen3.6-35B-A3B-FP8")
        self.assertEqual(config.local_retrieval_model_count, 2)
        self.assertEqual(config.local_retrieval_parallel_agents, 2)
        self.assertEqual(config.local_retrieval_max_parallel_tool_calls, 8)
        self.assertEqual(config.max_schools_parallel, 8)
        self.assertEqual(config.max_sonnet_parallel, 8)
        self.assertEqual(config.cv_path, "input/cv.md")
        self.assertEqual(config.context_path, "input/context.md")
        self.assertEqual(config.schools_path, "input/schools.json")
        self.assertEqual(
            config.local_retrieval_base_urls,
            (
                "http://127.0.0.1:8001/v1",
                "http://127.0.0.1:8002/v1",
            ),
        )

    def test_local_model_count_must_match_endpoint_count(self) -> None:
        config = Config(
            anthropic_api_key="anthropic",
            brave_api_key="brave",
            haiku_model="haiku",
            sonnet_model="sonnet",
            local_retrieval_model="qwen",
            retrieval_backend="local_qwen_vllm",
            local_retrieval_model_count=2,
            local_retrieval_base_urls=("http://127.0.0.1:8001/v1",),
            local_retrieval_api_key="",
            local_retrieval_timeout=60,
            max_retrieval_turns=2,
            max_search_results=3,
            max_page_chars=1000,
            local_retrieval_parallel_agents=1,
            local_retrieval_max_parallel_tool_calls=8,
            cv_path="input/cv.md",
            context_path="input/context.md",
            schools_path="input/schools.json",
            retry_gap_fill=True,
            gap_fill_max_turns=1,
            max_schools_parallel=1,
            max_sonnet_parallel=1,
            http_timeout=10,
            http_retries=0,
            output_dir="output",
            logs_dir="",
        )

        self.assertIn(
            "retrieval.local_model_count must match the number of "
            "retrieval.local_base_urls endpoints",
            config.validate(),
        )

    def test_api_retrieval_backend_ignores_local_endpoint_topology(self) -> None:
        config = Config(
            anthropic_api_key="anthropic",
            brave_api_key="brave",
            haiku_model="haiku",
            sonnet_model="sonnet",
            local_retrieval_model="qwen",
            retrieval_backend="anthropic_haiku",
            local_retrieval_model_count=0,
            local_retrieval_base_urls=(),
            local_retrieval_api_key="",
            local_retrieval_timeout=0,
            max_retrieval_turns=2,
            max_search_results=3,
            max_page_chars=1000,
            local_retrieval_parallel_agents=0,
            local_retrieval_max_parallel_tool_calls=0,
            cv_path="input/cv.md",
            context_path="input/context.md",
            schools_path="input/schools.json",
            retry_gap_fill=True,
            gap_fill_max_turns=1,
            max_schools_parallel=1,
            max_sonnet_parallel=1,
            http_timeout=10,
            http_retries=0,
            output_dir="output",
            logs_dir="",
        )

        self.assertEqual(config.validate(), [])
        self.assertFalse(config.uses_local_retrieval)
        self.assertEqual(config.retrieval_model, "haiku")

    def test_local_model_count_defaults_to_endpoint_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "retrieval:",
                        "  local_base_urls:",
                        "    - http://127.0.0.1:8001/v1",
                        "    - http://127.0.0.1:8002/v1",
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
                config = Config.load(config_path)

        self.assertEqual(config.local_retrieval_model_count, 2)
        self.assertEqual(
            config.local_retrieval_base_urls,
            ("http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"),
        )

    def test_deploy_config_env_uses_config_yaml_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  local_retrieval: test/model",
                        "retrieval:",
                        "  local_model_count: 2",
                        "  local_base_urls:",
                        "    - http://127.0.0.1:9101/v1",
                        "    - http://127.0.0.1:9102/v1",
                        "deploy:",
                        "  host: 0.0.0.0",
                        "  vllm_args:",
                        "    - --trust-remote-code",
                        "    - --dtype",
                        "    - auto",
                        "  log_dir: node-logs/vllm",
                        "  micromamba_env: test-env",
                        "  python_version: '3.11'",
                        "  pip_packages:",
                        "    - vllm==1.0.0",
                    ]
                ),
                encoding="utf-8",
            )
            script = Path(__file__).parents[1] / "deploy" / "config-env.py"

            output = subprocess.check_output(
                [sys.executable, str(script), str(config_path)], text=True
            )

        exports = _parse_shell_exports(output)
        self.assertEqual(exports["DEPLOY_MODEL_ID"], "test/model")
        self.assertEqual(exports["DEPLOY_MODEL_COUNT"], "2")
        self.assertEqual(exports["DEPLOY_VLLM_PORTS"], "9101 9102")
        self.assertEqual(
            exports["DEPLOY_VLLM_ENDPOINTS"],
            "http://127.0.0.1:9101/v1 http://127.0.0.1:9102/v1",
        )
        self.assertEqual(exports["DEPLOY_VLLM_ARGS"], "--trust-remote-code --dtype auto")
        self.assertTrue(exports["DEPLOY_VLLM_LOG_DIR"].endswith("node-logs/vllm"))
        self.assertEqual(exports["DEPLOY_MICROMAMBA_ENV"], "test-env")
        self.assertEqual(
            exports["DEPLOY_SYSTEM_PACKAGES"],
            "curl git build-essential tmux libcairo2 libpango-1.0-0 "
            "libpangoft2-1.0-0 libgdk-pixbuf-2.0-0 shared-mime-info",
        )
        self.assertEqual(exports["DEPLOY_PIP_PACKAGES"], "vllm==1.0.0")


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
        self.assertIsNone(profile.requirements.gre_policy)
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
            requirements={
                "gre_required": False,
                "gre_policy": "Not Considered",
                "statement_of_purpose": True,
            },
            sources=["https://example.edu/mscs"],
        )
        judge = JudgeReport(
            overall_quality=QualityRating.PARTIAL,
            flagged_fields=[{"field": "deadline", "reason": "Prior-cycle deadline."}],
            notes="Usable but verify deadline.",
        )

        markdown = render_school_markdown(profile, judge=judge)

        self.assertIn("**Deadline**: December 1 [unverified]", markdown)
        self.assertIn("- **GRE**: Not Considered", markdown)
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
        self.assertIn("| 2 | No Fit | MS | N/A | N/A | N/A | N/A |", markdown)

    def test_pdf_path_mirrors_markdown_report_name(self) -> None:
        dirs = report_dirs(Path("output"))
        pdf_path = pdf_path_for_markdown(Path("output/markdown/example_profile.md"), dirs)

        self.assertEqual(pdf_path, Path("output/pdf/example_profile.pdf"))

    def test_gre_policy_is_normalized_and_rendered_in_summary(self) -> None:
        required = SchoolProfile(
            school_name="Required U",
            program_name="PhD CS",
            requirements={"gre_policy": "required"},
        )
        considered = SchoolProfile(
            school_name="Considered U",
            program_name="MS CS",
            requirements={"gre_required": "GRE optional but considered"},
        )
        not_considered = SchoolProfile(
            school_name="No GRE U",
            program_name="MS DS",
            requirements={"gre_required": False},
        )

        markdown = render_summary_table([
            (required, None),
            (considered, None),
            (not_considered, None),
        ])

        self.assertIn(
            "| Rank | School | Program | Fit Score | Confidence | GRE | Deadline |",
            markdown,
        )
        self.assertIn("| Required U | PhD CS | N/A | N/A | Required | N/A |", markdown)
        self.assertIn("| Considered U | MS CS | N/A | N/A | Considered | N/A |", markdown)
        self.assertIn("| No GRE U | MS DS | N/A | N/A | Not Considered | N/A |", markdown)


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
