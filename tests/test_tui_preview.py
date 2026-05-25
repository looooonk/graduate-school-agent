from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

from grad_agent.config import Config
from grad_agent.events import SchoolDone, SchoolStarted, StageStarted
from grad_agent.tui import _Renderable, _split_school_program
from tests.tui_demo import generate_demo_steps, load_configured_schools, render_demo_snapshot


class TuiPreviewTests(unittest.TestCase):
    def test_demo_snapshot_uses_configured_school_list(self) -> None:
        config, schools = self._make_preview_config()
        snapshot = render_demo_snapshot(
            config=config,
            schools=schools,
            width=120,
            seed=3,
            max_steps=12,
        )

        self.assertIn("Graduate School Research Agent", snapshot)
        self.assertIn("Alpha University - MS Robotics", snapshot)
        self.assertIn("Beta Institute", snapshot)
        self.assertIn("PhD Computer Scie", snapshot)
        self.assertIn("judge anthropic_sonnet/claude-sonnet-4-6", snapshot)
        self.assertIn("local 2 models, 2 agents/school, 4", snapshot)
        self.assertIn("tools/turn, 2 endpoints", snapshot)
        self.assertIn("parallel 3", snapshot)
        self.assertIn("schools, 4 judge/fit", snapshot)
        self.assertIn("full", snapshot)
        self.assertIn("b", snapshot)
        self.assertIn("gap-fill", snapshot)
        self.assertIn("Fetched faculty directory and research group pages", snapshot)

    def test_demo_events_are_derived_from_configured_schools(self) -> None:
        config, schools = self._make_preview_config()
        events = [step.event for step in generate_demo_steps(schools, config, seed=4)]
        expected_labels = {f"{school} - {program}" for school, program in schools}
        event_labels = {getattr(event, "school", "") for event in events}

        self.assertGreater(len(events), 0)
        self.assertTrue(event_labels <= expected_labels)
        self.assertEqual(expected_labels, event_labels)

    def test_configured_schools_are_loaded_from_input_schools_field(self) -> None:
        config, schools = self._make_preview_config()

        self.assertEqual(schools, load_configured_schools(config))

    def test_renderable_removes_finished_schools_but_keeps_totals(self) -> None:
        renderable = _Renderable(total=2)
        renderable.on_event(SchoolStarted(school="Alpha - MS", idx=1, total=2))
        renderable.on_event(SchoolStarted(school="Beta - PhD", idx=2, total=2))
        renderable.on_event(SchoolDone(school="Alpha - MS", success=True, elapsed=12.0, cost=0.12))

        snapshot = self._render(renderable)

        self.assertIn("1 / 2 schools", snapshot)
        self.assertIn("$0.1200", snapshot)
        self.assertNotIn("Alpha - MS", snapshot)
        self.assertIn("Beta - PhD", snapshot)

    def test_renderable_limits_running_school_rows_to_eight(self) -> None:
        renderable = _Renderable(total=10)
        for idx in range(1, 11):
            renderable.on_event(SchoolStarted(school=f"School {idx}", idx=idx, total=10))

        snapshot = self._render(renderable)

        for idx in range(1, 9):
            self.assertIn(f"School {idx}", snapshot)
        self.assertNotIn("School 9", snapshot)
        self.assertNotIn("School 10", snapshot)
        self.assertIn("Showing 8 of 10 running schools", snapshot)

    def test_renderable_keeps_log_panel_at_fixed_row(self) -> None:
        one_school = _Renderable(total=8)
        one_school.on_event(SchoolStarted(school="School 1", idx=1, total=8))

        eight_schools = _Renderable(total=8)
        for idx in range(1, 9):
            eight_schools.on_event(SchoolStarted(school=f"School {idx}", idx=idx, total=8))

        self.assertEqual(
            self._log_panel_line(self._render(one_school)),
            self._log_panel_line(self._render(eight_schools)),
        )

    def test_progress_bar_uses_stage_colors(self) -> None:
        renderable = _Renderable(total=4)
        renderable.on_event(SchoolStarted(school="Alpha - MS", idx=1, total=4))
        renderable.on_event(StageStarted(school="Alpha - MS", stage="retrieval"))
        renderable.on_event(SchoolStarted(school="Beta - PhD", idx=2, total=4))
        renderable.on_event(StageStarted(school="Beta - PhD", stage="gap_fill"))
        renderable.on_event(SchoolDone(school="Gamma - MS", success=True, elapsed=1.0, cost=0.01))

        bar = renderable._render_progress_bar(width=8)
        styles = [span.style for span in bar.spans]

        self.assertIn("green", styles)
        self.assertIn("blue", styles)
        self.assertIn("magenta", styles)
        self.assertIn("dim", styles)
        self.assertIn("█", bar.plain)
        self.assertIn("░", bar.plain)

    def test_log_labels_split_into_school_and_program_columns(self) -> None:
        self.assertEqual(
            _split_school_program("Alpha University — MS Robotics"),
            ("Alpha University", "MS Robotics"),
        )
        self.assertEqual(
            _split_school_program("Beta Institute - PhD Computer Science"),
            ("Beta Institute", "PhD Computer Science"),
        )

    def _render(self, renderable: _Renderable) -> str:
        console = Console(width=120, record=True, color_system=None, file=io.StringIO())
        console.print(renderable)
        return console.export_text()

    def _log_panel_line(self, snapshot: str) -> int:
        for idx, line in enumerate(snapshot.splitlines()):
            if " Log " in line:
                return idx
        raise AssertionError("log panel not found")

    def _make_preview_config(self) -> tuple[Config, list[tuple[str, str]]]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        schools_path = root / "schools.json"
        schools = [
            ("Alpha University", "MS Robotics"),
            ("Beta Institute", "PhD Computer Science"),
            ("Gamma College", "MS Data Science"),
        ]
        schools_path.write_text(
            json.dumps([{"school": school, "program": program} for school, program in schools]),
            encoding="utf-8",
        )
        config_path = root / "config.yaml"
        config_path.write_text(
            f"""
models:
  local_retrieval: Qwen/Test
input:
  schools: {schools_path}
retrieval:
  backend: local_qwen_vllm
  max_turns: 6
  local_model_count: 2
  local_parallel_agents: 2
  local_max_parallel_tool_calls: 4
  local_base_urls:
    - http://127.0.0.1:8001/v1
    - http://127.0.0.1:8002/v1
judge:
  gap_fill_max_turns: 3
concurrency:
  max_schools_parallel: 3
  max_sonnet_parallel: 4
""",
            encoding="utf-8",
        )
        return Config.load(config_path), schools


if __name__ == "__main__":
    unittest.main()
