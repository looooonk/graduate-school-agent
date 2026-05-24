from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

from grad_agent.config import Config
from grad_agent.events import SchoolDone, SchoolStarted
from grad_agent.tui import _Renderable
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
        self.assertIn("Beta Institute - PhD Computer Science", snapshot)
        self.assertIn("local 2 models, 2 agents/school, 4 tools/turn, 2 endpoints", snapshot)
        self.assertIn("parallel 3", snapshot)
        self.assertIn("schools, 4 Sonnet", snapshot)
        self.assertIn("full", snapshot)
        self.assertIn("batch", snapshot)
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

    def _render(self, renderable: _Renderable) -> str:
        console = Console(width=120, record=True, color_system=None, file=io.StringIO())
        console.print(renderable)
        return console.export_text()

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
