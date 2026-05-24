from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from grad_agent.config import Config
from tests.tui_demo import generate_demo_steps, load_configured_schools, render_demo_snapshot


class TuiPreviewTests(unittest.TestCase):
    def test_demo_snapshot_uses_configured_school_list(self) -> None:
        config, schools = self._make_preview_config()
        snapshot = render_demo_snapshot(
            config=config,
            schools=schools,
            width=120,
            seed=3,
            max_steps=80,
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
