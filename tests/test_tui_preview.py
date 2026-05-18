from __future__ import annotations

import unittest

from grad_agent.tui import demo_tui_events, render_demo_snapshot


class TuiPreviewTests(unittest.TestCase):
    def test_demo_snapshot_contains_filled_tui_sections(self) -> None:
        snapshot = render_demo_snapshot(width=120)

        self.assertIn("Graduate School Research Agent", snapshot)
        self.assertIn("Stanford University - MS Computer Science", snapshot)
        self.assertIn("MIT - PhD Electrical Engineering and Computer Science", snapshot)
        self.assertIn("local 4 models, 4 agents/school, 8 tools/turn, 4 endpoints", snapshot)
        self.assertIn("parallel 8 schools, 8 Sonnet", snapshot)
        self.assertIn("full 8/25, adm 7/25", snapshot)
        self.assertIn("batch 4", snapshot)
        self.assertIn("gap-fill", snapshot)
        self.assertIn("Fetched faculty directory and research group pages", snapshot)

    def test_demo_events_do_not_require_runtime_inputs(self) -> None:
        events = demo_tui_events()

        self.assertGreater(len(events), 0)
        self.assertTrue(all(getattr(event, "school", "") for event in events))


if __name__ == "__main__":
    unittest.main()
