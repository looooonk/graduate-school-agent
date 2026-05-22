from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from grad_agent import cli
from grad_agent.config import Config
from grad_agent.pipeline import fit, judge, retrieval
from grad_agent.pipeline.prompts import (
    fit_user_prompt,
    judge_user_prompt,
    retrieval_turn_status,
    retrieval_user_prompt,
)
from grad_agent.pipeline.tools import (
    _strip_html,
    dispatch_tool,
    handle_fetch_page,
    handle_web_search,
)
from grad_agent.reporting.summary import load_profile_summaries


def _test_config(**overrides: object) -> Config:
    values = {
        "anthropic_api_key": "not-used",
        "brave_api_key": "brave-test-key",
        "haiku_model": "haiku-test",
        "sonnet_model": "sonnet-test",
        "local_retrieval_model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "retrieval_backend": "anthropic_haiku",
        "local_retrieval_model_count": 1,
        "local_retrieval_base_urls": ("http://127.0.0.1:8001/v1",),
        "local_retrieval_api_key": "",
        "local_retrieval_timeout": 30,
        "max_retrieval_turns": 2,
        "max_search_results": 3,
        "max_page_chars": 20,
        "local_retrieval_parallel_agents": 1,
        "local_retrieval_max_parallel_tool_calls": 8,
        "retry_gap_fill": True,
        "gap_fill_max_turns": 2,
        "max_schools_parallel": 1,
        "max_sonnet_parallel": 1,
        "http_timeout": 5,
        "http_retries": 0,
        "output_dir": "output",
        "logs_dir": "",
    }
    values.update(overrides)
    return Config(**values)


class FakeTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, name: str, args: dict[str, object], tool_id: str = "tool-1") -> None:
        self.id = tool_id
        self.name = name
        self.input = args


class FakeMessages:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake response queued")
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.messages = FakeMessages(responses)


class FakeLocalClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    async def create(self, _http: object, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake local response queued")
        return self.responses.pop(0)


def fake_response(text: str, *, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[FakeTextBlock(text)],
        usage=SimpleNamespace(input_tokens=12, output_tokens=6),
    )


def fake_local_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        content=[FakeTextBlock(text)],
        stop_reason="stop",
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )


class PromptTests(unittest.TestCase):
    def test_retrieval_prompt_includes_context_and_budget_when_present(self) -> None:
        prompt = retrieval_user_prompt("Example", "PhD CS", "target NLP", max_turns=4)

        self.assertIn("## Applicant Context", prompt)
        self.assertIn("target NLP", prompt)
        self.assertIn("**School**: Example", prompt)
        self.assertIn("**Program**: PhD CS", prompt)
        self.assertIn("budget of **4 turns**", prompt)

    def test_retrieval_turn_status_flags_final_and_near_final_turns(self) -> None:
        self.assertIn("Start wrapping up.", retrieval_turn_status(2, 5))
        self.assertIn("budget exhausted", retrieval_turn_status(5, 5))

    def test_judge_and_fit_prompts_include_context_only_when_non_empty(self) -> None:
        self.assertNotIn("## Applicant Context", judge_user_prompt("{}", ""))
        self.assertIn("deadline policy", judge_user_prompt("{}", "Fall 2027"))
        self.assertNotIn("## Applicant Context", fit_user_prompt("cv", "{}", ""))
        self.assertIn("## Applicant Context", fit_user_prompt("cv", "{}", "advisor preference"))


class ToolHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_tool_reports_unknown_tool(self) -> None:
        result = await dispatch_tool("missing", {}, _test_config(), SimpleNamespace(), "School")

        self.assertEqual(result, "Unknown tool: missing")

    async def test_handle_fetch_page_rejects_relative_urls_without_http_call(self) -> None:
        class FakeHttp:
            called = False

            async def get(self, *_args: object, **_kwargs: object) -> object:
                self.called = True
                raise AssertionError("Should not be called")

        http = FakeHttp()

        result = await handle_fetch_page({"url": "/relative"}, _test_config(), http, "School")

        self.assertEqual(result, "Failed to fetch page: url must be an absolute http(s) URL.")
        self.assertFalse(http.called)

    async def test_handle_fetch_page_strips_html_and_truncates(self) -> None:
        class FakeResponse:
            headers = {"content-type": "text/html; charset=utf-8"}
            text = (
                "<html><style>.x{}</style><body>Hello&nbsp;world<script>x()</script> "
                "more text</body></html>"
            )

            def raise_for_status(self) -> None:
                return None

        class FakeHttp:
            async def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
                return FakeResponse()

        result = await handle_fetch_page(
            {"url": "https://example.edu/page"},
            _test_config(max_page_chars=11),
            FakeHttp(),
            "School",
        )

        self.assertEqual(result, "Hello world\n\n[... truncated ...]")

    async def test_handle_web_search_formats_results_and_uses_configured_count(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "web": {
                        "results": [
                            {
                                "title": "Admissions",
                                "url": "https://example.edu/admissions",
                                "description": "Deadlines and requirements.",
                            }
                        ]
                    }
                }

        class FakeHttp:
            def __init__(self) -> None:
                self.kwargs: dict[str, object] | None = None

            async def get(self, *_args: object, **kwargs: object) -> FakeResponse:
                self.kwargs = kwargs
                return FakeResponse()

        http = FakeHttp()

        result = await handle_web_search(
            {"query": "Example MS CS deadline"},
            _test_config(max_search_results=4),
            http,
            "School",
        )

        self.assertIn("1. [Admissions](https://example.edu/admissions)", result)
        self.assertIn("Deadlines and requirements.", result)
        assert http.kwargs is not None
        self.assertEqual(http.kwargs["params"], {"q": "Example MS CS deadline", "count": 4})

    def test_strip_html_removes_script_style_tags_and_unescapes_entities(self) -> None:
        text = _strip_html("<style>x</style><p>A&nbsp;B</p><script>bad()</script><p>C</p>")

        self.assertIn("A B", text)
        self.assertIn("C", text)
        self.assertNotIn("\xa0", text)
        self.assertNotIn("bad()", text)


class PipelineStageTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_judge_parses_fake_response_without_api_endpoint(self) -> None:
        client = FakeClient(
            [
                fake_response(
                    json.dumps(
                        {
                            "overall_quality": "partial",
                            "flagged_fields": [{"field": "deadline", "reason": "prior cycle"}],
                            "suggested_queries": ["Example deadline"],
                            "notes": "verify",
                        }
                    )
                )
            ]
        )

        report, stats = await judge.run_judge(
            retrieval.SchoolProfile(school_name="Example", program_name="MS CS"),
            _test_config(),
            client,
            context_text="Fall 2027",
        )

        self.assertEqual(report.overall_quality.value, "partial")
        self.assertEqual(report.flagged_fields[0].field, "deadline")
        self.assertEqual(stats.api_calls, 1)
        self.assertEqual(client.messages.calls[0]["model"], "sonnet-test")

    async def test_run_fit_parses_fake_response_without_api_endpoint(self) -> None:
        client = FakeClient(
            [
                fake_response(
                    json.dumps(
                        {
                            "overall_score": 0.64,
                            "research_alignment": "Good NLP overlap.",
                            "advisor_candidates": ["Jane Smith - NLP"],
                            "competitiveness": "Competitive.",
                            "gaps": "Needs clearer SOP.",
                            "confidence": "medium",
                        }
                    )
                )
            ]
        )

        assessment, stats = await fit.run_fit_assessment(
            "CV text",
            retrieval.SchoolProfile(school_name="Example", program_name="MS CS"),
            _test_config(),
            client,
        )

        self.assertEqual(assessment.overall_score, 0.64)
        self.assertEqual(assessment.confidence.value, "medium")
        self.assertEqual(stats.api_calls, 1)
        self.assertEqual(client.messages.calls[0]["model"], "sonnet-test")

    async def test_run_retrieval_handles_tool_use_then_final_profile_with_fakes(self) -> None:
        client = FakeClient(
            [
                SimpleNamespace(
                    stop_reason="tool_use",
                    content=[
                        FakeTextBlock("Searching."),
                        FakeToolUseBlock("web_search", {"query": "Example MS CS"}),
                    ],
                    usage=SimpleNamespace(input_tokens=5, output_tokens=2),
                ),
                fake_response(
                    json.dumps(
                        {
                            "school_name": "Wrong name",
                            "program_name": "Wrong program",
                            "deadline": "December 1",
                            "sources": ["https://example.edu"],
                        }
                    )
                ),
            ]
        )
        events: list[object] = []

        async def fake_dispatch(
            name: str,
            args: dict[str, object],
            _config: Config,
            _http: object,
            _school: str,
        ) -> str:
            self.assertEqual(name, "web_search")
            self.assertEqual(args, {"query": "Example MS CS"})
            return "1. Example result"

        with patch.object(retrieval, "dispatch_tool", fake_dispatch):
            profile, stats = await retrieval.run_retrieval(
                "Example University",
                "MS CS",
                _test_config(max_retrieval_turns=2),
                client,
                SimpleNamespace(),
                on_event=events.append,
            )

        self.assertEqual(profile.school_name, "Example University")
        self.assertEqual(profile.program_name, "MS CS")
        self.assertEqual(profile.deadline, "December 1")
        self.assertEqual(stats.api_calls, 2)
        self.assertEqual(stats.tool_calls, 1)
        self.assertEqual(len(events), 3)

    async def test_run_retrieval_uses_local_json_tool_protocol(self) -> None:
        local_client = FakeLocalClient(
            [
                fake_local_response(
                    json.dumps(
                        {
                            "tool": "web_search",
                            "args": {"query": "Example MS CS deadline"},
                        }
                    )
                ),
                fake_local_response(
                    json.dumps(
                        {
                            "school_name": "Wrong",
                            "program_name": "Wrong",
                            "deadline": "December 1",
                            "sources": ["https://example.edu"],
                        }
                    )
                ),
            ]
        )

        async def fake_dispatch(
            name: str,
            args: dict[str, object],
            _config: Config,
            _http: object,
            _school: str,
        ) -> str:
            self.assertEqual(name, "web_search")
            self.assertEqual(args, {"query": "Example MS CS deadline"})
            return "1. Example result"

        with patch.object(retrieval, "dispatch_tool", fake_dispatch):
            profile, stats = await retrieval.run_retrieval(
                "Example University",
                "MS CS",
                _test_config(retrieval_backend="local_qwen_vllm"),
                FakeClient([]),
                SimpleNamespace(),
                local_client=local_client,
            )

        self.assertEqual(profile.school_name, "Example University")
        self.assertEqual(profile.deadline, "December 1")
        self.assertEqual(stats.model, "Qwen/Qwen3.6-35B-A3B-FP8")
        self.assertEqual(stats.api_calls, 2)
        self.assertEqual(stats.tool_calls, 1)
        self.assertEqual(local_client.calls[0]["model"], "Qwen/Qwen3.6-35B-A3B-FP8")

    async def test_run_retrieval_uses_local_batch_tool_protocol(self) -> None:
        local_client = FakeLocalClient(
            [
                fake_local_response(
                    json.dumps(
                        {
                            "tools": [
                                {
                                    "tool": "web_search",
                                    "args": {"query": "Example MS CS deadline"},
                                },
                                {
                                    "tool": "web_search",
                                    "args": {"query": "Example MS CS faculty"},
                                },
                            ],
                        }
                    )
                ),
                fake_local_response(
                    json.dumps(
                        {
                            "school_name": "Wrong",
                            "program_name": "Wrong",
                            "deadline": "December 1",
                            "sources": ["https://example.edu"],
                        }
                    )
                ),
            ]
        )
        calls: list[tuple[str, dict[str, object]]] = []

        async def fake_dispatch(
            name: str,
            args: dict[str, object],
            _config: Config,
            _http: object,
            _school: str,
        ) -> str:
            calls.append((name, args))
            return "result"

        with patch.object(retrieval, "dispatch_tool", fake_dispatch):
            profile, stats = await retrieval.run_retrieval(
                "Example University",
                "MS CS",
                _test_config(retrieval_backend="local_qwen_vllm"),
                FakeClient([]),
                SimpleNamespace(),
                local_client=local_client,
            )

        self.assertEqual(profile.deadline, "December 1")
        self.assertEqual(stats.api_calls, 2)
        self.assertEqual(stats.tool_calls, 2)
        self.assertEqual(
            calls,
            [
                ("web_search", {"query": "Example MS CS deadline"}),
                ("web_search", {"query": "Example MS CS faculty"}),
            ],
        )


class CliTests(unittest.TestCase):
    def test_load_schools_reads_valid_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schools.json"
            path.write_text(
                json.dumps(
                    [
                        {"school": "A", "program": "MS CS"},
                        {"school": "B", "program": "PhD CS"},
                    ]
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(schools=path, school=None, program=None)

            self.assertEqual(cli._load_schools(args), [("A", "MS CS"), ("B", "PhD CS")])

    def test_load_schools_requires_program_for_inline_school(self) -> None:
        args = argparse.Namespace(schools=None, school="A", program=None)

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            cli._load_schools(args)

        self.assertEqual(exc.exception.code, 1)

    def test_summary_from_does_not_require_cv_or_schools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "example_ms_cs_profile.md"
            profile.write_text(
                "\n".join(
                    [
                        "# Example University \u2014 MS CS",
                        "",
                        "**Deadline**: January 15, 2027 [unverified]  ",
                        "**Application fee**: $75",
                        "",
                        "---",
                        "",
                        "## Requirements",
                        "",
                        "- **GRE**: Not Considered",
                        "",
                        "---",
                        "",
                        "## Fit Summary",
                        "",
                        "- **Overall score**: 0.83 / 1.00",
                        "- **Confidence**: high",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with (
                patch.object(sys, "argv", ["grad-agent", "--summary-from", str(root)]),
                contextlib.redirect_stdout(stdout),
            ):
                cli.main()

            summary = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn(
                "| 1 | Example University | MS CS | 0.83 | high | Not Considered |",
                summary,
            )
            self.assertIn("Wrote", stdout.getvalue())

    def test_load_profile_summaries_parses_rendered_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "example_profile.md"
            path.write_text(
                "\n".join(
                    [
                        "# Example - PhD CS",
                        "**Deadline**: *not found*",
                        "- **GRE**: Required",
                    ]
                ),
                encoding="utf-8",
            )

            profile, fit = load_profile_summaries(path)[0]

            self.assertEqual(profile.school_name, "Example")
            self.assertEqual(profile.program_name, "PhD CS")
            self.assertIsNone(profile.deadline)
            self.assertEqual(profile.requirements.gre_policy.value, "Required")
            self.assertIsNone(fit)


if __name__ == "__main__":
    unittest.main()
