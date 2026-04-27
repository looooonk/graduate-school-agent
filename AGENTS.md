# Repository Instructions

## Code Style

- Be succinct in code, but do not compress logic into hard-to-read tricks.
- Prefer existing helpers and local patterns over new abstractions.
- Keep comments sparse. Do not comment code that is obvious to a programmer.
- Use ASCII in comments. Replace non-ASCII punctuation or symbols with plain ASCII when editing comments.
- Do not add delimiter comments such as long separator lines. Split files or functions when structure is getting too large.
- Do not commit secrets, generated outputs, local inputs, or trajectory logs.

## Project Overview

This repository implements a graduate school research agent using the Anthropic Python SDK. It researches graduate programs, evaluates profile quality, assesses applicant fit against a CV, and writes Markdown reports.

The installed CLI entry point is:

```bash
grad-agent
```

Primary runtime dependencies are `anthropic`, `pydantic`, `httpx`, `pyyaml`, `python-dotenv`, and `rich`.

## Current Architecture

Pipeline per school:

```text
retrieval (Haiku) -> judge (Sonnet) + fit (Sonnet) -> Markdown output
                    -> optional gap-fill (Haiku) -> re-judge + re-fit
```

Important behavior:

- `retrieval` uses Claude Haiku with two Anthropic tools: `web_search` and `fetch_page`.
- `web_search` calls Brave Search.
- `fetch_page` fetches HTTP(S) URLs with `httpx`, strips HTML, and truncates to `config.max_page_chars`.
- `judge` and `fit` run concurrently for a single school with `asyncio.gather`.
- Gap-fill runs only when enabled, the judge returns `insufficient`, and suggested queries are present.
- Schools are currently processed sequentially in `run_all_schools`. `max_schools_parallel` is loaded from config and CLI but is not used for parallel execution yet.
- `http.retries` is validated in config but is not currently applied by the tool handlers.
- Anthropic rate-limit retries are handled by `grad_agent.util.retry.api_create_with_retry`.

## Package Layout

```text
grad_agent/
  cli.py                 argparse CLI entry point
  config.py              YAML, .env, and environment config loading
  events.py              pipeline event dataclasses
  models.py              Pydantic schemas
  tui.py                 Rich live terminal UI
  pipeline/
    retrieval.py         Haiku retrieval loop
    judge.py             Sonnet quality judge
    fit.py               Sonnet CV-aware fit assessor
    runner.py            per-school orchestration and output writing
    prompts.py           system and user prompts
    tools.py             Brave search and page fetch tool handlers
  reporting/
    markdown.py          report and summary rendering
    stats.py             token, cost, timing, and run statistics
    trajectory.py        per-school JSONL trajectory logger
  util/
    json.py              model JSON extraction
    log.py               structured logging
    retry.py             Anthropic rate-limit retry helper
tests/
  test_config_models_reporting.py
  test_pipeline_tools_and_cli.py
  test_regressions.py
```

## Configuration

`Config.load()` merges:

1. built-in defaults,
2. `config.yaml`,
3. environment variables loaded through `.env`,
4. explicit override values from the CLI.

Secrets are environment-only:

```text
ANTHROPIC_API_KEY
BRAVE_API_KEY
```

Do not add API keys to `config.yaml`, tests, docs examples with real values, or trajectory logs.

Current `config.yaml` keys:

```yaml
models:
  haiku: claude-haiku-4-5-20251001
  sonnet: claude-sonnet-4-6

retrieval:
  max_turns: 25
  max_search_results: 5
  max_page_chars: 30000

judge:
  retry_gap_fill: true
  gap_fill_max_turns: 5

concurrency:
  max_schools_parallel: 3

http:
  timeout: 20
  retries: 2

output:
  dir: output

logs:
  dir: logs
```

Set `logs.dir: ""` to disable trajectory logging.

## CLI Behavior

Supported school inputs:

```bash
grad-agent --schools input/schools.json --cv input/cv.md
grad-agent --school "MIT" --program "PhD EECS" --cv input/cv.md
```

Useful flags:

- `--config PATH`: use a custom YAML config.
- `--output DIR`: override `output.dir`.
- `--context PATH`: inject applicant context into all stages.
- `--max-turns N`: override retrieval turn budget.
- `--max-parallel N`: override config only; current execution remains sequential.
- `--no-gap-fill`: disable insufficient-profile gap-fill.
- `--verbose`: disable the TUI and enable debug logs.

Default `--context input/context.md` is skipped if missing. A user-specified context path must exist.

## Inputs and Outputs

Inputs:

- `input/schools.json`: list of `{"school": "...", "program": "..."}` objects.
- `input/cv.md`: applicant CV in plain text or Markdown.
- `input/context.md`: optional applicant context.

Outputs:

- `output/{school}_{program}_profile.md`: one Markdown report per school.
- `output/summary.md`: ranked Markdown summary table.
- `logs/{YYYY-MM-DDTHHMMSS}/{school_slug}.jsonl`: optional trajectory logs.

`input/`, `output/`, and `logs/` are expected to contain local or generated data. Avoid relying on non-example files in tests.

## Data Models

The main Pydantic models live in `grad_agent/models.py`:

- `SchoolProfile`
- `Requirements`
- `ApplicantReports`
- `JudgeReport`
- `FlaggedField`
- `FitAssessment`
- `SchoolResult`

Model validators intentionally accept common LLM output variants, such as:

- string values for list fields,
- dict deadlines flattened into a string,
- dict advisor entries converted into display strings,
- dict source maps converted into URL lists,
- list notes joined into one string.

Preserve these coercions unless a schema change is deliberate and covered by tests.

## Reporting and Ranking

`reporting/markdown.py` renders:

- one complete school report from `SchoolProfile`, optional `JudgeReport`, and optional `FitAssessment`,
- one summary table ranked by confidence-adjusted fit score.

Summary priority weights are:

- high confidence: `1.0`
- medium confidence: `0.85`
- low confidence: `0.65`

`runner.calibrate_fit_confidence()` lowers fit confidence after judge results:

- `insufficient` forces low confidence,
- `partial` caps high confidence at medium.

## TUI and Events

When stderr is a TTY and `--verbose` is not set, `cli.py` starts `PipelineTUI`.

Events are defined in `grad_agent/events.py`:

- `SchoolStarted`
- `StageStarted`
- `TurnProgress`
- `ToolCalled`
- `SchoolDone`

The TUI replaces root log handlers while active, then leaves the final Rich display visible.

## Trajectory Logging

`TrajectoryLogger` writes one JSONL file per school per run. Record types include:

- `stage_start`
- `stage_end`
- `api_response`
- `tool_result`
- `profile`
- `judge_report`
- `fit_assessment`

Trajectory logs include full model content and tool results. Treat them as potentially sensitive local artifacts.

## Tests

Run all tests with:

```bash
python3 -m unittest
```

The tests use fakes for Anthropic and HTTP behavior. Do not make tests depend on live network calls or real API keys.

When changing prompts, model schemas, output rendering, config behavior, or CLI parsing, update or add focused regression tests in `tests/`.

## Development Notes

- Keep prompt text in `grad_agent/pipeline/prompts.py`; do not inline new stage prompts elsewhere.
- Keep external tool schemas and handlers together in `grad_agent/pipeline/tools.py`.
- Tool handlers should return strings suitable for LLM consumption, not structured Python objects.
- Prefer `extract_json_object()` for parsing model JSON instead of duplicating parsing logic.
- Use `SchoolProfile.model_validate`, `JudgeReport.model_validate`, and `FitAssessment.model_validate` at API boundaries.
- Preserve partial-failure behavior in `run_school`: retrieval failure returns a stub result; judge, fit, and gap-fill errors are captured in stats instead of crashing the entire run.
- Keep generated output paths filesystem-safe through `_safe_filename()`.
