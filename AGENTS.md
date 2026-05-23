# Repository Instructions

## Code Style

- Be succinct in code, but do not compress logic into hard-to-read tricks.
- Prefer existing helpers and local patterns over new abstractions.
- Keep comments sparse. Do not comment code that is obvious to a programmer.
- Use ASCII in comments. Replace non-ASCII punctuation or symbols with plain ASCII when editing comments.
- Do not add delimiter comments such as long separator lines. Split files or functions when structure is getting too large.
- Do not commit secrets, generated outputs, local inputs, or trajectory logs.

## Project Overview

This repository implements a graduate school research agent using local Qwen retrieval through vLLM plus Anthropic Sonnet for judging and fit assessment. It researches graduate programs, evaluates profile quality, assesses applicant fit against a CV, and writes Markdown and PDF reports.

The installed CLI entry point is:

```bash
grad-agent
```

Primary Python runtime dependencies are `anthropic`, `pydantic`, `httpx`, `pyyaml`, `python-dotenv`, `rich`, `markdown`, and `weasyprint`. Default retrieval also expects one or more local OpenAI-compatible vLLM servers running `Qwen/Qwen3.6-35B-A3B-FP8`.

## Development Environment

Use the local micromamba environment named `graduate-school-agent` for development commands:

```bash
micromamba run -n graduate-school-agent python3 -m unittest
micromamba run -n graduate-school-agent ruff check .
```

The environment name is local-specific. If it is unavailable, use an equivalent Python 3.11+ environment with the project dependencies and `ruff` installed.

## Current Architecture

Pipeline per school:

```text
retrieval (local Qwen vLLM by default) -> judge (Sonnet) + fit (Sonnet) -> Markdown/PDF output
                                      -> optional gap-fill (same retrieval backend) -> re-judge + re-fit
```

Important behavior:

- Default `retrieval` uses local `Qwen/Qwen3.6-35B-A3B-FP8` through the configured OpenAI-compatible vLLM endpoints.
- The alternate retrieval backend is Claude Haiku via Anthropic native tool calls. Select it with `retrieval.backend: anthropic_haiku` or `--retrieval-backend anthropic_haiku`.
- Local Qwen retrieval uses a strict JSON command loop for two tools: `web_search` and `fetch_page`. It may emit either one tool command or a batched `tools` list per turn.
- Local retrieval defaults to two parallel agents per school on a 2 x H100 topology. Agents focus on full profile, admissions, faculty, and applicant-report evidence, then merge into one `SchoolProfile`.
- Anthropic Haiku retrieval can emit multiple native tool-use blocks in one response; tool handlers run concurrently.
- `web_search` calls Brave Search.
- `fetch_page` fetches HTTP(S) URLs with `httpx`, strips HTML, and truncates to `config.max_page_chars`.
- `judge` and `fit` run concurrently for a single school with `asyncio.gather`, including the post-gap-fill re-judge and re-fit pass.
- Gap-fill runs only when enabled, the judge returns `insufficient`, and suggested queries are present.
- Schools run with bounded concurrency from `max_schools_parallel`; concurrent Sonnet judge/fit calls are bounded separately by `max_sonnet_parallel`.
- `http.retries` is applied to local vLLM endpoint failover but is not currently applied by the fetch/search tool handlers.
- Anthropic rate-limit retries are handled by `grad_agent.util.retry.api_create_with_retry`.

## Package Layout

```text
grad_agent/
  cli.py                 argparse CLI entry point
  config.py              YAML, .env, and environment config loading
  events.py              pipeline event dataclasses
  llm/
    vllm.py              OpenAI-compatible local vLLM client
  models.py              Pydantic schemas
  tui.py                 Rich live terminal UI
  pipeline/
    retrieval.py         Anthropic or local-vLLM retrieval loop
    judge.py             Sonnet quality judge
    fit.py               Sonnet CV-aware fit assessor
    runner.py            per-school orchestration and output writing
    prompts.py           system and user prompts
    tools.py             Brave search and page fetch tool handlers
  reporting/
    markdown.py          report and summary rendering
    pdf.py               Markdown-to-PDF rendering and report directory helpers
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
VLLM_API_KEY  # optional, only when local vLLM servers require bearer auth
```

Do not add API keys to `config.yaml`, tests, docs examples with real values, or trajectory logs.

Current `config.yaml` keys:

```yaml
models:
  haiku: claude-haiku-4-5-20251001
  sonnet: claude-sonnet-4-6
  local_retrieval: Qwen/Qwen3.6-35B-A3B-FP8

retrieval:
  backend: local_qwen_vllm
  max_turns: 25
  max_search_results: 5
  max_page_chars: 30000
  local_model_count: 2
  local_parallel_agents: 2
  local_max_parallel_tool_calls: 8
  local_base_urls:
    - http://127.0.0.1:8001/v1
    - http://127.0.0.1:8002/v1
  local_timeout: 600

judge:
  retry_gap_fill: true
  gap_fill_max_turns: 5

concurrency:
  max_schools_parallel: 8
  max_sonnet_parallel: 8

http:
  timeout: 20
  retries: 2

output:
  dir: output

logs:
  dir: logs

deploy:
  host: 0.0.0.0
  vllm_args:
    - --trust-remote-code
  log_dir: logs/vllm
  micromamba_env: graduate-school-agent
  python_version: "3.11"
  system_packages:
    - curl
    - git
    - build-essential
    - tmux
    - libcairo2
    - libpango-1.0-0
    - libpangoft2-1.0-0
    - libgdk-pixbuf-2.0-0
    - shared-mime-info
  pip_packages:
    - vllm

```

Set `logs.dir: ""` to disable trajectory logging.

## Local vLLM Usage

Default runtime expects `retrieval.local_model_count` independent vLLM servers, one per GPU. The default config uses two local model copies:

```text
http://127.0.0.1:8001/v1
http://127.0.0.1:8002/v1
```

For fewer or more GPUs, set `retrieval.local_model_count` to the number of model copies and provide the same number of endpoints in `retrieval.local_base_urls`. Set `retrieval.local_parallel_agents` to the desired per-school local fanout, usually the endpoint count. The app validates that model count and endpoint count match, round-robins local retrieval calls across the endpoints, and lets local vLLM handle batching.

Deployment helpers live in `deploy/`:

```bash
deploy/start-vllm.sh
```

Health check:

```bash
deploy/healthcheck.sh
```

Run with the default local retrieval backend:

```bash
grad-agent --schools input/schools.json --cv input/cv.md
```

Switch retrieval back to Anthropic Haiku when local endpoints are unavailable:

```bash
grad-agent --schools input/schools.json --cv input/cv.md --retrieval-backend anthropic_haiku
```

The Python app does not launch or supervise vLLM. Keep server startup and node setup in `deploy/` scripts and docs.

## CLI Behavior

Supported school inputs:

```bash
grad-agent --schools input/schools.json --cv input/cv.md
grad-agent --school "MIT" --program "PhD EECS" --cv input/cv.md
```

Useful flags:

- `--config PATH`: use a custom YAML config.
- `--output DIR`: override the report output root from `output.dir`.
- `--context PATH`: inject applicant context into all stages.
- `--max-turns N`: override retrieval turn budget.
- `--max-parallel N`: override max concurrent school pipelines.
- `--retrieval-backend {anthropic_haiku,local_qwen_vllm}`: override retrieval backend.
- `--no-gap-fill`: disable insufficient-profile gap-fill.
- `--verbose`: disable the TUI and enable debug logs.

Default `--context input/context.md` is skipped if missing. A user-specified context path must exist.

## Inputs and Outputs

Inputs:

- `input/schools.json`: list of `{"school": "...", "program": "..."}` objects.
- `input/cv.md`: applicant CV in plain text or Markdown.
- `input/context.md`: optional applicant context.

Outputs:

- `output/markdown/{school}_{program}_profile.md`: one Markdown report per school.
- `output/markdown/summary.md`: ranked Markdown summary table.
- `output/pdf/{school}_{program}_profile.pdf`: PDF version of each school report.
- `output/pdf/summary.pdf`: PDF version of the summary table.
- `logs/{YYYY-MM-DDTHHMMSS}/{school_slug}.jsonl`: optional trajectory logs.

`input/`, `output/`, and `logs/` are expected to contain local or generated data. Avoid relying on non-example files in tests.

## Data Models

The main Pydantic models live in `grad_agent/models.py`:

- `SchoolProfile`
- `Requirements`
- `GREPolicy`
- `ApplicantReports`
- `JudgeReport`
- `FlaggedField`
- `FitAssessment`
- `SchoolResult`

Model validators intentionally accept common LLM output variants, such as:

- string values for list fields,
- dict deadlines flattened into a string,
- GRE policy variants normalized to `Required`, `Considered`, or `Not Considered`,
- dict advisor entries converted into display strings,
- dict source maps converted into URL lists,
- list notes joined into one string.

Preserve these coercions unless a schema change is deliberate and covered by tests.

## Reporting and Ranking

`reporting/markdown.py` renders:

- one complete school report from `SchoolProfile`, optional `JudgeReport`, and optional `FitAssessment`,
- one summary table ranked by confidence-adjusted fit score, including a `GRE` column from `requirements.gre_policy`.

`reporting/pdf.py` converts generated Markdown reports to simple styled PDFs with `markdown` and `weasyprint`. Future report writes should keep raw Markdown under `output/markdown/` and matching PDFs under `output/pdf/`.

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
- `TurnProgress` with optional stage and worker labels for parallel retrieval
- `ToolCalled` with optional stage, worker, and tool batch size
- `SchoolDone`

The TUI replaces root log handlers while active, then leaves the final Rich display visible. Its header summarizes retrieval backend, local vLLM topology, local agent and tool fanout, and school/Sonnet concurrency; school rows show per-worker retrieval turns for local parallel runs.

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
micromamba run -n graduate-school-agent python3 -m unittest
```

Run lint checks with:

```bash
micromamba run -n graduate-school-agent ruff check .
```

All source changes should pass unit tests and `ruff check .`. Do not run the full `grad-agent` program for routine validation because it can spend API tokens. The tests use fakes for Anthropic and HTTP behavior. Do not make tests depend on live network calls or real API keys.

When changing prompts, model schemas, output rendering, config behavior, or CLI parsing, update or add focused regression tests in `tests/`.

## Development Notes

- Keep prompt text and local retrieval protocol text in `grad_agent/pipeline/prompts.py`; do not inline new stage prompts elsewhere.
- Keep external tool schemas and handlers together in `grad_agent/pipeline/tools.py`.
- Tool handlers should return strings suitable for LLM consumption, not structured Python objects.
- Keep local vLLM endpoint calling in `grad_agent/llm/vllm.py`. Do not make the package responsible for launching vLLM processes.
- Prefer `extract_json_object()` for parsing model JSON instead of duplicating parsing logic.
- Use `SchoolProfile.model_validate`, `JudgeReport.model_validate`, and `FitAssessment.model_validate` at API boundaries.
- Preserve partial-failure behavior in `run_school`: retrieval failure returns a stub result; judge, fit, and gap-fill errors are captured in stats instead of crashing the entire run.
- Keep generated output paths filesystem-safe through `_safe_filename()`.
