# Repository Instructions

## Code Style

- Be succinct in code, but do not compress logic into hard-to-read tricks.
- Prefer existing helpers and local patterns over new abstractions.
- Keep comments sparse. Do not comment code that is obvious to a programmer.
- Use ASCII in comments. Replace non-ASCII punctuation or symbols with plain ASCII when editing comments.
- Do not add delimiter comments such as long separator lines. Split files or functions when structure is getting too large.
- Do not commit secrets, generated outputs, local inputs, or trajectory logs.

## Project Overview

This repository implements a graduate school research agent. It researches graduate programs, evaluates profile quality, assesses applicant fit against a CV, and writes Markdown and PDF reports.

Default pipeline:

```text
retrieval backend -> judge (Sonnet) + fit (Sonnet) -> Markdown/PDF
                  -> optional gap-fill -> re-judge + re-fit
```

The installed CLI entry point is:

```bash
grad-agent
```

Primary Python dependencies are `anthropic`, `pydantic`, `httpx`, `pyyaml`, `python-dotenv`, `rich`, `markdown`, and `weasyprint`. Default retrieval also expects one or more OpenAI-compatible local endpoints running `Qwen/Qwen3.6-35B-A3B-FP8` through vLLM or an equivalent server.

## Development Environment

Use the local micromamba environment named `graduate-school-agent` for routine checks:

```bash
micromamba run -n graduate-school-agent python3 -m unittest
micromamba run -n graduate-school-agent ruff check .
```

If that environment is unavailable, use an equivalent Python 3.11+ environment with the project dependencies and `ruff`.

Do not run the full `grad-agent` program for routine validation because it can spend API tokens.

## Architecture Notes

- Retrieval is dispatched through a modular backend registry. `retrieval.backend` names an implementation, not a one-off conditional path.
- The currently supported retrieval implementations are `local_qwen_vllm` and `anthropic_haiku`.
- `local_qwen_vllm` uses local OpenAI-compatible chat completions, currently configured for Qwen/vLLM. It uses a strict JSON command loop for `web_search` and `fetch_page` and may emit one tool command or a batched `tools` list per turn.
- Local retrieval defaults to two parallel agents per school on a 2 x H100 topology. Agents gather evidence across profile, admissions, faculty, and applicant reports, then merge into one `SchoolProfile`.
- `anthropic_haiku` uses Claude Haiku native tool calls through the Anthropic Messages API. It may emit multiple native tool-use blocks in one response; tool handlers run concurrently.
- `web_search` calls Brave Search. `fetch_page` uses `httpx`, strips HTML, and truncates to `config.max_page_chars`.
- Judge and fit run concurrently for a school, including the post-gap-fill pass.
- Gap-fill runs only when enabled, the judge returns `insufficient`, and suggested queries are present.
- Schools run with bounded concurrency from `max_schools_parallel`; concurrent Sonnet calls are bounded separately by `max_sonnet_parallel`.
- `http.retries` applies to local endpoint failover, not to fetch/search handlers.
- API-based retrieval calls should use `grad_agent.util.retry.api_create_with_retry` for rate-limit retry and exponential backoff behavior.

## Package Layout

```text
grad_agent/
  cli.py                 argparse CLI entry point
  cli_support.py         CLI input/config override helpers
  config.py              YAML, .env, and environment config loading
  events.py              pipeline event dataclasses
  retrieval_registry.py  retrieval backend metadata and supported ids
  llm/vllm.py            OpenAI-compatible local retrieval client
  models.py              Pydantic schemas
  tui.py                 Rich live terminal UI
  pipeline/
    retrieval.py         retrieval stage dispatch
    retrieval_backends.py concrete API and local retrieval implementations
    local_retrieval.py   local JSON tool-command retrieval loops
    gap_fill.py          targeted insufficient-profile retrieval
    tool_loop.py         shared tool command execution and events
    confidence.py        judge-aware fit confidence calibration
    judge.py             Sonnet quality judge
    fit.py               Sonnet CV-aware fit assessor
    runner.py            per-school orchestration and output writing
    prompts.py           system and user prompts
    tools.py             Brave search and page fetch handlers
  reporting/
    markdown.py          report and summary rendering
    paths.py             filesystem-safe report path helpers
    pdf.py               Markdown-to-PDF rendering
    stats.py             token, cost, timing, and run statistics
    trajectory.py        per-school JSONL trajectory logger
  util/
    json.py              model JSON extraction
    log.py               structured logging
    retry.py             Anthropic rate-limit retry helper
tests/
  preview_tui.py          no-token TUI preview CLI
  tui_demo.py             config-driven fake TUI event simulation
resources/
  demo.tape               VHS script for the README TUI demo
  demo.gif                rendered README TUI demo
```

## Configuration

`Config.load()` merges built-in defaults, `config.yaml`, `.env`, and explicit CLI overrides.

Secrets are environment-only:

```text
ANTHROPIC_API_KEY
BRAVE_API_KEY
VLLM_API_KEY  # optional, only when local vLLM servers require bearer auth
```

Do not add API keys to `config.yaml`, tests, docs examples with real values, or trajectory logs.

Important config areas:

- `models.*`: Claude and local retrieval model names.
- `input.cv`, `input.context`, `input.schools`: default input paths.
- `retrieval.backend`: registered retrieval backend id. Current options are `local_qwen_vllm` and `anthropic_haiku`.
- `retrieval.max_turns`, `max_search_results`, `max_page_chars`: retrieval budgets.
- `retrieval.local_model_count`: expected number of local model copies.
- `retrieval.local_base_urls`: local OpenAI-compatible endpoints; length must match `local_model_count`.
- `retrieval.local_parallel_agents`: per-school local retrieval fanout.
- `retrieval.local_max_parallel_tool_calls`: concurrent tool calls from one local turn.
- `judge.retry_gap_fill`, `judge.gap_fill_max_turns`: gap-fill behavior.
- `concurrency.max_schools_parallel`: school pipeline concurrency.
- `concurrency.max_sonnet_parallel`: concurrent Sonnet judge/fit calls.
- `logs.dir`: set to `""` to disable trajectory logging.
- `deploy.*`: settings consumed by deployment scripts.

## Retrieval Backend Usage

Retrieval backends are registered in `grad_agent/retrieval_registry.py` and implemented in `grad_agent/pipeline/retrieval_backends.py`. To add a backend, add its metadata, implement the `RetrievalBackend.run()` protocol, register the implementation in `_BACKEND_IMPLEMENTATIONS`, and add focused tests for config validation, model selection, dispatch, and endpoint-specific tool-call behavior.

API-based retrieval implementations should keep endpoint calling and retry behavior inside their backend class or a small LLM client helper. Local implementations should use OpenAI-compatible chat completions where possible and keep local process startup outside the Python package.

## Local Endpoint Usage

Default runtime expects `retrieval.local_model_count` independent local OpenAI-compatible servers, one per GPU. The default config uses:

```text
http://127.0.0.1:8001/v1
http://127.0.0.1:8002/v1
```

For fewer or more GPUs, set `retrieval.local_model_count` to the number of model copies and provide the same number of endpoints in `retrieval.local_base_urls`. Set `retrieval.local_parallel_agents` to the desired per-school fanout, usually the endpoint count.

Deployment helpers live in `deploy/`:

```bash
deploy/start-vllm.sh
deploy/healthcheck.sh
```

The Python app does not launch or supervise local model servers. Keep server startup and node setup in `deploy/` scripts and docs.

## CLI Behavior

Supported school inputs:

```bash
grad-agent
grad-agent --schools input/schools.json --cv input/cv.md
grad-agent --school "MIT" --program "PhD EECS" --cv input/cv.md
```

Useful flags:

- `--config PATH`: use a custom YAML config.
- `--output DIR`: override report output root.
- `--schools PATH`, `--cv PATH`, `--context PATH`: override input paths.
- `--max-turns N`: override retrieval turn budget.
- `--max-parallel N`: override max concurrent school pipelines.
- `--retrieval-backend {anthropic_haiku,local_qwen_vllm}`: override retrieval backend implementation.
- `--no-gap-fill`: disable insufficient-profile gap-fill.
- `--verbose`: disable the TUI and enable debug logs.

Configured `input.context: input/context.md` is skipped if missing. Any other configured or CLI-specified context path must exist.

## Inputs and Outputs

Inputs:

- `input.schools`: configured school list path; default `input/schools.json`.
- `input.cv`: configured applicant CV path; default `input/cv.md`.
- `input.context`: configured optional applicant context path; default `input/context.md`.

Outputs:

- `output/markdown/{school}_{program}_profile.md`
- `output/markdown/summary.md`
- `output/pdf/{school}_{program}_profile.pdf`
- `output/pdf/summary.pdf`
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

Model validators intentionally accept common LLM output variants, including string/list coercions, flattened deadline dicts, normalized GRE policy variants, dict advisor entries, dict source maps, and list notes. Preserve these coercions unless a schema change is deliberate and covered by tests.

## Reporting and Ranking

- `reporting/markdown.py` renders complete school reports and the ranked summary table.
- `reporting/pdf.py` converts generated Markdown reports to simple styled PDFs.
- Keep raw Markdown under `output/markdown/` and matching PDFs under `output/pdf/`.
- Summary ranking uses confidence-adjusted fit scores: high `1.0`, medium `0.85`, low `0.65`.
- `pipeline.confidence.calibrate_fit_confidence()` forces low confidence for insufficient profiles and caps partial profiles at medium confidence.

## TUI and Events

When stderr is a TTY and `--verbose` is not set, `cli.py` starts `PipelineTUI`.

Events are defined in `grad_agent/events.py`:

- `SchoolStarted`
- `StageStarted`
- `TurnProgress` with optional stage and worker labels
- `ToolCalled` with optional stage, worker, and tool batch size
- `SchoolDone`

The TUI replaces root log handlers while active and leaves the final Rich display visible. Its progress bar is colored by school state, matching table stage colors. Its compact school table shows currently running schools only, keeps each school to one row, and caps visible school rows at 8 when more are running.

No-token TUI previews are driven from `tests/preview_tui.py` and `tests/tui_demo.py`, not from `grad_agent/tui.py`. The preview loads `Config`, reads `input.schools`, and simulates bounded school concurrency, random latency, retrieval turns, tool calls, judge/fit, and gap-fill events with seeded randomness. Keep hard-coded fake log message text and preview-only scheduling in `tests/tui_demo.py`; keep `grad_agent/tui.py` focused on rendering and consuming real `PipelineEvent` objects.

## Trajectory Logging

`TrajectoryLogger` writes one JSONL file per school per run. Record types include stage boundaries, API responses, tool results, profiles, judge reports, and fit assessments.

Trajectory logs include full model content and tool results. Treat them as potentially sensitive local artifacts.

## Tests

Run all tests and lint checks with:

```bash
micromamba run -n graduate-school-agent python3 -m unittest
micromamba run -n graduate-school-agent ruff check .
```

The tests use fakes for Anthropic and HTTP behavior. Do not make tests depend on live network calls or real API keys.

When changing prompts, model schemas, output rendering, config behavior, or CLI parsing, update or add focused regression tests in `tests/`.

Use `python3 -m tests.preview_tui` or `python3 -m tests.preview_tui --snapshot` for TUI visual checks without API calls. The preview accepts `--config PATH` and `--seed N`; it should remain config-driven and should not duplicate school fixtures in production code.

## Development Notes

- Keep prompt text and local retrieval protocol text in `grad_agent/pipeline/prompts.py`.
- Keep external tool schemas and handlers in `grad_agent/pipeline/tools.py`.
- Keep shared retrieval tool execution and `ToolCalled` event behavior in `grad_agent/pipeline/tool_loop.py`.
- Keep backend selection metadata in `grad_agent/retrieval_registry.py`.
- Keep concrete retrieval implementation classes in `grad_agent/pipeline/retrieval_backends.py`; `retrieval.py` should remain thin stage dispatch.
- Keep local JSON tool-command behavior in `grad_agent/pipeline/local_retrieval.py`.
- Keep targeted insufficient-profile retrieval in `grad_agent/pipeline/gap_fill.py`.
- Tool handlers should return strings suitable for LLM consumption, not structured Python objects.
- Keep local OpenAI-compatible endpoint calling in `grad_agent/llm/vllm.py`. Do not make the package responsible for launching local model processes.
- Prefer `extract_json_object()` for parsing model JSON.
- Use `SchoolProfile.model_validate`, `JudgeReport.model_validate`, and `FitAssessment.model_validate` at API boundaries.
- Preserve partial-failure behavior in `run_school`: retrieval failure returns a stub result; judge, fit, and gap-fill errors are captured in stats instead of crashing the run.
- Keep generated output paths filesystem-safe through `reporting.paths.safe_filename()`.
- Do not edit `TODO.md` unless told to.
