# Repository Instructions

## Code Style

- Be succinct in code, but do not compress logic into hard-to-read tricks.
- Prefer existing helpers and local patterns over new abstractions.
- Keep comments sparse. Do not comment code that is obvious to a programmer.
- Use ASCII in comments. Replace non-ASCII punctuation or symbols with plain ASCII when editing comments.
- Do not add delimiter comments such as long separator lines. Split files or functions when structure is getting too large.
- Do not commit secrets, generated outputs, local inputs, or trajectory logs.

## Documentation Structure

- Keep `README.md` focused on users: what the tool does, how to install it, how to provide inputs, how to run it, where outputs go, and the minimum backend choices needed to start.
- Keep implementation details, extension points, package layout, registry behavior, test strategy, and internal data contracts in `AGENTS.md` or focused subdirectory READMEs.
- Prefer subdirectory READMEs when the detail is useful to someone operating or extending one subsystem, such as deployment-specific setup in `deploy/README.md`.
- Do not duplicate long backend implementation notes in `README.md`; link or point readers to the relevant technical document instead.

## Project Overview

This repository implements a graduate school research agent. It researches graduate programs, evaluates profile quality, assesses applicant fit against a CV, and writes Markdown and PDF reports.

Default pipeline:

```text
retrieval backend -> judge backend + fit (Sonnet) -> Markdown/PDF
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
- The currently supported retrieval implementations are `local_qwen_vllm`, `local_openai_compatible`, `openai_compatible`, `anthropic_haiku`, and `anthropic_sonnet`.
- Judge is dispatched through a modular backend registry. `judge.backend` names an implementation; current options are `anthropic_sonnet`, `anthropic_haiku`, and `openai_compatible`.
- `local_qwen_vllm` uses local OpenAI-compatible chat completions, currently configured for Qwen/vLLM. It uses a strict JSON command loop for `web_search` and `fetch_page` and may emit one tool command or a batched `tools` list per turn.
- `local_openai_compatible` uses the same local endpoint path without tying the backend id to Qwen.
- `openai_compatible` uses remote OpenAI-compatible chat completions with the same JSON command loop. Configure it with `models.openai_retrieval`, `retrieval.openai_base_urls`, and `OPENAI_API_KEY` or `OPENAI_COMPATIBLE_API_KEY`.
- Local retrieval defaults to two parallel agents per school on a 2 x H100 topology. Agents gather evidence across profile, admissions, faculty, and applicant reports, then merge into one `SchoolProfile`.
- `anthropic_haiku` and `anthropic_sonnet` use native tool calls through the Anthropic Messages API. They may emit multiple native tool-use blocks in one response; tool handlers run concurrently.
- `web_search` calls Brave Search. `fetch_page` uses `httpx`, strips HTML, and truncates to `config.max_page_chars`.
- Judge and fit run concurrently for a school, including the post-gap-fill pass.
- Gap-fill runs only when enabled, the judge returns `insufficient`, and suggested queries are present.
- Schools run with bounded concurrency from `max_schools_parallel`; concurrent judge/fit calls are bounded separately by `max_sonnet_parallel`.
- `http.retries` applies to local endpoint failover, not to fetch/search handlers.
- API-based retrieval calls should use `grad_agent.util.retry.api_create_with_retry` for rate-limit retry and exponential backoff behavior.

## Package Layout

```text
grad_agent/
  cli.py                 argparse CLI entry point
  cli_support.py         CLI input/config override helpers
  config.py              YAML, .env, and environment config loading
  events.py              pipeline event dataclasses
  llm/vllm.py            OpenAI-compatible chat client
  models.py              Pydantic schemas
  tui.py                 Rich live terminal UI
  agents/
    retrieval/
      registry.py        retrieval backend metadata and supported ids
      service.py         retrieval dispatch
      gap_fill.py        targeted insufficient-profile retrieval
      local_loop.py      JSON tool-command retrieval loops
      prompts.py         retrieval and gap-fill prompts
      profile_merge.py   parallel worker profile merging
      tool_loop.py       shared retrieval tool execution and events
      tools.py           Brave search and page fetch handlers
      types.py           retrieval backend protocol and request type
      backends/          concrete retrieval implementations
    judge/
      registry.py        judge backend metadata and supported ids
      service.py         judge dispatch
      prompts.py         judge prompts
      types.py           judge backend protocol and request type
      backends/          concrete judge implementations
    fit/
      service.py         Sonnet CV-aware fit assessor
      confidence.py      judge-aware fit confidence calibration
      prompts.py         fit prompts
      scoring.py         deterministic fit score composition
  orchestration/
    runner.py            per-school orchestration and output writing
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
OPENAI_API_KEY  # when an OpenAI-compatible backend needs it
OPENAI_COMPATIBLE_API_KEY  # optional alternate key for compatible providers
OPENAI_JUDGE_API_KEY  # optional judge-specific compatible provider key
```

Do not add API keys to `config.yaml`, tests, docs examples with real values, or trajectory logs.

Important config areas:

- `models.*`: Claude, local retrieval, and OpenAI-compatible backend model names.
- `input.cv`, `input.context`, `input.schools`: default input paths.
- `retrieval.backend`: registered retrieval backend id. Current options are `local_qwen_vllm`, `local_openai_compatible`, `openai_compatible`, `anthropic_haiku`, and `anthropic_sonnet`.
- `judge.backend`: registered judge backend id. Current options are `anthropic_sonnet`, `anthropic_haiku`, and `openai_compatible`.
- `retrieval.max_turns`, `max_search_results`, `max_page_chars`: retrieval budgets.
- `retrieval.local_model_count`: expected number of local model copies.
- `retrieval.local_base_urls`: local OpenAI-compatible endpoints; length must match `local_model_count`.
- `retrieval.openai_base_urls`: remote OpenAI-compatible chat-completions endpoints.
- `judge.openai_base_urls`: remote OpenAI-compatible judge chat-completions endpoints.
- `retrieval.local_parallel_agents`: per-school local retrieval fanout.
- `retrieval.local_max_parallel_tool_calls`: concurrent tool calls from one local turn.
- `judge.retry_gap_fill`, `judge.gap_fill_max_turns`: gap-fill behavior.
- `concurrency.max_schools_parallel`: school pipeline concurrency.
- `concurrency.max_sonnet_parallel`: concurrent judge/fit calls.
- `logs.dir`: set to `""` to disable trajectory logging.
- `deploy.*`: settings consumed by deployment scripts.

OpenAI-compatible retrieval example:

```yaml
retrieval:
  backend: openai_compatible
  openai_base_urls:
    - https://api.openai.com/v1

models:
  openai_retrieval: gpt-4.1-mini
```

OpenAI-compatible judge example:

```yaml
judge:
  backend: openai_compatible
  openai_base_urls:
    - https://api.openai.com/v1

models:
  openai_judge: gpt-4.1
```

## Retrieval Backend Usage

Retrieval backends are registered in `grad_agent/agents/retrieval/registry.py` and implemented under `grad_agent/agents/retrieval/backends/`. To add a backend, add its metadata, implement the `RetrievalBackend.run()` protocol in a separate module in that directory, register the implementation in `_BACKEND_IMPLEMENTATIONS`, and add focused tests for config validation, model selection, dispatch, and endpoint-specific tool-call behavior.

API-based retrieval implementations should keep endpoint calling and retry behavior inside their backend class or a small LLM client helper. Local implementations should use OpenAI-compatible chat completions where possible and keep local process startup outside the Python package. The Anthropic backend uses native tool-use blocks; OpenAI-compatible backends use the JSON tool-command loop.

## Judge Backend Usage

Judge backends are registered in `grad_agent/agents/judge/registry.py` and implemented under `grad_agent/agents/judge/backends/`. To add a backend, add its metadata, implement the `JudgeBackend.run()` protocol in a separate module in that directory, register the implementation in `_BACKEND_IMPLEMENTATIONS`, and add focused tests for config validation, model selection, dispatch, and endpoint-specific response handling.

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
- `--retrieval-backend BACKEND`: override retrieval backend implementation.
- `--judge-backend BACKEND`: override judge backend implementation.
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
- `FitDimensionScore`
- `FitScoreBreakdown`
- `FitAssessment`
- `SchoolResult`

Model validators intentionally accept common LLM output variants, including string/list coercions, flattened deadline dicts, normalized GRE policy variants, dict advisor entries, dict source maps, and list notes. Preserve these coercions unless a schema change is deliberate and covered by tests.

Fit assessment uses a structured rubric. The fit agent should produce `score_breakdown` dimension scores and evidence, not an LLM-invented final real number. `FitAssessment` computes `overall_score` from the rubric, and `agents.fit.scoring.apply_program_fit_score()` reapplies program-type weights for MS-like and PhD-like targets.

## Reporting and Ranking

- `reporting/markdown.py` renders complete school reports and the ranked summary table.
- `reporting/pdf.py` converts generated Markdown reports to simple styled PDFs.
- Keep raw Markdown under `output/markdown/` and matching PDFs under `output/pdf/`.
- School reports show the computed overall fit score plus rubric dimension scores when the fit assessment includes a `score_breakdown`.
- Summary ranking uses confidence-adjusted fit scores: high `1.0`, medium `0.85`, low `0.65`.
- `agents.fit.confidence.calibrate_fit_confidence()` forces low confidence for insufficient profiles and caps partial profiles at medium confidence.

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

- Keep retrieval prompt text and local retrieval protocol text in `grad_agent/agents/retrieval/prompts.py`.
- Keep judge prompt text in `grad_agent/agents/judge/prompts.py`.
- Keep fit prompt text in `grad_agent/agents/fit/prompts.py`.
- Keep external retrieval tool schemas and handlers in `grad_agent/agents/retrieval/tools.py`.
- Keep shared retrieval tool execution and `ToolCalled` event behavior in `grad_agent/agents/retrieval/tool_loop.py`.
- Keep retrieval backend selection metadata in `grad_agent/agents/retrieval/registry.py`.
- Keep judge backend selection metadata in `grad_agent/agents/judge/registry.py`.
- Keep concrete retrieval implementation classes under `grad_agent/agents/retrieval/backends/`; `agents/retrieval/service.py` should remain thin dispatch.
- Keep concrete judge implementation classes under `grad_agent/agents/judge/backends/`; `agents/judge/service.py` should remain thin dispatch.
- Keep local JSON tool-command behavior in `grad_agent/agents/retrieval/local_loop.py`.
- Keep parallel worker profile merging in `grad_agent/agents/retrieval/profile_merge.py`.
- Keep targeted insufficient-profile retrieval in `grad_agent/agents/retrieval/gap_fill.py`.
- Tool handlers should return strings suitable for LLM consumption, not structured Python objects.
- Keep local OpenAI-compatible endpoint calling in `grad_agent/llm/vllm.py`. Do not make the package responsible for launching local model processes.
- Prefer `extract_json_object()` for parsing model JSON.
- Use `SchoolProfile.model_validate`, `JudgeReport.model_validate`, and `FitAssessment.model_validate` at API boundaries.
- Preserve partial-failure behavior in `run_school`: retrieval failure returns a stub result; judge, fit, and gap-fill errors are captured in stats instead of crashing the run.
- Keep generated output paths filesystem-safe through `reporting.paths.safe_filename()`.
- Do not edit `TODO.md` unless told to.
