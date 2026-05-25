# Graduate School Agent

An LLM research agent for graduate application planning. It gathers program information, checks source quality, assesses applicant fit against a CV, and writes Markdown and PDF reports.

The default pipeline uses a modular retrieval backend layer, then Claude Sonnet for profile judging and fit assessment. Registered retrieval implementations include local OpenAI-compatible endpoints, remote OpenAI-compatible APIs, and Anthropic tool-use models.

![Graduate School Agent TUI demo](resources/demo.gif)
Demo run of the TUI.

## Features

- Web retrieval with Brave Search and page fetching.
- Pluggable retrieval backends for local endpoints or API-based model calls.
- Local OpenAI-compatible retrieval with parallel agents and batched tool calls.
- Anthropic Haiku or Sonnet retrieval with native tool calls and retry/backoff handling.
- Remote OpenAI-compatible API retrieval for OpenAI, OpenRouter, Together, Groq, or similar chat-completions endpoints.
- Sonnet quality judging for missing, stale, contradictory, or weakly sourced fields.
- Sonnet CV-aware fit assessment.
- Optional gap-fill pass for insufficient profiles.
- Markdown and PDF reports plus a ranked summary.
- Rich terminal UI and optional JSONL trajectory logs.

## Requirements

- Python 3.11+
- Anthropic API key
- Brave Search API key
- WeasyPrint native libraries for PDF rendering
- Local OpenAI-compatible endpoints for the default retrieval backend

Install in editable mode:

```bash
python3 -m pip install -e .
```

Set secrets in your shell or a local `.env` file:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export BRAVE_API_KEY=BSA...
export OPENAI_API_KEY=sk-...
```

`config.yaml` is for non-secret settings only. Set `VLLM_API_KEY` only if your local endpoints require bearer-token auth. Set `OPENAI_COMPATIBLE_API_KEY` instead of `OPENAI_API_KEY` when using a non-OpenAI compatible API provider.

## Quick Start

Create inputs:

```bash
cp input/schools.example.json input/schools.json
$EDITOR input/schools.json
$EDITOR input/cv.md
```

Optional applicant context can be placed at `input/context.md` for subfield interests, advisor preferences, funding needs, geographic constraints, or scoring guidance.

Start and check local retrieval servers when using the default backend implementation:

```bash
deploy/start-vllm.sh
deploy/healthcheck.sh
```

Run the agent:

```bash
grad-agent
```

Run one school without editing `input/schools.json`:

```bash
grad-agent --school "MIT" --program "PhD Electrical Engineering and Computer Science" --cv input/cv.md
```

Swap retrieval backends from the CLI:

```bash
grad-agent --retrieval-backend anthropic_haiku
grad-agent --retrieval-backend anthropic_sonnet
grad-agent --retrieval-backend openai_compatible
grad-agent --retrieval-backend local_openai_compatible
```

## Inputs

`input/schools.json` is a list of school/program objects:

```json
[
  {
    "school": "Stanford University",
    "program": "MS Computer Science"
  }
]
```

`input/cv.md` is plain text or Markdown. Defaults come from `input.cv`, `input.context`, and `input.schools` in `config.yaml`; CLI flags override them. The default context path, `input/context.md`, is skipped if absent. Any custom context path must exist.

## CLI

Common commands:

```bash
grad-agent
grad-agent --schools input/schools.json --cv input/cv.md
grad-agent --school "School Name" --program "Program Name" --cv input/cv.md
```

Useful flags:

- `--config PATH`: load another YAML config file.
- `--output DIR`: override report output root.
- `--schools PATH`, `--cv PATH`, `--context PATH`: override input paths.
- `--max-turns N`: override retrieval turn budget.
- `--max-parallel N`: override max concurrent school pipelines.
- `--retrieval-backend BACKEND`: choose any registered retrieval backend implementation.
- `--no-gap-fill`: disable targeted gap-fill.
- `--verbose`: bypass the Rich TUI and print debug logs.

## Outputs

Reports are written under `output/` by default:

- `markdown/{school}_{program}_profile.md`
- `markdown/summary.md`
- `pdf/{school}_{program}_profile.pdf`
- `pdf/summary.pdf`

When `logs.dir` is non-empty, trajectory logs are written to `logs/{YYYY-MM-DDTHHMMSS}/`. These logs include full model responses and tool results, so treat them as sensitive local artifacts.

## Pipeline

```text
retrieval -> judge + fit -> Markdown/PDF
          -> optional gap-fill -> re-judge + re-fit
```

- `retrieval` produces a structured `SchoolProfile` using Brave Search and fetched web pages.
- `judge` evaluates completeness, source quality, consistency, freshness, and actionability.
- `fit` compares the profile with the applicant CV and optional context.
- `gap-fill` runs only when enabled and the judge marks the initial profile insufficient.

Schools run with bounded concurrency. Judge and fit calls run concurrently for each school and are separately bounded by `concurrency.max_sonnet_parallel`.

## Configuration

`Config.load()` merges built-in defaults, `config.yaml`, `.env`, and explicit CLI overrides. See `config.yaml` for current defaults.

- `retrieval.backend`: registered retrieval backend id. Current options are `local_qwen_vllm`, `local_openai_compatible`, `openai_compatible`, `anthropic_haiku`, and `anthropic_sonnet`.
- `input.*`: default CV, context, and schools paths.
- `models.local_retrieval`, `models.openai_retrieval`, `models.haiku`, and `models.sonnet`: model ids used by retrieval implementations.
- `retrieval.local_model_count` and `retrieval.local_base_urls`: local endpoint topology.
- `retrieval.openai_base_urls`: remote OpenAI-compatible API endpoints.
- `retrieval.local_parallel_agents`: per-school local retrieval fanout.
- `concurrency.*`: school and Sonnet concurrency limits.
- `logs.dir`: set to `""` to disable trajectory logging.

## Retrieval Backends

Retrieval is selected by `retrieval.backend` and dispatched through a registry in `grad_agent/retrieval_registry.py`. Concrete implementations live under `grad_agent/pipeline/retrieval_backends/`.

Current implementations:

- `local_qwen_vllm`: local OpenAI-compatible chat completions. The default model is `Qwen/Qwen3.6-35B-A3B-FP8`; calls round-robin across configured endpoints and fail over according to `http.retries`.
- `local_openai_compatible`: generic local OpenAI-compatible chat completions. Use this id when the local model is not Qwen.
- `openai_compatible`: remote OpenAI-compatible chat completions. Set `models.openai_retrieval`, `retrieval.openai_base_urls`, and `OPENAI_API_KEY` or `OPENAI_COMPATIBLE_API_KEY`.
- `anthropic_haiku`: Anthropic Messages API using native tool-use blocks and the configured Haiku model.
- `anthropic_sonnet`: Anthropic Messages API using native tool-use blocks and the configured Sonnet model.

Hot-swap examples:

```yaml
retrieval:
  backend: openai_compatible
  openai_base_urls:
    - https://api.openai.com/v1

models:
  openai_retrieval: gpt-4.1-mini
```

To add another retrieval option, add a backend spec, implement the `RetrievalBackend.run()` protocol in a new module under `grad_agent/pipeline/retrieval_backends/`, register it in that package's `_BACKEND_IMPLEMENTATIONS`, and add focused tests for dispatch, model selection, and any endpoint-specific tool-call behavior.

## Local Endpoints

The default retrieval model is `Qwen/Qwen3.6-35B-A3B-FP8`. The default config expects two endpoints:

```text
http://127.0.0.1:8001/v1
http://127.0.0.1:8002/v1
```

For a different GPU count or local serving layout, set `retrieval.local_model_count`, `retrieval.local_parallel_agents`, and `retrieval.local_base_urls` consistently. The Python app validates endpoint count, round-robins calls, and applies local endpoint failover according to `http.retries`; it does not launch or supervise local model processes.

Deployment helpers:

```bash
deploy/setup-node.sh
deploy/start-vllm.sh
deploy/healthcheck.sh
```

See `deploy/README.md` for the deployment-specific flow.

## Development

Run tests and lint checks in the local micromamba environment:

```bash
micromamba run -n graduate-school-agent python3 -m unittest
micromamba run -n graduate-school-agent ruff check .
```

If that environment is unavailable, use an equivalent Python 3.11+ environment and run `python3 -m unittest` plus `ruff check .`.

Preview the TUI without API calls:

```bash
python3 -m tests.preview_tui
python3 -m tests.preview_tui --snapshot
```

The preview uses the school list from `input.schools` in `config.yaml`. Use `--config PATH` to preview another configuration and `--seed N` for repeatable simulated progress.
The live TUI colors the progress bar by school state, matching the table stage colors. Its compact table shows currently running schools only, keeps each school to one row, and caps visible school rows at 8 when concurrency is higher.

Docs assets live in `resources/`. `resources/demo.tape` is the VHS script used to render `resources/demo.gif`.
