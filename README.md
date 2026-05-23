# Graduate School Agent

An LLM research agent for graduate application planning. It gathers program information, checks source quality, assesses applicant fit against a CV, and writes Markdown and PDF reports.

The default pipeline uses local Qwen retrieval through OpenAI-compatible vLLM endpoints, then Claude Sonnet for profile judging and fit assessment. Claude Haiku can be used as the retrieval backend when local vLLM is unavailable.

## Features

- Web retrieval with Brave Search and page fetching.
- Local Qwen/vLLM retrieval with parallel agents and batched tool calls.
- Optional Anthropic Haiku retrieval backend.
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
- Local OpenAI-compatible vLLM endpoints for the default retrieval backend

Install in editable mode:

```bash
python3 -m pip install -e .
```

Set secrets in your shell or a local `.env` file:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export BRAVE_API_KEY=BSA...
```

`config.yaml` is for non-secret settings only. Set `VLLM_API_KEY` only if your vLLM endpoints require bearer-token auth.

## Quick Start

Create inputs:

```bash
cp input/schools.example.json input/schools.json
$EDITOR input/schools.json
$EDITOR input/cv.md
```

Optional applicant context can be placed at `input/context.md` for subfield interests, advisor preferences, funding needs, geographic constraints, or scoring guidance.

Start and check local retrieval servers when using the default backend:

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

Use Anthropic Haiku for retrieval instead of local Qwen:

```bash
grad-agent --retrieval-backend anthropic_haiku
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
grad-agent --summary-from output/markdown
```

Useful flags:

- `--config PATH`: load another YAML config file.
- `--output DIR`: override report output root.
- `--schools PATH`, `--cv PATH`, `--context PATH`: override input paths.
- `--max-turns N`: override retrieval turn budget.
- `--max-parallel N`: override max concurrent school pipelines.
- `--retrieval-backend {anthropic_haiku,local_qwen_vllm}`: choose retrieval backend.
- `--no-gap-fill`: disable targeted gap-fill.
- `--summary-from PATH`: rebuild summary reports from existing profile Markdown without model calls.
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

- `retrieval.backend`: `local_qwen_vllm` or `anthropic_haiku`.
- `input.*`: default CV, context, and schools paths.
- `retrieval.local_model_count` and `retrieval.local_base_urls`: local vLLM topology.
- `retrieval.local_parallel_agents`: per-school local retrieval fanout.
- `concurrency.*`: school and Sonnet concurrency limits.
- `logs.dir`: set to `""` to disable trajectory logging.

## Local vLLM

The default retrieval model is `Qwen/Qwen3.6-35B-A3B-FP8`. The default config expects two endpoints:

```text
http://127.0.0.1:8001/v1
http://127.0.0.1:8002/v1
```

For a different GPU count, set `retrieval.local_model_count`, `retrieval.local_parallel_agents`, and `retrieval.local_base_urls` consistently. The Python app validates endpoint count, round-robins calls, and applies vLLM failover according to `http.retries`; it does not launch or supervise vLLM processes.

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
