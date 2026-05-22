# Graduate School Agent

An LLM-based research agent that gathers graduate program information, evaluates source quality, scores applicant fit against a CV, and writes Markdown reports.

The agent is intended for application planning: it searches official program pages, faculty pages, and informal applicant reports, then produces a structured profile for each school plus a ranked summary.

## Features

- Agentic retrieval using Claude Haiku or local Qwen through vLLM, with parallel local retrieval agents and batched `web_search` / `fetch_page` tools.
- Quality judging with Claude Sonnet to flag missing, stale, contradictory, or weakly sourced fields.
- CV-aware fit assessment with Claude Sonnet.
- Optional gap-fill pass when the judge rates a profile as insufficient.
- Markdown output for each school and a confidence-adjusted summary table.
- Optional JSONL trajectory logs with model responses and tool results.
- Rich terminal UI for interactive runs.

## Requirements

- Python 3.11 or newer
- Anthropic API key
- Brave Search API key
- One or more local OpenAI-compatible vLLM endpoints for default retrieval, or `--retrieval-backend anthropic_haiku`

Install the package in editable mode:

```bash
python3 -m pip install -e .
```

Set secrets in your shell or in a local `.env` file:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export BRAVE_API_KEY=BSA...
```

`config.yaml` stores non-secret runtime settings only.

## Quick Start

Create the inputs:

```bash
cp input/schools.example.json input/schools.json
$EDITOR input/schools.json
$EDITOR input/cv.md
```

Optional applicant context can be placed at `input/context.md`. It is injected into retrieval, judge, fit, and gap-fill prompts, and is useful for target subfields, advisor preferences, funding needs, geographic constraints, or scoring guidance.

Start the local retrieval servers if you are using the default config:

```bash
deploy/start-vllm.sh
```

In another shell, check the configured endpoints:

```bash
deploy/healthcheck.sh
```

Run the agent with local Qwen retrieval and Sonnet judge/fit:

```bash
grad-agent --schools input/schools.json --cv input/cv.md
```

Run a single school without a JSON file:

```bash
grad-agent --school "MIT" --program "PhD Electrical Engineering and Computer Science" --cv input/cv.md
```

To use Anthropic Haiku for retrieval instead of local Qwen:

```bash
grad-agent --schools input/schools.json --cv input/cv.md --retrieval-backend anthropic_haiku
```

## Input Format

`input/schools.json` must be a list of objects with `school` and `program` keys:

```json
[
  {
    "school": "Stanford University",
    "program": "MS Computer Science"
  },
  {
    "school": "MIT",
    "program": "PhD Electrical Engineering and Computer Science"
  }
]
```

`input/cv.md` is plain text or Markdown. The default context path is `input/context.md`; if that default file is absent, it is skipped. If you pass `--context some/path.md`, that file must exist.

## CLI Options

```bash
grad-agent --schools input/schools.json --cv input/cv.md
grad-agent --school "School Name" --program "Program Name" --cv input/cv.md
grad-agent --summary-from output
```

Useful options:

- `--config PATH`: load a different YAML config file.
- `--output DIR`: override the configured Markdown output directory.
- `--context PATH`: use an applicant context file.
- `--max-turns N`: override retrieval turn budget.
- `--max-parallel N`: override max concurrent school pipelines.
- `--retrieval-backend {anthropic_haiku,local_qwen_vllm}`: choose Claude Haiku retrieval or local vLLM Qwen retrieval.
- `--no-gap-fill`: disable targeted gap-fill on insufficient profiles.
- `--summary-from PATH`: rebuild `summary.md` from existing rendered profile Markdown without model calls.
- `--verbose`: bypass the Rich TUI and print debug logs to stderr.

## Output

By default, output is written to `output/`:

- `{school}_{program}_profile.md`: one report per school.
- `summary.md`: ranked table across all schools.

When `logs.dir` is non-empty in `config.yaml`, trajectory logs are written to `logs/{YYYY-MM-DDTHHMMSS}/`. Each school gets one JSONL file containing stage events, full model responses, tool results, final profiles, judge reports, and fit assessments.

Each profile includes:

1. Header with school, program, deadline, and fee.
2. Formal requirements.
3. Research areas and advisor candidates.
4. Essay prompts, when found.
5. Applicant landscape from informal reports, when found.
6. Fit summary.
7. Quality assessment and flags.
8. Sources and notes.

## Pipeline

Each school runs through this pipeline:

```text
retrieval (local Qwen by default) -> judge (Sonnet) + fit (Sonnet) -> Markdown output
                                  -> optional gap-fill (same retrieval backend) -> re-judge + re-fit
```

Stage details:

- `retrieval`: the configured retrieval backend searches the web through Brave and fetches pages with HTTPX, then emits a `SchoolProfile` JSON object.
- `local_qwen_vllm` retrieval calls the OpenAI-compatible vLLM chat completions API. By default, four local agents run per school against the configured endpoints, each focused on a different evidence slice, and their profiles are merged.
- Local retrieval can issue batched JSON tool commands in a single turn; the app executes those searches or fetches concurrently.
- `anthropic_haiku` retrieval uses Anthropic native tool calls with the same tool handlers. Multiple tool-use blocks in one model response are executed concurrently.
- `judge`: Claude Sonnet evaluates profile coverage, source quality, consistency, program match, cycle freshness, and actionability.
- `fit`: Claude Sonnet compares the profile against the applicant CV and optional context.
- `gap-fill`: if enabled, runs targeted retrieval using the judge's suggested queries when the initial profile is insufficient.

Judge and fit run concurrently for a school, including after gap-fill. Schools run with bounded concurrency from `max_schools_parallel`, and concurrent Sonnet calls are additionally bounded by `max_sonnet_parallel`.

## Configuration

Default `config.yaml`:

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
  local_model_count: 4
  local_parallel_agents: 4
  local_max_parallel_tool_calls: 8
  local_base_urls:
    - http://127.0.0.1:8001/v1
    - http://127.0.0.1:8002/v1
    - http://127.0.0.1:8003/v1
    - http://127.0.0.1:8004/v1
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
    - --gpu-memory-utilization
    - "0.95"
    - --enable-prefix-caching
    - --enable-chunked-prefill
    - --max-num-batched-tokens
    - "32768"
    - --max-num-seqs
    - "256"
    - --kv-cache-dtype
    - fp8_e5m2
  log_dir: logs/vllm
  micromamba_env: graduate-school-agent
  python_version: "3.11"
  system_packages:
    - curl
    - git
    - build-essential
  pip_packages:
    - vllm
```

Notes:

- `.env` is loaded automatically from the current working tree.
- `ANTHROPIC_API_KEY` is still required because judge and fit use Sonnet.
- `BRAVE_API_KEY` is required for retrieval web search in both backends.
- `retrieval.backend` accepts `local_qwen_vllm` or `anthropic_haiku`.
- `retrieval.local_model_count` is the number of local model copies the app expects. It must equal the number of `retrieval.local_base_urls` endpoints.
- `retrieval.local_parallel_agents` controls how many independent local retrieval agents run per school. The default is 4 for a 4 x A100 setup.
- `retrieval.local_max_parallel_tool_calls` caps the number of batched local tool commands executed concurrently from one model turn.
- Local vLLM retrieval uses the OpenAI-compatible `/chat/completions` API and round-robins across `retrieval.local_base_urls`.
- `concurrency.max_sonnet_parallel` caps concurrent Sonnet judge and fit calls separately from school pipeline concurrency.
- Deployment scripts read non-secret deployment settings from `config.yaml`.
- Set `VLLM_API_KEY` only if the vLLM servers require bearer-token authentication.
- `http.retries` is applied to local vLLM endpoint failover, but not currently applied by the fetch/search tool handlers.
- Set `logs.dir: ""` to disable trajectory logging.

## Local vLLM Deployment

The default retrieval model is `Qwen/Qwen3.6-35B-A3B-FP8`. The local topology is `retrieval.local_model_count` independent vLLM servers, one per GPU. The default topology targets a 4 x A100 instance with four local model copies:

```text
GPU 0 -> http://127.0.0.1:8001/v1
GPU 1 -> http://127.0.0.1:8002/v1
GPU 2 -> http://127.0.0.1:8003/v1
GPU 3 -> http://127.0.0.1:8004/v1
```

For fewer GPUs, set `retrieval.local_model_count` to the available GPU count, provide the same number of endpoints, and reduce `retrieval.local_parallel_agents` to match. The deployment scripts read the model count and endpoints directly. For two GPUs:

```yaml
retrieval:
  local_model_count: 2
  local_parallel_agents: 2
  local_base_urls:
    - http://127.0.0.1:8001/v1
    - http://127.0.0.1:8002/v1
```

The app does not launch or supervise vLLM processes. It validates that `retrieval.local_model_count` matches `retrieval.local_base_urls`, then round-robins retrieval calls across those endpoints and fails over according to `http.retries`. With local retrieval, a single school can consume multiple endpoints concurrently because the retrieval fanout assumes local batching capacity and no provider rate limit.

On a fresh GPU node, install system packages, micromamba, the agent package, and vLLM with:

```bash
deploy/setup-node.sh
```

Then start the configured vLLM servers:

```bash
deploy/start-vllm.sh
```

Launch settings such as host, vLLM args, log directory, environment name, and setup packages live under `deploy` in `config.yaml`. See `deploy/README.md` for the deployment-specific flow.

## Development

Run the test suite:

```bash
python3 -m unittest
```

Preview the Rich TUI with fake data and no API calls:

```bash
python3 -m tests.preview_tui
```

The TUI header shows retrieval backend, local vLLM model/endpoint topology,
local agent fanout, tool-call fanout, and school/Sonnet concurrency. Each
school row shows the active stage, per-worker retrieval turns for parallel
local runs, largest observed tool batch, elapsed time, and final cost.

For a static text snapshot instead of a live preview:

```bash
python3 -m tests.preview_tui --snapshot
```

The package layout is:

```text
grad_agent/
  cli.py                 argparse CLI entry point
  config.py              YAML, .env, and environment config loading
  events.py              pipeline events consumed by the TUI
  llm/
    vllm.py              OpenAI-compatible local vLLM client
  models.py              Pydantic schemas
  tui.py                 Rich live terminal UI
  pipeline/
    retrieval.py         Anthropic or local-vLLM retrieval agent loop
    judge.py             Sonnet quality judge
    fit.py               Sonnet fit assessor
    runner.py            per-school orchestration
    prompts.py           system and user prompts
    tools.py             Brave search and page fetch tools
  reporting/
    markdown.py          Markdown rendering
    stats.py             token, cost, and timing stats
    trajectory.py        JSONL trajectory logging
  util/
    json.py              model JSON extraction helpers
    log.py               structured logging
    retry.py             Anthropic rate-limit retry helper
```
