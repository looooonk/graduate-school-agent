# Graduate School Agent

An Anthropic SDK based research agent that gathers graduate program information, evaluates source quality, scores applicant fit against a CV, and writes Markdown reports.

The agent is intended for application planning: it searches official program pages, faculty pages, and informal applicant reports, then produces a structured profile for each school plus a ranked summary.

## Features

- Agentic retrieval loop using Claude Haiku with `web_search` and `fetch_page` tools.
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

Run the agent:

```bash
grad-agent --schools input/schools.json --cv input/cv.md
```

Run a single school without a JSON file:

```bash
grad-agent --school "MIT" --program "PhD Electrical Engineering and Computer Science" --cv input/cv.md
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
```

Useful options:

- `--config PATH`: load a different YAML config file.
- `--output DIR`: override the configured Markdown output directory.
- `--context PATH`: use an applicant context file.
- `--max-turns N`: override retrieval turn budget.
- `--max-parallel N`: parsed into config, but current execution is sequential.
- `--no-gap-fill`: disable targeted gap-fill on insufficient profiles.
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
retrieval (Haiku) -> judge (Sonnet) + fit (Sonnet) -> Markdown output
                    -> optional gap-fill (Haiku) -> re-judge + re-fit
```

Stage details:

- `retrieval`: Claude Haiku searches the web through Brave and fetches pages with HTTPX, then emits a `SchoolProfile` JSON object.
- `judge`: Claude Sonnet evaluates profile coverage, source quality, consistency, program match, cycle freshness, and actionability.
- `fit`: Claude Sonnet compares the profile against the applicant CV and optional context.
- `gap-fill`: if enabled, runs targeted retrieval using the judge's suggested queries when the initial profile is insufficient.

Judge and fit run concurrently for a school. Schools themselves currently run sequentially in `run_all_schools`, despite the `max_schools_parallel` configuration field.

## Configuration

Default `config.yaml`:

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

Notes:

- `.env` is loaded automatically from the current working tree.
- Missing API keys fail validation before the pipeline starts.
- `http.retries` is validated but not currently applied by the fetch/search tool handlers.
- Set `logs.dir: ""` to disable trajectory logging.

## Development

Run the test suite:

```bash
python3 -m unittest
```

Preview the Rich TUI with fake data and no API calls:

```bash
python3 -m tests.preview_tui
```

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
  models.py              Pydantic schemas
  tui.py                 Rich live terminal UI
  pipeline/
    retrieval.py         Haiku retrieval agent loop
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
