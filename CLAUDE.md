# Graduate School Research Agent

An agentic system built on the Anthropic Python SDK that autonomously researches graduate school programs and produces structured Markdown profiles. See `DESIGN.md` for the original architecture specification.

## Quick Start

```bash
# Install
pip install -e .

# Set API keys
export ANTHROPIC_API_KEY=sk-ant-...
export BRAVE_API_KEY=BSA...

# Place your CV, school list, and optional context in input/
cp input/schools.example.json input/schools.json  # edit with your schools
cp input/context.example.md input/context.md      # edit with your focus areas
# Create input/cv.md with your CV

# Run
grad-agent --schools input/schools.json --cv input/cv.md
# context.md is loaded automatically if present; pass --context to use a different path
```

## Architecture

Three-stage pipeline per school, each school fully independent:

```
retrieval (Haiku) → judge (Sonnet) + fit (Sonnet) in parallel → markdown output
                     ↓ (if "insufficient")
                  gap-fill (Haiku) → re-judge + re-fit
```

### Package Layout

```
grad_agent/
├── cli.py              # Argparse CLI entry point (grad-agent command)
├── config.py           # YAML + env-based Config dataclass
├── models.py           # Shared Pydantic models
├── pipeline/
│   ├── retrieval.py    # Stage 1 — Haiku iterative retrieval agent
│   ├── judge.py        # Stage 2 — Sonnet quality/coverage assessment
│   ├── fit.py          # Stage 3 — Sonnet CV-aware fit scoring
│   ├── runner.py       # Per-school orchestration + multi-school launcher
│   ├── prompts.py      # All system/user prompts for the three stages
│   └── tools.py        # Tool definitions + handlers (web_search, fetch_page)
├── reporting/
│   ├── markdown.py     # Renders Markdown reports and summary table
│   └── stats.py        # Token/cost/timing statistics collection
└── util/
    └── log.py          # Structured logging with per-school context
```

### Data Flow

1. **Input**: School list (`input/schools.json`) + applicant CV (`input/cv.md`) + optional context (`input/context.md`)
2. **Stage 1** (`pipeline/retrieval.py`): Haiku model iteratively calls `web_search` and `fetch_page` tools to populate a `SchoolProfile`. Runs up to `max_turns` (default 25). Context is injected into the initial user message to focus the search.
3. **Stage 2** (`pipeline/judge.py`): Sonnet evaluates profile quality → `JudgeReport` with pass/partial/insufficient rating and flagged fields. Context is injected to prioritise gaps relevant to the applicant's subfield.
4. **Stage 3** (`pipeline/fit.py`): Sonnet cross-references CV against profile → `FitAssessment` with 0.0–1.0 score. Context supplements the CV with goals and constraints not visible in the CV itself.
5. **Gap-fill** (optional): If judge rates "insufficient" and `retry_gap_fill: true`, re-runs targeted retrieval using judge's suggested queries, then re-evaluates (context passed through here as well).
6. **Output**: Per-school Markdown file + `summary.md` with priority-ranked table in `output/`.

Stages 2 and 3 run concurrently via `asyncio.gather`. Schools run sequentially to avoid rate limits.

## Configuration

Non-secret settings live in `config.yaml` (YAML). API keys are environment-only (`.env`).

### config.yaml

```yaml
models:
  haiku: claude-haiku-4-5-20251001
  sonnet: claude-sonnet-4-6

retrieval:
  max_turns: 15
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
```

### Environment Variables (secrets only)

```
ANTHROPIC_API_KEY=sk-ant-...
BRAVE_API_KEY=BSA...
```

### CLI Overrides

`--max-turns`, `--max-parallel`, `--no-gap-fill`, `--output`, `--config` override the corresponding YAML settings for a single run.

`--context` overrides the default `input/context.md` path; errors if the specified file does not exist.

## Input / Output

**Input directory** (`input/`, gitignored except examples):
- `schools.json` — `[{"school": "...", "program": "..."}, ...]`
- `cv.md` — applicant's CV in plain text or Markdown
- `context.md` *(optional)* — free-form applicant context injected into every pipeline stage; use it to specify target subfields, advisor preferences, funding requirements, geographic constraints, or scoring guidance. See `context.example.md` for a template. Silently skipped if absent.

**Output directory** (`output/`, gitignored):
- `{school_name}_{program}_profile.md` — full report per school
- `summary.md` — priority-ranked table across all schools

Each profile contains: header, requirements, research & faculty, essay prompts, applicant landscape, fit summary, quality flags, and sources.

## Design Deviations

Decisions resolved from `DESIGN.md` open questions:

- **Turn budget**: 25, configurable via `retrieval.max_turns`
- **Gap-fill on insufficient**: Enabled by default, configurable via `judge.retry_gap_fill`
- **Search API**: Brave Search (cost-effective, good coverage, simple API)
- **Output format**: Both per-school files and a consolidated summary table

## Dependencies

- `anthropic` — Claude API client
- `pydantic` — structured data validation
- `httpx` — async HTTP for tool execution (Brave Search + page fetches)
- `pyyaml` — YAML config loading
