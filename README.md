# Graduate School Agent

Graduate School Agent researches graduate programs, evaluates the quality of the
information it found, compares each program against an applicant CV, and writes
Markdown and PDF reports.

![Graduate School Agent TUI demo](resources/demo.gif)

## What You Get

- Program profiles with admissions, funding, faculty, requirements, and source notes.
- Applicant-fit assessments based on a CV and optional context.
- A ranked summary across all requested schools.
- Markdown and PDF reports under `output/`.
- A terminal UI for monitoring long runs.

## Requirements

- Python 3.11+
- Anthropic API key
- Brave Search API key
- WeasyPrint native libraries for PDF generation
- Local OpenAI-compatible retrieval endpoints, unless you choose an API retrieval backend

Install the package:

```bash
python3 -m pip install -e .
```

Set secrets in your shell or in a local `.env` file:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export BRAVE_API_KEY=BSA...
```

If you use an OpenAI-compatible API backend, also set one of:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_COMPATIBLE_API_KEY=...
```

Do not put API keys in `config.yaml`.

## Quick Start

Create your inputs:

```bash
cp input/schools.example.json input/schools.json
$EDITOR input/schools.json
$EDITOR input/cv.md
```

`input/schools.json` is a list of school/program pairs:

```json
[
  {
    "school": "Stanford University",
    "program": "MS Computer Science"
  }
]
```

`input/cv.md` can be plain text or Markdown. You can also create
`input/context.md` for preferences such as research interests, advisor
preferences, location constraints, funding needs, or scoring guidance. The
default `input/context.md` file is optional and is skipped when missing.

Start local retrieval servers if you are using the default backend:

```bash
deploy/start-vllm.sh
deploy/healthcheck.sh
```

Then run:

```bash
grad-agent
```

For a one-off school without editing `input/schools.json`:

```bash
grad-agent --school "MIT" --program "PhD Electrical Engineering and Computer Science" --cv input/cv.md
```

## Choosing a Retrieval Backend

The default backend expects local OpenAI-compatible endpoints configured in
`config.yaml`. See `deploy/README.md` for the local vLLM workflow.

To run without local retrieval servers, choose an API backend:

```bash
grad-agent --retrieval-backend anthropic_haiku
grad-agent --retrieval-backend anthropic_sonnet
grad-agent --retrieval-backend openai_compatible
```

For `openai_compatible`, set the provider endpoint and model in `config.yaml`
before running.

## Common Options

```bash
grad-agent --schools input/schools.json --cv input/cv.md
grad-agent --config custom-config.yaml
grad-agent --output output/my-run
grad-agent --max-parallel 4
grad-agent --max-turns 15
grad-agent --no-gap-fill
grad-agent --verbose
```

Useful flags:

- `--schools PATH`, `--cv PATH`, `--context PATH`: override input files.
- `--school NAME`, `--program NAME`: run one school/program pair.
- `--retrieval-backend BACKEND`: choose the retrieval model path.
- `--judge-backend BACKEND`: choose the profile-quality judge.
- `--max-turns N`: limit retrieval turns per school.
- `--max-parallel N`: limit concurrent school runs.
- `--no-gap-fill`: skip the extra pass for insufficient profiles.
- `--verbose`: disable the terminal UI and print debug logs.

## Outputs

Reports are written under `output/` by default:

```text
output/markdown/{school}_{program}_profile.md
output/markdown/summary.md
output/pdf/{school}_{program}_profile.pdf
output/pdf/summary.pdf
```

When `logs.dir` is set, run logs are written under `logs/`. These logs can
include full model responses, fetched page content, and applicant details, so
treat them as sensitive local files.

## Configuration

`config.yaml` holds non-secret defaults for inputs, outputs, model names,
backend selection, retrieval budgets, and concurrency. CLI flags override the
configured values for a run.

Common settings:

- `input.cv`, `input.context`, `input.schools`: default input paths.
- `output.dir`: report output directory.
- `retrieval.backend`: retrieval backend id.
- `judge.backend`: judge backend id.
- `retrieval.max_turns`: retrieval budget per school.
- `concurrency.max_schools_parallel`: concurrent school pipelines.
- `logs.dir`: set to `""` to disable trajectory logs.

## Development Checks

For routine validation:

```bash
micromamba run -n graduate-school-agent python3 -m unittest
micromamba run -n graduate-school-agent ruff check .
```

The full `grad-agent` command can spend API tokens, so use tests and the TUI
preview for routine development checks:

```bash
python3 -m tests.preview_tui
```
