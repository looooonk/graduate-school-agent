# Graduate School Research Agent — Design Document

## Overview

An agentic system that autonomously researches graduate school programs and produces
a structured profile document for each school. Each school is treated as a fully
independent, self-contained job. There is no cross-school orchestrator; the outer
runner is a simple launcher that dispatches one agent per school.

The system is composed of three distinct stages per school:

1. **Haiku Retrieval Agent** — iterative web research and document synthesis
2. **Sonnet Judge** — single-pass quality and coverage assessment
3. **Sonnet Fit Assessor** — CV-aware fit scoring against the school profile

---

## Stage 1: Haiku Retrieval Agent

### Responsibility

Produce a complete `SchoolProfile` document for a single target school by
iteratively querying the web, fetching relevant pages, and extracting structured
information until all required fields are adequately populated.

### Inputs

- School name and program name (e.g. "Columbia University — MS Computer Science")
- A fixed schema defining which fields must be filled (see Output Schema below)

### Behavior

The agent runs a loop:

1. Issue a search query targeting a specific field gap or source type
2. Optionally fetch full page content from returned URLs
3. Extract relevant information and attempt to fill schema fields
4. Identify remaining gaps and issue further queries until the schema is
   sufficiently complete or a maximum turn budget is reached

Source types the agent should prioritize, in rough order:

- Official program page (deadlines, requirements, tuition, application portal)
- Faculty and lab pages (research areas, advisor candidates)
- GradCafe and Reddit threads (applicant experiences, informal stats)
- Departmental news or event pages (recent highlights, culture signals)
- Past or sample essay prompts (from program pages or applicant blogs)

Each extracted field should be stored alongside its source URL to support
manual verification.

### Output Schema

```
SchoolProfile:
  school_name:          str
  program_name:         str
  deadline:             date
  application_fee:      str
  requirements:
    gre_required:       bool
    gpa_minimum:        str (if stated)
    statement_of_purpose: bool
    recommendations:    int
    other:              list[str]
  essay_prompts:        list[str]
  research_areas:       list[str]
  advisor_candidates:   list[str]  # names + brief note on research fit
  applicant_reports:
    typical_gpa:        str
    typical_gre:        str (if applicable)
    acceptance_signals: str  # qualitative summary from GradCafe/Reddit
  sources:              list[str]  # URL per field where possible
  notes:                str        # anything notable that doesn't fit above
```

### Output Format

A populated `SchoolProfile` instance serialized to JSON, passed directly to
Stage 2 and Stage 3.

---

## Stage 2: Sonnet Judge

### Responsibility

Perform a single-pass quality and coverage assessment of the `SchoolProfile`
produced by Stage 1. The judge does not re-do retrieval; it evaluates what
was found.

### Inputs

- The full `SchoolProfile` JSON from Stage 1
- A judge prompt specifying evaluation criteria

### Evaluation Criteria

- **Coverage**: Are all required fields populated? Which are missing or thin?
- **Source quality**: Does any field rely on a single anecdotal source where
  multiple corroborating sources would be expected?
- **Consistency**: Are there contradictions across sources within the same field?
- **Confidence flags**: Fields that the judge considers unverified or low-confidence
  should be explicitly flagged for manual review (deadlines in particular)

### Output Schema

```
JudgeReport:
  overall_quality:    "pass" | "partial" | "insufficient"
  flagged_fields:     list[{ field: str, reason: str }]
  suggested_queries:  list[str]  # optional re-queries for Haiku if gaps are critical
  notes:              str
```

The `suggested_queries` field enables an optional second Haiku loop for targeted
gap-filling, if the judge deems the gaps significant enough to warrant it. Whether
to trigger this loop is a configurable policy decision, not automatic.

---

## Stage 3: Sonnet Fit Assessor

### Responsibility

Cross-reference the applicant's CV against the `SchoolProfile` and produce a
structured fit assessment with justification. This stage runs independently of
Stage 2 and can execute in parallel with it.

### Inputs

- The applicant's CV (provided once, reused across all schools)
- The full `SchoolProfile` JSON from Stage 1

### Assessment Dimensions

- **Research alignment**: How well do the applicant's research areas and projects
  map to the program's stated focus and available advisors?
- **Advisor fit**: Are there specific named faculty whose work overlaps with the
  applicant's background?
- **Profile competitiveness**: How does the applicant's stats and background
  compare to informal applicant reports in the profile?
- **Gaps**: Where is the applicant's profile weak relative to this program's
  apparent expectations or culture?

### Output Schema

```
FitAssessment:
  overall_score:          float  # 0.0–1.0
  research_alignment:     str    # qualitative justification
  advisor_candidates:     list[str]  # ranked by fit
  competitiveness:        str    # qualitative relative to applicant reports
  gaps:                   str
  confidence:             "high" | "medium" | "low"  # based on profile completeness
```

`confidence` should be set to "low" if the `SchoolProfile` has significant gaps
flagged by the judge, so downstream prioritization accounts for data quality.

---

## Final Output Per School

After all three stages complete, the outputs are merged into a single
human-readable Markdown document:

```
{school_name}_profile.md
```

Structure:

1. **Header**: School name, program, deadline (flagged if unverified)
2. **Requirements**: All formal requirements in a compact list
3. **Research & Faculty**: Research areas and advisor candidates
4. **Essay Prompts**: Verbatim if retrieved, otherwise notes on expected format
5. **Applicant Landscape**: Summary of GradCafe/Reddit signals
6. **Fit Summary**: Score, justification, advisor matches, gaps
7. **Quality Flags**: Any fields the judge flagged for manual review
8. **Sources**: Full list of URLs used

---

## Execution Model

```
for school in target_schools:
    profile  = haiku_retrieval_agent(school)
    judge    = sonnet_judge(profile)          # parallel with fit if desired
    fit      = sonnet_fit_assessor(cv, profile)
    write_markdown(profile, judge, fit)
```

Each school is fully independent. Parallelism across schools is possible but
should be rate-limited against shared search APIs.

---

## Cost Profile (approximate, 20 schools)

| Stage              | Model  | Est. cost per school | Est. total (20 schools) |
|--------------------|--------|----------------------|--------------------------|
| Haiku retrieval    | Haiku  | ~$0.07               | ~$1.40                   |
| Sonnet judge       | Sonnet | ~$0.03               | ~$0.60                   |
| Sonnet fit         | Sonnet | ~$0.04               | ~$0.80                   |
| **Total**          |        | **~$0.14**           | **~$2.80**               |

Estimates assume ~50K input tokens and ~4K output tokens per Haiku run,
and ~6–10K input / ~1K output per Sonnet call. Costs scale with page fetch
aggressiveness.

---

## Open Decisions for Implementation

- Maximum turn budget for Haiku retrieval loop (suggested: 15 turns)
- Whether to trigger a second Haiku loop when the judge flags critical gaps
- Which search API to use (Brave, Serper, Tavily, etc.)
- Whether `FitAssessment.confidence` should gate the school's priority ranking
- Output format preference: one Markdown file per school, or a consolidated
  multi-school summary with a priority-ranked table at the top
