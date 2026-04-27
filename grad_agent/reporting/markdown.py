"""Markdown output generation.

Merges SchoolProfile + JudgeReport + FitAssessment into a single
human-readable Markdown document per the DESIGN.md specification.
"""

from __future__ import annotations

from grad_agent.models import FitAssessment, JudgeReport, SchoolProfile


def render_school_markdown(
    profile: SchoolProfile,
    judge: JudgeReport | None = None,
    fit: FitAssessment | None = None,
) -> str:
    """Render a complete Markdown report for one school."""
    sections: list[str] = []

    # --- Header ---
    deadline_str = profile.deadline or "*not found*"
    if judge and any(f.field == "deadline" for f in judge.flagged_fields):
        deadline_str += " [unverified]"

    sections.append(
        f"# {profile.school_name} — {profile.program_name}\n\n"
        f"**Deadline**: {deadline_str}  \n"
        f"**Application fee**: {profile.application_fee or '*not found*'}\n"
    )

    # --- Requirements ---
    req = profile.requirements
    req_lines = ["## Requirements\n"]
    req_lines.append(f"- **GRE required**: {_yn(req.gre_required)}")
    if req.gpa_minimum:
        req_lines.append(f"- **GPA minimum**: {req.gpa_minimum}")
    req_lines.append(f"- **Statement of Purpose**: {_yn(req.statement_of_purpose)}")
    if req.recommendations is not None:
        req_lines.append(f"- **Recommendations**: {req.recommendations}")
    for other in req.other:
        req_lines.append(f"- {other}")
    sections.append("\n".join(req_lines) + "\n")

    # --- Research & Faculty ---
    rf_lines = ["## Research & Faculty\n"]
    if profile.research_areas:
        rf_lines.append("### Research Areas\n")
        for area in profile.research_areas:
            rf_lines.append(f"- {area}")
    if profile.advisor_candidates:
        rf_lines.append("\n### Advisor Candidates\n")
        for advisor in profile.advisor_candidates:
            rf_lines.append(f"- {advisor}")
    sections.append("\n".join(rf_lines) + "\n")

    # --- Essay Prompts ---
    if profile.essay_prompts:
        ep_lines = ["## Essay Prompts\n"]
        for i, prompt in enumerate(profile.essay_prompts, 1):
            ep_lines.append(f"{i}. {prompt}")
        sections.append("\n".join(ep_lines) + "\n")

    # --- Applicant Landscape ---
    ar = profile.applicant_reports
    if ar.typical_gpa or ar.typical_gre or ar.acceptance_signals:
        al_lines = ["## Applicant Landscape\n"]
        if ar.typical_gpa:
            al_lines.append(f"- **Typical GPA**: {ar.typical_gpa}")
        if ar.typical_gre:
            al_lines.append(f"- **Typical GRE**: {ar.typical_gre}")
        if ar.acceptance_signals:
            al_lines.append(f"\n{ar.acceptance_signals}")
        sections.append("\n".join(al_lines) + "\n")

    # --- Fit Summary ---
    if fit:
        fit_lines = ["## Fit Summary\n"]
        fit_lines.append(f"- **Overall score**: {fit.overall_score:.2f} / 1.00")
        fit_lines.append(f"- **Confidence**: {fit.confidence.value}")
        fit_lines.append(f"\n### Research Alignment\n\n{fit.research_alignment}")
        if fit.advisor_candidates:
            fit_lines.append("\n### Advisor Matches\n")
            for a in fit.advisor_candidates:
                fit_lines.append(f"- {a}")
        fit_lines.append(f"\n### Competitiveness\n\n{fit.competitiveness}")
        fit_lines.append(f"\n### Gaps\n\n{fit.gaps}")
        sections.append("\n".join(fit_lines) + "\n")

    # --- Quality Flags ---
    if judge:
        qf_lines = [f"## Quality Assessment ({judge.overall_quality.value})\n"]
        if judge.flagged_fields:
            for flag in judge.flagged_fields:
                qf_lines.append(f"- **{flag.field}**: {flag.reason}")
        if judge.notes:
            qf_lines.append(f"\n{judge.notes}")
        sections.append("\n".join(qf_lines) + "\n")

    # --- Sources ---
    if profile.sources:
        src_lines = ["## Sources\n"]
        for i, url in enumerate(profile.sources, 1):
            src_lines.append(f"{i}. {url}")
        sections.append("\n".join(src_lines) + "\n")

    # --- Notes ---
    if profile.notes:
        sections.append(f"## Notes\n\n{profile.notes}\n")

    return "\n---\n\n".join(sections)


def render_summary_table(
    results: list[tuple[SchoolProfile, FitAssessment | None]],
) -> str:
    """Render a priority-ranked summary table across all schools."""
    # Sort by fit score descending, schools without fit last
    ranked = sorted(
        results,
        key=lambda r: r[1].overall_score if r[1] else -1.0,
        reverse=True,
    )

    lines = [
        "# Graduate School Research — Summary\n",
        "| Rank | School | Program | Fit Score | Confidence | Deadline |",
        "|------|--------|---------|-----------|------------|----------|",
    ]
    for i, (profile, fit) in enumerate(ranked, 1):
        score = f"{fit.overall_score:.2f}" if fit else "N/A"
        conf = fit.confidence.value if fit else "N/A"
        deadline = profile.deadline or "N/A"
        lines.append(
            f"| {i} | {_md_cell(profile.school_name)} | {_md_cell(profile.program_name)} | "
            f"{score} | {conf} | {_md_cell(deadline)} |"
        )

    return "\n".join(lines) + "\n"


def _yn(val: str | bool | None) -> str:
    if val is None:
        return "*not found*"
    if isinstance(val, str):
        return val
    return "Yes" if val else "No"


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
