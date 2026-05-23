"""PDF rendering for generated Markdown reports."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class ReportDirs:
    root: Path
    markdown_dir: Path
    pdf_dir: Path


PDF_CSS = """
@page {
  size: Letter;
  margin: 0.72in 0.68in;
  @bottom-right {
    color: #6b7280;
    content: counter(page);
    font-size: 9pt;
  }
}

body {
  color: #202733;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
}

h1 {
  color: #162033;
  font-size: 22pt;
  font-weight: 700;
  line-height: 1.15;
  margin: 0 0 0.28in;
}

h2 {
  border-bottom: 1px solid #d5dbe5;
  color: #27364d;
  font-size: 15pt;
  margin: 0.3in 0 0.12in;
  padding-bottom: 0.04in;
}

h3 {
  color: #3c4a5f;
  font-size: 11.5pt;
  margin: 0.18in 0 0.08in;
}

p {
  margin: 0 0 0.1in;
}

ul,
ol {
  margin: 0.04in 0 0.12in 0.22in;
  padding-left: 0.15in;
}

li {
  margin: 0.025in 0;
}

hr {
  border: 0;
  border-top: 1px solid #e2e8f0;
  margin: 0.24in 0;
}

table {
  border-collapse: collapse;
  font-size: 9.2pt;
  margin: 0.1in 0 0.2in;
  width: 100%;
}

th,
td {
  border: 1px solid #d7dde7;
  overflow-wrap: break-word;
  padding: 0.055in 0.07in;
  text-align: left;
  vertical-align: top;
}

th {
  background: #edf2f7;
  color: #1f2a3d;
  font-weight: 700;
}

tr:nth-child(even) td {
  background: #f8fafc;
}

th:nth-child(1),
td:nth-child(1),
th:nth-child(4),
td:nth-child(4),
th:nth-child(5),
td:nth-child(5) {
  white-space: nowrap;
}

a {
  color: #0b5cad;
  overflow-wrap: anywhere;
  text-decoration: none;
}

code {
  background: #f1f5f9;
  border-radius: 3px;
  color: #334155;
  font-family: "Courier New", monospace;
  font-size: 9.4pt;
  padding: 0.01in 0.035in;
}

strong {
  color: #182235;
}

body,
p,
li {
  overflow-wrap: break-word;
}
"""


def report_dirs(output_root: Path) -> ReportDirs:
    root = Path(output_root)
    return ReportDirs(
        root=root,
        markdown_dir=root / "markdown",
        pdf_dir=root / "pdf",
    )


def ensure_report_dirs(output_root: Path) -> ReportDirs:
    dirs = report_dirs(output_root)
    dirs.markdown_dir.mkdir(parents=True, exist_ok=True)
    dirs.pdf_dir.mkdir(parents=True, exist_ok=True)
    return dirs


def write_markdown_report(markdown_path: Path, markdown_text: str, dirs: ReportDirs) -> Path:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return render_markdown_pdf(markdown_path, pdf_path_for_markdown(markdown_path, dirs))


def render_markdown_tree(markdown_dir: Path, pdf_dir: Path) -> list[Path]:
    dirs = ReportDirs(root=markdown_dir.parent, markdown_dir=markdown_dir, pdf_dir=pdf_dir)
    return [render_markdown_pdf(path, pdf_path_for_markdown(path, dirs))
            for path in sorted(markdown_dir.rglob("*.md"))]


def render_markdown_pdf(markdown_path: Path, pdf_path: Path) -> Path:
    try:
        import markdown as markdown_lib
        from weasyprint import CSS, HTML
    except ImportError as exc:
        raise RuntimeError(
            "PDF reporting requires the markdown and weasyprint packages. "
            "Install the project dependencies before generating reports."
        ) from exc

    markdown_text = markdown_path.read_text(encoding="utf-8")
    body = markdown_lib.markdown(
        markdown_text,
        extensions=["extra", "sane_lists", "toc"],
        output_format="html5",
    )
    title = _document_title(markdown_text)
    html = (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>{escape(title)}</title>\n"
        "</head>\n"
        f"<body>{body}</body>\n"
        "</html>\n"
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(markdown_path.parent.resolve())).write_pdf(
        pdf_path,
        stylesheets=[CSS(string=PDF_CSS)],
    )
    return pdf_path


def pdf_path_for_markdown(markdown_path: Path, dirs: ReportDirs) -> Path:
    try:
        relative = markdown_path.relative_to(dirs.markdown_dir)
    except ValueError:
        relative = Path(markdown_path.name)
    return (dirs.pdf_dir / relative).with_suffix(".pdf")


def _document_title(markdown_text: str) -> str:
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Graduate School Research Report"
