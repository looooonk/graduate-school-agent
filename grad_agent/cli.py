"""CLI entry point for the graduate school research agent.

Usage:
    grad-agent --schools input/schools.json --cv input/cv.md
    grad-agent --school "MIT" --program "MS Computer Science" --cv input/cv.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from grad_agent.config import Config
from grad_agent.events import EventCallback
from grad_agent.pipeline.runner import run_all_schools
from grad_agent.util.log import setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research graduate school programs using AI agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  grad-agent --schools input/schools.json --cv input/cv.md\n"
            '  grad-agent --school "MIT" --program "MS CS" --cv input/cv.md\n'
        ),
    )

    # School specification — either a JSON file or inline
    school_group = parser.add_mutually_exclusive_group(required=True)
    school_group.add_argument(
        "--schools",
        type=Path,
        help='Path to JSON file: [{"school": "...", "program": "..."}, ...]',
    )
    school_group.add_argument(
        "--school",
        type=str,
        help="Single school name (use with --program)",
    )

    parser.add_argument(
        "--program",
        type=str,
        help="Program name (required when --school is used)",
    )
    parser.add_argument(
        "--cv",
        type=Path,
        default=Path("input/cv.md"),
        help="Path to the applicant's CV (default: input/cv.md)",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path("input/context.md"),
        help=(
            "Path to applicant context file injected into every pipeline stage "
            "(default: input/context.md; silently skipped if absent)"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for Markdown reports (overrides config.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max retrieval turns",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Override max parallel schools",
    )
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="Disable automatic gap-fill when judge rates profile as insufficient",
    )

    return parser.parse_args()


def _load_schools(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Parse school specifications from CLI args."""
    if args.schools:
        path: Path = args.schools
        if not path.exists():
            print(f"Error: schools file not found: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Error: schools file is not valid JSON: {exc}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(data, list):
            print("Error: schools JSON must be a list of objects", file=sys.stderr)
            sys.exit(1)
        schools = []
        for entry in data:
            if not isinstance(entry, dict):
                print(
                    f"Error: each entry in schools JSON must be an object. Got: {entry}",
                    file=sys.stderr,
                )
                sys.exit(1)
            school = entry.get("school")
            program = entry.get("program")
            if not school or not program:
                print(
                    f"Error: each entry in schools JSON must have 'school' and 'program' keys. "
                    f"Got: {entry}",
                    file=sys.stderr,
                )
                sys.exit(1)
            schools.append((school, program))
        return schools

    # Single school mode
    if not args.program:
        print("Error: --program is required when using --school", file=sys.stderr)
        sys.exit(1)
    return [(args.school, args.program)]


def main() -> None:
    args = _parse_args()
    setup_logging(verbose=args.verbose)
    log = logging.getLogger(__name__)

    # Load CV
    cv_path: Path = args.cv
    if not cv_path.exists():
        print(f"Error: CV file not found: {cv_path}", file=sys.stderr)
        sys.exit(1)
    cv_text = cv_path.read_text(encoding="utf-8")

    # Load optional applicant context
    context_path: Path = args.context
    if context_path.exists():
        context_text = context_path.read_text(encoding="utf-8")
        log.info("Loaded applicant context from %s", context_path)
    elif args.context != Path("input/context.md"):
        # User explicitly specified a path that doesn't exist — treat as an error
        print(f"Error: context file not found: {context_path}", file=sys.stderr)
        sys.exit(1)
    else:
        context_text = ""

    # Load school list
    schools = _load_schools(args)
    log.info("Loaded %d school(s) to research", len(schools))

    # Build config from YAML + env, with CLI overrides
    overrides: dict[str, object] = {}
    if args.max_turns is not None:
        overrides["max_retrieval_turns"] = args.max_turns
    if args.max_parallel is not None:
        overrides["max_schools_parallel"] = args.max_parallel
    if args.no_gap_fill:
        overrides["retry_gap_fill"] = False
    if args.output is not None:
        overrides["output_dir"] = str(args.output)

    config = Config.load(yaml_path=args.config, overrides=overrides)
    errors = config.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    log.info(
        "Configuration: haiku=%s, sonnet=%s, max_turns=%d, parallel=%d",
        config.haiku_model, config.sonnet_model,
        config.max_retrieval_turns, config.max_schools_parallel,
    )

    # Start TUI when running interactively; fall back to plain logging otherwise.
    # --verbose implies plain logging so debug output remains readable.
    tui = None
    on_event: EventCallback | None = None
    if sys.stderr.isatty() and not args.verbose:
        try:
            from grad_agent.tui import PipelineTUI
            tui = PipelineTUI(total=len(schools))
            tui.start()
            on_event = tui.on_event
        except ImportError:
            pass  # rich not installed — plain logging already set up above

    # Run pipeline
    try:
        collector = asyncio.run(run_all_schools(schools, cv_text, config, context_text, on_event))
    finally:
        if tui is not None:
            tui.stop()

    # Print summary
    print(collector.summary())

    # Exit with error code if any school failed
    if any(not s.success for s in collector.schools):
        sys.exit(1)


if __name__ == "__main__":
    main()
