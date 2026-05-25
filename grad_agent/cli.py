"""CLI entry point for the graduate school research agent.

Usage:
    grad-agent --schools input/schools.json --cv input/cv.md
    grad-agent --school "MIT" --program "MS Computer Science" --cv input/cv.md
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from grad_agent.agents.judge.registry import judge_backend_ids
from grad_agent.agents.retrieval.registry import retrieval_backend_ids
from grad_agent.cli_support import config_overrides, load_schools, read_context, read_required_text
from grad_agent.config import Config
from grad_agent.events import EventCallback
from grad_agent.orchestration.runner import run_all_schools
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

    school_group = parser.add_mutually_exclusive_group()
    school_group.add_argument(
        "--schools",
        type=Path,
        default=None,
        help='Path to JSON file: [{"school": "...", "program": "..."}, ...] (overrides config)',
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
        default=None,
        help="Path to the applicant's CV (overrides config)",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=None,
        help=(
            "Path to applicant context file injected into every pipeline stage "
            "(overrides config; input/context.md is silently skipped if absent)"
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
        help="Output root for Markdown and PDF reports (overrides config.yaml)",
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
        "--retrieval-backend",
        choices=retrieval_backend_ids(),
        default=None,
        help="Override retrieval backend implementation",
    )
    parser.add_argument(
        "--judge-backend",
        choices=judge_backend_ids(),
        default=None,
        help="Override judge backend implementation",
    )
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="Disable automatic gap-fill when judge rates profile as insufficient",
    )

    return parser.parse_args()


def _load_schools(
    args: argparse.Namespace, schools_path: Path | None = None
) -> list[tuple[str, str]]:
    return load_schools(args, schools_path)


def main() -> None:
    args = _parse_args()
    setup_logging(verbose=args.verbose)
    log = logging.getLogger(__name__)

    config = Config.load(yaml_path=args.config, overrides=config_overrides(args))

    cv_path = Path(config.cv_path)
    cv_text = read_required_text(cv_path, "CV")

    context_path = Path(config.context_path)
    context_exists = context_path.exists()
    context_text = read_context(context_path)
    if context_exists:
        log.info("Loaded applicant context from %s", context_path)

    schools = _load_schools(args, Path(config.schools_path))
    log.info("Loaded %d school(s) to research", len(schools))

    errors = config.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    retrieval_topology = (
        f", local_models={config.local_retrieval_model_count}, "
        f"local_agents={config.local_retrieval_parallel_agents}"
        if config.uses_local_retrieval
        else ""
    )
    log.info(
        "Configuration: retrieval=%s/%s%s, judge=%s/%s, sonnet=%s, max_turns=%d, "
        "schools_parallel=%d, judge_fit_parallel=%d",
        config.retrieval_backend,
        config.retrieval_model,
        retrieval_topology,
        config.judge_backend,
        config.judge_model,
        config.sonnet_model,
        config.max_retrieval_turns,
        config.max_schools_parallel,
        config.max_sonnet_parallel,
    )

    tui = None
    on_event: EventCallback | None = None
    if sys.stderr.isatty() and not args.verbose:
        try:
            from grad_agent.tui import PipelineTUI
            tui = PipelineTUI(total=len(schools), config=config)
            tui.start()
            on_event = tui.on_event
        except ImportError:
            pass

    try:
        collector = asyncio.run(run_all_schools(schools, cv_text, config, context_text, on_event))
    finally:
        if tui is not None:
            tui.stop()

    print(collector.summary())

    if any(not s.success for s in collector.schools):
        sys.exit(1)


if __name__ == "__main__":
    main()
