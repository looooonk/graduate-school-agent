"""Structured logging setup with per-school context propagation."""

from __future__ import annotations

import logging
import sys
from typing import Any

# ANSI escape helpers
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BRED   = "\033[1;31m"  # bold red

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG:    _DIM,
    logging.INFO:     _GREEN,
    logging.WARNING:  _YELLOW,
    logging.ERROR:    _RED,
    logging.CRITICAL: _BRED,
}

class ColorFormatter(logging.Formatter):
    """Formatter that applies ANSI colors when writing to a TTY.

    Falls back to plain text when stdout/stderr is redirected, so piped or
    captured output stays machine-readable.
    """

    def __init__(self, *, use_color: bool) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        school = getattr(record, "school", "[global]")
        level = record.levelname
        name = record.name
        msg = record.getMessage()

        if record.exc_info:
            msg = msg + "\n" + self.formatException(record.exc_info)

        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            return (
                f"{color}{_BOLD}{level:<8}{_RESET} | "
                f"{_CYAN}{school:<40}{_RESET} | "
                f"{_DIM}{name}{_RESET} | "
                f"{color}{msg}{_RESET}"
            )

        return f"{level:<8} | {school:<40} | {name} | {msg}"


class SchoolContextFilter(logging.Filter):
    """Injects a `school` field into every log record.

    Records without a school context get "[global]" so log formats stay uniform.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "school"):
            record.school = "[global]"  # type: ignore[attr-defined]
        return True


def setup_logging(*, verbose: bool = False) -> None:
    """Configure root logger with color-aware structured formatting.

    Call once at process startup. Color is enabled only when stderr is a TTY.
    """
    level = logging.DEBUG if verbose else logging.INFO
    use_color = sys.stderr.isatty()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColorFormatter(use_color=use_color))
    handler.addFilter(SchoolContextFilter())

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on repeated calls (e.g. in tests)
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_school_logger(name: str, school: str) -> logging.LoggerAdapter[Any]:
    """Return a logger adapter that tags every message with the school name."""
    return logging.LoggerAdapter(logging.getLogger(name), {"school": school})
