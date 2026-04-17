"""Structured logging setup with per-school context propagation."""

from __future__ import annotations

import logging
import sys
from typing import Any


class SchoolContextFilter(logging.Filter):
    """Injects a `school` field into every log record.

    Records without a school context get "[global]" so log formats stay uniform.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "school"):
            record.school = "[global]"  # type: ignore[attr-defined]
        return True


_FORMAT = "%(asctime)s | %(levelname)-8s | %(school)-40s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(*, verbose: bool = False) -> None:
    """Configure root logger with structured formatting.

    Call once at process startup.
    """
    level = logging.DEBUG if verbose else logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
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
