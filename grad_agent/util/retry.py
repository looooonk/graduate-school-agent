"""Async retry helper with exponential backoff for Anthropic rate-limit errors."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar

import anthropic

logger = logging.getLogger(__name__)

T = TypeVar("T")

_BASE_DELAY = 60.0   # seconds — rate limit window is 1 minute
_BACKOFF_MULTIPLIER = 1.5
_MAX_RETRIES = 4


async def api_create_with_retry(
    fn: Callable[[], Coroutine[Any, Any, T]],
    *,
    max_retries: int = _MAX_RETRIES,
) -> T:
    """Call *fn* (a zero-arg coroutine factory) retrying on RateLimitError.

    Waits *retry-after* seconds from the response header when available,
    otherwise doubles the delay starting from ``_BASE_DELAY``.

    Args:
        fn: Zero-argument async callable that performs the API request.
        max_retries: Maximum number of retry attempts after the first failure.

    Raises:
        anthropic.RateLimitError: If all retries are exhausted.
        Exception: Any non-rate-limit exception is re-raised immediately.
    """
    delay = _BASE_DELAY
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except anthropic.RateLimitError as exc:
            if attempt == max_retries:
                logger.error("Rate limit exceeded after %d retries — giving up", max_retries)
                raise

            # Honour the server's retry-after header when present
            retry_after: float | None = None
            if exc.response is not None:
                raw = exc.response.headers.get("retry-after") or exc.response.headers.get(
                    "x-ratelimit-reset-requests"
                )
                if raw is not None:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        pass

            wait = retry_after if retry_after is not None else delay
            logger.warning(
                "Rate limit hit (attempt %d/%d) — waiting %.0fs before retry",
                attempt + 1,
                max_retries + 1,
                wait,
            )
            await asyncio.sleep(wait)
            delay *= _BACKOFF_MULTIPLIER
