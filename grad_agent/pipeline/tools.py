"""Tool definitions and execution handlers for the retrieval agent.

Provides two tools:
  - web_search: queries Brave Search API and returns a list of results
  - fetch_page: fetches a URL and returns its text content (truncated)

Each tool is defined as an Anthropic tool-use schema dict *and* has a
corresponding async handler that performs the actual I/O.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from grad_agent.config import Config
from grad_agent.util.log import get_school_logger

# ---------------------------------------------------------------------------
# Anthropic tool-use definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Search the web for information. Returns a list of results with "
            "title, URL, and snippet. Use targeted queries to fill specific "
            "fields of the school profile."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": (
            "Fetch the text content of a web page by URL. Returns the page "
            "body with HTML tags stripped, truncated to a maximum character "
            "limit. Use this to read full page content after identifying a "
            "relevant URL from search results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
            },
            "required": ["url"],
        },
    },
]


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s{3,}")


def _strip_html(raw: str) -> str:
    """Rough HTML → plaintext conversion. Good enough for LLM consumption."""
    raw = _SCRIPT_STYLE_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = _WS_RE.sub("\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def handle_web_search(
    args: dict[str, Any],
    config: Config,
    http: httpx.AsyncClient,
    school: str,
) -> str:
    """Execute a Brave Search API query and return formatted results."""
    log = get_school_logger("tools.web_search", school)
    query = str(args.get("query", "")).strip()
    if not query:
        return "Search failed: missing query."
    log.info("Searching: %s", query)

    try:
        resp = await http.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": config.max_search_results},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": config.brave_api_key,
            },
            timeout=config.http_timeout,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("Search failed: %s", exc)
        return f"Search failed: {exc}"

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        log.warning("Search returned invalid JSON: %s", exc)
        return f"Search failed: invalid JSON response ({exc})"

    results = data.get("web", {}).get("results", [])
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("description", "")
        lines.append(f"{i}. [{title}]({url})\n   {snippet}")

    formatted = "\n\n".join(lines)
    log.debug("Got %d results", len(results))
    return formatted


async def handle_fetch_page(
    args: dict[str, Any],
    config: Config,
    http: httpx.AsyncClient,
    school: str,
) -> str:
    """Fetch a URL and return stripped text content."""
    log = get_school_logger("tools.fetch_page", school)
    url = str(args.get("url", "")).strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "Failed to fetch page: url must be an absolute http(s) URL."
    log.info("Fetching: %s", url)

    try:
        resp = await http.get(
            url,
            timeout=config.http_timeout,
            follow_redirects=True,
            headers={"User-Agent": "GradSchoolResearchBot/0.1 (academic research)"},
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("Fetch failed for %s: %s", url, exc)
        return f"Failed to fetch page: {exc}"

    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type:
        text = _strip_html(resp.text)
    elif "text/" in content_type or "application/json" in content_type:
        text = resp.text
    else:
        return f"Unsupported content type: {content_type}"

    if len(text) > config.max_page_chars:
        text = text[: config.max_page_chars] + "\n\n[... truncated ...]"

    return text


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "web_search": handle_web_search,
    "fetch_page": handle_fetch_page,
}


async def dispatch_tool(
    name: str,
    args: dict[str, Any],
    config: Config,
    http: httpx.AsyncClient,
    school: str,
) -> str:
    """Route a tool call to its handler. Returns the tool result as a string."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    return await handler(args, config, http, school)
