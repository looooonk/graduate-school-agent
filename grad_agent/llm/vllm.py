"""OpenAI-compatible chat client for local vLLM retrieval workers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx

from grad_agent.config import Config


@dataclass(frozen=True)
class LocalChatResponse:
    """Small response shape matching the fields used by retrieval logging."""

    text: str
    stop_reason: str
    usage: Any
    endpoint: str

    @property
    def content(self) -> list[Any]:
        return [SimpleNamespace(type="text", text=self.text)]


class LocalVLLMClient:
    """Round-robin client for one or more OpenAI-compatible vLLM endpoints."""

    def __init__(
        self,
        base_urls: tuple[str, ...],
        *,
        api_key: str = "",
        timeout: int = 600,
        retries: int = 0,
    ) -> None:
        if not base_urls:
            raise ValueError("At least one vLLM endpoint is required")
        self._base_urls = tuple(url.rstrip("/") for url in base_urls)
        self._api_key = api_key
        self._timeout = timeout
        self._retries = retries
        self._next = 0
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: Config) -> LocalVLLMClient:
        return cls(
            config.local_retrieval_endpoints,
            api_key=config.local_retrieval_api_key,
            timeout=config.local_retrieval_timeout,
            retries=config.http_retries,
        )

    async def create(
        self,
        http: httpx.AsyncClient,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.2,
    ) -> LocalChatResponse:
        """Create a chat completion, failing over across endpoints."""
        attempts = max(1, len(self._base_urls) * (self._retries + 1))
        last_exc: Exception | None = None
        for _ in range(attempts):
            endpoint = await self._next_endpoint()
            try:
                data = await self._post_chat_completion(
                    http,
                    endpoint,
                    {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                return _parse_response(data, endpoint)
            except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                last_exc = exc
        raise RuntimeError(f"All local vLLM endpoints failed: {last_exc}") from last_exc

    async def _next_endpoint(self) -> str:
        async with self._lock:
            endpoint = self._base_urls[self._next]
            self._next = (self._next + 1) % len(self._base_urls)
            return endpoint

    async def _post_chat_completion(
        self,
        http: httpx.AsyncClient,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = await http.post(
            _chat_completions_url(endpoint),
            json=payload,
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()


def _chat_completions_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _parse_response(data: dict[str, Any], endpoint: str) -> LocalChatResponse:
    choice = data["choices"][0]
    message = choice.get("message", {})
    text = message.get("content") or ""
    usage = data.get("usage", {})
    return LocalChatResponse(
        text=text,
        stop_reason=choice.get("finish_reason") or "stop",
        usage=SimpleNamespace(
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
        ),
        endpoint=endpoint,
    )
