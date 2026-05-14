"""Ollama-backed LLM client adapter.

Implements the :class:`app.core.interfaces.LLMClient` protocol against an
Ollama HTTP endpoint. Network failures and non-200 responses surface as
:class:`app.core.errors.LLMUnavailableError`, which the FastAPI handler
maps to HTTP 503 (DESIGN.md §10).

Two methods:

- :meth:`generate` — single blocking call (``stream: false``); used by the
  summariser, where there's no live UX.
- :meth:`stream` — yields tokens as they arrive (``stream: true``); used
  by the chat path so the frontend can render tokens live.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.core.errors import LLMUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
# Tell Ollama to keep the model loaded between calls so the map-reduce
# summary path doesn't pay model-load cost 4 or 5 times in a row.
DEFAULT_KEEP_ALIVE = "30m"


class OllamaLLMClient:
    """Talks to a local Ollama instance via its REST API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        http_client: httpx.AsyncClient,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._http = http_client
        self._timeout = timeout_seconds
        self._keep_alive = keep_alive

    # --- non-streaming -----------------------------------------------------

    async def generate(
        self, *, system: str, user: str, max_tokens: int = 512
    ) -> str:
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "system": system,
            "prompt": user,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"num_predict": max_tokens},
        }
        logger.info("[ollama-llm] POST %s model=%s", url, self._model)
        try:
            response = await self._http.post(url, json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama unreachable at {url}: {type(exc).__name__}: {exc!r}"
            ) from exc

        if response.status_code != 200:
            raise LLMUnavailableError(
                f"Ollama at {url} returned status {response.status_code}: "
                f"{response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMUnavailableError("Ollama returned invalid JSON") from exc

        text = data.get("response", "")
        return text.strip() if isinstance(text, str) else ""

    # --- streaming ---------------------------------------------------------

    async def stream(
        self, *, system: str, user: str, max_tokens: int = 512
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token.

        Yields partial response chunks (each is a few tokens of text).
        Raises :class:`LLMUnavailableError` on transport failure, non-200
        status, or malformed line content.
        """
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "system": system,
            "prompt": user,
            "stream": True,
            "keep_alive": self._keep_alive,
            "options": {"num_predict": max_tokens},
        }
        logger.info("[ollama-llm] STREAM POST %s model=%s", url, self._model)
        try:
            async with self._http.stream(
                "POST", url, json=payload, timeout=self._timeout
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise LLMUnavailableError(
                        f"Ollama at {url} returned status "
                        f"{response.status_code}: "
                        f"{body.decode('utf-8', errors='replace')[:200]}"
                    )
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMUnavailableError(
                            f"Ollama streaming returned non-JSON line: "
                            f"{line[:100]!r}"
                        ) from exc
                    chunk = data.get("response", "")
                    if isinstance(chunk, str) and chunk:
                        yield chunk
                    if data.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama unreachable at {url}: {type(exc).__name__}: {exc!r}"
            ) from exc
