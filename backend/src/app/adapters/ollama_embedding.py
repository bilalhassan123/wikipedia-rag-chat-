"""Ollama-backed embedding client adapter.

Implements the :class:`app.core.interfaces.EmbeddingClient` protocol against
Ollama's ``/api/embeddings`` endpoint. One HTTP call per text — Ollama's
embeddings endpoint does not currently accept a batch parameter, so we
sequence calls in the caller (acceptable for single-article ingestion).
"""

from __future__ import annotations

import logging

import httpx

from app.core.errors import LLMUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_KEEP_ALIVE = "30m"


class OllamaEmbeddingClient:
    """Talks to a local Ollama instance for text embeddings."""

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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        url = f"{self._base_url}/api/embeddings"
        logger.info(
            "[ollama-embed] POST %s model=%s n_texts=%d", url, self._model, len(texts)
        )
        vectors: list[list[float]] = []
        for i, text in enumerate(texts):
            payload = {
                "model": self._model,
                "prompt": text,
                "keep_alive": self._keep_alive,
            }
            try:
                response = await self._http.post(url, json=payload, timeout=self._timeout)
            except httpx.HTTPError as exc:
                raise LLMUnavailableError(
                    f"Ollama embeddings unreachable at {url} "
                    f"(text {i + 1}/{len(texts)}): {type(exc).__name__}: {exc!r}"
                ) from exc

            if response.status_code != 200:
                raise LLMUnavailableError(
                    f"Ollama embeddings at {url} returned status "
                    f"{response.status_code}: {response.text[:200]}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise LLMUnavailableError(
                    "Ollama embeddings returned invalid JSON"
                ) from exc

            embedding = data.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise LLMUnavailableError(
                    "Ollama embeddings response missing 'embedding' field"
                )
            vectors.append([float(x) for x in embedding])

        return vectors
