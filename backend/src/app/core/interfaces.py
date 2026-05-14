"""Module contracts: data classes and protocols.

Business logic depends on the protocols defined here. Concrete adapters
(Ollama, Qdrant, ...) live under ``app.adapters`` and are wired in via
dependency injection at startup. Swapping an LLM runtime or vector store
means writing a new adapter, not changing this file.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk of article body text with a stable id and origin metadata."""

    id: str
    text: str
    chunk_index: int
    article_title: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by a vector-store search, carrying its similarity score."""

    chunk: Chunk
    score: float


@runtime_checkable
class LLMClient(Protocol):
    """A local generative LLM. Concrete impl: ``OllamaLLMClient``.

    Implementations expose two flavours of generation:
    - :meth:`generate` returns the full answer once complete (used by
      summarisation, where there is no live UX);
    - :meth:`stream` yields partial token chunks as they are generated
      (used by the chat path so the UI can render tokens live).
    """

    async def generate(self, *, system: str, user: str, max_tokens: int = 512) -> str: ...

    def stream(
        self, *, system: str, user: str, max_tokens: int = 512
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """A local text embedding model. Concrete impl: ``OllamaEmbeddingClient``."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """An indexed vector collection, scoped to one article at a time (FR-10)."""

    async def reset(self) -> None: ...
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    async def search(self, query_vector: list[float], k: int) -> list[RetrievedChunk]: ...
