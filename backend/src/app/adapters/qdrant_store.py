"""Qdrant-backed vector store adapter.

Implements the :class:`app.core.interfaces.VectorStore` protocol against a
Qdrant collection. The collection is scoped to one article at a time
(FR-10): :meth:`reset` recreates it empty, :meth:`upsert` writes
chunks+vectors, :meth:`search` returns the top-k by cosine similarity.

Failures surface as :class:`app.core.errors.VectorStoreUnavailableError`,
which the FastAPI handler maps to HTTP 503 (DESIGN.md §10).
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.core.errors import VectorStoreUnavailableError
from app.core.interfaces import Chunk, RetrievedChunk


def _chunk_id_to_point_id(chunk_id: str) -> int:
    """Convert our 16-hex-char chunk id to a uint64 Qdrant point id."""
    return int(chunk_id, 16)


class QdrantVectorStore:
    """Adapter over a single Qdrant collection."""

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._vector_size = vector_size
        self._distance = distance

    async def reset(self) -> None:
        """Drop the collection if present, then recreate it empty."""
        try:
            await self._client.recreate_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=self._distance
                ),
            )
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant reset failed: {exc}"
            ) from exc

    async def upsert(
        self, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch"
            )
        if not chunks:
            return

        points = [
            PointStruct(
                id=_chunk_id_to_point_id(chunk.id),
                vector=vector,
                payload={
                    "id": chunk.id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "article_title": chunk.article_title,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            await self._client.upsert(
                collection_name=self._collection, points=points
            )
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant upsert failed: {exc}"
            ) from exc

    async def search(
        self, query_vector: list[float], k: int
    ) -> list[RetrievedChunk]:
        try:
            results = await self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=k,
            )
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Qdrant search failed: {exc}"
            ) from exc

        return [
            RetrievedChunk(
                chunk=Chunk(
                    id=point.payload["id"],
                    text=point.payload["text"],
                    chunk_index=point.payload["chunk_index"],
                    article_title=point.payload["article_title"],
                ),
                score=float(point.score),
            )
            for point in results
        ]
