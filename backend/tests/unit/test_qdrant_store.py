"""Unit tests for the Qdrant vector store adapter."""

from collections import namedtuple
from unittest.mock import AsyncMock

import pytest
from qdrant_client.http.models import Distance, VectorParams

from app.adapters.qdrant_store import (
    QdrantVectorStore,
    _chunk_id_to_point_id,
)
from app.core.errors import VectorStoreUnavailableError
from app.core.interfaces import Chunk, RetrievedChunk, VectorStore

COLLECTION = "test_collection"
VECTOR_SIZE = 4

FakePoint = namedtuple("FakePoint", ["id", "score", "payload"])


def _store(client) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        vector_size=VECTOR_SIZE,
    )


def _chunk(i: int, text: str = "text", title: str = "Foo") -> Chunk:
    return Chunk(
        id=f"{i:016x}",  # 16 hex chars
        text=text,
        chunk_index=i,
        article_title=title,
    )


# --- chunk_id -> point id conversion -----------------------------------------

class TestChunkIdConversion:
    def test_converts_zero(self):
        assert _chunk_id_to_point_id("0000000000000000") == 0

    def test_converts_max_uint64_minus_one(self):
        assert _chunk_id_to_point_id("ffffffffffffffff") == 2**64 - 1

    def test_round_trip_format(self):
        for value in [0, 1, 0xDEADBEEF, 2**64 - 1]:
            hex_str = format(value, "016x")
            assert _chunk_id_to_point_id(hex_str) == value


# --- reset --------------------------------------------------------------------

class TestReset:
    async def test_calls_recreate_collection_with_vector_params(self):
        client = AsyncMock()
        client.recreate_collection = AsyncMock(return_value=None)

        await _store(client).reset()

        client.recreate_collection.assert_awaited_once()
        kwargs = client.recreate_collection.await_args.kwargs
        assert kwargs["collection_name"] == COLLECTION
        params: VectorParams = kwargs["vectors_config"]
        assert params.size == VECTOR_SIZE
        assert params.distance == Distance.COSINE

    async def test_failure_raises_vector_store_unavailable(self):
        client = AsyncMock()
        client.recreate_collection = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )

        with pytest.raises(VectorStoreUnavailableError):
            await _store(client).reset()


# --- upsert -------------------------------------------------------------------

class TestUpsert:
    async def test_empty_inputs_short_circuit_with_no_call(self):
        client = AsyncMock()
        client.upsert = AsyncMock(return_value=None)

        await _store(client).upsert([], [])

        client.upsert.assert_not_awaited()

    async def test_mismatched_lengths_raise_value_error(self):
        client = AsyncMock()
        with pytest.raises(ValueError):
            await _store(client).upsert([_chunk(0)], [[0.1], [0.2]])

    async def test_each_chunk_becomes_a_point_with_payload_and_vector(self):
        client = AsyncMock()
        client.upsert = AsyncMock(return_value=None)

        chunks = [_chunk(0, text="alpha"), _chunk(1, text="beta")]
        vectors = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        await _store(client).upsert(chunks, vectors)

        client.upsert.assert_awaited_once()
        kwargs = client.upsert.await_args.kwargs
        assert kwargs["collection_name"] == COLLECTION
        points = kwargs["points"]
        assert len(points) == 2
        assert points[0].id == _chunk_id_to_point_id(chunks[0].id)
        assert points[0].vector == vectors[0]
        assert points[0].payload == {
            "id": chunks[0].id,
            "text": "alpha",
            "chunk_index": 0,
            "article_title": "Foo",
        }
        assert points[1].id == _chunk_id_to_point_id(chunks[1].id)
        assert points[1].payload["text"] == "beta"

    async def test_failure_raises_vector_store_unavailable(self):
        client = AsyncMock()
        client.upsert = AsyncMock(side_effect=RuntimeError("network down"))

        with pytest.raises(VectorStoreUnavailableError):
            await _store(client).upsert([_chunk(0)], [[0.0, 0.0, 0.0, 0.0]])


# --- search -------------------------------------------------------------------

class TestSearch:
    async def test_calls_search_with_query_and_limit(self):
        client = AsyncMock()
        client.search = AsyncMock(return_value=[])

        await _store(client).search([0.1, 0.2, 0.3, 0.4], k=5)

        client.search.assert_awaited_once_with(
            collection_name=COLLECTION,
            query_vector=[0.1, 0.2, 0.3, 0.4],
            limit=5,
        )

    async def test_results_are_mapped_to_retrieved_chunks(self):
        client = AsyncMock()
        client.search = AsyncMock(
            return_value=[
                FakePoint(
                    id=1,
                    score=0.92,
                    payload={
                        "id": "0000000000000001",
                        "text": "first",
                        "chunk_index": 0,
                        "article_title": "Foo",
                    },
                ),
                FakePoint(
                    id=2,
                    score=0.71,
                    payload={
                        "id": "0000000000000002",
                        "text": "second",
                        "chunk_index": 1,
                        "article_title": "Foo",
                    },
                ),
            ]
        )

        results = await _store(client).search([0.0, 0.0, 0.0, 0.0], k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert results[0].chunk.text == "first"
        assert results[0].score == pytest.approx(0.92)
        assert results[1].chunk.text == "second"
        assert results[1].score == pytest.approx(0.71)

    async def test_empty_results_yield_empty_list(self):
        client = AsyncMock()
        client.search = AsyncMock(return_value=[])

        assert await _store(client).search([0.0] * 4, k=5) == []

    async def test_failure_raises_vector_store_unavailable(self):
        client = AsyncMock()
        client.search = AsyncMock(side_effect=RuntimeError("collection missing"))

        with pytest.raises(VectorStoreUnavailableError):
            await _store(client).search([0.0] * 4, k=5)


# --- protocol conformance -----------------------------------------------------

class TestProtocolConformance:
    async def test_satisfies_vectorstore_protocol(self):
        client = AsyncMock()
        store = _store(client)
        assert isinstance(store, VectorStore)
