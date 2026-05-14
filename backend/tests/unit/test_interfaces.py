"""Unit tests for the core protocols and dataclasses."""

from dataclasses import FrozenInstanceError

import pytest

from app.core.interfaces import (
    Chunk,
    EmbeddingClient,
    LLMClient,
    RetrievedChunk,
    VectorStore,
)


class TestChunk:
    def test_holds_all_fields(self):
        c = Chunk(id="abc123", text="hello world", chunk_index=0, article_title="Foo")
        assert c.id == "abc123"
        assert c.text == "hello world"
        assert c.chunk_index == 0
        assert c.article_title == "Foo"

    def test_is_frozen(self):
        c = Chunk(id="abc123", text="hello", chunk_index=0, article_title="Foo")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            c.text = "mutated"  # type: ignore[misc]

    def test_equality_by_value(self):
        a = Chunk(id="x", text="t", chunk_index=1, article_title="A")
        b = Chunk(id="x", text="t", chunk_index=1, article_title="A")
        assert a == b
        assert hash(a) == hash(b)

    def test_distinct_when_any_field_differs(self):
        base = Chunk(id="x", text="t", chunk_index=1, article_title="A")
        assert base != Chunk(id="y", text="t", chunk_index=1, article_title="A")
        assert base != Chunk(id="x", text="u", chunk_index=1, article_title="A")
        assert base != Chunk(id="x", text="t", chunk_index=2, article_title="A")
        assert base != Chunk(id="x", text="t", chunk_index=1, article_title="B")


class TestRetrievedChunk:
    def test_wraps_chunk_with_score(self):
        c = Chunk(id="x", text="t", chunk_index=0, article_title="A")
        r = RetrievedChunk(chunk=c, score=0.87)
        assert r.chunk is c
        assert r.score == pytest.approx(0.87)

    def test_is_frozen(self):
        c = Chunk(id="x", text="t", chunk_index=0, article_title="A")
        r = RetrievedChunk(chunk=c, score=0.5)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            r.score = 0.9  # type: ignore[misc]


class _FakeLLM:
    async def generate(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        return f"sys={system}|usr={user}|max={max_tokens}"

    async def stream(self, *, system: str, user: str, max_tokens: int = 512):
        yield f"sys={system}|usr={user}|max={max_tokens}"


class _FakeEmbedding:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]


class _FakeVectorStore:
    async def reset(self) -> None:
        return None

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        return None

    async def search(self, query_vector: list[float], k: int) -> list[RetrievedChunk]:
        return []


class TestProtocolConformance:
    """The protocols are @runtime_checkable so adapters can be sanity-checked
    structurally at startup if ever needed. These tests pin that contract."""

    def test_fake_llm_satisfies_llmclient(self):
        assert isinstance(_FakeLLM(), LLMClient)

    def test_fake_embedding_satisfies_embeddingclient(self):
        assert isinstance(_FakeEmbedding(), EmbeddingClient)

    def test_fake_store_satisfies_vectorstore(self):
        assert isinstance(_FakeVectorStore(), VectorStore)

    def test_unrelated_object_rejected_for_llmclient(self):
        assert not isinstance(object(), LLMClient)

    def test_unrelated_object_rejected_for_vectorstore(self):
        class _MissingMethods:
            async def reset(self) -> None: ...

        assert not isinstance(_MissingMethods(), VectorStore)
