"""Unit tests for the Ollama embedding adapter."""

import json

import httpx
import pytest

from app.adapters.ollama_embedding import OllamaEmbeddingClient
from app.core.errors import LLMUnavailableError
from app.core.interfaces import EmbeddingClient

OLLAMA_URL = "http://ollama:11434"
EMBED_MODEL = "nomic-embed-text"


def _mock_async_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestOllamaEmbeddingClient:
    async def test_empty_input_short_circuits_with_no_http_call(self):
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"embedding": [1.0]}, request=request)

        async with _mock_async_client(handler) as client:
            ec = OllamaEmbeddingClient(
                base_url=OLLAMA_URL, model=EMBED_MODEL, http_client=client
            )
            assert await ec.embed([]) == []

        assert called is False

    async def test_embed_posts_to_api_embeddings_per_text(self):
        captured_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.read()))
            assert str(request.url) == f"{OLLAMA_URL}/api/embeddings"
            return httpx.Response(
                200, json={"embedding": [0.1, 0.2, 0.3]}, request=request
            )

        async with _mock_async_client(handler) as client:
            ec = OllamaEmbeddingClient(
                base_url=OLLAMA_URL, model=EMBED_MODEL, http_client=client
            )
            vectors = await ec.embed(["hello", "world"])

        assert len(vectors) == 2
        assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        assert len(captured_payloads) == 2
        assert captured_payloads[0]["model"] == EMBED_MODEL
        assert captured_payloads[0]["prompt"] == "hello"
        assert captured_payloads[1]["prompt"] == "world"

    async def test_embed_payload_includes_keep_alive(self):
        captured_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.read()))
            return httpx.Response(
                200, json={"embedding": [0.0]}, request=request
            )

        async with _mock_async_client(handler) as client:
            ec = OllamaEmbeddingClient(
                base_url=OLLAMA_URL, model=EMBED_MODEL, http_client=client
            )
            await ec.embed(["x"])

        assert captured_payloads[0].get("keep_alive") == "30m"

    async def test_embedding_values_are_floats(self):
        async with _mock_async_client(
            lambda req: httpx.Response(
                200, json={"embedding": [1, 2, 3]}, request=req
            )
        ) as client:
            ec = OllamaEmbeddingClient(
                base_url=OLLAMA_URL, model=EMBED_MODEL, http_client=client
            )
            vectors = await ec.embed(["x"])

        assert vectors == [[1.0, 2.0, 3.0]]
        assert all(isinstance(v, float) for v in vectors[0])

    async def test_5xx_raises_llm_unavailable(self):
        async with _mock_async_client(
            lambda req: httpx.Response(503, text="busy", request=req)
        ) as client:
            ec = OllamaEmbeddingClient(
                base_url=OLLAMA_URL, model=EMBED_MODEL, http_client=client
            )
            with pytest.raises(LLMUnavailableError):
                await ec.embed(["x"])

    async def test_network_error_raises_llm_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated")

        async with _mock_async_client(handler) as client:
            ec = OllamaEmbeddingClient(
                base_url=OLLAMA_URL, model=EMBED_MODEL, http_client=client
            )
            with pytest.raises(LLMUnavailableError):
                await ec.embed(["x"])

    async def test_invalid_json_raises_llm_unavailable(self):
        async with _mock_async_client(
            lambda req: httpx.Response(200, text="<html>", request=req)
        ) as client:
            ec = OllamaEmbeddingClient(
                base_url=OLLAMA_URL, model=EMBED_MODEL, http_client=client
            )
            with pytest.raises(LLMUnavailableError):
                await ec.embed(["x"])

    async def test_missing_embedding_field_raises_llm_unavailable(self):
        async with _mock_async_client(
            lambda req: httpx.Response(200, json={"done": True}, request=req)
        ) as client:
            ec = OllamaEmbeddingClient(
                base_url=OLLAMA_URL, model=EMBED_MODEL, http_client=client
            )
            with pytest.raises(LLMUnavailableError):
                await ec.embed(["x"])

    async def test_empty_embedding_field_raises_llm_unavailable(self):
        async with _mock_async_client(
            lambda req: httpx.Response(200, json={"embedding": []}, request=req)
        ) as client:
            ec = OllamaEmbeddingClient(
                base_url=OLLAMA_URL, model=EMBED_MODEL, http_client=client
            )
            with pytest.raises(LLMUnavailableError):
                await ec.embed(["x"])

    async def test_base_url_with_trailing_slash_is_normalised(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"embedding": [0.0]}, request=request)

        async with _mock_async_client(handler) as client:
            ec = OllamaEmbeddingClient(
                base_url=f"{OLLAMA_URL}/", model=EMBED_MODEL, http_client=client
            )
            await ec.embed(["x"])

        assert captured["url"] == f"{OLLAMA_URL}/api/embeddings"

    async def test_satisfies_embeddingclient_protocol(self):
        async with _mock_async_client(
            lambda req: httpx.Response(200, json={"embedding": [0.0]}, request=req)
        ) as client:
            ec = OllamaEmbeddingClient(
                base_url=OLLAMA_URL, model=EMBED_MODEL, http_client=client
            )
            assert isinstance(ec, EmbeddingClient)
