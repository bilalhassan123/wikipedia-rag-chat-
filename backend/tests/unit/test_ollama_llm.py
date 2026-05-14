"""Unit tests for the Ollama LLM adapter."""

import json

import httpx
import pytest

from app.adapters.ollama_llm import OllamaLLMClient
from app.core.errors import LLMUnavailableError
from app.core.interfaces import LLMClient

OLLAMA_URL = "http://ollama:11434"
MODEL = "phi3:mini"


def _mock_async_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _capturing_handler(captured: dict, *, status: int = 200, body: dict | str | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = request.read()
        if isinstance(body, dict):
            return httpx.Response(status, json=body, request=request)
        return httpx.Response(status, text=body or "", request=request)

    return handler


class TestOllamaLLMClient:
    async def test_generate_posts_to_api_generate_endpoint(self):
        captured: dict = {}
        async with _mock_async_client(
            _capturing_handler(captured, body={"response": "Hello!", "done": True})
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            result = await llm.generate(system="be helpful", user="hi")

        assert captured["method"] == "POST"
        assert captured["url"] == f"{OLLAMA_URL}/api/generate"
        assert result == "Hello!"

    async def test_generate_payload_contains_system_user_model_and_options(self):
        captured: dict = {}
        async with _mock_async_client(
            _capturing_handler(captured, body={"response": "ok"})
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            await llm.generate(system="sys text", user="user text", max_tokens=200)

        body = json.loads(captured["body"])
        assert body["model"] == MODEL
        assert body["system"] == "sys text"
        assert body["prompt"] == "user text"
        assert body["stream"] is False
        assert body["options"]["num_predict"] == 200

    async def test_generate_payload_includes_keep_alive_to_prevent_reload(self):
        captured: dict = {}
        async with _mock_async_client(
            _capturing_handler(captured, body={"response": "ok"})
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            await llm.generate(system="s", user="u")

        body = json.loads(captured["body"])
        assert body.get("keep_alive") == "30m"

    async def test_default_max_tokens_is_512(self):
        captured: dict = {}
        async with _mock_async_client(
            _capturing_handler(captured, body={"response": "ok"})
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            await llm.generate(system="s", user="u")

        body = json.loads(captured["body"])
        assert body["options"]["num_predict"] == 512

    async def test_response_is_stripped(self):
        async with _mock_async_client(
            lambda req: httpx.Response(200, json={"response": "  hi  \n"}, request=req)
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            assert await llm.generate(system="s", user="u") == "hi"

    async def test_non_string_response_returns_empty(self):
        async with _mock_async_client(
            lambda req: httpx.Response(200, json={"response": None}, request=req)
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            assert await llm.generate(system="s", user="u") == ""

    async def test_missing_response_field_returns_empty(self):
        async with _mock_async_client(
            lambda req: httpx.Response(200, json={"done": True}, request=req)
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            assert await llm.generate(system="s", user="u") == ""

    async def test_5xx_raises_llm_unavailable(self):
        async with _mock_async_client(
            lambda req: httpx.Response(503, text="busy", request=req)
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            with pytest.raises(LLMUnavailableError):
                await llm.generate(system="s", user="u")

    async def test_4xx_raises_llm_unavailable(self):
        async with _mock_async_client(
            lambda req: httpx.Response(400, text="bad request", request=req)
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            with pytest.raises(LLMUnavailableError):
                await llm.generate(system="s", user="u")

    async def test_network_error_raises_llm_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated")

        async with _mock_async_client(handler) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            with pytest.raises(LLMUnavailableError):
                await llm.generate(system="s", user="u")

    async def test_invalid_json_raises_llm_unavailable(self):
        async with _mock_async_client(
            lambda req: httpx.Response(200, text="not json at all", request=req)
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            with pytest.raises(LLMUnavailableError):
                await llm.generate(system="s", user="u")

    async def test_base_url_with_trailing_slash_is_normalised(self):
        captured: dict = {}
        async with _mock_async_client(
            _capturing_handler(captured, body={"response": "ok"})
        ) as client:
            llm = OllamaLLMClient(
                base_url=f"{OLLAMA_URL}/", model=MODEL, http_client=client
            )
            await llm.generate(system="s", user="u")

        assert captured["url"] == f"{OLLAMA_URL}/api/generate"

    async def test_satisfies_llmclient_protocol(self):
        async with _mock_async_client(
            lambda req: httpx.Response(200, json={"response": "x"}, request=req)
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            assert isinstance(llm, LLMClient)


# --- Streaming ----------------------------------------------------------------


def _ndjson(*objects: dict) -> bytes:
    """Pack JSON objects into Ollama's newline-delimited streaming format."""
    return ("\n".join(json.dumps(o) for o in objects) + "\n").encode("utf-8")


class TestOllamaLLMStream:
    async def test_stream_yields_one_chunk_per_response_line(self):
        body = _ndjson(
            {"response": "Hello", "done": False},
            {"response": " ", "done": False},
            {"response": "world", "done": False},
            {"response": "!", "done": True},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, request=request)

        async with _mock_async_client(handler) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            chunks = [c async for c in llm.stream(system="s", user="u")]

        assert chunks == ["Hello", " ", "world", "!"]

    async def test_stream_payload_sets_stream_true_and_keep_alive(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.read()
            return httpx.Response(
                200,
                content=_ndjson({"response": "x", "done": True}),
                request=request,
            )

        async with _mock_async_client(handler) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            async for _ in llm.stream(system="s", user="u"):
                pass

        body = json.loads(captured["body"])
        assert body["stream"] is True
        assert body["keep_alive"] == "30m"
        assert body["model"] == MODEL

    async def test_stream_stops_at_done_marker(self):
        # Lines after done=True should never reach the consumer.
        body = _ndjson(
            {"response": "a", "done": False},
            {"response": "b", "done": True},
            {"response": "should-not-arrive", "done": False},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, request=request)

        async with _mock_async_client(handler) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            chunks = [c async for c in llm.stream(system="s", user="u")]

        assert chunks == ["a", "b"]

    async def test_stream_skips_lines_with_no_response_text(self):
        body = _ndjson(
            {"response": "", "done": False},
            {"response": "real", "done": False},
            {"done": True},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, request=request)

        async with _mock_async_client(handler) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            chunks = [c async for c in llm.stream(system="s", user="u")]

        assert chunks == ["real"]

    async def test_stream_5xx_raises_llm_unavailable(self):
        async with _mock_async_client(
            lambda req: httpx.Response(503, text="busy", request=req)
        ) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            with pytest.raises(LLMUnavailableError):
                async for _ in llm.stream(system="s", user="u"):
                    pass

    async def test_stream_network_error_raises_llm_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated")

        async with _mock_async_client(handler) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            with pytest.raises(LLMUnavailableError):
                async for _ in llm.stream(system="s", user="u"):
                    pass

    async def test_stream_malformed_json_line_raises_llm_unavailable(self):
        body = b'{"response": "ok", "done": false}\nNOT-JSON\n'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, request=request)

        async with _mock_async_client(handler) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            with pytest.raises(LLMUnavailableError):
                async for _ in llm.stream(system="s", user="u"):
                    pass

    async def test_stream_skips_blank_lines_in_body(self):
        # Defensive against a stray blank line in the stream — verifies
        # the loop's empty-line continue branch.
        body = (
            b'{"response": "first", "done": false}\n'
            b'\n'
            b'   \n'
            b'{"response": "second", "done": true}\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, request=request)

        async with _mock_async_client(handler) as client:
            llm = OllamaLLMClient(base_url=OLLAMA_URL, model=MODEL, http_client=client)
            chunks = [c async for c in llm.stream(system="s", user="u")]

        assert chunks == ["first", "second"]
