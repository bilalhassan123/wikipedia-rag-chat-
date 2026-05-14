"""Tests for the FastAPI routes, with all adapters faked via
``app.dependency_overrides``.

The TestClient is constructed without ``with``, which skips the lifespan
handler — we never touch the real Ollama or Qdrant clients.
"""

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import deps
from app.core.interfaces import Chunk, RetrievedChunk
from app.core.rag import REFUSAL_PHRASE
from app.main import create_app

WIKIPEDIA_FIXTURES = (
    Path(__file__).parent.parent / "fixtures" / "wikipedia"
)


def _load(name: str) -> str:
    return (WIKIPEDIA_FIXTURES / name).read_text(encoding="utf-8")


# --- fakes ------------------------------------------------------------------

class _FakeLLM:
    def __init__(
        self,
        response: str = "summary or answer",
        *,
        stream_chunks: list[str] | None = None,
    ):
        self.response = response
        self.stream_chunks = (
            stream_chunks if stream_chunks is not None else [response]
        )

    async def generate(self, *, system, user, max_tokens=512):
        return self.response

    async def stream(self, *, system, user, max_tokens=512):
        for chunk in self.stream_chunks:
            yield chunk


class _FakeEmbedding:
    def __init__(self, vector=None):
        self.vector = list(vector) if vector is not None else [0.1] * 768

    async def embed(self, texts):
        return [list(self.vector) for _ in texts]


class _FakeVectorStore:
    def __init__(self, search_results=None):
        self.search_results = search_results or []
        self.reset_called = False
        self.upserted_chunks: list[Chunk] = []
        self.upserted_vectors: list[list[float]] = []

    async def reset(self):
        self.reset_called = True

    async def upsert(self, chunks, vectors):
        self.upserted_chunks = list(chunks)
        self.upserted_vectors = list(vectors)

    async def search(self, query_vector, k):
        return self.search_results


def _wiki_http(*, status: int = 200, body: str = "") -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _make_test_client(
    *,
    llm: _FakeLLM,
    embedding: _FakeEmbedding,
    vector_store: _FakeVectorStore,
    wiki_http: httpx.AsyncClient | None = None,
) -> TestClient:
    app = create_app()
    app.dependency_overrides[deps.get_llm_client] = lambda: llm
    app.dependency_overrides[deps.get_embedding_client] = lambda: embedding
    app.dependency_overrides[deps.get_vector_store] = lambda: vector_store
    if wiki_http is not None:
        app.dependency_overrides[deps.get_wiki_http] = lambda: wiki_http
    return TestClient(app)


# --- /api/article -----------------------------------------------------------

class TestIndexArticleRoute:
    def test_happy_path_returns_summary_and_chunk_count(self):
        llm = _FakeLLM(response="A short summary about Einstein.")
        embedding = _FakeEmbedding()
        store = _FakeVectorStore()
        wiki_http = _wiki_http(status=200, body=_load("canonical.html"))

        client = _make_test_client(
            llm=llm, embedding=embedding, vector_store=store, wiki_http=wiki_http
        )
        response = client.post(
            "/api/article",
            json={"url": "https://en.wikipedia.org/wiki/Albert_Einstein"},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["title"] == "Albert Einstein"
        assert data["summary"] == "A short summary about Einstein."
        assert data["chunk_count"] >= 1

    def test_happy_path_resets_and_upserts_vector_store(self):
        llm = _FakeLLM(response="summary")
        embedding = _FakeEmbedding()
        store = _FakeVectorStore()
        wiki_http = _wiki_http(status=200, body=_load("canonical.html"))

        client = _make_test_client(
            llm=llm, embedding=embedding, vector_store=store, wiki_http=wiki_http
        )
        response = client.post(
            "/api/article",
            json={"url": "https://en.wikipedia.org/wiki/Albert_Einstein"},
        )
        assert response.status_code == 200
        assert store.reset_called is True
        assert len(store.upserted_chunks) == response.json()["chunk_count"]
        assert len(store.upserted_vectors) == response.json()["chunk_count"]

    def test_invalid_url_returns_400(self):
        llm = _FakeLLM()
        embedding = _FakeEmbedding()
        store = _FakeVectorStore()
        wiki_http = _wiki_http(status=200)

        client = _make_test_client(
            llm=llm, embedding=embedding, vector_store=store, wiki_http=wiki_http
        )
        response = client.post(
            "/api/article", json={"url": "https://example.com/not-wikipedia"}
        )
        assert response.status_code == 400
        assert "error" in response.json()

    def test_disambiguation_returns_422(self):
        llm = _FakeLLM()
        embedding = _FakeEmbedding()
        store = _FakeVectorStore()
        wiki_http = _wiki_http(status=200, body=_load("disambiguation.html"))

        client = _make_test_client(
            llm=llm, embedding=embedding, vector_store=store, wiki_http=wiki_http
        )
        response = client.post(
            "/api/article",
            json={"url": "https://en.wikipedia.org/wiki/Mercury"},
        )
        assert response.status_code == 422

    def test_thin_article_returns_422(self):
        llm = _FakeLLM()
        embedding = _FakeEmbedding()
        store = _FakeVectorStore()
        wiki_http = _wiki_http(status=200, body=_load("stub.html"))

        client = _make_test_client(
            llm=llm, embedding=embedding, vector_store=store, wiki_http=wiki_http
        )
        response = client.post(
            "/api/article",
            json={"url": "https://en.wikipedia.org/wiki/Stub"},
        )
        assert response.status_code == 422

    def test_wikipedia_404_returns_404(self):
        llm = _FakeLLM()
        embedding = _FakeEmbedding()
        store = _FakeVectorStore()
        wiki_http = _wiki_http(status=404)

        client = _make_test_client(
            llm=llm, embedding=embedding, vector_store=store, wiki_http=wiki_http
        )
        response = client.post(
            "/api/article",
            json={"url": "https://en.wikipedia.org/wiki/Nonexistent"},
        )
        assert response.status_code == 404

    def test_missing_url_returns_422_pydantic(self):
        llm = _FakeLLM()
        embedding = _FakeEmbedding()
        store = _FakeVectorStore()

        client = _make_test_client(
            llm=llm,
            embedding=embedding,
            vector_store=store,
            wiki_http=_wiki_http(status=200),
        )
        response = client.post("/api/article", json={})
        assert response.status_code == 422  # Pydantic validation


# --- /api/article/stream (Server-Sent Events) ------------------------------

class TestIndexArticleStreamRoute:
    """The streaming variant emits one Server-Sent Event per pipeline phase
    plus a final ``result`` event. Errors during streaming arrive as
    in-band ``error`` events rather than HTTP error responses."""

    def test_emits_progress_phases_and_result(self):
        llm = _FakeLLM(response="A short summary.")
        embedding = _FakeEmbedding()
        store = _FakeVectorStore()
        wiki_http = _wiki_http(status=200, body=_load("canonical.html"))

        client = _make_test_client(
            llm=llm, embedding=embedding, vector_store=store, wiki_http=wiki_http
        )
        response = client.post(
            "/api/article/stream",
            json={"url": "https://en.wikipedia.org/wiki/Albert_Einstein"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        body = response.text
        # Each phase emits an SSE block; the final event is "result".
        for phase in ("fetching", "fetched", "split", "result"):
            assert f"event: {phase}\n" in body
        assert '"summary": "A short summary."' in body
        assert "Albert Einstein" in body

    def test_emits_in_band_error_event_for_invalid_url(self):
        client = _make_test_client(
            llm=_FakeLLM(),
            embedding=_FakeEmbedding(),
            vector_store=_FakeVectorStore(),
            wiki_http=_wiki_http(status=200),
        )
        # Invalid URL is rejected by the scraper inside the pipeline; the
        # streaming route catches the AppError and emits an in-band event
        # (rather than a 4xx response, since the stream has already started).
        response = client.post(
            "/api/article/stream",
            json={"url": "https://example.com/not-wikipedia"},
        )
        assert response.status_code == 200
        body = response.text
        assert "event: error" in body
        assert "InvalidUrlError" in body


# --- /api/chat --------------------------------------------------------------

def _retrieved(i: int, text: str = "sample") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=f"{i:016x}",
            text=text,
            chunk_index=i,
            article_title="Foo",
        ),
        score=0.9 - i * 0.05,
    )


class TestChatRoute:
    def test_happy_path_returns_answer_and_evidence(self):
        llm = _FakeLLM(response="The answer is 42.")
        embedding = _FakeEmbedding()
        results = [_retrieved(0, "alpha"), _retrieved(1, "beta")]
        store = _FakeVectorStore(search_results=results)

        client = _make_test_client(
            llm=llm, embedding=embedding, vector_store=store
        )
        response = client.post(
            "/api/chat", json={"question": "What is the answer?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The answer is 42."
        assert len(data["evidence"]) == 2
        assert data["evidence"][0]["text"] == "alpha"
        assert data["evidence"][0]["chunk_index"] == 0
        assert data["evidence"][0]["score"] == pytest.approx(0.9)

    def test_refusal_passthrough(self):
        """If retrieval returns chunks that don't support the question and
        the LLM refuses, the route surfaces the verbatim refusal phrase."""
        llm = _FakeLLM(response=REFUSAL_PHRASE)
        embedding = _FakeEmbedding()
        store = _FakeVectorStore(
            search_results=[_retrieved(0, "totally unrelated")]
        )

        client = _make_test_client(
            llm=llm, embedding=embedding, vector_store=store
        )
        response = client.post(
            "/api/chat", json={"question": "Off-topic question?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == REFUSAL_PHRASE
        assert data["evidence"][0]["text"] == "totally unrelated"

    def test_empty_question_returns_422_pydantic(self):
        client = _make_test_client(
            llm=_FakeLLM(),
            embedding=_FakeEmbedding(),
            vector_store=_FakeVectorStore(),
        )
        response = client.post("/api/chat", json={"question": ""})
        assert response.status_code == 422

    def test_missing_question_returns_422_pydantic(self):
        client = _make_test_client(
            llm=_FakeLLM(),
            embedding=_FakeEmbedding(),
            vector_store=_FakeVectorStore(),
        )
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_chat_request_accepts_history_and_passes_it_to_pipeline(self):
        # Capture the prompt the LLM sees so we can verify the history
        # made it through the route -> pipeline -> prompt path.
        captured_prompts: list[str] = []

        class _CapturingLLM:
            async def generate(self, *, system, user, max_tokens=512):
                captured_prompts.append(user)
                return "answer"

            async def stream(self, *, system, user, max_tokens=512):
                yield "answer"

        client = _make_test_client(
            llm=_CapturingLLM(),
            embedding=_FakeEmbedding(),
            vector_store=_FakeVectorStore(search_results=[_retrieved(0, "ctx")]),
        )
        response = client.post(
            "/api/chat",
            json={
                "question": "and where?",
                "history": [
                    {"role": "user", "text": "When was he born?"},
                    {"role": "assistant", "text": "14 March 1879."},
                ],
            },
        )
        assert response.status_code == 200
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "CONVERSATION SO FAR" in prompt
        assert "You: When was he born?" in prompt
        assert "Assistant: 14 March 1879." in prompt
        assert "QUESTION: and where?" in prompt

    def test_invalid_history_role_returns_422(self):
        client = _make_test_client(
            llm=_FakeLLM(),
            embedding=_FakeEmbedding(),
            vector_store=_FakeVectorStore(),
        )
        response = client.post(
            "/api/chat",
            json={
                "question": "q",
                "history": [{"role": "system", "text": "x"}],
            },
        )
        assert response.status_code == 422


# --- /api/chat/stream (Server-Sent Events) ---------------------------------

class TestChatStreamRoute:
    """The streaming chat variant emits one ``evidence`` event up front,
    then one ``token`` event per LLM stream chunk, then a ``done``
    terminator. Errors during streaming arrive as in-band ``error``
    events rather than HTTP error responses."""

    def test_emits_evidence_then_tokens_then_done(self):
        llm = _FakeLLM(stream_chunks=["Hello", " world", "!"])
        embedding = _FakeEmbedding()
        store = _FakeVectorStore(
            search_results=[_retrieved(0, "ctx alpha"), _retrieved(1, "ctx beta")]
        )

        client = _make_test_client(
            llm=llm, embedding=embedding, vector_store=store
        )
        response = client.post(
            "/api/chat/stream", json={"question": "Tell me about it"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        body = response.text
        # Phase events appear in order: evidence -> token (xN) -> done.
        assert "event: evidence\n" in body
        assert "event: token\n" in body
        assert "event: done\n" in body
        # Each fake stream chunk surfaces as a token event.
        assert '"token": "Hello"' in body
        assert '"token": " world"' in body
        assert '"token": "!"' in body
        # Evidence carries the chunk text.
        assert "ctx alpha" in body

    def test_includes_history_in_streaming_request(self):
        captured_prompts: list[str] = []

        class _CapturingLLM:
            async def generate(self, *, system, user, max_tokens=512):
                return "x"

            async def stream(self, *, system, user, max_tokens=512):
                captured_prompts.append(user)
                yield "ok"

        client = _make_test_client(
            llm=_CapturingLLM(),
            embedding=_FakeEmbedding(),
            vector_store=_FakeVectorStore(search_results=[_retrieved(0, "ctx")]),
        )
        response = client.post(
            "/api/chat/stream",
            json={
                "question": "and where?",
                "history": [{"role": "user", "text": "When was he born?"}],
            },
        )
        assert response.status_code == 200
        assert len(captured_prompts) == 1
        assert "CONVERSATION SO FAR" in captured_prompts[0]
        assert "You: When was he born?" in captured_prompts[0]
        assert "QUESTION: and where?" in captured_prompts[0]

    def test_app_error_during_stream_surfaces_as_in_band_error_event(self):
        """If the LLM stream raises an AppError mid-stream, the route
        catches it and emits an `error` SSE event rather than tearing
        down the connection with an HTTP error status."""
        from app.core.errors import LLMUnavailableError

        class _RaisingLLM:
            async def generate(self, *, system, user, max_tokens=512):
                return "ignored"

            async def stream(self, *, system, user, max_tokens=512):
                # Yield nothing first; raise on iteration.
                raise LLMUnavailableError("simulated outage")
                yield  # pragma: no cover - unreachable, satisfies generator typing

        client = _make_test_client(
            llm=_RaisingLLM(),
            embedding=_FakeEmbedding(),
            vector_store=_FakeVectorStore(search_results=[_retrieved(0, "ctx")]),
        )
        response = client.post("/api/chat/stream", json={"question": "q"})
        # The HTTP status is still 200 because the stream had already opened.
        assert response.status_code == 200
        body = response.text
        assert "event: error" in body
        assert "simulated outage" in body
        assert "LLMUnavailableError" in body


# --- health -----------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self):
        client = _make_test_client(
            llm=_FakeLLM(),
            embedding=_FakeEmbedding(),
            vector_store=_FakeVectorStore(),
        )
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
