"""Unit tests for the RAG pipeline.

The rubric-critical contract (FR-14): when retrieval returns chunks that
do not support the question, the pipeline must surface the model's
verbatim refusal phrase, not a fabricated answer. The integration test
at ``tests/integration`` exercises this end-to-end against the real
Ollama runtime; the tests here pin the prompt structure and the
pipeline's pass-through of the refusal.
"""

import pytest

from app.core.interfaces import Chunk, RetrievedChunk
from app.core.rag import (
    CHAT_SYSTEM_PROMPT,
    DEFAULT_TOP_K,
    REFUSAL_PHRASE,
    ChatAnswer,
    HistoryTurn,
    answer_question,
    answer_question_stream,
)


class _FakeEmbedding:
    def __init__(self, vector=(0.1, 0.2, 0.3, 0.4)):
        self._vector = list(vector)
        self.calls: list[list[str]] = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [list(self._vector) for _ in texts]


class _FakeVectorStore:
    def __init__(self, results: list[RetrievedChunk] | None = None):
        self.results = results or []
        self.calls: list[dict] = []

    async def reset(self):
        return None

    async def upsert(self, chunks, vectors):
        return None

    async def search(self, query_vector, k):
        self.calls.append({"query_vector": query_vector, "k": k})
        return self.results


class _FakeLLM:
    def __init__(
        self,
        response: str = "stub answer",
        *,
        stream_chunks: list[str] | None = None,
    ):
        self.response = response
        # Default streaming behaviour: emit the full response as one chunk.
        # Tests can override stream_chunks to verify token-by-token wiring.
        self.stream_chunks = (
            stream_chunks if stream_chunks is not None else [response]
        )
        self.calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def generate(self, *, system, user, max_tokens=512):
        self.calls.append(
            {"system": system, "user": user, "max_tokens": max_tokens}
        )
        return self.response

    async def stream(self, *, system, user, max_tokens=512):
        self.stream_calls.append(
            {"system": system, "user": user, "max_tokens": max_tokens}
        )
        for chunk in self.stream_chunks:
            yield chunk


def _r(i: int, text: str = "sample text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=f"{i:016x}",
            text=text,
            chunk_index=i,
            article_title="Foo",
        ),
        score=0.9 - i * 0.05,
    )


class TestPipelineFlow:
    async def test_embeds_the_question(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM(response="ok")
        await answer_question(
            question="What did Einstein discover?",
            embedding_client=emb,
            vector_store=store,
            llm=llm,
        )
        assert emb.calls == [["What did Einstein discover?"]]

    async def test_default_top_k_is_5(self):
        assert DEFAULT_TOP_K == 5
        emb = _FakeEmbedding(vector=(0.5, 0.5))
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM()
        await answer_question(
            question="q", embedding_client=emb, vector_store=store, llm=llm
        )
        assert store.calls[0] == {"query_vector": [0.5, 0.5], "k": 5}

    async def test_custom_top_k_propagates(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM()
        await answer_question(
            question="q",
            embedding_client=emb,
            vector_store=store,
            llm=llm,
            top_k=3,
        )
        assert store.calls[0]["k"] == 3

    async def test_uses_chat_system_prompt(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM()
        await answer_question(
            question="q", embedding_client=emb, vector_store=store, llm=llm
        )
        assert llm.calls[0]["system"] == CHAT_SYSTEM_PROMPT

    async def test_user_prompt_contains_numbered_excerpts(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(
            results=[_r(0, "alpha"), _r(1, "beta"), _r(2, "gamma")]
        )
        llm = _FakeLLM()
        await answer_question(
            question="q", embedding_client=emb, vector_store=store, llm=llm
        )
        user = llm.calls[0]["user"]
        assert "[1] alpha" in user
        assert "[2] beta" in user
        assert "[3] gamma" in user

    async def test_user_prompt_includes_question(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM()
        await answer_question(
            question="What is X?",
            embedding_client=emb,
            vector_store=store,
            llm=llm,
        )
        assert "QUESTION: What is X?" in llm.calls[0]["user"]

    async def test_returns_chat_answer_with_evidence(self):
        emb = _FakeEmbedding()
        results = [_r(0), _r(1)]
        store = _FakeVectorStore(results=results)
        llm = _FakeLLM(response="The answer is 42.")
        ans = await answer_question(
            question="q", embedding_client=emb, vector_store=store, llm=llm
        )
        assert isinstance(ans, ChatAnswer)
        assert ans.answer == "The answer is 42."
        assert ans.evidence == results

    async def test_no_retrieval_results_still_calls_llm(self):
        """With no chunks the pipeline still asks the LLM; the system
        prompt forces the refusal phrase. Tested end-to-end in T17."""
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[])
        llm = _FakeLLM(response=REFUSAL_PHRASE)
        ans = await answer_question(
            question="q", embedding_client=emb, vector_store=store, llm=llm
        )
        assert ans.answer == REFUSAL_PHRASE
        assert ans.evidence == []
        assert "(no excerpts retrieved)" in llm.calls[0]["user"]


class TestRefusalContract:
    """FR-14 contract: the prompt instructs the model to reply with the
    verbatim refusal phrase when retrieval is unsupportive."""

    def test_refusal_phrase_pinned(self):
        assert REFUSAL_PHRASE == "I cannot find that in the article."

    def test_system_prompt_contains_refusal_phrase(self):
        assert REFUSAL_PHRASE in CHAT_SYSTEM_PROMPT

    def test_system_prompt_forbids_prior_knowledge(self):
        assert "prior knowledge" in CHAT_SYSTEM_PROMPT.lower()

    def test_system_prompt_pins_excerpts_as_only_article_source(self):
        # The excerpts must be the only allowed source of article facts.
        assert "EXCERPTS" in CHAT_SYSTEM_PROMPT
        assert "only allowed source" in CHAT_SYSTEM_PROMPT.lower()

    async def test_pipeline_returns_refusal_verbatim_when_llm_refuses(self):
        emb = _FakeEmbedding()
        irrelevant = [
            _r(0, "Quantum mechanics describes very small particles."),
            _r(1, "Photons are quanta of light."),
        ]
        store = _FakeVectorStore(results=irrelevant)
        llm = _FakeLLM(response=REFUSAL_PHRASE)
        ans = await answer_question(
            question="What is the capital of France?",
            embedding_client=emb,
            vector_store=store,
            llm=llm,
        )
        assert ans.answer == REFUSAL_PHRASE
        # Evidence is still surfaced even when the answer refuses — the
        # frontend can show what was searched.
        assert ans.evidence == irrelevant


class TestConversationHistory:
    """Multi-turn memory: prior turns are passed to the model so follow-ups
    like 'tell me more' or 'and his second wife?' can be understood."""

    async def test_no_history_omits_conversation_section_from_prompt(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM()
        await answer_question(
            question="q", embedding_client=emb, vector_store=store, llm=llm
        )
        assert "CONVERSATION SO FAR" not in llm.calls[0]["user"]

    async def test_history_is_rendered_into_user_prompt(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM()
        history = [
            HistoryTurn(role="user", text="When was Einstein born?"),
            HistoryTurn(role="assistant", text="14 March 1879."),
        ]
        await answer_question(
            question="And where?",
            embedding_client=emb,
            vector_store=store,
            llm=llm,
            history=history,
        )
        user = llm.calls[0]["user"]
        assert "CONVERSATION SO FAR:" in user
        assert "You: When was Einstein born?" in user
        assert "Assistant: 14 March 1879." in user
        assert "QUESTION: And where?" in user

    async def test_history_appears_before_current_question(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM()
        history = [HistoryTurn(role="user", text="prev")]
        await answer_question(
            question="now",
            embedding_client=emb,
            vector_store=store,
            llm=llm,
            history=history,
        )
        user = llm.calls[0]["user"]
        assert user.index("CONVERSATION SO FAR") < user.index("QUESTION:")

    def test_system_prompt_describes_history_role(self):
        # The prompt must (a) mention CONVERSATION SO FAR, (b) instruct the
        # model to use it for follow-ups and meta-questions, while still
        # forbidding it as a source of *article* facts.
        assert "CONVERSATION SO FAR" in CHAT_SYSTEM_PROMPT
        assert "follow-up" in CHAT_SYSTEM_PROMPT.lower()


class TestStreamingPipeline:
    """`answer_question_stream` mirrors `answer_question` but yields events
    as they happen: one `evidence` event up front, then `token` events
    from the LLM stream, then a `done` terminator."""

    async def test_emits_evidence_event_first(self):
        emb = _FakeEmbedding()
        results = [_r(0, "alpha"), _r(1, "beta")]
        store = _FakeVectorStore(results=results)
        llm = _FakeLLM(stream_chunks=["x"])

        events = []
        async for event in answer_question_stream(
            question="q", embedding_client=emb, vector_store=store, llm=llm
        ):
            events.append(event)

        assert events[0]["phase"] == "evidence"
        assert len(events[0]["evidence"]) == 2
        assert events[0]["evidence"][0]["text"] == "alpha"
        assert events[0]["evidence"][0]["chunk_index"] == 0
        assert events[0]["evidence"][0]["score"] == pytest.approx(0.9)

    async def test_emits_one_token_event_per_stream_chunk(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM(stream_chunks=["Hello", " ", "world", "!"])

        token_events = [
            event
            async for event in answer_question_stream(
                question="q",
                embedding_client=emb,
                vector_store=store,
                llm=llm,
            )
            if event["phase"] == "token"
        ]
        assert [e["token"] for e in token_events] == [
            "Hello", " ", "world", "!",
        ]

    async def test_terminates_with_done_event(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM(stream_chunks=["a", "b"])

        events = [
            event
            async for event in answer_question_stream(
                question="q",
                embedding_client=emb,
                vector_store=store,
                llm=llm,
            )
        ]
        assert events[-1]["phase"] == "done"

    async def test_streaming_pipeline_uses_chat_system_prompt(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM(stream_chunks=["x"])
        async for _ in answer_question_stream(
            question="q",
            embedding_client=emb,
            vector_store=store,
            llm=llm,
        ):
            pass
        assert llm.stream_calls[0]["system"] == CHAT_SYSTEM_PROMPT

    async def test_streaming_pipeline_passes_history_to_prompt(self):
        emb = _FakeEmbedding()
        store = _FakeVectorStore(results=[_r(0)])
        llm = _FakeLLM(stream_chunks=["x"])
        async for _ in answer_question_stream(
            question="follow up",
            embedding_client=emb,
            vector_store=store,
            llm=llm,
            history=[HistoryTurn(role="user", text="prev")],
        ):
            pass
        prompt = llm.stream_calls[0]["user"]
        assert "CONVERSATION SO FAR" in prompt
        assert "You: prev" in prompt
