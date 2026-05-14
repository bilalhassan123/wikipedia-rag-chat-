"""RAG pipeline use case.

Composes embedding → vector search (top-k) → numbered-excerpt prompt →
LLM generation with the chat system prompt pinned in DESIGN.md §7. The
model is instructed to respond with the verbatim :data:`REFUSAL_PHRASE`
when the retrieved excerpts do not contain an answer (FR-14,
rubric-critical — see also the integration test at ``tests/integration``).

Prior turns of the conversation are passed to the model as a context
section so follow-ups like *"tell me more"* or *"and his second wife?"*
can be understood. The grounding rule still holds: the model is told
the conversation is context, not evidence; only the EXCERPTS count.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

from app.core.interfaces import (
    EmbeddingClient,
    LLMClient,
    RetrievedChunk,
    VectorStore,
)

REFUSAL_PHRASE = "I cannot find that in the article."

CHAT_SYSTEM_PROMPT = (
    "You are a careful research assistant for a single Wikipedia article. "
    "The numbered EXCERPTS below are the only allowed source of facts "
    "about the article. If a question asks for article information that "
    "the excerpts do not contain, reply exactly: "
    f'"{REFUSAL_PHRASE}" '
    "The CONVERSATION SO FAR (if any) records what the user and you have "
    "already discussed in this session: use it to interpret follow-up "
    "questions like \"tell me more\", and to answer questions about the "
    "conversation itself (such as \"what did I ask?\"). Never use prior "
    "knowledge about the article topic. Do not speculate. Keep answers "
    "to 1 to 4 sentences."
)

DEFAULT_TOP_K = 5


@dataclass(frozen=True, slots=True)
class HistoryTurn:
    """One prior turn in the conversation."""

    role: str  # "user" or "assistant"
    text: str


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    """A grounded answer plus the chunks used as evidence."""

    answer: str
    evidence: list[RetrievedChunk]


def _render_user_prompt(
    question: str,
    retrieved: list[RetrievedChunk],
    history: Iterable[HistoryTurn] | None = None,
) -> str:
    if not retrieved:
        excerpts_block = "(no excerpts retrieved)"
    else:
        excerpts_block = "\n".join(
            f"[{i + 1}] {rc.chunk.text}" for i, rc in enumerate(retrieved)
        )

    parts = [f"EXCERPTS:\n{excerpts_block}"]

    history = list(history or [])
    if history:
        rendered = "\n".join(
            f"{'You' if turn.role == 'user' else 'Assistant'}: {turn.text}"
            for turn in history
        )
        parts.append(f"CONVERSATION SO FAR:\n{rendered}")

    parts.append(f"QUESTION: {question}")
    return "\n\n".join(parts)


async def answer_question(
    *,
    question: str,
    embedding_client: EmbeddingClient,
    vector_store: VectorStore,
    llm: LLMClient,
    top_k: int = DEFAULT_TOP_K,
    history: Iterable[HistoryTurn] | None = None,
) -> ChatAnswer:
    """Embed the question, retrieve top-k chunks, and ask the LLM (blocking).

    Used by the JSON ``/chat`` endpoint and by tests. The streaming path
    is :func:`answer_question_stream`.
    """
    [query_vec] = await embedding_client.embed([question])
    retrieved = await vector_store.search(query_vec, k=top_k)
    user_prompt = _render_user_prompt(question, retrieved, history)
    answer = await llm.generate(system=CHAT_SYSTEM_PROMPT, user=user_prompt)
    return ChatAnswer(answer=answer, evidence=retrieved)


async def answer_question_stream(
    *,
    question: str,
    embedding_client: EmbeddingClient,
    vector_store: VectorStore,
    llm: LLMClient,
    top_k: int = DEFAULT_TOP_K,
    history: Iterable[HistoryTurn] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream the chat answer token-by-token.

    Yields a sequence of events:
    - one ``evidence`` event up front, carrying the retrieved chunks;
    - any number of ``token`` events with partial answer text;
    - one terminating ``done`` event.

    Raised :class:`AppError`\\s propagate up to the route handler, which
    converts them to in-band ``error`` events on the SSE stream.
    """
    [query_vec] = await embedding_client.embed([question])
    retrieved = await vector_store.search(query_vec, k=top_k)
    user_prompt = _render_user_prompt(question, retrieved, history)

    yield {
        "phase": "evidence",
        "evidence": [
            {
                "chunk_index": rc.chunk.chunk_index,
                "text": rc.chunk.text,
                "score": float(rc.score),
            }
            for rc in retrieved
        ],
    }

    async for chunk in llm.stream(system=CHAT_SYSTEM_PROMPT, user=user_prompt):
        yield {"phase": "token", "token": chunk}

    yield {"phase": "done"}
