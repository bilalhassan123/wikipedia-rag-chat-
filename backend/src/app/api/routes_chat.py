"""POST /api/chat (JSON) and POST /api/chat/stream (Server-Sent Events).

Both routes share the same retrieval + prompt assembly. The JSON variant
calls the LLM blocking and returns the full answer in one response (used
by direct API clients and tests). The streaming variant emits an
``evidence`` event up front then pipes per-token events to the browser
so the chat bubble grows live.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.errors import AppError
from app.core.interfaces import EmbeddingClient, LLMClient, VectorStore
from app.core.rag import HistoryTurn, answer_question, answer_question_stream
from app.deps import (
    Settings,
    get_embedding_client,
    get_llm_client,
    get_settings,
    get_vector_store,
)
from app.schemas import ChatEvidence, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_history(payload: ChatRequest) -> list[HistoryTurn]:
    return [HistoryTurn(role=h.role, text=h.text) for h in payload.history]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    vector_store: VectorStore = Depends(get_vector_store),
) -> ChatResponse:
    result = await answer_question(
        question=payload.question,
        embedding_client=embedding_client,
        vector_store=vector_store,
        llm=llm,
        top_k=settings.top_k,
        history=_to_history(payload),
    )
    return ChatResponse(
        answer=result.answer,
        evidence=[
            ChatEvidence(
                chunk_index=rc.chunk.chunk_index,
                text=rc.chunk.text,
                score=rc.score,
            )
            for rc in result.evidence
        ],
    )


def _format_sse(event: dict[str, Any]) -> str:
    phase = event.get("phase", "message")
    return f"event: {phase}\ndata: {json.dumps(event)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    vector_store: VectorStore = Depends(get_vector_store),
) -> StreamingResponse:
    """Streaming chat — Server-Sent Events with per-token deltas."""

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in answer_question_stream(
                question=payload.question,
                embedding_client=embedding_client,
                vector_store=vector_store,
                llm=llm,
                top_k=settings.top_k,
                history=_to_history(payload),
            ):
                yield _format_sse(event)
        except AppError as exc:
            yield _format_sse(
                {
                    "phase": "error",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            )
        except Exception as exc:  # pragma: no cover  -- defensive catch-all
            logger.exception("Unexpected error in /chat/stream")
            yield _format_sse(
                {
                    "phase": "error",
                    "error": "Unexpected server error",
                    "type": type(exc).__name__,
                }
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
