"""POST /api/article (JSON) and POST /api/article/stream (Server-Sent Events).

Both endpoints share the same underlying ingestion pipeline. The JSON
variant collects the final result and returns it as a single response
(used by tests and direct API clients). The streaming variant pipes per-
phase progress events to the browser so the UI can render live status
while a long ingestion is in flight.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.chunker import chunk_article
from app.core.errors import AppError
from app.core.interfaces import EmbeddingClient, LLMClient, VectorStore
from app.core.scraper import scrape
from app.core.summariser import summarise_article
from app.deps import (
    Settings,
    get_embedding_client,
    get_llm_client,
    get_settings,
    get_vector_store,
    get_wiki_http,
)
from app.schemas import IndexArticleRequest, IndexArticleResponse

logger = logging.getLogger(__name__)
router = APIRouter()


async def _index_pipeline(
    url: str,
    *,
    settings: Settings,
    wiki_http: httpx.AsyncClient,
    llm: LLMClient,
    embedding_client: EmbeddingClient,
    vector_store: VectorStore,
) -> AsyncIterator[dict[str, Any]]:
    """Run the full ingestion pipeline, yielding a progress event at each
    phase and a final ``result`` event with the response payload."""

    yield {"phase": "fetching", "message": "Reading article from Wikipedia"}
    article = await scrape(url, client=wiki_http)
    logger.info(
        "[article] scraped title=%r body=%d chars",
        article.title, len(article.body_text),
    )
    yield {
        "phase": "fetched",
        "message": f"Read {len(article.body_text):,} characters",
        "title": article.title,
    }

    chunks = chunk_article(
        text=article.body_text,
        article_title=article.title,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    logger.info("[article] split into %d sections", len(chunks))
    yield {
        "phase": "split",
        "message": f"Split into {len(chunks)} sections",
        "chunk_count": len(chunks),
    }

    yield {"phase": "resetting", "message": "Preparing knowledge base"}
    await vector_store.reset()

    # Embeddings first, then summary. Sequential keeps only one model
    # hot in Ollama at a time — much friendlier to a 16 GB CPU box than
    # running both in parallel. See NOTES.md.
    yield {
        "phase": "processing_started",
        "message": f"Processing {len(chunks)} sections",
    }
    if chunks:
        vectors = await embedding_client.embed([c.text for c in chunks])
        await vector_store.upsert(chunks, vectors)
    yield {
        "phase": "processing_ready",
        "message": "Knowledge base ready",
    }

    yield {
        "phase": "summary_started",
        "message": "Generating summary (this can take a few minutes for long articles)",
    }
    summary = await summarise_article(
        title=article.title, body=article.body_text, llm=llm
    )
    yield {"phase": "summary_ready", "message": "Summary ready"}

    yield {
        "phase": "result",
        "title": article.title,
        "summary": summary,
        "chunk_count": len(chunks),
    }


@router.post("/article", response_model=IndexArticleResponse)
async def index_article(
    payload: IndexArticleRequest,
    settings: Settings = Depends(get_settings),
    wiki_http: httpx.AsyncClient = Depends(get_wiki_http),
    llm: LLMClient = Depends(get_llm_client),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    vector_store: VectorStore = Depends(get_vector_store),
) -> IndexArticleResponse:
    """Non-streaming JSON variant. Used by direct API clients and tests."""
    final: dict[str, Any] = {}
    async for event in _index_pipeline(
        payload.url,
        settings=settings,
        wiki_http=wiki_http,
        llm=llm,
        embedding_client=embedding_client,
        vector_store=vector_store,
    ):
        if event.get("phase") == "result":
            final = event

    return IndexArticleResponse(
        title=final["title"],
        summary=final["summary"],
        chunk_count=final["chunk_count"],
    )


def _format_sse(event: dict[str, Any]) -> str:
    phase = event.get("phase", "message")
    return f"event: {phase}\ndata: {json.dumps(event)}\n\n"


@router.post("/article/stream")
async def index_article_stream(
    payload: IndexArticleRequest,
    settings: Settings = Depends(get_settings),
    wiki_http: httpx.AsyncClient = Depends(get_wiki_http),
    llm: LLMClient = Depends(get_llm_client),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    vector_store: VectorStore = Depends(get_vector_store),
) -> StreamingResponse:
    """Streaming variant — emits Server-Sent Events with per-phase progress.

    Errors during the stream are sent as in-band ``error`` events rather
    than HTTP error responses, since the connection is already open.
    """

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in _index_pipeline(
                payload.url,
                settings=settings,
                wiki_http=wiki_http,
                llm=llm,
                embedding_client=embedding_client,
                vector_store=vector_store,
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
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected error in /article/stream")
            yield _format_sse(
                {
                    "phase": "error",
                    "error": "Unexpected server error",
                    "type": type(exc).__name__,
                }
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
