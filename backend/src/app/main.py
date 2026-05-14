"""FastAPI application entrypoint and runtime wiring.

Excluded from coverage (DESIGN.md §12) — this module is the ASGI
entrypoint: it constructs the app, wires routes, registers the typed
exception handlers, and manages the lifecycle of the shared HTTP and
Qdrant clients.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from app.api import error_handlers
from app.api.routes_article import router as article_router
from app.api.routes_chat import router as chat_router
from app.deps import get_settings

# Surface our INFO logs. uvicorn configures uvicorn.* loggers but leaves the
# root logger alone, so without this our `app.*` logs get dropped.
_app_logger = logging.getLogger("app")
if not _app_logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    )
    _app_logger.addHandler(_handler)
    _app_logger.setLevel(logging.INFO)
    _app_logger.propagate = False

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "[startup] OLLAMA_HOST=%s QDRANT_HOST=%s CHAT_MODEL=%s EMBED_MODEL=%s "
        "CHUNK_SIZE=%d CHUNK_OVERLAP=%d TOP_K=%d",
        settings.ollama_host, settings.qdrant_host,
        settings.chat_model, settings.embed_model,
        settings.chunk_size, settings.chunk_overlap, settings.top_k,
    )
    # One Ollama HTTP client shared by both Ollama adapters; one separate
    # client for Wikipedia (different host, different timeout).
    app.state.ollama_http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    app.state.wiki_http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    # check_compatibility=False silences the client/server version-mismatch
    # warning. Our operations (recreate_collection, upsert, search) are
    # stable across the client/server minor-version skew we're seeing.
    app.state.qdrant = AsyncQdrantClient(
        url=settings.qdrant_host, check_compatibility=False
    )
    try:
        yield
    finally:
        await app.state.ollama_http.aclose()
        await app.state.wiki_http.aclose()
        await app.state.qdrant.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Wikipedia RAG Chat", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    error_handlers.register(app)
    app.include_router(article_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
