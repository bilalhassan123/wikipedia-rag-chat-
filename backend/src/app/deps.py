"""Dependency-injection providers.

Excluded from coverage (DESIGN.md §12) — this module is framework glue
that wires concrete adapters to the protocols at request time. The
underlying adapters and use-cases are tested directly elsewhere.
"""

from __future__ import annotations

from functools import lru_cache

import httpx
from fastapi import Depends, Request
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import AsyncQdrantClient

from app.adapters.ollama_embedding import OllamaEmbeddingClient
from app.adapters.ollama_llm import OllamaLLMClient
from app.adapters.qdrant_store import QdrantVectorStore
from app.core.interfaces import EmbeddingClient, LLMClient, VectorStore

NOMIC_EMBED_DIM = 768  # nomic-embed-text vector size


class Settings(BaseSettings):
    """Runtime configuration. Reads from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    ollama_host: str = "http://ollama:11434"
    chat_model: str = "qwen2.5:3b"
    embed_model: str = "nomic-embed-text"
    qdrant_host: str = "http://qdrant:6333"
    qdrant_collection: str = "wiki_article"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_ollama_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.ollama_http


def get_wiki_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.wiki_http


def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    return request.app.state.qdrant


def get_llm_client(
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_ollama_http),
) -> LLMClient:
    return OllamaLLMClient(
        base_url=settings.ollama_host,
        model=settings.chat_model,
        http_client=http_client,
    )


def get_embedding_client(
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_ollama_http),
) -> EmbeddingClient:
    return OllamaEmbeddingClient(
        base_url=settings.ollama_host,
        model=settings.embed_model,
        http_client=http_client,
    )


def get_vector_store(
    settings: Settings = Depends(get_settings),
    qdrant: AsyncQdrantClient = Depends(get_qdrant_client),
) -> VectorStore:
    return QdrantVectorStore(
        client=qdrant,
        collection_name=settings.qdrant_collection,
        vector_size=NOMIC_EMBED_DIM,
    )
