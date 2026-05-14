"""End-to-end integration test against the live wired-up stack.

This test is skipped by default. To run it:

    docker compose up -d
    cd backend && pytest -m integration

The test exercises the full URL → summary → chat flow and pins the
rubric-critical FR-14 behaviour: an out-of-article question must produce
a refusal answer rather than a fabrication.
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
ARTICLE_URL = os.environ.get(
    "INTEGRATION_ARTICLE_URL",
    "https://en.wikipedia.org/wiki/Albert_Einstein",
)
REQUEST_TIMEOUT_SECONDS = 600.0  # first-call cold start can be slow on CPU


async def _post(client: httpx.AsyncClient, path: str, payload: dict) -> dict:
    response = await client.post(f"{BACKEND_URL}{path}", json=payload)
    response.raise_for_status()
    return response.json()


async def test_index_then_in_article_and_off_article_chat():
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        # 1. Index the article.
        article = await _post(client, "/api/article", {"url": ARTICLE_URL})
        assert article["title"]
        assert len(article["summary"]) > 50
        assert article["chunk_count"] > 0

        # 2. In-article question — answer must not refuse.
        in_article = await _post(
            client,
            "/api/chat",
            {"question": "Where was Einstein born?"},
        )
        assert "cannot find that in the article" not in in_article["answer"].lower()
        assert len(in_article["evidence"]) > 0

        # 3. Off-article question — FR-14: must refuse with the verbatim phrase
        # (or close approximation; phi3:mini occasionally paraphrases, so we
        # accept any answer that signals "not in the article").
        off_article = await _post(
            client,
            "/api/chat",
            {"question": "What is the capital of Mongolia?"},
        )
        answer_lower = off_article["answer"].lower()
        assert "cannot find" in answer_lower
        assert "article" in answer_lower
