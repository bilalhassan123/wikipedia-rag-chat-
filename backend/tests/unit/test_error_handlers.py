"""Tests for FastAPI exception handlers.

Each typed application error must map to the HTTP status code documented
in DESIGN.md §10 and surface a JSON body containing the error message.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.error_handlers import STATUS_BY_ERROR, register
from app.core.errors import (
    ArticleNotFoundError,
    DisambiguationError,
    InvalidUrlError,
    LLMUnavailableError,
    ThinArticleError,
    VectorStoreUnavailableError,
    WikipediaUnreachableError,
)


def _client_with_raising_route(exc: Exception) -> TestClient:
    app = FastAPI()
    register(app)

    @app.get("/raise")
    def _route():
        raise exc

    return TestClient(app)


@pytest.mark.parametrize(
    "exc, expected_status, expected_message",
    [
        (InvalidUrlError("bad url"), 400, "bad url"),
        (ArticleNotFoundError("no such article"), 404, "no such article"),
        (WikipediaUnreachableError("wiki down"), 502, "wiki down"),
        (DisambiguationError("disambig page"), 422, "disambig page"),
        (ThinArticleError("too short"), 422, "too short"),
        (LLMUnavailableError("llm not ready"), 503, "llm not ready"),
        (VectorStoreUnavailableError("qdrant down"), 503, "qdrant down"),
    ],
)
def test_each_typed_error_maps_to_documented_status(
    exc, expected_status, expected_message
):
    client = _client_with_raising_route(exc)
    response = client.get("/raise")
    assert response.status_code == expected_status
    body = response.json()
    assert body == {"error": expected_message}


def test_status_table_matches_design_doc():
    """Pin the status mapping so any drift from DESIGN.md §10 fails a test."""
    assert STATUS_BY_ERROR == {
        InvalidUrlError: 400,
        ArticleNotFoundError: 404,
        WikipediaUnreachableError: 502,
        DisambiguationError: 422,
        ThinArticleError: 422,
        LLMUnavailableError: 503,
        VectorStoreUnavailableError: 503,
    }


def test_register_attaches_handler_for_every_typed_error():
    """Every key in the status table is wired up by :func:`register`."""
    app = FastAPI()
    register(app)
    registered_types = set(app.exception_handlers.keys())
    for exc_type in STATUS_BY_ERROR:
        assert exc_type in registered_types
