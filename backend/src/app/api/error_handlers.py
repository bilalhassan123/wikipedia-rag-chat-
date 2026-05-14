"""FastAPI exception handlers for typed application errors.

A single table-driven handler maps each :class:`AppError` subclass to the
HTTP status code documented in DESIGN.md §10. Adding a new typed error
means adding a row here; no naked ``except`` ever appears in route code.
"""

from __future__ import annotations

from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import (
    ArticleNotFoundError,
    DisambiguationError,
    InvalidUrlError,
    LLMUnavailableError,
    ThinArticleError,
    VectorStoreUnavailableError,
    WikipediaUnreachableError,
)

STATUS_BY_ERROR: Final[dict[type[Exception], int]] = {
    InvalidUrlError: 400,
    ArticleNotFoundError: 404,
    WikipediaUnreachableError: 502,
    DisambiguationError: 422,
    ThinArticleError: 422,
    LLMUnavailableError: 503,
    VectorStoreUnavailableError: 503,
}


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map a typed :class:`AppError` to a JSON HTTP response."""
    status = STATUS_BY_ERROR.get(type(exc), 500)
    return JSONResponse(status_code=status, content={"error": str(exc)})


def register(app: FastAPI) -> None:
    """Attach :func:`app_error_handler` to every typed error in the table."""
    for exc_type in STATUS_BY_ERROR:
        app.add_exception_handler(exc_type, app_error_handler)
