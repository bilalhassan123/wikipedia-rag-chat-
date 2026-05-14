"""Typed exceptions raised by the domain layer.

Each subclass maps 1:1 to an HTTP response in the FastAPI exception
handlers (see DESIGN.md §10 and :mod:`app.api.error_handlers`). Adding a
new failure mode means adding a class here AND a row in the handler's
status table — never a naked ``except``.
"""


class AppError(Exception):
    """Base class for all typed application errors."""


class InvalidUrlError(AppError):
    """The URL is malformed, non-Wikipedia, or otherwise unsupported (HTTP 400)."""


class ArticleNotFoundError(AppError):
    """Wikipedia returned a 404 for the requested article (HTTP 404)."""


class WikipediaUnreachableError(AppError):
    """Wikipedia could not be reached: network failure or 5xx (HTTP 502)."""


class DisambiguationError(AppError):
    """The URL resolves to a disambiguation page, not a real article (HTTP 422)."""


class ThinArticleError(AppError):
    """The scraped article body is below the minimum length to chat with (HTTP 422)."""


class LLMUnavailableError(AppError):
    """The local LLM runtime is not reachable or not ready (HTTP 503)."""


class VectorStoreUnavailableError(AppError):
    """The vector store is not reachable or not ready (HTTP 503)."""
