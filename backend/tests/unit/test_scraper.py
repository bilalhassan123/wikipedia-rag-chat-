"""Unit tests for the scraper.

Network is mocked via ``httpx.MockTransport`` and HTML fixtures live under
``tests/fixtures/wikipedia/``. No real Wikipedia calls.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import httpx
import pytest

from app.core.errors import (
    ArticleNotFoundError,
    DisambiguationError,
    InvalidUrlError,
    ThinArticleError,
    WikipediaUnreachableError,
)
from app.core.scraper import (
    THIN_ARTICLE_THRESHOLD,
    ScrapedArticle,
    fetch_article_html,
    parse_article_html,
    scrape,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "wikipedia"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _mock_client(*, status: int, body: str = "") -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` that replies with the same status/body for every request."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- URL validation ---------------------------------------------------------

class TestUrlValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "",
            "   ",
            "not a url at all",
            "ftp://en.wikipedia.org/wiki/Foo",
            "https://www.google.com/",
            "https://de.wikipedia.org/wiki/Foo",
            "https://wikipedia.org/wiki/Foo",
            "https://en.wikipedia.org/",
            "https://en.wikipedia.org/Article",
            "https://en.wikipedia.org/wiki/",
        ],
    )
    async def test_invalid_url_rejected(self, url):
        async with _mock_client(status=200, body="<html></html>") as client:
            with pytest.raises(InvalidUrlError):
                await fetch_article_html(url, client=client)

    @pytest.mark.parametrize(
        "url",
        [
            "https://en.wikipedia.org/wiki/Albert_Einstein",
            "https://en.m.wikipedia.org/wiki/Albert_Einstein",
            "http://en.wikipedia.org/wiki/Albert_Einstein",
        ],
    )
    async def test_valid_urls_accepted(self, url):
        html = _load("canonical.html")
        async with _mock_client(status=200, body=html) as client:
            title, _ = await fetch_article_html(url, client=client)
            assert title == "Albert Einstein"


# --- HTML parsing -----------------------------------------------------------

class TestHtmlParsing:
    def test_canonical_article_parses(self):
        html = _load("canonical.html")
        result = parse_article_html(html, title="Albert Einstein")
        assert isinstance(result, ScrapedArticle)
        assert result.title == "Albert Einstein"
        assert "German-born theoretical physicist" in result.body_text
        assert "Early life" in result.body_text
        assert "Career" in result.body_text

    def test_canonical_article_drops_chrome(self):
        html = _load("canonical.html")
        result = parse_article_html(html, title="Albert Einstein")
        # Each chrome block in the fixture carries a tagged sentinel string;
        # none of them should survive the cleaner.
        assert "INFOBOX_CONTENT" not in result.body_text
        assert "REFERENCE_TEXT" not in result.body_text
        assert "NAVBOX_CONTENT" not in result.body_text
        assert "FIGURE_CAPTION" not in result.body_text

    def test_disambiguation_page_raises(self):
        html = _load("disambiguation.html")
        with pytest.raises(DisambiguationError):
            parse_article_html(html, title="Mercury")

    def test_thin_article_raises(self):
        html = _load("stub.html")
        with pytest.raises(ThinArticleError) as excinfo:
            parse_article_html(html, title="Tiny Stub")
        assert str(THIN_ARTICLE_THRESHOLD) in str(excinfo.value)

    def test_scraped_article_is_frozen(self):
        html = _load("canonical.html")
        article = parse_article_html(html, title="Albert Einstein")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            article.title = "mutated"  # type: ignore[misc]


# --- Fetch with mock transport ----------------------------------------------

class TestFetch:
    async def test_404_raises_article_not_found(self):
        async with _mock_client(status=404) as client:
            with pytest.raises(ArticleNotFoundError):
                await fetch_article_html(
                    "https://en.wikipedia.org/wiki/Nonexistent", client=client
                )

    async def test_500_raises_wikipedia_unreachable(self):
        async with _mock_client(status=500) as client:
            with pytest.raises(WikipediaUnreachableError):
                await fetch_article_html(
                    "https://en.wikipedia.org/wiki/Foo", client=client
                )

    async def test_unexpected_status_raises_wikipedia_unreachable(self):
        async with _mock_client(status=403) as client:
            with pytest.raises(WikipediaUnreachableError):
                await fetch_article_html(
                    "https://en.wikipedia.org/wiki/Foo", client=client
                )

    async def test_network_error_raises_wikipedia_unreachable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(WikipediaUnreachableError):
                await fetch_article_html(
                    "https://en.wikipedia.org/wiki/Foo", client=client
                )

    async def test_canonical_url_returns_title_and_html(self):
        html = _load("canonical.html")
        async with _mock_client(status=200, body=html) as client:
            title, returned = await fetch_article_html(
                "https://en.wikipedia.org/wiki/Albert_Einstein", client=client
            )
        assert title == "Albert Einstein"
        assert returned == html

    async def test_redirect_resolves_canonical_title(self):
        """A redirect from /Einstein to /Albert_Einstein should surface the
        canonical title in the result."""
        html = _load("canonical.html")
        canonical_url = (
            "https://en.wikipedia.org/api/rest_v1/page/html/Albert_Einstein"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/Einstein"):
                return httpx.Response(
                    301,
                    headers={"Location": canonical_url},
                    request=request,
                )
            return httpx.Response(200, text=html, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            title, _ = await fetch_article_html(
                "https://en.wikipedia.org/wiki/Einstein", client=client
            )
        assert title == "Albert Einstein"


# --- End-to-end scrape ------------------------------------------------------

class TestScrapeEndToEnd:
    async def test_canonical_url_yields_scraped_article(self):
        html = _load("canonical.html")
        async with _mock_client(status=200, body=html) as client:
            article = await scrape(
                "https://en.wikipedia.org/wiki/Albert_Einstein", client=client
            )
        assert article.title == "Albert Einstein"
        assert "German-born theoretical physicist" in article.body_text
        assert len(article.body_text) >= THIN_ARTICLE_THRESHOLD

    async def test_disambiguation_url_raises(self):
        html = _load("disambiguation.html")
        async with _mock_client(status=200, body=html) as client:
            with pytest.raises(DisambiguationError):
                await scrape(
                    "https://en.wikipedia.org/wiki/Mercury", client=client
                )

    async def test_thin_article_url_raises(self):
        html = _load("stub.html")
        async with _mock_client(status=200, body=html) as client:
            with pytest.raises(ThinArticleError):
                await scrape(
                    "https://en.wikipedia.org/wiki/Tiny", client=client
                )

    async def test_404_url_raises(self):
        async with _mock_client(status=404) as client:
            with pytest.raises(ArticleNotFoundError):
                await scrape(
                    "https://en.wikipedia.org/wiki/DoesNotExist", client=client
                )
