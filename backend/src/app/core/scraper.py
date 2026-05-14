"""Wikipedia article scraper.

Resolves an English Wikipedia URL to clean article body text via the
Wikipedia REST API (Parsoid HTML), dropping infoboxes, references,
navigation chrome, and figure captions. See DESIGN.md §6 for the
end-to-end data flow and §10 for error mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from app.core.errors import (
    ArticleNotFoundError,
    DisambiguationError,
    InvalidUrlError,
    ThinArticleError,
    WikipediaUnreachableError,
)

WIKIPEDIA_HOSTS = frozenset({"en.wikipedia.org", "en.m.wikipedia.org"})
REST_API_BASE = "https://en.wikipedia.org/api/rest_v1/page/html"
USER_AGENT = "WikipediaRagChat/0.1 (educational; contact: example@example.com)"
THIN_ARTICLE_THRESHOLD = 200

_DROP_SELECTORS: tuple[str, ...] = (
    "table.infobox",
    "table.metadata",
    "table.mbox-small",
    "table.navbox",
    "div.navbox",
    "div.hatnote",
    "div.thumb",
    "ol.references",
    "div.reflist",
    "div.mw-references-wrap",
    "sup.reference",
    "sup.noprint",
    "figure",
    "style",
    "script",
)
_CONTENT_SELECTOR = "p, h1, h2, h3, h4, h5, h6"
_REST_API_PATH_PREFIX = "/api/rest_v1/page/html/"


@dataclass(frozen=True, slots=True)
class ScrapedArticle:
    """A cleaned Wikipedia article ready for chunking."""

    title: str
    body_text: str


def _extract_title_from_url(url: str) -> str:
    """Validate that *url* is an English Wikipedia article URL and return
    the (still URL-encoded) title segment."""
    if not isinstance(url, str) or not url.strip():
        raise InvalidUrlError("URL must be a non-empty string.")
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise InvalidUrlError(
            f"URL must use http or https scheme, got: {parsed.scheme!r}"
        )
    if parsed.hostname not in WIKIPEDIA_HOSTS:
        raise InvalidUrlError(
            "Only English Wikipedia URLs are supported "
            f"(en.wikipedia.org, en.m.wikipedia.org). Got host: {parsed.hostname!r}"
        )
    if not parsed.path.startswith("/wiki/"):
        raise InvalidUrlError(
            f"URL path must be of the form /wiki/<title>. Got: {parsed.path!r}"
        )
    title = parsed.path[len("/wiki/") :].strip("/")
    if not title:
        raise InvalidUrlError("URL is missing an article title.")
    return title


async def fetch_article_html(
    url: str, *, client: httpx.AsyncClient
) -> tuple[str, str]:
    """Fetch the Parsoid HTML for *url* via the Wikipedia REST API.

    Returns ``(resolved_title, html)`` where ``resolved_title`` is in
    human-readable form (underscores replaced with spaces, percent-encoding
    decoded). Wikipedia redirects are followed transparently and the
    canonical title is recovered from the final URL when possible.
    """
    title = _extract_title_from_url(url)
    api_url = f"{REST_API_BASE}/{title}"
    try:
        response = await client.get(
            api_url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        raise WikipediaUnreachableError(
            f"Network error fetching {api_url}: {exc}"
        ) from exc

    if response.status_code == 404:
        raise ArticleNotFoundError(
            f"Wikipedia returned 404 for article: "
            f"{unquote(title).replace('_', ' ')}"
        )
    if response.status_code != 200:
        raise WikipediaUnreachableError(
            f"Wikipedia returned status {response.status_code} for {title!r}"
        )

    final_path = urlparse(str(response.url)).path
    if final_path.startswith(_REST_API_PATH_PREFIX):
        title = final_path[len(_REST_API_PATH_PREFIX) :]

    return unquote(title).replace("_", " "), response.text


def parse_article_html(html: str, *, title: str) -> ScrapedArticle:
    """Parse Parsoid HTML and return a cleaned :class:`ScrapedArticle`.

    Drops chrome (infoboxes, references, navboxes, captions, footnotes),
    detects disambiguation pages, and enforces the thin-article threshold.
    """
    soup = BeautifulSoup(html, "html.parser")

    if soup.find("link", attrs={"rel": "mw:PageProp/disambiguation"}) or soup.find(
        "meta", attrs={"property": "mw:PageProp/disambiguation"}
    ):
        raise DisambiguationError(
            f"{title} is a disambiguation page; pick a specific article."
        )

    for selector in _DROP_SELECTORS:
        for el in soup.select(selector):
            el.decompose()

    body = soup.body or soup
    parts: list[str] = []
    for el in body.select(_CONTENT_SELECTOR):
        text = el.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)

    body_text = "\n\n".join(parts).strip()

    if len(body_text) < THIN_ARTICLE_THRESHOLD:
        raise ThinArticleError(
            f"Article body is only {len(body_text)} characters; "
            f"minimum is {THIN_ARTICLE_THRESHOLD}."
        )

    return ScrapedArticle(title=title, body_text=body_text)


async def scrape(url: str, *, client: httpx.AsyncClient) -> ScrapedArticle:
    """Fetch and parse an English Wikipedia article from *url*."""
    title, html = await fetch_article_html(url, client=client)
    return parse_article_html(html, title=title)
