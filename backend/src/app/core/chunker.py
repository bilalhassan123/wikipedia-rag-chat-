"""Chunker: break article body text into overlapping chunks for retrieval.

The strategy is a sliding window with a backward search for a "good" break
point in the second half of the window. We prefer breaking on paragraph
boundaries first, then newlines, then sentence ends, then word boundaries,
and only as a last resort hard-split mid-word. See DESIGN.md §8.
"""

from __future__ import annotations

import hashlib

from app.core.interfaces import Chunk

# Locked defaults — see DESIGN §8. Overridable per call but the build
# does not turn these knobs.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Separator priority: paragraph → newline → sentence → word.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping pieces no longer than *chunk_size*.

    Each piece (after the first) starts ``overlap`` characters before the
    end of the previous piece, so a sentence that crosses a chunk boundary
    is not orphaned.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        if end == n:
            chunks.append(text[start:end])
            break

        # Look for the latest "good" break point in the second half of the
        # window. This guarantees chunks are at least chunk_size // 2 long
        # while still preferring semantically clean boundaries.
        chunk_end = end
        search_floor = start + chunk_size // 2
        for sep in _SEPARATORS:
            idx = text.rfind(sep, search_floor, end)
            if idx != -1:
                chunk_end = idx + len(sep)
                break

        chunks.append(text[start:chunk_end])
        # Advance with overlap; the max() guards against infinite loops on
        # degenerate (chunk_size, overlap) combinations.
        start = max(chunk_end - overlap, start + 1)

    return chunks


def _chunk_id(article_title: str, chunk_index: int) -> str:
    """Stable id derived from (article_title, chunk_index).

    16 hex chars of sha256 — 64 bits, plenty for collision avoidance within
    one article and across re-indexes.
    """
    raw = f"{article_title}::{chunk_index}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def chunk_article(
    *,
    text: str,
    article_title: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk an article body into ``Chunk`` records ready for embedding.

    The chunker is a pure function: given the same arguments it produces
    the same chunks with the same ids. Empty / whitespace-only input
    yields an empty list.
    """
    pieces = _split_text(text, chunk_size, overlap)
    return [
        Chunk(
            id=_chunk_id(article_title, i),
            text=piece,
            chunk_index=i,
            article_title=article_title,
        )
        for i, piece in enumerate(pieces)
    ]
