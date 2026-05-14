"""Summarisation use case.

For short articles we issue a single LLM call. For long articles we run
**map-reduce**: split the body into segments that each fit in
``phi3:mini``'s 4K-token context, summarise each segment in its own
call, then ask the model to combine the partial summaries into a single
coherent 5–8 sentence summary. This trades latency for coverage —
versus the prior naive truncation, we now cover up to four times as
much of the article (capped at ``SUMMARY_SEGMENT_MAX_CHARS *
SUMMARY_MAX_SEGMENTS`` chars total).

See DESIGN.md §7 for the prompt designs and §13 for the honest note on
phi3:mini's limits and the latency cost of this approach.
"""

from __future__ import annotations

import logging

from app.core.interfaces import LLMClient

logger = logging.getLogger(__name__)

# Per-segment input cap. qwen2.5:3b has a 32K-token (~128K char) window;
# 12 K chars of input (~3 K tokens) plus a ~150-token partial summary
# plus the system prompt comfortably fits with room for the model's
# working state. Most Wikipedia articles fit in a single segment, so
# the map-reduce path only kicks in for genuinely long articles.
SUMMARY_SEGMENT_MAX_CHARS = 12000

# Bound how many segments we emit, so the worst case stays interactive
# on CPU. With 4 segments + 1 reduce call, total LLM calls = 5.
SUMMARY_MAX_SEGMENTS = 4

# Convenience: the maximum body coverage of the map-reduce summary.
SUMMARY_MAX_BODY_CHARS = SUMMARY_SEGMENT_MAX_CHARS * SUMMARY_MAX_SEGMENTS

# --- Prompts ---------------------------------------------------------------

# Single-shot summary (used when body fits in one segment).
SUMMARY_SYSTEM_PROMPT = (
    "You are a precise summariser. Summarise the Wikipedia article below "
    "in 5 to 8 sentences. Use only information from the article. Do not "
    "include facts you may know from elsewhere. Do not invent details. "
    "Output only the summary, no preamble, no headings."
)

# Map step: summarise a single segment briefly.
PARTIAL_SUMMARY_SYSTEM_PROMPT = (
    "You are summarising one part of a longer Wikipedia article. In 2 to "
    "3 sentences, capture the key facts from THIS SEGMENT only. Use only "
    "information present in the segment. Do not invent. Output only the "
    "partial summary, no preamble."
)

# Reduce step: combine partial summaries into a single coherent summary.
REDUCE_SUMMARY_SYSTEM_PROMPT = (
    "You are combining several partial summaries of one Wikipedia article "
    "into a single coherent summary. Merge them into 5 to 8 sentences "
    "that read as one summary, removing redundancy and contradictions. "
    "Use only information from the partial summaries. Output only the "
    "final summary, no preamble."
)

SUMMARY_USER_TEMPLATE = "ARTICLE TITLE: {title}\n\nARTICLE:\n{body}"
PARTIAL_USER_TEMPLATE = "ARTICLE TITLE: {title}\n\nSEGMENT:\n{body}"


# --- Splitting -------------------------------------------------------------

def _split_for_summary(
    body: str,
    *,
    segment_max_chars: int = SUMMARY_SEGMENT_MAX_CHARS,
    max_segments: int = SUMMARY_MAX_SEGMENTS,
) -> list[str]:
    """Split *body* into up to *max_segments* segments of <= *segment_max_chars*.

    Cuts prefer paragraph (``\\n\\n``) boundaries, then newlines, then
    word boundaries; falls back to a hard cut. Anything past
    ``max_segments * segment_max_chars`` is dropped — covered in
    :data:`SUMMARY_MAX_BODY_CHARS` and DESIGN.md §13.
    """
    body = body.strip()
    if not body:
        return []
    if len(body) <= segment_max_chars:
        return [body]

    segments: list[str] = []
    start = 0
    while start < len(body) and len(segments) < max_segments:
        end = min(start + segment_max_chars, len(body))
        if end < len(body):
            # Look for a clean break in the second half of the window.
            search_floor = start + segment_max_chars // 2
            cut = -1
            for sep in ("\n\n", "\n", ". ", " "):
                cut = body.rfind(sep, search_floor, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        segment = body[start:end].strip()
        if segment:
            segments.append(segment)
        start = end

    return segments


# --- LLM call helpers ------------------------------------------------------

async def _llm_call(
    *,
    llm: LLMClient,
    system: str,
    user: str,
    max_tokens: int,
) -> str:
    return await llm.generate(system=system, user=user, max_tokens=max_tokens)


# --- Public API ------------------------------------------------------------

async def summarise_article(
    *,
    title: str,
    body: str,
    llm: LLMClient,
    max_tokens: int = 256,
) -> str:
    """Produce a 5–8 sentence summary of *body* using *llm*.

    Short bodies use a single LLM call. Long bodies run map-reduce: a
    partial summary per segment, then one reduce call to combine.
    """
    segments = _split_for_summary(body)
    if not segments:
        return ""

    if len(segments) == 1:
        logger.info(
            "[summariser] single-call summary (%d chars)", len(segments[0])
        )
        return await _llm_call(
            llm=llm,
            system=SUMMARY_SYSTEM_PROMPT,
            user=SUMMARY_USER_TEMPLATE.format(title=title, body=segments[0]),
            max_tokens=max_tokens,
        )

    logger.info(
        "[summariser] map-reduce: %d segments (%d chars covered, %d chars dropped)",
        len(segments),
        sum(len(s) for s in segments),
        max(0, len(body.strip()) - sum(len(s) for s in segments)),
    )

    # Map: one brief partial summary per segment.
    partials: list[str] = []
    for i, segment in enumerate(segments, start=1):
        logger.info(
            "[summariser] segment %d/%d (%d chars)",
            i, len(segments), len(segment),
        )
        partial = await _llm_call(
            llm=llm,
            system=PARTIAL_SUMMARY_SYSTEM_PROMPT,
            user=PARTIAL_USER_TEMPLATE.format(title=title, body=segment),
            max_tokens=180,
        )
        partials.append(partial.strip())

    # Reduce: combine partials into one coherent summary.
    logger.info(
        "[summariser] combining %d partial summaries", len(partials)
    )
    combined = "\n\n".join(
        f"PART {i}:\n{p}" for i, p in enumerate(partials, start=1)
    )
    reduce_user = f"ARTICLE TITLE: {title}\n\n{combined}"
    return await _llm_call(
        llm=llm,
        system=REDUCE_SUMMARY_SYSTEM_PROMPT,
        user=reduce_user,
        max_tokens=max_tokens,
    )
