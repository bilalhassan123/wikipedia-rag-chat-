"""Unit tests for the summariser use case (map-reduce)."""

from app.core.summariser import (
    PARTIAL_SUMMARY_SYSTEM_PROMPT,
    REDUCE_SUMMARY_SYSTEM_PROMPT,
    SUMMARY_MAX_BODY_CHARS,
    SUMMARY_MAX_SEGMENTS,
    SUMMARY_SEGMENT_MAX_CHARS,
    SUMMARY_SYSTEM_PROMPT,
    _split_for_summary,
    summarise_article,
)


class _FakeLLM:
    """Records every generate() call. Returns a sequence of canned responses."""

    def __init__(self, responses: list[str] | str = "stub"):
        self.responses = (
            responses if isinstance(responses, list) else [responses]
        )
        self.calls: list[dict] = []
        self._index = 0

    async def generate(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        self.calls.append(
            {"system": system, "user": user, "max_tokens": max_tokens}
        )
        if self._index < len(self.responses):
            response = self.responses[self._index]
            self._index += 1
        else:
            response = self.responses[-1]
        return response

    async def stream(self, *, system: str, user: str, max_tokens: int = 512):
        # Summariser only uses generate(); this stream stub exists purely
        # to satisfy the LLMClient protocol's runtime_checkable shape.
        yield await self.generate(system=system, user=user, max_tokens=max_tokens)


# --- Constants -------------------------------------------------------------

class TestConstants:
    def test_segment_max_chars_is_12000(self):
        # qwen2.5:3b's 32K context lets us send a much larger segment than
        # phi3:mini could; 12K chars (~3K tokens) leaves comfortable
        # headroom for the system prompt + the answer budget.
        assert SUMMARY_SEGMENT_MAX_CHARS == 12000

    def test_max_segments_is_4(self):
        assert SUMMARY_MAX_SEGMENTS == 4

    def test_max_body_chars_is_segment_times_segments(self):
        assert SUMMARY_MAX_BODY_CHARS == SUMMARY_SEGMENT_MAX_CHARS * SUMMARY_MAX_SEGMENTS


# --- Splitting -------------------------------------------------------------

class TestSplitForSummary:
    def test_empty_input_returns_no_segments(self):
        assert _split_for_summary("") == []
        assert _split_for_summary("   \n\n\t  ") == []

    def test_short_body_yields_one_segment(self):
        body = "A short article body."
        assert _split_for_summary(body) == [body]

    def test_at_threshold_yields_one_segment(self):
        body = "x" * SUMMARY_SEGMENT_MAX_CHARS
        result = _split_for_summary(body)
        assert len(result) == 1

    def test_long_body_yields_multiple_segments(self):
        body = "x" * (SUMMARY_SEGMENT_MAX_CHARS * 2 + 500)
        result = _split_for_summary(body)
        assert 2 <= len(result) <= SUMMARY_MAX_SEGMENTS

    def test_no_segment_exceeds_max(self):
        body = "y" * (SUMMARY_SEGMENT_MAX_CHARS * 5)
        for segment in _split_for_summary(body):
            assert len(segment) <= SUMMARY_SEGMENT_MAX_CHARS

    def test_segment_count_is_capped(self):
        body = "z" * (SUMMARY_SEGMENT_MAX_CHARS * 10)
        result = _split_for_summary(body)
        assert len(result) <= SUMMARY_MAX_SEGMENTS

    def test_overflow_beyond_cap_is_dropped(self):
        body = "z" * (SUMMARY_SEGMENT_MAX_CHARS * 10)
        result = _split_for_summary(body)
        covered = sum(len(s) for s in result)
        assert covered <= SUMMARY_MAX_BODY_CHARS

    def test_split_prefers_paragraph_boundary(self):
        cut_at = int(SUMMARY_SEGMENT_MAX_CHARS * 0.75)
        body = "a" * cut_at + "\n\n" + "b" * SUMMARY_SEGMENT_MAX_CHARS
        result = _split_for_summary(body)
        assert len(result) >= 2
        # First segment ends at the paragraph break (after stripping).
        assert result[0] == "a" * cut_at
        # Second segment starts with the b's.
        assert result[1].startswith("b")


# --- Single-call path ------------------------------------------------------

class TestSingleCallPath:
    async def test_short_body_makes_exactly_one_llm_call(self):
        llm = _FakeLLM(responses=["the summary"])
        result = await summarise_article(
            title="Foo", body="A short body.", llm=llm
        )
        assert result == "the summary"
        assert len(llm.calls) == 1

    async def test_single_call_uses_full_summary_prompt(self):
        llm = _FakeLLM()
        await summarise_article(title="Foo", body="A short body.", llm=llm)
        assert llm.calls[0]["system"] == SUMMARY_SYSTEM_PROMPT
        assert "ARTICLE:" in llm.calls[0]["user"]
        assert "A short body." in llm.calls[0]["user"]

    async def test_default_max_tokens_is_256(self):
        llm = _FakeLLM()
        await summarise_article(title="x", body="y", llm=llm)
        assert llm.calls[0]["max_tokens"] == 256

    async def test_empty_body_returns_empty_string_without_calling_llm(self):
        llm = _FakeLLM()
        result = await summarise_article(title="x", body="", llm=llm)
        assert result == ""
        assert llm.calls == []


# --- Map-reduce path -------------------------------------------------------

class TestMapReducePath:
    async def test_long_body_triggers_map_reduce(self):
        body = "x" * (SUMMARY_SEGMENT_MAX_CHARS * 2)
        llm = _FakeLLM(
            responses=[
                "partial 1",
                "partial 2",
                "final combined summary",
            ]
        )
        result = await summarise_article(title="Foo", body=body, llm=llm)
        # 2 partials + 1 reduce = 3 calls.
        assert len(llm.calls) == 3
        assert result == "final combined summary"

    async def test_each_segment_uses_partial_prompt(self):
        body = "x" * (SUMMARY_SEGMENT_MAX_CHARS * 2)
        llm = _FakeLLM(
            responses=["partial 1", "partial 2", "combined"]
        )
        await summarise_article(title="Foo", body=body, llm=llm)
        # First two calls are the map step.
        for partial_call in llm.calls[:2]:
            assert partial_call["system"] == PARTIAL_SUMMARY_SYSTEM_PROMPT
            assert "SEGMENT:" in partial_call["user"]

    async def test_reduce_call_uses_reduce_prompt_and_includes_partials(self):
        body = "x" * (SUMMARY_SEGMENT_MAX_CHARS * 2)
        llm = _FakeLLM(
            responses=["alpha partial", "beta partial", "combined"]
        )
        await summarise_article(title="Foo", body=body, llm=llm)
        reduce_call = llm.calls[-1]
        assert reduce_call["system"] == REDUCE_SUMMARY_SYSTEM_PROMPT
        assert "alpha partial" in reduce_call["user"]
        assert "beta partial" in reduce_call["user"]
        assert "PART 1:" in reduce_call["user"]
        assert "PART 2:" in reduce_call["user"]

    async def test_segment_count_capped_at_max_segments(self):
        # Body large enough to require more than max segments — extra is
        # dropped, not surfaced as additional LLM calls.
        body = "x" * (SUMMARY_SEGMENT_MAX_CHARS * (SUMMARY_MAX_SEGMENTS + 3))
        llm = _FakeLLM(responses=["p1", "p2", "p3", "p4", "combined"])
        await summarise_article(title="Foo", body=body, llm=llm)
        # SUMMARY_MAX_SEGMENTS partials + 1 reduce.
        assert len(llm.calls) == SUMMARY_MAX_SEGMENTS + 1

    async def test_partial_summaries_use_brief_token_budget(self):
        body = "x" * (SUMMARY_SEGMENT_MAX_CHARS * 2)
        llm = _FakeLLM(responses=["p1", "p2", "combined"])
        await summarise_article(title="Foo", body=body, llm=llm)
        # Partials get a smaller budget than the final reduce.
        for partial_call in llm.calls[:2]:
            assert partial_call["max_tokens"] == 180
        # Reduce uses the user-supplied (default 256).
        assert llm.calls[-1]["max_tokens"] == 256

    async def test_title_propagates_to_every_call(self):
        body = "x" * (SUMMARY_SEGMENT_MAX_CHARS * 2)
        llm = _FakeLLM(responses=["p1", "p2", "combined"])
        await summarise_article(title="Albert Einstein", body=body, llm=llm)
        for call in llm.calls:
            assert "Albert Einstein" in call["user"]


# --- Prompt invariants -----------------------------------------------------

class TestPromptInvariants:
    """Pin the rubric-critical grounding constraints in each prompt."""

    def test_single_call_prompt_pins_grounding(self):
        assert "only information from the article" in SUMMARY_SYSTEM_PROMPT.lower()
        assert "do not invent" in SUMMARY_SYSTEM_PROMPT.lower()
        assert "5 to 8 sentences" in SUMMARY_SYSTEM_PROMPT

    def test_partial_prompt_pins_segment_only_grounding(self):
        assert "this segment" in PARTIAL_SUMMARY_SYSTEM_PROMPT.lower()
        assert "do not invent" in PARTIAL_SUMMARY_SYSTEM_PROMPT.lower()

    def test_reduce_prompt_pins_partials_only_grounding(self):
        assert "partial summaries" in REDUCE_SUMMARY_SYSTEM_PROMPT.lower()
        assert "5 to 8 sentences" in REDUCE_SUMMARY_SYSTEM_PROMPT
