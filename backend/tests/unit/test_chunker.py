"""Unit tests for the chunker."""

import pytest

from app.core.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_article
from app.core.interfaces import Chunk


class TestDefaults:
    def test_chunk_size_is_locked_to_800(self):
        assert CHUNK_SIZE == 800

    def test_chunk_overlap_is_locked_to_120(self):
        assert CHUNK_OVERLAP == 120


class TestEdgeCases:
    def test_empty_text_returns_no_chunks(self):
        assert chunk_article(text="", article_title="Foo") == []

    def test_whitespace_only_text_returns_no_chunks(self):
        assert chunk_article(text="   \n\n\t  ", article_title="Foo") == []

    def test_short_text_yields_single_chunk(self):
        text = "Hello world."
        chunks = chunk_article(text=text, article_title="Foo")
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_exactly_chunk_size_yields_single_chunk(self):
        text = "x" * CHUNK_SIZE
        chunks = chunk_article(text=text, article_title="Foo")
        assert len(chunks) == 1
        assert chunks[0].text == text


class TestStructure:
    def test_chunks_are_chunk_dataclass_instances(self):
        chunks = chunk_article(text="hello", article_title="Foo")
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_indices_are_sequential_from_zero(self):
        chunks = chunk_article(text="x" * 3000, article_title="Foo")
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_article_title_propagates_to_every_chunk(self):
        chunks = chunk_article(text="x" * 3000, article_title="Albert Einstein")
        assert all(c.article_title == "Albert Einstein" for c in chunks)

    def test_chunks_never_exceed_chunk_size(self):
        chunks = chunk_article(text="x" * 5000, article_title="Foo")
        assert all(len(c.text) <= CHUNK_SIZE for c in chunks)


class TestOverlap:
    def test_consecutive_chunks_share_exact_overlap_with_no_separators(self):
        # No separators of any kind → hard char split, exact overlap.
        text = "x" * 2400
        chunks = chunk_article(text=text, article_title="Foo")
        assert len(chunks) >= 2
        for i in range(1, len(chunks)):
            assert chunks[i].text[:CHUNK_OVERLAP] == chunks[i - 1].text[-CHUNK_OVERLAP:]

    def test_long_text_yields_multiple_chunks(self):
        chunks = chunk_article(text="abc " * 1000, article_title="Foo")
        assert len(chunks) >= 2


class TestBoundaryPreference:
    def test_paragraph_boundaries_are_preferred(self):
        # Each paragraph is 700 chars; without paragraph-awareness the
        # splitter would land mid-paragraph at char 800.
        para = "x" * 700
        text = (para + "\n\n") * 3
        chunks = chunk_article(text=text, article_title="Foo")
        assert chunks[0].text.endswith("\n\n")

    def test_falls_back_to_word_boundary_without_paragraphs(self):
        text = "word " * 400  # 2000 chars, only " " as separator
        chunks = chunk_article(text=text, article_title="Foo")
        for c in chunks[:-1]:
            assert c.text.endswith(" ")


class TestStableIds:
    def test_same_input_yields_same_ids(self):
        a = chunk_article(text="x" * 1500, article_title="Foo")
        b = chunk_article(text="x" * 1500, article_title="Foo")
        assert [c.id for c in a] == [c.id for c in b]

    def test_different_article_title_changes_ids(self):
        a = chunk_article(text="x" * 1500, article_title="Foo")
        b = chunk_article(text="x" * 1500, article_title="Bar")
        assert [c.id for c in a] != [c.id for c in b]

    def test_ids_within_one_article_are_unique(self):
        chunks = chunk_article(text="x" * 5000, article_title="Foo")
        ids = [c.id for c in chunks]
        assert len(set(ids)) == len(ids)


class TestConfigurability:
    def test_custom_chunk_size_respected(self):
        chunks = chunk_article(
            text="x" * 1000, article_title="Foo", chunk_size=200, overlap=20
        )
        assert all(len(c.text) <= 200 for c in chunks)

    def test_custom_overlap_respected(self):
        chunks = chunk_article(
            text="x" * 1000, article_title="Foo", chunk_size=200, overlap=20
        )
        assert len(chunks) >= 2
        for i in range(1, len(chunks)):
            assert chunks[i].text[:20] == chunks[i - 1].text[-20:]

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_article(text="abc", article_title="Foo", chunk_size=100, overlap=100)
        with pytest.raises(ValueError):
            chunk_article(text="abc", article_title="Foo", chunk_size=100, overlap=200)

    def test_default_args_match_locked_constants(self):
        text = "x" * 3000
        a = chunk_article(text=text, article_title="Foo")
        b = chunk_article(
            text=text,
            article_title="Foo",
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
        assert [(c.id, c.text) for c in a] == [(c.id, c.text) for c in b]
