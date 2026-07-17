"""Unit tests for src/utils.py (spec Section 6) and, once implemented, the
retrieval / generation pieces of the pipeline.

Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    char_to_page,
    deterministic_id,
    generate_id,
    recursive_split,
    split_sentences,
)

LOREM = (
    "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu."
)

PARAGRAPHS = "\n\n".join([LOREM] * 5)


# --- recursive_split: chunk size limits ---


def test_recursive_split_respects_max_chunk_size():
    chunks = recursive_split(PARAGRAPHS, chunk_size=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 100


def test_recursive_split_prefers_paragraph_boundaries():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = recursive_split(text, chunk_size=20)
    # each paragraph should stay intact (not split mid-sentence) since it
    # fits well within the chunk size
    assert any(c.strip() == "Para one." for c in chunks)


def test_recursive_split_falls_back_to_sentence_boundary():
    # one long "paragraph" (no \n\n) made of short sentences
    text = "One. Two. Three. Four. Five. Six. Seven. Eight."
    chunks = recursive_split(text, chunk_size=15)
    for c in chunks:
        assert len(c) <= 15
    # The greedy packer merges multiple short sentences into one chunk
    # when they fit (e.g. "One. Two. " both fit in 15 chars) rather than
    # wasting capacity with one sentence per chunk -- so the invariant to
    # check isn't "each sentence is its own chunk", it's "no chunk ever
    # cuts a sentence mid-word".
    assert "".join(chunks) == text
    for c in chunks:
        assert c.rstrip().endswith(".")


def test_recursive_split_hard_cut_for_unsplittable_text():
    text = "x" * 500  # no paragraph or sentence boundaries at all
    chunks = recursive_split(text, chunk_size=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_recursive_split_no_content_lost_without_overlap():
    chunks = recursive_split(PARAGRAPHS, chunk_size=120)
    assert "".join(chunks) == PARAGRAPHS


def test_recursive_split_empty_text():
    assert recursive_split("", chunk_size=100) == []


def test_recursive_split_short_text_single_chunk():
    text = "Just a short sentence."
    chunks = recursive_split(text, chunk_size=1000)
    assert chunks == [text]


# --- recursive_split: overlap behavior ---


def test_recursive_split_overlap_respects_max_chunk_size():
    chunks = recursive_split(PARAGRAPHS, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 100


def test_recursive_split_overlap_carries_trailing_context():
    chunks = recursive_split("x" * 500, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-20:]
        assert chunks[i].startswith(prev_tail)


def test_recursive_split_overlap_loses_no_content():
    # every character of the source must appear in at least one chunk,
    # i.e. stitching chunks back together (accounting for overlap) covers
    # the whole document with no gaps.
    text = "x" * 900
    overlap = 30
    chunk_size = 100
    chunks = recursive_split(text, chunk_size=chunk_size, overlap=overlap)
    stitched = chunks[0]
    for c in chunks[1:]:
        stitched += c[overlap:]
    assert stitched == text


def test_recursive_split_invalid_overlap_rejected():
    import pytest

    with pytest.raises(ValueError):
        recursive_split("hello world", chunk_size=10, overlap=10)


# --- deterministic_id ---


def test_deterministic_id_stable_for_same_input():
    a = deterministic_id("doc.pdf", 0, "some text")
    b = deterministic_id("doc.pdf", 0, "some text")
    assert a == b


def test_deterministic_id_varies_with_doc_name():
    a = deterministic_id("doc_a.pdf", 0, "some text")
    b = deterministic_id("doc_b.pdf", 0, "some text")
    assert a != b


def test_deterministic_id_varies_with_index():
    a = deterministic_id("doc.pdf", 0, "some text")
    b = deterministic_id("doc.pdf", 1, "some text")
    assert a != b


def test_deterministic_id_varies_with_text():
    a = deterministic_id("doc.pdf", 0, "some text")
    b = deterministic_id("doc.pdf", 0, "other text")
    assert a != b


# --- generate_id ---


def test_generate_id_is_hex_uuid():
    gid = generate_id()
    assert len(gid) == 32
    int(gid, 16)  # raises if not valid hex


def test_generate_id_is_random():
    assert generate_id() != generate_id()


# --- split_sentences ---


def test_split_sentences_basic():
    sentences = split_sentences("One. Two? Three!")
    assert sentences == ["One.", "Two?", "Three!"]


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


# --- char_to_page ---


def test_char_to_page_resolves_correct_page():
    page_map = [(0, 1), (100, 2), (250, 3)]
    assert char_to_page(0, page_map) == 1
    assert char_to_page(50, page_map) == 1
    assert char_to_page(100, page_map) == 2
    assert char_to_page(249, page_map) == 2
    assert char_to_page(300, page_map) == 3


def test_char_to_page_empty_map_defaults_to_one():
    assert char_to_page(0, []) == 1
