"""Shared low-level helpers: text splitting, id generation, sentence
splitting, and character-offset -> page-number resolution.
"""

from __future__ import annotations

import hashlib
import re
import uuid

Chunk = tuple[str, int]  # (chunk_text, start_offset_in_original_text)

# Boundary preference order: paragraph, then sentence. A recursive call
# only ever tries separators *later* in this tuple than the one that
# produced its input piece -- never the same one again. Without that, a
# piece ending exactly in its own separator (e.g. text ending in "\n\n",
# or any single-occurrence edge case) regenerates an identical
# (part, "") split forever, since reattaching the separator for
# reconstruction is what makes it "contain" that separator again.
_SEPARATORS: tuple[str, ...] = ("\n\n", ". ")


def _split_into_units(
    text: str,
    max_unit_size: int,
    base_offset: int = 0,
    separators: tuple[str, ...] = _SEPARATORS,
) -> list[Chunk]:
    """Recursively break `text` into pieces each <= max_unit_size,
    preferring '\\n\\n' paragraph boundaries, then '. ' sentence
    boundaries, then hard character cuts. Returns (piece, start_offset)
    pairs where start_offset is relative to the ORIGINAL text passed to
    recursive_split_with_offsets (tracked via base_offset).

    Concatenating the returned pieces in order exactly reconstructs `text`
    (no characters are dropped or altered), which is what lets ingest.py
    resolve exact page numbers for every chunk.
    """
    if not text:
        return []
    if len(text) <= max_unit_size:
        return [(text, base_offset)]

    if not separators:
        return [
            (text[i : i + max_unit_size], base_offset + i)
            for i in range(0, len(text), max_unit_size)
        ]

    separator, remaining = separators[0], separators[1:]
    if separator not in text:
        return _split_into_units(text, max_unit_size, base_offset, remaining)

    pieces: list[Chunk] = []
    offset = base_offset
    parts = text.split(separator)
    for i, part in enumerate(parts):
        # Reattach the separator to every part except the last, so
        # concatenation losslessly reconstructs the original text.
        piece = part if i == len(parts) - 1 else part + separator
        if piece:
            pieces.extend(_split_into_units(piece, max_unit_size, offset, remaining))
        offset += len(piece)
    return pieces


def recursive_split_with_offsets(text: str, chunk_size: int, overlap: int = 0) -> list[Chunk]:
    """Same behavior as recursive_split, but also returns each chunk's
    starting character offset within the original `text`. Used internally
    by ingest.py to resolve real page numbers; recursive_split() below is
    the plain public API described in the spec.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []

    # Reserve `overlap` characters of headroom in every chunk (after the
    # first) so that prepending trailing context from the previous chunk
    # never pushes a chunk over chunk_size, and no source text is ever
    # dropped to make room for it.
    effective_size = chunk_size - overlap if overlap else chunk_size

    units = _split_into_units(text, effective_size)

    # Greedily pack units into base segments, each <= effective_size.
    base_segments: list[Chunk] = []
    current_text = ""
    current_start = 0
    have_current = False
    for unit_text, unit_offset in units:
        if not have_current:
            current_text, current_start, have_current = unit_text, unit_offset, True
        elif len(current_text) + len(unit_text) <= effective_size:
            current_text += unit_text
        else:
            base_segments.append((current_text, current_start))
            current_text, current_start = unit_text, unit_offset
    if have_current:
        base_segments.append((current_text, current_start))

    if overlap == 0 or len(base_segments) <= 1:
        return base_segments

    result: list[Chunk] = [base_segments[0]]
    for i in range(1, len(base_segments)):
        prev_text, _ = base_segments[i - 1]
        cur_text, cur_start = base_segments[i]
        prefix = prev_text[-overlap:]
        result.append((prefix + cur_text, cur_start - len(prefix)))
    return result


def recursive_split(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split `text` into chunks of at most `chunk_size` characters,
    preferring to break on paragraph boundaries ('\\n\\n'), then sentence
    boundaries ('. '), then falling back to hard character cuts. When
    `overlap` > 0, each chunk after the first is prefixed with the last
    `overlap` characters of the previous chunk for trailing context (the
    chunk_size max is still respected -- overlap eats into a chunk's own
    budget rather than extending it).
    """
    return [chunk_text for chunk_text, _ in recursive_split_with_offsets(text, chunk_size, overlap)]


def generate_id() -> str:
    """Return a random UUID4 hex string."""
    return uuid.uuid4().hex


def deterministic_id(doc_name: str, index: object, text: str) -> str:
    """Return a stable, content-addressed id for a chunk.

    Deviation D1: the spec uses random UUID4 ids for parent/child chunks,
    which makes re-ingesting a PDF non-idempotent (duplicate vectors) and
    makes any eval dataset referencing those ids break on re-ingest. Using
    a hash of (doc_name, index, text) instead means the same PDF always
    produces the same ids, so upserts stay idempotent and eval references
    stay stable across re-ingests.
    """
    raw = f"{doc_name}:{index}:{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no external NLP dependency)."""
    if not text or not text.strip():
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def char_to_page(offset: int, page_map: list[tuple[int, int]]) -> int:
    """Resolve a character offset (within the concatenated document text)
    to its page number.

    `page_map` is a list of (start_offset, page_number) pairs, sorted
    ascending by start_offset, where start_offset marks where that page's
    text begins in the concatenated document. Returns the page number of
    the last entry whose start_offset <= offset.
    """
    if not page_map:
        return 1
    page = page_map[0][1]
    for start_offset, page_num in page_map:
        if offset >= start_offset:
            page = page_num
        else:
            break
    return page
