"""Ingestion pipeline: PDF -> page-mapped text -> parent chunks -> child
chunks -> embeddings -> Chroma + parent_store.json.

Implements instruction.md Section 7, with deviations D1 (deterministic
ids), D4 (cosine distance), D5 (page ranges instead of a single page),
D9 (purge-before-reingest for idempotency), and D10 (JSON parent store
instead of pickle). See README.md "Deviations from spec" for details.
"""

from __future__ import annotations

import json
import os

import chromadb
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

from src.config import (
    CHILD_CHUNK_SIZE,
    CHILD_COLLECTION_NAME,
    CHILD_OVERLAP,
    CHROMA_DISTANCE,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    NORMALIZE_EMBEDDINGS,
    PARENT_CHUNK_SIZE,
    PARENT_STORE_PATH,
)
from src.utils import char_to_page, deterministic_id, recursive_split_with_offsets

# Page separator inserted between pages when concatenating; its length is
# counted in the running character offset so char_to_page stays accurate.
_PAGE_SEPARATOR = "\n\n"


def _extract_pages_and_map(pdf_path: str) -> tuple[str, list[tuple[int, int]]]:
    """Return (full_text, page_map) where page_map is a list of
    (start_offset, page_number) marking where each page's text begins in
    full_text (1-indexed page numbers, matching how a human would cite a
    page in the PDF)."""
    doc = fitz.open(pdf_path)
    try:
        parts: list[str] = []
        page_map: list[tuple[int, int]] = []
        cursor = 0
        for page_index in range(len(doc)):
            page_number = page_index + 1
            page_text = doc[page_index].get_text()
            page_map.append((cursor, page_number))
            parts.append(page_text)
            cursor += len(page_text)
            parts.append(_PAGE_SEPARATOR)
            cursor += len(_PAGE_SEPARATOR)
        return "".join(parts), page_map
    finally:
        doc.close()


def _load_parent_store() -> dict:
    if os.path.exists(PARENT_STORE_PATH):
        with open(PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_parent_store(store: dict) -> None:
    os.makedirs(os.path.dirname(PARENT_STORE_PATH), exist_ok=True)
    with open(PARENT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def _get_child_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=CHILD_COLLECTION_NAME,
        metadata={"hnsw:space": CHROMA_DISTANCE},  # D4: bge wants cosine, not Chroma's default L2
    )


def ingest(pdf_path: str) -> dict:
    """Ingest a single PDF into the parent store + Chroma child collection.
    Returns a small summary dict for CLI / smoke-test reporting.

    Safe to re-run on the same file (D9): any existing chunks for this
    doc_name are purged first, so re-ingesting is idempotent rather than
    additive.
    """
    doc_name = os.path.basename(pdf_path)

    os.makedirs(os.path.dirname(CHROMA_PATH) or ".", exist_ok=True)
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = _get_child_collection(client)

    # D9: purge stale entries for this doc before re-adding, so re-ingest
    # is idempotent instead of accumulating duplicates/orphans.
    collection.delete(where={"doc_name": doc_name})
    parent_store = _load_parent_store()
    parent_store = {pid: p for pid, p in parent_store.items() if p.get("doc_name") != doc_name}

    full_text, page_map = _extract_pages_and_map(pdf_path)

    parent_chunks = recursive_split_with_offsets(full_text, PARENT_CHUNK_SIZE)

    child_ids: list[str] = []
    child_texts: list[str] = []
    child_metadatas: list[dict] = []

    for p_index, (parent_text, parent_offset) in enumerate(parent_chunks):
        parent_id = deterministic_id(doc_name, p_index, parent_text)
        page_start = char_to_page(parent_offset, page_map)
        page_end = char_to_page(parent_offset + max(len(parent_text) - 1, 0), page_map)
        parent_store[parent_id] = {
            "text": parent_text,
            "doc_name": doc_name,
            "page_start": page_start,
            "page_end": page_end,
            # D5: kept as a friendly display string; page_start/page_end
            # are the source of truth since a parent chunk can span pages.
            "source_page": page_start if page_start == page_end else f"{page_start}-{page_end}",
        }

        child_chunks = recursive_split_with_offsets(parent_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)
        for c_index, (child_text, child_local_offset) in enumerate(child_chunks):
            if not child_text.strip():
                continue
            child_id = deterministic_id(doc_name, f"{p_index}_{c_index}", child_text)
            child_global_offset = parent_offset + child_local_offset
            child_page = char_to_page(child_global_offset, page_map)
            child_ids.append(child_id)
            child_texts.append(child_text)
            child_metadatas.append(
                {
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "source_page": child_page,
                    "doc_name": doc_name,
                }
            )

    _save_parent_store(parent_store)

    if child_texts:
        model = SentenceTransformer(EMBEDDING_MODEL)
        # No BGE_QUERY_PREFIX on documents -- only queries get the prefix.
        embeddings = model.encode(
            child_texts,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,  # D4
            show_progress_bar=False,
        ).tolist()
        collection.upsert(
            ids=child_ids,
            embeddings=embeddings,
            documents=child_texts,
            metadatas=child_metadatas,
        )

    summary = {
        "doc_name": doc_name,
        "num_pages": len(page_map),
        "num_parent_chunks": len(parent_chunks),
        "num_child_chunks": len(child_texts),
        "collection_count": collection.count(),
        "parent_store_size": len(parent_store),
    }
    print(
        f"Ingested '{doc_name}': {summary['num_pages']} pages, "
        f"{summary['num_parent_chunks']} parent chunks, "
        f"{summary['num_child_chunks']} child chunks. "
        f"Chroma collection count = {summary['collection_count']}, "
        f"parent_store total entries = {summary['parent_store_size']}."
    )
    return summary
