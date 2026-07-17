"""Retrieval pipeline: embed query -> Chroma vector search over child
chunks -> FlashRank rerank -> unique parent chunks, best-first.

Implements instruction.md Section 8, with deviation D3 (rerank the short
child chunks, not the long parent chunks -- the cross-encoder truncates at
512 tokens, so reranking ~2000-char parents only ever judges their head)
and D4 (cosine distance space). See README.md "Deviations from spec".
"""

from __future__ import annotations

import json
import os

import chromadb
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer

from src.config import (
    BGE_QUERY_PREFIX,
    CHILD_COLLECTION_NAME,
    CHROMA_DISTANCE,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    NORMALIZE_EMBEDDINGS,
    PARENT_STORE_PATH,
    RERANKER_MODEL,
    TOP_K_RERANKED,
    TOP_K_VECTOR_SEARCH,
)

_embedder: SentenceTransformer | None = None
_ranker: Ranker | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name=RERANKER_MODEL)
    return _ranker


def _get_child_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=CHILD_COLLECTION_NAME,
        metadata={"hnsw:space": CHROMA_DISTANCE},
    )


def _load_parent_store() -> dict:
    if os.path.exists(PARENT_STORE_PATH):
        with open(PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _vector_search(query: str) -> list[dict]:
    """Top TOP_K_VECTOR_SEARCH child chunks by cosine similarity, ordered
    nearest-first."""
    model = _get_embedder()
    collection = _get_child_collection()
    query_embedding = model.encode(
        [BGE_QUERY_PREFIX + query],
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_VECTOR_SEARCH,
        include=["documents", "metadatas", "distances"],
    )
    if not results["ids"] or not results["ids"][0]:
        return []
    candidates = []
    for cid, text, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        candidates.append({"child_id": cid, "text": text, "meta": meta, "distance": dist})
    return candidates


def _dedupe_parents_by_score(
    ordered_items: list[dict], score_key: str, k: int
) -> list[dict]:
    """Walk `ordered_items` (already sorted best-first) and keep the first
    (best) occurrence of each unique parent_id, up to k parents."""
    parent_store = _load_parent_store()
    seen: set[str] = set()
    out: list[dict] = []
    for item in ordered_items:
        meta = item.get("meta") or {}
        parent_id = meta.get("parent_id")
        if not parent_id or parent_id in seen:
            continue
        parent = parent_store.get(parent_id)
        if not parent:
            continue
        seen.add(parent_id)
        out.append(
            {
                "parent_id": parent_id,
                "text": parent["text"],
                "page_start": parent["page_start"],
                "page_end": parent["page_end"],
                "source_page": parent["source_page"],
                "doc_name": parent["doc_name"],
                score_key: item["score"],
            }
        )
        if len(out) >= k:
            break
    return out


def retrieve(query: str) -> list[dict]:
    """Vector search -> rerank children -> unique parents, best-first.

    Returns a list of dicts:
    [{"parent_id", "text", "page_start", "page_end", "source_page",
      "doc_name", "rerank_score"}], ordered best-first, length <=
    TOP_K_RERANKED.
    """
    candidates = _vector_search(query)
    if not candidates:
        return []

    ranker = _get_ranker()
    passages = [
        {"id": c["child_id"], "text": c["text"], "meta": c["meta"]} for c in candidates
    ]
    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    # FlashRank returns results sorted best-first with a "score" field.
    reranked_items = [{"meta": r.get("meta") or {}, "score": r.get("score")} for r in reranked]

    return _dedupe_parents_by_score(reranked_items, "rerank_score", TOP_K_RERANKED)


def raw_vector_parents(query: str, k: int = TOP_K_RERANKED) -> list[dict]:
    """Naive top-k unique parents in pure vector-distance order, with NO
    reranking. Used only by eval/retrieval_metrics.py to measure the
    reranker's lift over plain vector search (pre-rerank baseline)."""
    candidates = _vector_search(query)
    if not candidates:
        return []
    # Chroma cosine "distance" is 1 - cosine_similarity, so smaller is
    # better; results already arrive nearest-first, but sort explicitly
    # to be robust to backend ordering changes.
    candidates = sorted(candidates, key=lambda c: c["distance"])
    ranked_items = [{"meta": c["meta"], "score": -c["distance"]} for c in candidates]
    return _dedupe_parents_by_score(ranked_items, "vector_score", k)
