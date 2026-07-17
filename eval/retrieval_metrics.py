"""Objective retrieval metrics (instruction.md Section 11) -- ground-truth
based, no model judgment involved. This is the rigorous backbone of the
eval: ids either match the hand-labeled relevant_parent_ids or they don't.
"""

from __future__ import annotations

from src.config import TOP_K_RERANKED, TOP_K_VECTOR_SEARCH
from src.retrieval import raw_vector_parents, retrieve


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> bool:
    """True if any retrieved id is in relevant_ids."""
    if not relevant_ids:
        return False
    relevant_set = set(relevant_ids)
    return any(rid in relevant_set for rid in retrieved_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1/rank of the first relevant hit (rank is 1-indexed), else 0."""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of relevant_ids found anywhere in retrieved_ids."""
    if not relevant_ids:
        return 1.0
    relevant_set = set(relevant_ids)
    found = sum(1 for rid in relevant_set if rid in retrieved_ids)
    return found / len(relevant_set)


def _average(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def evaluate_retrieval(eval_dataset: list[dict]) -> dict:
    """For each answerable question, run both:
    - the raw top-K_VECTOR_SEARCH vector search (pre-rerank), and
    - the final top-K_RERANKED post-rerank result,
    compute Hit Rate@k / MRR / Recall@k for both, and compute
    rerank_lift = hit_rate(post-rerank top-4) - hit_rate(naive top-4 of the
    raw vector search) -- i.e. an apples-to-apples top-4-vs-top-4
    comparison, not top-4-vs-top-15. Returns averaged metrics across all
    answerable questions plus the lift and per-question detail.

    Unanswerable questions (answerable=False / no relevant_parent_ids) are
    skipped here since there is no retrieval ground truth to score against
    -- their behavior is covered separately by refusal accuracy.
    """
    per_question = []
    for entry in eval_dataset:
        relevant_ids = entry.get("relevant_parent_ids") or []
        if not entry.get("answerable", True) or not relevant_ids:
            continue

        query = entry["question"]
        pre_15 = raw_vector_parents(query, k=TOP_K_VECTOR_SEARCH)
        pre_4 = raw_vector_parents(query, k=TOP_K_RERANKED)
        post_4 = retrieve(query)

        pre_15_ids = [c["parent_id"] for c in pre_15]
        pre_4_ids = [c["parent_id"] for c in pre_4]
        post_4_ids = [c["parent_id"] for c in post_4]

        per_question.append(
            {
                "question": query,
                "hit_rate_pre15": hit_rate_at_k(pre_15_ids, relevant_ids),
                "mrr_pre15": reciprocal_rank(pre_15_ids, relevant_ids),
                "recall_pre15": recall_at_k(pre_15_ids, relevant_ids),
                "hit_rate_pre4": hit_rate_at_k(pre_4_ids, relevant_ids),
                "hit_rate_post4": hit_rate_at_k(post_4_ids, relevant_ids),
                "mrr_post4": reciprocal_rank(post_4_ids, relevant_ids),
                "recall_post4": recall_at_k(post_4_ids, relevant_ids),
            }
        )

    if not per_question:
        return {"num_questions": 0, "per_question": []}

    hit_rate_pre4 = _average(per_question, "hit_rate_pre4")
    hit_rate_post4 = _average(per_question, "hit_rate_post4")

    return {
        "num_questions": len(per_question),
        "hit_rate_at_15_pre_rerank": _average(per_question, "hit_rate_pre15"),
        "mrr_at_15_pre_rerank": _average(per_question, "mrr_pre15"),
        "recall_at_15_pre_rerank": _average(per_question, "recall_pre15"),
        "hit_rate_at_4_pre_rerank": hit_rate_pre4,
        "hit_rate_at_4_post_rerank": hit_rate_post4,
        "mrr_at_4_post_rerank": _average(per_question, "mrr_post4"),
        "recall_at_4_post_rerank": _average(per_question, "recall_post4"),
        "rerank_lift": hit_rate_post4 - hit_rate_pre4,
        "per_question": per_question,
    }
