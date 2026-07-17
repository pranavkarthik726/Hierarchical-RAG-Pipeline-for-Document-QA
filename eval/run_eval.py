"""Run the full pipeline (retrieve -> generate_answer) over every entry in
eval/eval_dataset.json, compute retrieval metrics (Section 11), the Groq
LLM-judge generation metrics + citation grounding, and refusal accuracy on
the deliberately-unanswerable subset. Prints a summary table to stdout and
writes per-question detail to eval/eval_report.json.

Run with: python -m eval.run_eval
"""

from __future__ import annotations

import json
import os
import time

from eval.generation_metrics import judge_answer, refusal_check
from eval.retrieval_metrics import evaluate_retrieval
from src.generation import generate_answer
from src.retrieval import retrieve

EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
EVAL_REPORT_PATH = os.path.join(os.path.dirname(__file__), "eval_report.json")


def _load_dataset() -> list[dict]:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_eval() -> dict:
    dataset = _load_dataset()
    print(f"Running eval over {len(dataset)} questions...\n")

    retrieval_report = evaluate_retrieval(dataset)

    per_question_gen = []
    for i, entry in enumerate(dataset, start=1):
        question = entry["question"]
        # For unanswerable questions, ground_truth_answer should itself be
        # phrased as "not available in the document" -- that way a correct
        # refusal is judged as a CORRECT answer (matching the ground
        # truth) rather than needing to be specially excluded from the
        # correctness/relevance averages.
        ground_truth = entry.get("ground_truth_answer", "")
        answerable = entry.get("answerable", True)

        t0 = time.perf_counter()
        chunks = retrieve(question)
        retrieve_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = generate_answer(question, chunks)
        generate_time = time.perf_counter() - t0

        answer = result["answer"]
        refused = refusal_check(answer)

        judge = judge_answer(
            question=question,
            answer=answer,
            context=[c["text"] for c in chunks],
            ground_truth=ground_truth,
        )

        per_question_gen.append(
            {
                "question": question,
                "answerable": answerable,
                "answer": answer,
                "ground_truth_answer": ground_truth,
                "refused": refused,
                "grounded_citation_rate": result["grounded_citation_rate"],
                "citations": result["citations"],
                "retrieve_time_sec": retrieve_time,
                "generate_time_sec": generate_time,
                "judge": judge,
            }
        )
        status = "judge-failed" if judge is None else "ok"
        flag = " [refused]" if refused else ""
        print(f"  [{i}/{len(dataset)}] {status}{flag} {question[:70]}")

    judged_rows = [r for r in per_question_gen if r["judge"] is not None]
    faithfulness_scores = [r["judge"]["faithfulness"] for r in judged_rows]
    correctness_scores = [r["judge"]["correctness"] for r in judged_rows]
    relevance_scores = [r["judge"]["relevance"] for r in judged_rows]
    grounding_scores = [r["grounded_citation_rate"] for r in per_question_gen]

    unanswerable_rows = [r for r in per_question_gen if not r["answerable"]]
    refusal_accuracy = (
        sum(1 for r in unanswerable_rows if r["refused"]) / len(unanswerable_rows)
        if unanswerable_rows
        else None
    )

    summary = {
        "num_questions": len(dataset),
        "num_answerable": len(dataset) - len(unanswerable_rows),
        "num_unanswerable": len(unanswerable_rows),
        "num_judge_failures": len(per_question_gen) - len(judged_rows),
        "retrieval": retrieval_report,
        "generation": {
            "faithfulness": _average(faithfulness_scores),
            "correctness": _average(correctness_scores),
            "relevance": _average(relevance_scores),
        },
        "grounded_citation_rate": _average(grounding_scores),
        "refusal_accuracy_on_unanswerable": refusal_accuracy,
        "avg_retrieve_time_sec": _average([r["retrieve_time_sec"] for r in per_question_gen]),
        "avg_generate_time_sec": _average([r["generate_time_sec"] for r in per_question_gen]),
    }

    _print_summary_table(summary)

    report = {"summary": summary, "per_question": per_question_gen}
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull per-question report written to {EVAL_REPORT_PATH}")

    return report


def _print_summary_table(summary: dict) -> None:
    r = summary["retrieval"]
    print("\n" + "=" * 62)
    print("EVAL SUMMARY")
    print("=" * 62)
    print(
        f"Questions: {summary['num_questions']} "
        f"({summary['num_answerable']} answerable, "
        f"{summary['num_unanswerable']} unanswerable)"
    )
    if summary["num_judge_failures"]:
        print(
            f"WARNING: {summary['num_judge_failures']} judge call(s) failed to "
            "parse and were excluded from generation averages"
        )

    print(f"\n-- Retrieval (n={r.get('num_questions', 0)} answerable questions) --")
    print(f"  Hit Rate@15 (pre-rerank):  {r.get('hit_rate_at_15_pre_rerank', 0):.3f}")
    print(f"  MRR@15      (pre-rerank):  {r.get('mrr_at_15_pre_rerank', 0):.3f}")
    print(f"  Recall@15   (pre-rerank):  {r.get('recall_at_15_pre_rerank', 0):.3f}")
    print(f"  Hit Rate@4  (pre-rerank):  {r.get('hit_rate_at_4_pre_rerank', 0):.3f}")
    print(f"  Hit Rate@4  (post-rerank): {r.get('hit_rate_at_4_post_rerank', 0):.3f}")
    print(f"  MRR@4       (post-rerank): {r.get('mrr_at_4_post_rerank', 0):.3f}")
    print(f"  Recall@4    (post-rerank): {r.get('recall_at_4_post_rerank', 0):.3f}")
    print(f"  Rerank lift (HR@4 post - HR@4 pre): {r.get('rerank_lift', 0):+.3f}")

    g = summary["generation"]
    print("\n-- Generation (Groq LLM-as-judge, 0-1 scale) --")
    print(f"  Faithfulness: {g['faithfulness']:.3f}")
    print(f"  Correctness:  {g['correctness']:.3f}")
    print(f"  Relevance:    {g['relevance']:.3f}")

    print("\n-- Safety checks --")
    print(f"  Citation grounding rate:         {summary['grounded_citation_rate']:.3f}")
    ra = summary["refusal_accuracy_on_unanswerable"]
    print(
        f"  Refusal accuracy (unanswerable): {ra:.3f}"
        if ra is not None
        else "  Refusal accuracy (unanswerable): n/a (no unanswerable questions in dataset)"
    )

    print("\n-- Timing --")
    print(f"  Avg retrieve time: {summary['avg_retrieve_time_sec']:.3f}s")
    print(f"  Avg generate time: {summary['avg_generate_time_sec']:.3f}s")
    print("=" * 62)


if __name__ == "__main__":
    run_eval()
