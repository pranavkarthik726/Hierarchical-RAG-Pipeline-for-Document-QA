"""Run the full pipeline (retrieve -> generate_answer) over every entry in
eval/apple_10k_eval_dataset.json, compute retrieval metrics, Gemini
LLM-as-judge generation metrics (faithfulness / correctness / relevance),
citation grounding, and refusal accuracy on the unanswerable subset.

Rate-limit constraints (Gemini free tier):
  - 5 RPM  -> sleep 12 seconds between judge calls
  - 20 RPD -> at most 20 judge calls per calendar day

With 31 questions this means TWO runs are needed:
  Run 1  -> judges questions 1-20, saves checkpoint
  Run 2  -> loads checkpoint, judges remaining 11 questions, writes final report

The runner automatically detects the checkpoint and resumes from where it
stopped.  Delete eval/apple_eval_checkpoint.json to start fresh.

Prints a summary table to stdout.
Writes per-question detail to eval/apple_eval_report.json (final) and
eval/apple_eval_checkpoint.json (incremental progress after every judge call).

Run with: python -m eval.run_apple_eval
"""

from __future__ import annotations

import json
import os
import time

from eval.generation_metrics import gemini_judge_answer, refusal_check
from eval.retrieval_metrics import evaluate_retrieval
from src.generation import generate_answer
from src.retrieval import retrieve

EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "apple_10k_eval_dataset.json")
EVAL_REPORT_PATH = os.path.join(os.path.dirname(__file__), "apple_eval_report.json")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "apple_eval_checkpoint.json")

# Gemini free-tier limits
_GEMINI_RPM_LIMIT = 5           # requests per minute
_GEMINI_RPD_LIMIT = 20          # requests per day  <-- hard daily ceiling
_JUDGE_SLEEP_SECONDS = 60.0 / _GEMINI_RPM_LIMIT  # = 12.0 s between calls


def _load_dataset() -> list[dict]:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_checkpoint() -> dict[str, dict]:
    """Load previously judged results keyed by question text.
    Returns an empty dict if no checkpoint exists."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_checkpoint(judged: dict[str, dict]) -> None:
    """Persist incremental judge results so a re-run can skip already-judged
    questions and stay within the 20 RPD daily limit."""
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(judged, f, indent=2, ensure_ascii=False)


def run_eval() -> dict:
    dataset = _load_dataset()
    checkpoint = _load_checkpoint()

    already_judged = len(checkpoint)
    remaining_budget = max(0, _GEMINI_RPD_LIMIT - already_judged)

    print(f"Running Apple 10-K eval over {len(dataset)} questions...\n")
    print(f"Gemini free-tier limits: {_GEMINI_RPM_LIMIT} RPM / {_GEMINI_RPD_LIMIT} RPD")
    print(f"  Questions already judged (checkpoint): {already_judged}")
    print(f"  Remaining judge budget today:          {remaining_budget}")
    print(f"  Sleep between judge calls:             {_JUDGE_SLEEP_SECONDS:.0f}s\n")

    if already_judged > 0 and remaining_budget == 0:
        print(
            "WARNING: Daily judge budget exhausted (20 RPD).  "
            "Re-run tomorrow to judge remaining questions.\n"
            "Retrieval and generation will still run for all questions.\n"
        )

    # ------------------------------------------------------------------
    # Retrieval metrics (no model judgment, no rate-limit concern)
    # ------------------------------------------------------------------
    retrieval_report = evaluate_retrieval(dataset)

    # ------------------------------------------------------------------
    # Generation: retrieve -> generate -> Gemini judge (with RPD cap)
    # ------------------------------------------------------------------
    per_question_gen = []
    judge_calls_this_run = 0

    for i, entry in enumerate(dataset, start=1):
        question = entry["question"]
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

        # --- Judge: use checkpoint if available, otherwise call Gemini ---
        if question in checkpoint:
            judge = checkpoint[question]
            judge_source = "checkpoint"
        elif judge_calls_this_run < remaining_budget:
            print(f"  [{i}/{len(dataset)}] judging (Gemini): {question[:65]}")
            judge = gemini_judge_answer(
                question=question,
                answer=answer,
                context=[c["text"] for c in chunks],
                ground_truth=ground_truth,
            )
            judge_calls_this_run += 1

            # Save to checkpoint immediately so progress survives a crash
            if judge is not None:
                checkpoint[question] = judge
                _save_checkpoint(checkpoint)

            # Rate-limit pacing: sleep after every judge call
            if judge_calls_this_run < remaining_budget and i < len(dataset):
                print(f"    [rate-limit] sleeping {_JUDGE_SLEEP_SECONDS:.0f}s ...")
                time.sleep(_JUDGE_SLEEP_SECONDS)

            status = "judge-failed" if judge is None else "ok"
            flag = " [refused]" if refused else ""
            print(f"    -> {status}{flag}")
            judge_source = "gemini"
        else:
            # Daily budget exhausted -- skip judging for this question
            judge = None
            judge_source = "skipped (RPD limit)"
            print(
                f"  [{i}/{len(dataset)}] SKIP judge (daily budget used): "
                f"{question[:60]}"
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
                "judge_source": judge_source,
            }
        )

    # ------------------------------------------------------------------
    # Aggregate scores
    # ------------------------------------------------------------------
    judged_rows = [r for r in per_question_gen if r["judge"] is not None]
    skipped_rows = [r for r in per_question_gen if r["judge_source"].startswith("skipped")]

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
        "dataset": "apple_10k_eval_dataset.json",
        "judge": "gemini",
        "num_questions": len(dataset),
        "num_answerable": len(dataset) - len(unanswerable_rows),
        "num_unanswerable": len(unanswerable_rows),
        "num_judged": len(judged_rows),
        "num_judge_skipped_rpd_limit": len(skipped_rows),
        "num_judge_failures": len(per_question_gen) - len(judged_rows) - len(skipped_rows),
        "retrieval": retrieval_report,
        "generation": {
            "faithfulness": _average(faithfulness_scores),
            "correctness": _average(correctness_scores),
            "relevance": _average(relevance_scores),
            "note": (
                f"Averaged over {len(judged_rows)}/{len(dataset)} judged questions"
                + (
                    f" -- {len(skipped_rows)} skipped due to 20 RPD limit, re-run to complete"
                    if skipped_rows
                    else ""
                )
            ),
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
    if skipped_rows:
        print(
            f"\n*** {len(skipped_rows)} questions were NOT judged due to the 20 RPD limit. ***"
            f"\n*** Re-run tomorrow -- the checkpoint will resume from where we stopped. ***"
        )
    elif os.path.exists(CHECKPOINT_PATH):
        print("\nAll questions judged. You may delete apple_eval_checkpoint.json if desired.")

    return report


def _print_summary_table(summary: dict) -> None:
    r = summary["retrieval"]
    print("\n" + "=" * 65)
    print("APPLE 10-K EVAL SUMMARY  (Gemini judge)")
    print("=" * 65)
    print(
        f"Questions: {summary['num_questions']} "
        f"({summary['num_answerable']} answerable, "
        f"{summary['num_unanswerable']} unanswerable)"
    )
    print(
        f"Judged:    {summary['num_judged']}/{summary['num_questions']}"
        + (
            f"  [{summary['num_judge_skipped_rpd_limit']} skipped - RPD limit, re-run tomorrow]"
            if summary["num_judge_skipped_rpd_limit"]
            else ""
        )
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
    print(f"\n-- Generation (Gemini LLM-as-judge, 0-1 scale) --")
    print(f"  Faithfulness: {g['faithfulness']:.3f}")
    print(f"  Correctness:  {g['correctness']:.3f}")
    print(f"  Relevance:    {g['relevance']:.3f}")
    print(f"  Note: {g['note']}")

    print("\n-- Safety checks --")
    print(f"  Citation grounding rate:         {summary['grounded_citation_rate']:.3f}")
    ra = summary["refusal_accuracy_on_unanswerable"]
    print(
        f"  Refusal accuracy (unanswerable): {ra:.3f}"
        if ra is not None
        else "  Refusal accuracy (unanswerable): n/a"
    )

    print("\n-- Timing --")
    print(f"  Avg retrieve time: {summary['avg_retrieve_time_sec']:.3f}s")
    print(f"  Avg generate time: {summary['avg_generate_time_sec']:.3f}s")
    print("=" * 65)


if __name__ == "__main__":
    run_eval()
