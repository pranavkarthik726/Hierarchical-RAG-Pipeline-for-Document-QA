"""Generation-quality evaluation helpers.

Two judge implementations are available:
  - judge_answer()        -- original Groq LLM-as-judge (used by run_eval.py)
  - gemini_judge_answer() -- Gemini LLM-as-judge (used by run_apple_eval.py)

Both score faithfulness / correctness / relevance on a 0.0-1.0 scale.
A deterministic, model-free refusal_check() is shared by both runners.

Deviation D8 (Groq judge): the spec bans a "paid" LLM judge and otherwise
wants generation metrics from BAAI/bge-small-en-v1.5 cosine similarity alone.
Cosine can't distinguish a faithful paraphrase from an on-topic hallucination,
and it *penalizes correct refusals*. Since Groq is already a required, free
dependency, using it as a judge is strictly more defensible at zero extra cost.

Gemini judge (Apple 10-K eval): same rationale -- Gemini free tier (10 RPM)
gives a stronger, instruction-following judge at no cost.  The runner
(run_apple_eval.py) enforces a 6-second inter-call sleep to stay within the
10 RPM limit.
"""

from __future__ import annotations

import json
import os
import re

import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq

from src.config import GEMINI_JUDGE_MODEL, GROQ_FALLBACK_MODEL, JUDGE_MODEL, REFUSAL_PHRASE

load_dotenv()

# ---------------------------------------------------------------------------
# Shared prompt templates (identical for both judges)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator of a RAG (retrieval-augmented "
    "generation) system's answers. Score strictly and only using the "
    "information given below. Respond with ONLY a single JSON object, no "
    "markdown formatting and no extra commentary."
)

_JUDGE_USER_TEMPLATE = """Question:
{question}

Retrieved context (this is ALL the context the system had access to):
{context}

System's answer:
{answer}

Ground-truth reference answer:
{ground_truth}

Score the system's answer on three 0.0-1.0 scales (any value in that
range, e.g. 0.0, 0.25, 0.5, 0.75, 1.0):
- "faithfulness": is every claim in the answer actually supported by the
  retrieved context above (1.0 = fully supported, no hallucination; 0.0 =
  unsupported or contradicts the context)? A correct refusal ("not enough
  information") when the context genuinely lacks the answer is fully
  faithful (1.0).
- "correctness": does the answer convey the same facts as the ground-truth
  reference answer (1.0 = matches; 0.0 = wrong or missing the key facts)?
- "relevance": does the answer actually address the question asked (1.0 =
  directly addresses it; 0.0 = off-topic or non-responsive)?

Respond with ONLY this JSON object (no other text):
{{"faithfulness": <float>, "faithfulness_reason": "<one short sentence>",
  "correctness": <float>, "correctness_reason": "<one short sentence>",
  "relevance": <float>, "relevance_reason": "<one short sentence>"}}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _clamp01(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _build_scores(parsed: dict, model: str) -> dict:
    """Normalise a parsed judge JSON into a standard scores dict."""
    return {
        "faithfulness": _clamp01(parsed.get("faithfulness")),
        "faithfulness_reason": parsed.get("faithfulness_reason", ""),
        "correctness": _clamp01(parsed.get("correctness")),
        "correctness_reason": parsed.get("correctness_reason", ""),
        "relevance": _clamp01(parsed.get("relevance")),
        "relevance_reason": parsed.get("relevance_reason", ""),
        "judge_model": model,
    }


# ---------------------------------------------------------------------------
# Groq judge (original -- used by run_eval.py)
# ---------------------------------------------------------------------------

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def judge_answer(
    question: str, answer: str, context: list[str], ground_truth: str
) -> dict | None:
    """Run one Groq judge call scoring faithfulness / correctness /
    relevance for a single generated answer (temperature=0 for stable
    scores). Returns None if no parseable response was obtained after
    retries -- the caller should exclude the row from averages and note
    it in the report rather than crash the batch eval."""
    client = _get_groq_client()
    context_text = "\n\n".join(context) if context else "(no context was retrieved)"
    user_content = _JUDGE_USER_TEMPLATE.format(
        question=question, context=context_text, answer=answer, ground_truth=ground_truth
    )

    models = [JUDGE_MODEL, GROQ_FALLBACK_MODEL]
    for model_index, model in enumerate(models):
        is_last_model = model_index == len(models) - 1
        for attempt in range(2):  # one retry per model on a bad/unparseable response
            is_last_attempt = attempt == 1
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                )
            except Exception as e:  # noqa: BLE001 - Groq SDK error types vary by version
                status = getattr(e, "status_code", None)
                if status == 429 and not is_last_model:
                    break  # give up on this model, try the fallback model
                if is_last_attempt and is_last_model:
                    return None
                continue

            parsed = _extract_json(response.choices[0].message.content)
            if parsed is not None:
                return _build_scores(parsed, model)
            if is_last_attempt and is_last_model:
                return None
    return None


# ---------------------------------------------------------------------------
# Gemini judge (used by run_apple_eval.py)
# ---------------------------------------------------------------------------

_gemini_configured = False


def _ensure_gemini_configured() -> None:
    global _gemini_configured
    if not _gemini_configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        genai.configure(api_key=api_key)
        _gemini_configured = True


def gemini_judge_answer(
    question: str, answer: str, context: list[str], ground_truth: str
) -> dict | None:
    """Run one Gemini judge call scoring faithfulness / correctness /
    relevance for a single generated answer.

    The caller (run_apple_eval.py) is responsible for rate-limit pacing --
    sleep at least 6 seconds between calls to stay within the free-tier
    10 RPM ceiling.

    Returns None if no parseable response was obtained after retries.
    """
    _ensure_gemini_configured()
    context_text = "\n\n".join(context) if context else "(no context was retrieved)"
    prompt = (
        _JUDGE_SYSTEM_PROMPT
        + "\n\n"
        + _JUDGE_USER_TEMPLATE.format(
            question=question,
            context=context_text,
            answer=answer,
            ground_truth=ground_truth,
        )
    )

    model = genai.GenerativeModel(
        model_name=GEMINI_JUDGE_MODEL,
        generation_config=genai.types.GenerationConfig(temperature=0),
    )

    for attempt in range(3):  # up to 3 attempts on parse failure
        try:
            response = model.generate_content(prompt)
            raw_text = response.text
        except Exception as e:  # noqa: BLE001
            print(f"    [gemini-judge] attempt {attempt + 1} error: {e}")
            if attempt == 2:
                return None
            continue

        parsed = _extract_json(raw_text)
        if parsed is not None:
            return _build_scores(parsed, GEMINI_JUDGE_MODEL)

        print(f"    [gemini-judge] attempt {attempt + 1} could not parse JSON from response.")
        if attempt == 2:
            return None

    return None


# ---------------------------------------------------------------------------
# Refusal check (deterministic, shared by both runners)
# ---------------------------------------------------------------------------

def refusal_check(answer: str) -> bool:
    """Deterministic (no model) check for whether the answer refused to
    answer -- i.e. contains the refusal phrase the system prompt asks the
    model to use when context is insufficient."""
    return REFUSAL_PHRASE.lower() in answer.lower()
