"""Generation: format retrieved parent chunks as cited snippets, call Groq
for a grounded answer, then parse and validate its citations.

Implements instruction.md Section 9, plus a citation-grounding check (an
eval-strategy addition, not a spec deviation) and the spec's own
rate-limit fallback to GROQ_FALLBACK_MODEL.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from groq import Groq

from src.config import GROQ_FALLBACK_MODEL, GROQ_MODEL, REFUSAL_PHRASE

load_dotenv()

SYSTEM_PROMPT = (
    "You are a factual assistant. Answer the user's question using ONLY "
    "the provided context snippets. If the answer is not contained in the "
    "snippets, say you don't have enough information. Cite the snippet "
    "number(s) you used in the format [Snippet N]."
)

_CITATION_RE = re.compile(r"\[Snippets?\s+([^\]]+)\]", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+")

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        _client = Groq(api_key=api_key)
    return _client


def _format_page(chunk: dict) -> str:
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    if page_start is not None and page_end is not None and page_start != page_end:
        return f"p.{page_start}-{page_end}"
    page = chunk.get("source_page", page_start)
    return f"p.{page}"


def format_snippets(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Snippet {i}] ({chunk['doc_name']}, {_format_page(chunk)})"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def _call_groq(system_prompt: str, user_content: str) -> tuple[str, str]:
    """Call Groq chat completions, falling back to GROQ_FALLBACK_MODEL on a
    rate-limit error. Returns (answer_text, model_actually_used)."""
    client = _get_client()
    last_err: Exception | None = None
    for model in (GROQ_MODEL, GROQ_FALLBACK_MODEL):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return response.choices[0].message.content, model
        except Exception as e:  # noqa: BLE001 - Groq SDK error types vary by version
            last_err = e
            status = getattr(e, "status_code", None)
            is_rate_limited = status == 429 or "rate limit" in str(e).lower()
            if is_rate_limited and model != GROQ_FALLBACK_MODEL:
                continue
            raise
    raise last_err  # pragma: no cover - unreachable, satisfies type checkers


def _parse_citations(answer_text: str, num_chunks: int) -> tuple[list[int], list[int]]:
    """Return (valid_snippet_numbers, invalid_snippet_numbers) cited in
    answer_text, each sorted ascending and de-duplicated. "Invalid" means
    the model cited a snippet number that wasn't actually offered to it --
    a fabricated/out-of-range citation."""
    cited: set[int] = set()
    for match in _CITATION_RE.finditer(answer_text):
        for num_str in _NUM_RE.findall(match.group(1)):
            cited.add(int(num_str))
    valid = sorted(n for n in cited if 1 <= n <= num_chunks)
    invalid = sorted(n for n in cited if not (1 <= n <= num_chunks))
    return valid, invalid


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Generate a grounded, cited answer from the given (already retrieved
    and reranked) parent chunks.

    Returns {"answer", "citations": [{"doc_name","source_page"}],
    "raw_chunks_used", "grounded_citation_rate", "cited_snippets",
    "model_used"}.
    """
    if not chunks:
        # No context to answer from -- refuse without spending an LLM call.
        return {
            "answer": f"I {REFUSAL_PHRASE} to answer that question.",
            "citations": [],
            "raw_chunks_used": [],
            "grounded_citation_rate": 1.0,
            "cited_snippets": [],
            "model_used": None,
        }

    formatted = format_snippets(chunks)
    user_content = f"{formatted}\n\nQuestion: {query}"
    answer_text, model_used = _call_groq(SYSTEM_PROMPT, user_content)

    valid, invalid = _parse_citations(answer_text, len(chunks))
    total_cited = len(valid) + len(invalid)
    grounded_citation_rate = 1.0 if total_cited == 0 else len(valid) / total_cited

    citations = [
        {"doc_name": chunks[n - 1]["doc_name"], "source_page": chunks[n - 1]["source_page"]}
        for n in valid
    ]

    return {
        "answer": answer_text,
        "citations": citations,
        "raw_chunks_used": chunks,
        "grounded_citation_rate": grounded_citation_rate,
        "cited_snippets": valid,
        "model_used": model_used,
    }
