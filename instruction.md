You are an autonomous coding agent. Build the following project end-to-end, exactly as specified. Do not ask clarifying questions — every decision has already been made below. Work sequentially through the Build Order in Section 9 and do not skip steps.

# PROJECT: Enterprise-Grade Hierarchical RAG Analyzer (100% Free Stack)

## 1. Goal

Build a command-line RAG system that ingests a PDF, chunks it hierarchically (large "parent" chunks + small "child" chunks), retrieves via vector search + cross-encoder re-ranking, and generates grounded, cited answers using the Groq API. Then build a fully local evaluation harness (no paid LLM judge) that measures retrieval and generation quality.

## 2. Tech stack — use exactly these, no substitutions

- LLM: Groq API, model `llama-3.3-70b-versatile` (fallback `llama-3.1-8b-instant` if rate-limited). Requires env var `GROQ_API_KEY`.
- Embeddings: `sentence-transformers`, model `BAAI/bge-small-en-v1.5`, runs locally on CPU. No API key needed.
- Vector DB: `chromadb`, local persistent client, no server.
- Re-ranker: `flashrank`, model `ms-marco-MiniLM-L-12-v2`, local CPU.
- PDF parsing: `pymupdf` (`fitz`).
- Parent store: Python `pickle` file on disk.
- CLI: `typer`.
- No LangChain, no LlamaIndex, no RAGAS, no OpenAI. Plain Python only.

## 3. Create this exact project structure

```
rag-analyzer/
├── .env.example
├── requirements.txt
├── README.md
├── data/raw_pdfs/
├── storage/chroma_db/
├── storage/parent_store.pkl
├── src/config.py
├── src/utils.py
├── src/ingest.py
├── src/retrieval.py
├── src/generation.py
├── src/cli.py
├── eval/eval_dataset.json
├── eval/retrieval_metrics.py
├── eval/generation_metrics.py
├── eval/run_eval.py
└── tests/test_pipeline.py
```

## 4. requirements.txt — pin exactly these packages

```
groq
chromadb
sentence-transformers
flashrank
pymupdf
python-dotenv
typer
numpy
scikit-learn
diskcache
```

## 5. src/config.py — create with exactly these constants

```python
PARENT_CHUNK_SIZE = 2000
CHILD_CHUNK_SIZE = 400
CHILD_OVERLAP = 50
TOP_K_VECTOR_SEARCH = 15
TOP_K_RERANKED = 4
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHROMA_PATH = "storage/chroma_db"
PARENT_STORE_PATH = "storage/parent_store.pkl"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
```

## 6. src/utils.py — implement

- `recursive_split(text: str, chunk_size: int, overlap: int = 0) -> list[str]`: split preferring `\n\n` paragraph boundaries first, then sentence boundaries (`. `), then hard character cuts as last resort. Must respect `chunk_size` as a max, and add `overlap` characters of trailing context from the previous chunk when overlap > 0.
- `generate_id() -> str`: return a UUID4 hex string.
- Write unit tests in `tests/test_pipeline.py` asserting chunk sizes stay under limits and overlap works.

## 7. src/ingest.py — implement `ingest(pdf_path: str)` that does, in order

1. Open the PDF with `fitz.open(pdf_path)`, extract text per page, build a page-number map (character offset → page number) so citations can reference real page numbers later.
2. Concatenate all page text into one document string.
3. Call `recursive_split(text, PARENT_CHUNK_SIZE)` to get parent chunks. For each parent chunk generate a `parent_id`, and store `{parent_id: {"text": ..., "source_page": ..., "doc_name": ...}}` in a dict.
4. Persist that dict to `storage/parent_store.pkl` via `pickle.dump` (merge with existing content if the file already exists — do not overwrite other documents already ingested).
5. For each parent chunk, call `recursive_split(parent_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)` to get child chunks. Generate a `child_id` for each, tagged with its parent's `parent_id`.
6. Batch-encode all child chunk texts with `SentenceTransformer(EMBEDDING_MODEL)` — do NOT add `BGE_QUERY_PREFIX` to documents, only to queries later.
7. Upsert into a Chroma persistent collection at `CHROMA_PATH` (collection name `"child_chunks"`), storing metadata `{parent_id, child_id, source_page, doc_name}` per vector.
8. Expose this as CLI command `ingest` in `src/cli.py`: `python -m src.cli ingest --path data/raw_pdfs/<file>.pdf`.

## 8. src/retrieval.py — implement `retrieve(query: str) -> list[dict]`

1. Encode `BGE_QUERY_PREFIX + query` with the same embedding model.
2. Query the Chroma `"child_chunks"` collection for `TOP_K_VECTOR_SEARCH` nearest neighbors.
3. Map each result's `parent_id` metadata to the full parent text via `storage/parent_store.pkl`. De-duplicate parents (multiple children can map to the same parent).
4. Run FlashRank (`Ranker(model_name=RERANKER_MODEL)`) with the original `query` against the list of unique parent texts. Keep the top `TOP_K_RERANKED` by re-rank score.
5. Return a list of dicts: `[{"parent_id", "text", "source_page", "doc_name", "rerank_score"}]`, ordered best-first.

## 9. src/generation.py — implement `generate_answer(query: str, chunks: list[dict]) -> dict`

1. Format chunks as `[Snippet 1] (doc_name, page N)\n<text>\n\n[Snippet 2] ...`.
2. System prompt (use verbatim):
   ```
   You are a factual assistant. Answer the user's question using ONLY the provided context snippets. If the answer is not contained in the snippets, say you don't have enough information. Cite the snippet number(s) you used in the format [Snippet N].
   ```
3. Call Groq chat completions with `model=GROQ_MODEL`, `temperature=0.1`, passing system prompt + formatted snippets + user query.
4. Parse which `[Snippet N]` markers appear in the response; map back to `doc_name`/`source_page` for a clean citation list.
5. Return `{"answer": str, "citations": [{"doc_name", "source_page"}], "raw_chunks_used": chunks}`.
6. Expose CLI command `ask`: `python -m src.cli ask --query "..."`.

## 10. eval/eval_dataset.json — build this by hand after ingestion works

Create 20–30 entries in this exact schema, derived from a real test PDF you ingest:

```json
[
  {
    "question": "What was the total revenue in Q3?",
    "ground_truth_answer": "Total revenue in Q3 was $42.3 million.",
    "relevant_parent_ids": ["parent_a1b2c3"],
    "relevant_pages": [14]
  }
]
```

Include at least 3 deliberately unanswerable questions (facts not in the document) to test refusal behavior — mark these with `"relevant_parent_ids": []`.

## 11. eval/retrieval_metrics.py — implement these functions exactly

- `hit_rate_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> bool` — True if any overlap.
- `reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float` — `1/rank` of first relevant hit, else 0.
- `recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float` — fraction of relevant_ids found.
- `evaluate_retrieval(eval_dataset) -> dict` — for each question, run BOTH the raw top-15 vector search (pre-rerank) and the final top-4 post-rerank result; compute Hit Rate@k and MRR for both; compute `rerank_lift = hit_rate_post - hit_rate_pre_at_4` (compare post-rerank top-4 against the naive top-4 of the raw vector search, not the full top-15). Return averaged metrics across all questions plus the lift.

## 12. eval/generation_metrics.py — implement these functions exactly, using `BAAI/bge-small-en-v1.5` embeddings and cosine similarity only (no external LLM judge)

- `answer_relevance(question: str, answer: str) -> float` — cosine similarity between question and answer embeddings.
- `faithfulness(answer: str, context_chunks: list[str]) -> float` — split answer into sentences, for each compute max cosine similarity against context sentences, return the average.
- `answer_correctness(answer: str, ground_truth: str) -> float` — cosine similarity between answer and ground_truth embeddings.
- `refusal_check(answer: str) -> bool` — True if answer contains a refusal phrase like "don't have enough information".

## 13. eval/run_eval.py — implement

Loop over every entry in `eval_dataset.json`, run the full pipeline (`retrieve` → `generate_answer`), compute every metric from Sections 11–12, time each pipeline stage, and:
- Print a summary table (averages across all questions) to stdout.
- Write per-question results to `eval/eval_report.json` for debugging.
- Separately report refusal accuracy only over the deliberately-unanswerable subset.

## 14. .env.example

```
GROQ_API_KEY=your_groq_key_here
```

## 15. README.md — must include

- Setup instructions (`pip install -r requirements.txt`, set `.env`, run `ingest` then `ask`).
- Architecture diagram (ASCII is fine) showing the two-phase pipeline.
- A results table populated with the actual numbers from `run_eval.py` after you run it — do not leave placeholders once eval has run.

# 9. BUILD ORDER — execute in exactly this sequence

1. Scaffold the full folder structure, `requirements.txt`, `.env.example`.
2. Implement `src/config.py` and `src/utils.py` + tests; run tests, confirm they pass.
3. Implement `src/ingest.py`. Run it against one sample PDF placed in `data/raw_pdfs/`. Confirm: Chroma collection is populated (print collection count), and `parent_store.pkl` exists and loads correctly.
4. Implement `src/retrieval.py`. Manually test with 3–5 hand-typed questions; print retrieved chunks and sanity-check relevance before proceeding.
5. Implement `src/generation.py`. Wire the Groq API call. Confirm citations map correctly to real page numbers.
6. Implement `src/cli.py` wiring `ingest` and `ask` commands together. Confirm both commands run end-to-end from the terminal.
7. Build `eval/eval_dataset.json` by hand (20–30 Q&A pairs, including 3+ unanswerable questions) against the real ingested PDF.
8. Implement `eval/retrieval_metrics.py` and `eval/generation_metrics.py`.
9. Implement and run `eval/run_eval.py`. Confirm it produces a summary table and `eval_report.json` without errors.
10. Write `README.md` with real architecture notes and the real eval numbers from step 9.

# ACCEPTANCE CRITERIA — the project is done only when all of these are true

- [ ] `python -m src.cli ingest --path <pdf>` runs without error and populates both Chroma and the parent store.
- [ ] `python -m src.cli ask --query "<question>"` returns an answer with correct page citations.
- [ ] `eval/run_eval.py` runs end-to-end and prints Hit Rate@k, MRR, Recall@k, re-ranker lift, answer relevance, faithfulness, answer correctness, and refusal accuracy.
- [ ] All numbers in `README.md` are real (copy-pasted from an actual eval run), not placeholders.
- [ ] No OpenAI, LangChain, LlamaIndex, or RAGAS dependency anywhere in the codebase.
