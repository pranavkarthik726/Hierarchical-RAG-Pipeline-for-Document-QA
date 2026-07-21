"""Central configuration constants for the RAG pipeline.

Values marked (spec) come verbatim from instruction.md Section 5.
Values marked (deviation) are additions/overrides documented in README.md
under "Deviations from spec" (see also the plan file used to build this
project).
"""

# --- Chunking (spec) ---
PARENT_CHUNK_SIZE = 2000
CHILD_CHUNK_SIZE = 400
CHILD_OVERLAP = 50

# --- Retrieval (spec) ---
TOP_K_VECTOR_SEARCH = 15
TOP_K_RERANKED = 4

# --- Models (spec) ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Storage (spec, with D10 deviation) ---
CHROMA_PATH = "storage/chroma_db"
# D10: JSON instead of pickle -- safe to load, human-inspectable, and the
# parent store only ever holds plain text/dict data (no pickle security
# risk, no opaque binary format). Spec originally said
# storage/parent_store.pkl.
PARENT_STORE_PATH = "storage/parent_store.json"

# --- Prompting (spec) ---
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Deviations (not in spec) ---
# D8: free Groq LLM-as-judge for generation eval (faithfulness / correctness
# / relevance), run at temperature=0 for stable scores.
JUDGE_MODEL = "llama-3.3-70b-versatile"
# Spec's fallback model, used both for generation (rate-limit fallback) and
# as the judge's own fallback.
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"

# Gemini judge: used for Apple 10-K eval (run_apple_eval.py).
# Free tier limits: 5 RPM (sleep 12s between calls) / 20 RPD.
# 31 questions requires 2 runs across 2 days; checkpoint handles resume.
GEMINI_JUDGE_MODEL = "gemini-2.5-flash"

# D4: bge embeddings are trained for cosine similarity; Chroma's default
# space is L2, which silently degrades ranking quality for this model.
CHROMA_DISTANCE = "cosine"
NORMALIZE_EMBEDDINGS = True

CHILD_COLLECTION_NAME = "child_chunks"

# Substring the generation system prompt asks the model to use when it
# cannot answer from context; used by refusal_check() in eval and by the
# citation-grounding logic.
REFUSAL_PHRASE = "don't have enough information"
