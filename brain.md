# 🧠 Brain — Agentic RAG Project

> This document captures the *why* and *how* of the project —
> the mental model, design intuition, and trade-offs made at each step.
> Read this before touching the code.

---

## Core Idea

Most RAG systems do: **retrieve → stuff into prompt → generate**.

This one does: **retrieve → LLM judges if it's enough → rewrite query if not → retry → generate with forced citations**.

The "agentic" label means there's a **decision loop**. The system doesn't trust its first retrieval pass; it asks itself "is this actually enough to answer the question?" before committing to an answer. That's the differentiator.

---

## Why Each Piece Exists

### 🔍 Two-Stage Retrieval
- **Stage 1 — Vector search (ChromaDB)**: Fast, cheap, wide net. Grabs the 6 most *embedding-similar* chunks. The problem: embedding similarity ≠ answer relevance. A chunk about "contract termination clauses" and a chunk about "the word terminate in a book" might look equally similar in embedding space.
- **Stage 2 — Cross-encoder reranking**: Reads the actual question *together* with each chunk and scores true relevance. Much more accurate. Cuts the 6 candidates down to 3. Costs ~0.5–2s extra, no money, entirely local.

> **Intuition**: Recall wide, then precision-cut. Same pattern used in production search engines.

### ⚖️ LLM Judge Step
- After retrieval, a smaller/faster Groq model reads the retrieved chunks and answers: "Is this context actually sufficient to fully answer the question?"
- If NO: it proposes a rewritten query → retry (up to `MAX_RETRIEVAL_ATTEMPTS` times, default 1 to keep latency low)
- If YES: proceed to answer generation

> **Intuition**: Don't let the generator hallucinate its way through bad context. Catch it early at the judge step.

### 📑 Citation Enforcement
- Every chunk carries `source` (filename) + `page_number` metadata, attached at ingestion time
- The generator is **prompted** to cite `(Source: X, Page: Y)` after every factual claim
- `evaluate.py` independently checks whether cited pages were actually in the retrieved chunks (catches hallucinated citations)

> **Intuition**: The user needs to be able to trust the answer. Citations that can be verified against the source PDF are the trust mechanism.

### 🏠 All Local, Zero Cost
- Embeddings: `all-MiniLM-L6-v2` (sentence-transformers, local CPU)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local CPU)
- Vector DB: ChromaDB (local persistent files)
- LLM: Groq API (free tier, fast inference, JSON mode for the judge step)

> **Intuition**: Portfolio/demo scale doesn't need managed infra. Keep the bill at $0 and document what would need to change for production.

---

## How the Data Flows

```
PDF Upload
  ↓
PyMuPDF (fast) / pypdf (fallback) → per-page text with page numbers preserved
  ↓
Sliding-window chunking (800 chars, 150 overlap) → sentence-boundary-aware splits
  ↓
SentenceTransformers embeds each chunk → stored in ChromaDB with {source, page_number} metadata

User Question
  ↓
agent.answer_question()
  ↓
[1] retriever.retrieve() → ChromaDB vector search (wide pool) → cross-encoder rerank → top 3 chunks
  ↓
[2] _judge_sufficiency() → Groq (fast 8B model) → {sufficient: true/false, better_query: "..."}
  ↓ (if not sufficient & have better_query → rewrite query → go back to [1])
  ↓ (if sufficient or max attempts reached)
[3] _generate_cited_answer() → Groq (main model) → structured answer with (Source, Page) citations
  ↓
JSON response: {answer, sources_used[], retrieval_attempts[]}
```

---

## Known Intentional Limitations (Document these when demoing)

| Limitation | Why it exists | What would fix it |
|---|---|---|
| No OCR — scanned PDFs fail | Keeping scope small for v1 | Add `tesseract` + `pdf2image` in `ingest.py` |
| No conversation memory | Each question is stateless | Pass `chat_history` into the prompt in `agent.py` |
| No streaming responses | Groq non-streaming is simpler | Use `stream=True` + SSE on the frontend |
| Single global collection | No per-user isolation | Add `session_id` as a ChromaDB metadata filter |
| No explicit page limit | Only 100MB file size cap enforced | Add `if len(pages) > MAX_PAGES: raise ValueError(...)` in `ingest.py` |
| Cross-encoder disabled by default | Adds 0.5–2s latency | Set `RERANK_ENABLED=true` in `.env` for better accuracy |
| No delete/re-upload via UI | v1 scope | Backend already handles re-upload (deletes old chunks) |

---

## Configuration Mental Map

All tunable via `.env`:

| Var | Default | What it controls |
|---|---|---|
| `GROQ_API_KEY` | (required) | Access to Groq API |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Answer generation model |
| `GROQ_JUDGE_MODEL` | `llama-3.1-8b-instant` | Sufficiency judge model (can use a smaller faster one) |
| `MAX_RETRIEVAL_ATTEMPTS` | `1` | How many judge-retry loops before giving up |
| `TOP_K_CHUNKS` | `3` | Final chunks sent to the generator |
| `RERANK_ENABLED` | `false` | Enable cross-encoder reranking |
| `RERANK_CANDIDATE_POOL` | `6` | How many chunks vector search fetches before reranking |
| `MAX_FILE_SIZE_MB` | `100` | Upload size cap (~300–650 pages for typical text PDFs) |
| `CHUNK_SIZE` | `800` chars | Size of each text chunk |
| `CHUNK_OVERLAP` | `150` chars | Overlap to prevent context loss at chunk boundaries |

---

## Frontend Mental Model

The frontend is intentionally thin: HTML + vanilla JS + CSS. No framework.

Key UI concepts:
- **Pipeline visualizer**: Shows Searching → Reranking → Judging → Generating in real time while the query runs (purely cosmetic timer, not wired to real backend events)
- **Sources panel**: Collapsible, shows the actual chunk text used (so users can verify the LLM isn't making things up)
- **Agent trace**: Collapsible, shows how many retrieval attempts were made and why — the "proof of agentic behavior"
- **Background**: Subtle static dark radial glow — intentionally non-distracting

---

## What Would Change for Production

1. Replace ChromaDB local with Pinecone/Weaviate (multi-user, concurrent)
2. Add user auth + per-session document isolation
3. Enable streaming responses (SSE)
4. Move from Groq free tier to a production LLM endpoint with SLA
5. Add OCR for scanned PDFs
6. Add document delete UI
7. Tighten CORS `allow_origins` from `["*"]` to actual frontend domain
