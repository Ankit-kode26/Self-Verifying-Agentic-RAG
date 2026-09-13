# Architecture — Agentic RAG

> Technical blueprint: component map, data flow, decision log, and file reference.
> For design intuition and *why*, read `brain.md` first.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENTIC RAG SYSTEM                           │
│                                                                     │
│   ┌──────────┐   HTTP POST /ingest    ┌────────────────────────┐   │
│   │ Browser  │ ──────────────────────▶│  FastAPI (main.py)     │   │
│   │ Frontend │                        │  :8000                 │   │
│   │(HTML/JS) │ ◀── JSON response ─────│                        │   │
│   │          │                        └────────┬───────────────┘   │
│   │          │   HTTP POST /query              │                    │
│   │          │ ──────────────────────▶         │                    │
│   └──────────┘                        ┌────────▼───────────────┐   │
│                                       │     rag/ package       │   │
│                                       │  ┌─────────────────┐   │   │
│                                       │  │   ingest.py     │   │   │
│                                       │  │  PDF→chunks→    │   │   │
│                                       │  │  embed→store    │   │   │
│                                       │  └────────┬────────┘   │   │
│                                       │           │             │   │
│                                       │  ┌────────▼────────┐   │   │
│                                       │  │  retriever.py   │   │   │
│                                       │  │  vector search  │   │   │
│                                       │  │  + reranking    │   │   │
│                                       │  └────────┬────────┘   │   │
│                                       │           │             │   │
│                                       │  ┌────────▼────────┐   │   │
│                                       │  │   agent.py      │   │   │
│                                       │  │  judge→retry→   │   │   │
│                                       │  │  generate loop  │   │   │
│                                       │  └────────┬────────┘   │   │
│                                       └───────────┼────────────┘   │
│                                                   │                 │
│   ┌───────────────────────┐   ┌───────────────────▼─────────────┐  │
│   │  ChromaDB (local)     │   │        Groq API (cloud)         │  │
│   │  data/chroma_db/      │   │  llama-3.1-8b-instant           │  │
│   │  Persistent vector DB │   │  - Judge step (fast)            │  │
│   └───────────────────────┘   │  - Answer generation            │  │
│                                └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Map

### `backend/main.py` — API Layer
FastAPI application. Owns all HTTP endpoints. Delegates all logic to `rag/`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/ingest` | POST | Upload & ingest a PDF |
| `/documents` | GET | List ingested document names |
| `/query` | POST | Run the full agentic Q&A pipeline |

Key concerns handled here:
- File size enforcement (100MB cap, streamed in 64KB chunks)
- Filename sanitization (prevents path traversal)
- CORS (open for local dev; tighten before any real deployment)
- Background model warm-up at startup

---

### `backend/rag/config.py` — Settings
Single `Settings` class. All values read from `.env` via `os.getenv`.
Fails fast at import if `GROQ_API_KEY` is missing.

---

### `backend/rag/ingest.py` — PDF Ingestion Pipeline

```
PDF file path
   ↓
_extract_pages_fast()   ← PyMuPDF primary, pypdf fallback
   ↓
List[{page_number: int, text: str}]
   ↓
_chunk_text()           ← sliding window, sentence-boundary-aware
   ↓
List[str] chunks (skip < 50 chars)
   ↓
ChromaDB collection.add()
   with metadata: {source: filename, page_number: int}
```

**Idempotent**: Re-uploading the same filename deletes old chunks before re-inserting (via `_delete_existing_chunks()`).

---

### `backend/rag/retriever.py` — Two-Stage Retrieval

```
query string
   ↓
Stage 1: ChromaDB cosine similarity search
         → RERANK_CANDIDATE_POOL chunks (default 6)
   ↓
Stage 2: CrossEncoder scores each (query, chunk) pair
         → sorts by relevance_score
         → keeps TOP_K_CHUNKS (default 3)
   ↓
List[{text, source, page_number, relevance_score}]
```

If `RERANK_ENABLED=false`, Stage 2 is skipped; `vector_score` is used as `relevance_score`.

---

### `backend/rag/agent.py` — Agentic Judge-Retry-Generate Loop

```
question
   ↓
FOR attempt in range(MAX_RETRIEVAL_ATTEMPTS):
    chunks = retriever.retrieve(current_query)
    if no chunks → break
    verdict = _judge_sufficiency(question, chunks)
    if sufficient → break
    if better_query proposed → current_query = better_query → continue
    else → break  (judge didn't improve query, no point looping)
   ↓
_generate_cited_answer(question, chunks)  ← Groq main model
   ↓
_clean_answer()  ← strip leading dashes from LLM output
   ↓
{question, answer, sources_used[], retrieval_attempts[]}
```

---

## Data Flow — End to End

### Ingestion (upload)
```
Browser → POST /ingest (multipart PDF)
  → stream to data/uploads/<sanitized_name>.pdf
  → ingest_pdf() in thread pool
  → PyMuPDF extracts pages
  → chunk each page
  → embed all chunks (sentence-transformers, local)
  → store in ChromaDB with {source, page_number} metadata
  ← {source, pages_ingested, chunks_created}
```

### Query
```
Browser → POST /query {question: "..."}
  → answer_question() in thread pool
  → retrieve() → ChromaDB + CrossEncoder → top 3 chunks
  → _judge_sufficiency() → Groq 8B → {sufficient, better_query}
  → if not sufficient & better_query → retry retrieve()
  → _generate_cited_answer() → Groq main → answer text
  → _clean_answer() → post-process
  ← {answer, sources_used[], retrieval_attempts[]}
```

---

## File Structure

```
agentic-rag/
├── brain.md                     ← mental model, design intent (READ FIRST)
├── Architecture.md              ← this file: technical blueprint
├── start.ps1                    ← PowerShell script to start both frontend + backend
│
├── backend/
│   ├── main.py                  ← FastAPI app, HTTP endpoints
│   ├── evaluate.py              ← groundedness evaluation (offline script)
│   ├── requirements.txt
│   ├── .env                     ← secrets + config (not committed)
│   ├── test_questions.sample.json
│   └── rag/
│       ├── config.py            ← Settings: all env vars in one place
│       ├── ingest.py            ← PDF → chunks → embeddings → ChromaDB
│       ├── retriever.py         ← vector search + cross-encoder rerank
│       └── agent.py            ← judge → retry → cited answer
│
├── data/
│   ├── uploads/                 ← uploaded PDFs land here
│   └── chroma_db/               ← ChromaDB persistent store (auto-created)
│
├── frontend/
│   ├── index.html               ← app shell + DOM structure
│   ├── styles.css               ← design system + all styles
│   └── app.js                   ← all JS logic (no framework)
│
└── docs/
    ├── PRD.md                   ← product requirements
    ├── ARCHITECTURE.md          ← original architecture notes
    └── API_SPEC.md              ← HTTP API reference
```

---

## Key Design Decisions (Decision Log)

| Decision | Alternatives Considered | Why This |
|---|---|---|
| Local embeddings (sentence-transformers) | OpenAI Embeddings API | $0 cost, no rate limits, fine for demo scale |
| ChromaDB local persistent | Pinecone, Weaviate, pgvector | $0, no account, easy setup; documented as scale limitation |
| Groq for LLM | OpenAI, Anthropic, local Ollama | Free tier, fast inference (~1–3s), JSON mode for judge step |
| Two-stage retrieval (vector + reranker) | Single-stage vector only | Measurably better precision; reranker is free and local |
| Judge-retry capped at MAX_RETRIEVAL_ATTEMPTS | Uncapped retry | Prevents infinite loops; 1 attempt is fast enough for most questions |
| PyMuPDF as primary PDF extractor | pypdf only | 3–10× faster extraction; pypdf kept as fallback |
| Sliding-window chunking with sentence boundaries | Fixed-size chunks, semantic chunking | Good balance of simplicity vs. context preservation; no external library needed |
| Per-chunk metadata (source + page_number) | Document-level metadata only | Page-level citation is only possible if metadata is stored per chunk at ingest |
| No streaming responses (v1) | SSE streaming token-by-token | Simpler implementation; streaming is a clear v2 upgrade path |
| Vanilla JS frontend (no framework) | React, Vue | Minimal bundle, no build step, easier to deploy as a static file |

---

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Scanned/image PDFs fail (no OCR) | High — common PDF type | Add tesseract + pdf2image in v2 |
| No page limit (only 100MB file size cap) | Medium — very large docs slow to ingest | Add `MAX_PAGES` check in `ingest.py` if needed |
| Single global ChromaDB collection | High for multi-user | Add session/user filter on queries |
| No conversation memory | Medium | Pass history into `_generate_cited_answer` prompt |
| No streaming | Low (UX only) | Add SSE on `/query` + EventSource on frontend |
| CORS open `allow_origins=["*"]` | High for production | Lock down to specific origin before any deployment |
| Cross-encoder disabled by default (`RERANK_ENABLED=false`) | Medium — lower retrieval accuracy | Enable in `.env` for better results (costs ~1s latency) |
