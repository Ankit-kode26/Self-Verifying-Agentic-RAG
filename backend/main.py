"""
FastAPI backend for the Agentic RAG system.
No UI here by design — Antigravity/your frontend calls these endpoints.
Run with: uvicorn main:app --reload --port 8000
Docs auto-generated at http://localhost:8000/docs
"""
import asyncio
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.config import settings
from rag.ingest import ingest_file, list_ingested_sources, delete_source
from rag.agent import answer_question

app = FastAPI(
    title="Agentic RAG API",
    description="Self-correcting, citation-enforced RAG over your documents with conversational memory.",
    version="1.1.0",
)

# Wide open for local dev / frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    history: Optional[list[ChatMessage]] = []


class SourceChunk(BaseModel):
    source: str
    page_number: int
    relevance_score: float
    chunk_text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources_used: list
    retrieval_attempts: list


def _secure_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal attacks while preserving
    allowed extension."""
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    stem = unicodedata.normalize("NFKD", stem)
    stem = stem.replace("\\", "/")
    stem = stem.split("/")[-1]
    stem = re.sub(r"[^\w\s\-.]", "", stem).strip()
    stem = re.sub(r"\s+", "_", stem)
    if not stem:
        stem = "document"
    return f"{stem}{ext}"


@app.on_event("startup")
async def warmup_models():
    """Pre-load the embedding model and reranker in background threads at
    startup so the first ingest/query request doesn't pay the cold-start penalty."""
    async def _warmup():
        try:
            from rag.ingest import _get_embedder as get_ingest_embedder
            from rag.retriever import _get_embedder as get_retriever_embedder
            from rag.retriever import _get_reranker

            import logging
            logger = logging.getLogger("warmup")
            logger.info("Warming up embedding model...")
            await asyncio.to_thread(get_ingest_embedder)
            await asyncio.to_thread(get_retriever_embedder)
            logger.info("Warming up reranker model...")
            await asyncio.to_thread(_get_reranker)
            logger.info("Models warmed up. Ingestion and retrieval ready.")
        except Exception as e:
            import logging
            logging.getLogger("warmup").warning(f"Warmup failed (non-fatal): {e}")

    asyncio.create_task(_warmup())


@app.on_event("startup")
async def start_render_keep_alive():
    """Keeps Render free-tier web services from going to sleep.
    Render spins down after 15 minutes of inactivity. This background loop
    sends a self-ping to its public health endpoint every 5 minutes (300s)."""
    async def _ping_loop():
        # Wait 60 seconds after server starts up
        await asyncio.sleep(60)

        # RENDER_EXTERNAL_URL is automatically populated by Render (e.g. https://xyz.onrender.com)
        external_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("KEEP_ALIVE_URL")
        if not external_url:
            return

        ping_url = f"{external_url.rstrip('/')}/health"
        import logging
        import httpx
        logger = logging.getLogger("keep_alive")
        logger.info(f"Render keep-alive loop started for {ping_url}")

        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(ping_url)
                    logger.info(f"Keep-alive ping to {ping_url} returned {resp.status_code}")
            except Exception as e:
                logger.warning(f"Keep-alive self-ping failed: {e}")

    asyncio.create_task(_ping_loop())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload a document (.pdf, .txt, .md). It gets chunked, embedded, and stored locally."""
    allowed_exts = {".pdf", ".txt", ".md", ".csv", ".json", ".log"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            400,
            f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(allowed_exts))}"
        )

    safe_name = _secure_filename(file.filename)
    save_path = Path(settings.UPLOAD_DIR) / safe_name

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "Uploaded file is empty.")
        if len(content) > max_bytes:
            raise HTTPException(
                413,
                f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB."
            )
        with open(save_path, "wb") as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to save file: {str(e)}")

    try:
        result = await asyncio.to_thread(
            ingest_file, str(save_path), source_name=safe_name
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    return result


@app.get("/documents")
def get_documents():
    """List all documents currently searchable."""
    return {"sources": list_ingested_sources()}


@app.delete("/documents/{source_name}")
def remove_document(source_name: str):
    """Remove a document from the vector store and disk."""
    safe_name = _secure_filename(source_name)
    delete_source(safe_name)
    file_path = Path(settings.UPLOAD_DIR) / safe_name
    file_path.unlink(missing_ok=True)
    return {"status": "deleted", "source": safe_name}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question with conversational history. Runs the full agentic
    retrieve -> judge -> retry -> cited-answer pipeline."""
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    history_dicts = (
        [{"role": m.role, "content": m.content} for m in request.history]
        if request.history
        else None
    )

    result = await asyncio.to_thread(
        answer_question, request.question, history=history_dicts
    )
    return result

