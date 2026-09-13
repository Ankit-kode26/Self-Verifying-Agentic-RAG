"""
Ingestion pipeline: PDF -> text (with page numbers preserved) -> chunks
(with overlap) -> embeddings -> stored in local ChromaDB.

Page numbers are kept on every chunk's metadata so the agent can cite
"page 4" instead of just saying "the document says...".

Performance improvements:
- Uses PyMuPDF (fitz) as primary extractor — 3-10x faster than pypdf
- Falls back to pypdf if fitz is unavailable
- Reduces chunk size for faster embedding throughput
- Smaller overlap to reduce redundant chunk count
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions

from rag.config import settings


# ── Cached objects (avoid recreating on every call) ──────────
_embedder = None
_chroma_client = None
_collection = None


def _get_embedder():
    """Local, free sentence-transformers embedding function (no API calls).
    Cached after first call to avoid reloading the model repeatedly."""
    global _embedder
    if _embedder is None:
        _embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )
    return _embedder


def _get_collection():
    """Returns the ChromaDB collection, caching the client and collection
    objects to avoid the overhead of reconnecting on every request."""
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=_get_embedder(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _extract_pages_fast(pdf_path: str) -> List[Dict]:
    """Fast extraction using PyMuPDF (fitz) — 3-10x faster than pypdf.
    Falls back to pypdf if fitz is not installed."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text and text.strip():
                # Clean up common PDF extraction artifacts
                text = text.replace('\x00', '').strip()
                pages.append({"page_number": i, "text": text})
        doc.close()
        return pages
    except ImportError:
        return _extract_pages_pypdf(pdf_path)
    except Exception as e:
        # If fitz fails on this file, fall back to pypdf
        import logging
        logging.getLogger(__name__).warning(f"fitz extraction failed ({e}), falling back to pypdf")
        return _extract_pages_pypdf(pdf_path)


def _extract_pages_pypdf(pdf_path: str) -> List[Dict]:
    """Fallback extraction using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_number": i, "text": text})
    return pages


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple sliding-window chunking with overlap so context isn't
    lost at chunk boundaries (this matters a lot for long documents
    like contracts or research papers).
    
    Tries to split on sentence boundaries when possible for cleaner chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Try to break at a sentence boundary near the end
            for boundary in ['. ', '.\n', '! ', '? ', '\n\n']:
                idx = text.rfind(boundary, start + chunk_size // 2, end)
                if idx != -1:
                    end = idx + len(boundary)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _delete_existing_chunks(collection, source_name: str):
    """Remove all chunks from a previously ingested document so
    re-uploading the same file doesn't create duplicates."""
    try:
        existing = collection.get(
            where={"source": source_name},
            include=[]
        )
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        # If the collection is empty or source doesn't exist, that's fine
        pass


def _extract_text_file(file_path: str) -> List[Dict]:
    """Extracts text from plaintext files (.txt, .md, .csv) with encoding fallbacks,
    paginating into chunks of ~2500 characters so page citations work cleanly."""
    content = ""
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if not content or not content.strip():
        return []

    # Clean null bytes and carriage returns
    content = content.replace("\x00", "").replace("\r\n", "\n")

    # Split into logical pages (~2500 chars) preserving line breaks
    page_size = 2500
    pages = []
    lines = content.split("\n")
    current_page_text = []
    current_len = 0
    page_num = 1

    for line in lines:
        current_page_text.append(line)
        current_len += len(line) + 1
        if current_len >= page_size:
            text = "\n".join(current_page_text).strip()
            if text:
                pages.append({"page_number": page_num, "text": text})
                page_num += 1
            current_page_text = []
            current_len = 0

    if current_page_text:
        text = "\n".join(current_page_text).strip()
        if text:
            pages.append({"page_number": page_num, "text": text})

    return pages


def ingest_file(file_path: str, source_name: str | None = None) -> Dict:
    """
    Ingest a document (.pdf, .txt, .md): extract per-page text, chunk it,
    embed, and store in ChromaDB with metadata (source file name + page number).
    
    If a document with the same source_name already exists, its old
    chunks are deleted first to prevent duplicates on re-upload.
    """
    import logging
    logger = logging.getLogger(__name__)

    path = Path(file_path)
    source_name = source_name or path.name
    ext = path.suffix.lower()

    logger.info(f"Starting ingestion of {source_name} (type: {ext})")

    if ext == ".pdf":
        pages = _extract_pages_fast(file_path)
    elif ext in [".txt", ".md", ".csv", ".json", ".log"]:
        pages = _extract_text_file(file_path)
    else:
        # Try fast pdf first, then fallback to text
        try:
            pages = _extract_pages_fast(file_path)
        except Exception:
            pages = _extract_text_file(file_path)

    if not pages:
        raise ValueError(
            f"No extractable text found in {source_name}. "
            f"Please make sure the file contains readable text."
        )

    logger.info(f"Extracted {len(pages)} pages from {source_name}")

    collection = _get_collection()

    # Remove old chunks from this source if re-uploading
    _delete_existing_chunks(collection, source_name)

    ids, documents, metadatas = [], [], []
    for page in pages:
        for chunk in _chunk_text(page["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP):
            if len(chunk.strip()) < 50:  # Skip very short chunks (noise)
                continue
            ids.append(str(uuid.uuid4()))
            documents.append(chunk)
            metadatas.append({
                "source": source_name,
                "page_number": page["page_number"],
            })

    if not ids:
        raise ValueError(f"No usable text chunks extracted from {source_name}.")

    logger.info(f"Created {len(ids)} chunks, starting embedding...")

    # Batch insert in groups of 200 for faster throughput on typical docs
    # (smaller batches = more frequent progress updates)
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        logger.info(f"Embedded batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")

    logger.info(f"Ingestion complete: {source_name} — {len(pages)} pages, {len(ids)} chunks")

    return {
        "source": source_name,
        "pages_ingested": len(pages),
        "chunks_created": len(ids),
    }


# Backward compatibility alias
ingest_pdf = ingest_file


def list_ingested_sources() -> List[str]:
    """Returns unique source file names currently in the vector store."""
    collection = _get_collection()
    data = collection.get(include=["metadatas"])
    sources = {m["source"] for m in data["metadatas"]}
    return sorted(sources)


def delete_source(source_name: str) -> bool:
    """Removes all chunks of a source document from ChromaDB."""
    collection = _get_collection()
    _delete_existing_chunks(collection, source_name)
    return True

