"""
Retrieval layer, now a two-stage pipeline:

Stage 1 (recall): ChromaDB vector search pulls a WIDE candidate pool
(RERANK_CANDIDATE_POOL chunks) using fast cosine similarity. Vector
search is good at finding "probably relevant" chunks quickly but is
not great at fine-grained ranking.

Stage 2 (precision): a cross-encoder model reads the actual question
together with each candidate chunk (not just their embeddings) and
scores true relevance far more accurately. Only the top TOP_K_CHUNKS
after reranking are kept and sent to the LLM.

This two-stage design is a standard production RAG pattern: cheap
wide recall, then expensive-but-accurate reranking on a small set.
Both models run locally -- $0 cost, no extra API.
"""
from __future__ import annotations

from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

from rag.config import settings

# ── Cached objects (loaded once, reused across requests) ─────
# Creating a PersistentClient + collection on every request was a
# major performance bottleneck — each call re-opened the DB file.
_reranker: CrossEncoder | None = None
_embedder = None
_chroma_client = None
_collection = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.RERANK_MODEL)
    return _reranker


def _get_embedder():
    """Cached embedding function — avoids reloading the model per request."""
    global _embedder
    if _embedder is None:
        _embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )
    return _embedder


def _get_collection():
    """Returns the cached ChromaDB collection. The client and collection
    are created once and reused — this alone saves ~200ms per query."""
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


def _vector_search(query: str, n_results: int) -> List[Dict]:
    """Stage 1: fast, wide recall via embedding similarity."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "page_number": meta["page_number"],
            "vector_score": round(1 - distance, 4),
        })
    return chunks


def _rerank(query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    """Stage 2: cross-encoder re-scores each (query, chunk) pair directly."""
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c["relevance_score"] = round(float(score), 4)
        c.pop("vector_score", None)

    candidates.sort(key=lambda c: c["relevance_score"], reverse=True)
    return candidates[:top_k]


def retrieve(query: str, top_k: int | None = None) -> List[Dict]:
    """
    Returns a list of chunks:
    [{text, source, page_number, relevance_score}, ...]
    sorted by relevance (best first).

    If RERANK_ENABLED is true (default), relevance_score is the
    cross-encoder's score (typically -10 to 10 range, higher = better
    match). If reranking is off, relevance_score falls back to
    1 - cosine_distance from plain vector search.
    """
    top_k = top_k or settings.TOP_K_CHUNKS

    if not settings.RERANK_ENABLED:
        candidates = _vector_search(query, top_k)
        for c in candidates:
            c["relevance_score"] = c.pop("vector_score")
        return candidates

    candidates = _vector_search(query, settings.RERANK_CANDIDATE_POOL)
    return _rerank(query, candidates, top_k)
