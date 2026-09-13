"""
The core "agentic" piece. Plain RAG does: query -> retrieve -> generate.
This does: query -> retrieve -> LLM JUDGES if the retrieved chunks are
actually enough to answer -> if not, LLM REWRITES the query and retries
(up to MAX_RETRIEVAL_ATTEMPTS) -> only then generates a final answer,
and the answer is required to cite [source, page] for every claim.

This self-correction loop is what separates "agentic RAG" from a
plain vector-search-and-stuff-into-prompt pipeline.

Formatting improvements:
- Structured output with clear sections, numbered lists, paragraphs
- No leading dashes or em-dashes (removes AI-generated look)
- Rich chunk context shown in response for user transparency
"""
from __future__ import annotations

import json
import time
import logging
from typing import List, Dict

from groq import Groq

from rag.config import settings
from rag.retriever import retrieve

logger = logging.getLogger(__name__)
client = Groq(api_key=settings.GROQ_API_KEY)


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    model: str | None = None,
    history: List[Dict[str, str]] | None = None,
) -> str:
    """Call the Groq LLM with optional conversation history and model override."""
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=model or settings.GROQ_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        **kwargs,
    )
    return response.choices[0].message.content


def _contextualize_query(question: str, history: List[Dict[str, str]] | None = None) -> str:
    """
    Given the conversation history, reformulates follow-up questions
    (e.g., 'what about his second project?', 'elaborate on that') into a
    standalone, document-searchable query.
    If already standalone or no history, returns question as-is.
    """
    if not history:
        return question

    recent_history = history[-6:]
    history_lines = []
    for msg in recent_history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        # Truncate long assistant answers in history to keep token count low
        if len(content) > 300:
            content = content[:300] + "..."
        history_lines.append(f"{role}: {content}")

    history_text = "\n".join(history_lines)
    system_prompt = (
        "You are a query reformulation assistant for a document search engine. "
        "Given the recent conversation history and a user follow-up question, "
        "rephrase the follow-up question into a standalone, self-contained search query "
        "containing all necessary keywords, entity names, and topics needed to retrieve relevant document passages. "
        "If the question is already standalone, return it verbatim. "
        "Respond ONLY with the standalone search query text, without quotes or explanation."
    )
    user_prompt = f"Conversation History:\n{history_text}\n\nFollow-up Question: {question}"

    try:
        rewritten = _call_llm(system_prompt, user_prompt, model=settings.GROQ_MODEL)
        rewritten = rewritten.strip().strip('"').strip("'")
        if rewritten and len(rewritten) > 2:
            logger.info(f"Query contextualized: '{question}' -> '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query contextualization failed ({e}), using original question")

    return question


def _judge_sufficiency(question: str, chunks: List[Dict]) -> Dict:
    """
    Asks the LLM: is this retrieved context actually enough to answer
    the question fully and accurately? Returns a structured verdict.
    This is the "self-correction" checkpoint.
    """
    # Truncated to 200 chars per chunk — reduces input tokens
    # for faster judge inference without losing meaningful signal
    context_preview = "\n\n".join(
        f"[{c['source']} - page {c['page_number']}] {c['text'][:200]}"
        for c in chunks
    ) or "No context retrieved."

    system_prompt = (
        "You are a strict retrieval quality judge for a RAG system. "
        "Given a user question and retrieved context chunks, decide if the "
        "context is SUFFICIENT to fully and accurately answer the question. "
        "Respond ONLY in JSON with keys: "
        '{"sufficient": true|false, "reason": "short reason", '
        '"better_query": "a rewritten search query if not sufficient, else empty string"}'
    )
    user_prompt = f"Question: {question}\n\nRetrieved context:\n{context_preview}"

    raw = _call_llm(
        system_prompt,
        user_prompt,
        json_mode=True,
        model=settings.GROQ_JUDGE_MODEL,
    )
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Fail safe: if the judge itself breaks, treat context as sufficient
        # so the pipeline doesn't loop forever or crash.
        return {"sufficient": True, "reason": "judge_parse_failed", "better_query": ""}


def _generate_cited_answer(
    question: str, chunks: List[Dict], history: List[Dict[str, str]] | None = None
) -> str:
    """
    Generates the final answer with strict formatting rules and conversational continuity:
    - Structured output: bold headers, numbered lists, clear paragraphs
    - No leading dashes or em-dashes (removes AI-generated text feel)
    - Citation of [source, page] after every factual claim
    - Access to conversation history for follow-ups and clarifications
    """
    context_block = "\n\n".join(
        f"[Source: {c['source']}, Page: {c['page_number']}]\n{c['text']}"
        for c in chunks
    )

    system_prompt = (
        "You are a precise document Q&A assistant with conversational memory.\n"
        "You have access to the conversation history and the retrieved context chunks from the documents.\n\n"
        "STRICT FORMATTING RULES — follow these exactly:\n"
        "1. If the user question refers to previous conversation (e.g. 'what did you say earlier', 'elaborate on point 2', 'summarize your answer'), answer directly using the conversation history and context.\n"
        "2. Start with a short 1-2 sentence summary paragraph answering the question directly.\n"
        "3. If there are multiple points, use numbered lists: '1. ', '2. ', '3. ' etc.\n"
        "4. Group related points under **Bold Section Headers** when needed.\n"
        "5. Write in clear, direct sentences. Keep paragraphs to 2-3 sentences max.\n"
        "6. NEVER use a dash character (- or —) to start any line, sentence, or list item.\n"
        "7. NEVER use bullet points with dashes. Use numbered lists only.\n"
        "8. After every factual sentence or claim from the documents, add a citation: (Source: <name>, Page: <n>).\n"
        "9. If neither the retrieved context nor the conversation contains the answer, write exactly: "
        "'The provided documents do not contain enough information to answer this question.'\n"
        "10. Do not use outside knowledge. Do not make up page numbers.\n"
        "11. End with a concise summary sentence if the answer has multiple sections."
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    # Pass conversation history turns to Groq
    clean_history = []
    if history:
        for m in history[-6:]:
            role = m.get("role")
            content = m.get("content")
            if role in ["user", "assistant"] and content:
                clean_history.append({"role": role, "content": content})

    return _call_llm(system_prompt, user_prompt, history=clean_history)


def _clean_answer(text: str) -> str:
    """Post-process the answer to remove any remaining AI-generated artifacts
    like leading dashes that the model might still produce despite instructions."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.lstrip()
        # Replace leading dash+space with nothing (inline numbered format instead)
        if stripped.startswith('- ') or stripped.startswith('— '):
            indent = len(line) - len(stripped)
            line = ' ' * indent + stripped[2:]
        elif stripped.startswith('–'):
            indent = len(line) - len(stripped)
            line = ' ' * indent + stripped[1:].lstrip()
        cleaned.append(line)
    return '\n'.join(cleaned)


def answer_question(question: str, history: List[Dict[str, str]] | None = None) -> Dict:
    """
    Full agentic pipeline with conversational memory.
    1. Contextualize follow-up questions using conversation history into a standalone query.
    2. Retrieve relevant chunks using ChromaDB + Cross-Encoder reranker.
    3. Self-correcting loop: Judge context sufficiency and rewrite query if necessary.
    4. Generate cited answer maintaining conversation context.
    """
    pipeline_start = time.time()
    attempt_log = []

    # Step 1: Contextualize query if conversation history is available
    search_query = _contextualize_query(question, history)
    current_query = search_query
    chunks: List[Dict] = []

    # Step 2: Agentic retrieval loop with self-correction
    for attempt in range(1, settings.MAX_RETRIEVAL_ATTEMPTS + 1):
        t0 = time.time()
        chunks = retrieve(current_query)
        retrieval_ms = round((time.time() - t0) * 1000)

        if not chunks:
            attempt_log.append({
                "attempt": attempt, "query": current_query,
                "verdict": "no_chunks_found",
            })
            logger.info(f"Attempt {attempt}: no chunks found ({retrieval_ms}ms)")
            break

        t0 = time.time()
        verdict = _judge_sufficiency(current_query, chunks)
        judge_ms = round((time.time() - t0) * 1000)

        attempt_log.append({
            "attempt": attempt,
            "query": current_query,
            "sufficient": verdict.get("sufficient"),
            "reason": verdict.get("reason"),
        })

        logger.info(
            f"Attempt {attempt}: retrieval={retrieval_ms}ms, "
            f"judge={judge_ms}ms, sufficient={verdict.get('sufficient')}"
        )

        if verdict.get("sufficient"):
            break

        next_query = verdict.get("better_query") or current_query
        if next_query == current_query:
            break
        current_query = next_query

    # If chunks not found for the contextual query, try original query as fallback
    if not chunks and search_query != question:
        logger.info(f"Fallback search using original question: '{question}'")
        chunks = retrieve(question)
        if chunks:
            attempt_log.append({
                "attempt": len(attempt_log) + 1,
                "query": question,
                "verdict": "fallback_chunks_found",
            })

    # If still no chunks, but there is conversation history, check if the question is purely conversational
    if not chunks and history:
        t0 = time.time()
        raw_answer = _generate_cited_answer(question, [], history=history)
        final_answer = _clean_answer(raw_answer)
        return {
            "question": question,
            "answer": final_answer,
            "sources_used": [],
            "retrieval_attempts": attempt_log,
        }

    if not chunks:
        return {
            "question": question,
            "answer": "No relevant documents found. Please upload or ingest a document first.",
            "sources_used": [],
            "retrieval_attempts": attempt_log,
        }

    t0 = time.time()
    raw_answer = _generate_cited_answer(question, chunks, history=history)
    final_answer = _clean_answer(raw_answer)
    generate_ms = round((time.time() - t0) * 1000)
    total_ms = round((time.time() - pipeline_start) * 1000)

    logger.info(f"Answer generation: {generate_ms}ms | Total pipeline: {total_ms}ms")

    sources_used = [
        {
            "source": c["source"],
            "page_number": c["page_number"],
            "relevance_score": c["relevance_score"],
            "chunk_text": c["text"],
        }
        for c in chunks
    ]

    return {
        "question": question,
        "answer": final_answer,
        "sources_used": sources_used,
        "retrieval_attempts": attempt_log,
    }

