"""Chat API — retrieval + generation orchestration.

Design notes:
* Real vector search over the Chroma collection (embeddings stored at
  ingestion are actually queried now); lexical scan remains as a *logged*
  fallback, not a silent one.
* Failures raise proper HTTP errors (503) instead of HTTP 200 with
  an error field; no fabricated answers.
* Response carries `source_hashes` so the frontend approve flow can
  maintain cache invalidation.
* Link expansion: linked documents (annex/parent/sibling) widen the candidate
  pool before fusion — no score manipulation, answers still cite real chunks.
"""

from __future__ import annotations

import logging
import time
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core.config import get_config
from backend.services.agentic_refinement import maybe_refine
from backend.services.answer_generator import generate_answer
from backend.services.approved_answers import get_cached_answer
from backend.services.hybrid_retrieval import looks_like_toc, hybrid_search

LOG = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

CANDIDATE_MULTIPLIER = 4  # vector-search candidates per final top_k


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)


def _open_db():
    """Open the topic DB, ensuring the schema (chunks/document_links) exists.

    init_db() is idempotent (CREATE TABLE IF NOT EXISTS), so chat stays robust
    even when invoked before any ingestion ran.
    """
    from backend.core.database import get_connection, init_db

    init_db()
    return get_connection()


def _chunks_from_query_results(res: dict) -> list[dict]:
    """Normalize a chroma query() result into chunk dicts."""
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    raw_embs = res.get("embeddings")
    embs = (
        [[float(x) for x in vec] for vec in raw_embs[0]]
        if raw_embs is not None and len(raw_embs)
        else []
    )
    chunks = []
    for idx, (doc, meta) in enumerate(zip(docs, metas, strict=False)):
        meta = meta or {}
        chunks.append(
            {
                "text": doc,
                "chunk_id": meta.get("chunk_id", ""),
                "file_hash": meta.get("file_hash", ""),
                "file_name": meta.get("file_name", ""),
                "page_number": meta.get("page_number", 1),
                "section_number": meta.get("section_number", ""),
                "kind": meta.get("kind", "chunk"),
                "embedding": embs[idx] if idx < len(embs) else None,
            }
        )
    return chunks


def _chunks_from_get_results(res: dict) -> list[dict]:
    docs = res.get("documents", []) or []
    metas = res.get("metadatas", []) or []
    raw_embs = res.get("embeddings")
    embs = (
        [[float(x) for x in vec] for vec in raw_embs]
        if raw_embs is not None
        else []
    )
    chunks = []
    for idx, (doc, meta) in enumerate(zip(docs, metas, strict=False)):
        meta = meta or {}
        chunks.append(
            {
                "text": doc,
                "chunk_id": meta.get("chunk_id", ""),
                "file_hash": meta.get("file_hash", ""),
                "file_name": meta.get("file_name", ""),
                "page_number": meta.get("page_number", 1),
                "section_number": meta.get("section_number", ""),
                "kind": meta.get("kind", "chunk"),
                "embedding": embs[idx] if idx < len(embs) else None,
            }
        )
    return chunks


def _attach_embeddings(chunks: list[dict]) -> list[dict]:
    """Fetch stored embeddings for candidates that arrived without one.

    Vector hits carry their embedding; chunks added by link expansion come from
    the SQLite ``chunks`` table. Without this step the dense half of the fusion
    would silently degrade to the lexical score for those candidates (the
    original bug).
    """
    from backend.services.qa_generator import get_or_create_collection

    missing = [c for c in chunks if not c.get("embedding") and c.get("chunk_id")]
    if not missing:
        return chunks
    col = get_or_create_collection()
    if col is None:
        LOG.warning("vector store unavailable — %d candidates keep lexical-only scoring", len(missing))
        return chunks
    unique_ids = list(dict.fromkeys(str(c["chunk_id"]) for c in missing if c.get("chunk_id")))
    try:
        got = col.get(ids=unique_ids, include=["embeddings"])
    except Exception:
        LOG.warning("embedding lookup failed for %d link-expanded chunks", len(missing), exc_info=True)
        return chunks
    ids = list(got.get("ids") or [])
    raw = got.get("embeddings")
    by_id = {
        str(cid): [float(x) for x in emb]
        for cid, emb in zip(ids, raw if raw is not None else [], strict=False)
        if emb is not None
    }
    attached = 0
    for c in missing:
        emb = by_id.get(str(c.get("chunk_id")))
        if emb is not None:
            c["embedding"] = emb
            attached += 1
    if attached < len(missing):
        LOG.warning("%d/%d link-expanded chunks have no stored embedding", len(missing) - attached, len(missing))
    else:
        LOG.info("attached stored embeddings to %d link-expanded chunks", attached)
    return chunks


def _resolve_question_chunks(chunks: list[dict]) -> list[dict]:
    """Replace kind='question' candidates with their real chunk text.

    Question embeddings widen recall only; the LLM answers from the actual
    chunk text (resolved from the chunks table). Candidates whose chunk row
    is gone are dropped.
    """
    from backend.core.database import get_chunk_text

    if not any(c.get("kind") == "question" for c in chunks):
        return chunks
    out: list[dict] = []
    conn = _open_db()
    try:
        for c in chunks:
            if c.get("kind") != "question":
                out.append(c)
                continue
            row = get_chunk_text(conn, c.get("chunk_id", ""))
            if row is None:
                LOG.warning("question candidate %r has no chunk row — dropped", c.get("chunk_id"))
                continue
            out.append(row)
    finally:
        conn.close()
    return out


def _expand_with_linked_chunks(chunks: list[dict], max_chunks: int) -> list[dict]:
    """Widen the candidate pool with chunks of linked documents.

    Documents linked to any retrieved hash (annex/parent/sibling) contribute
    up to ``max_chunks`` additional candidates BEFORE fusion — no score
    boost; fusion still ranks every candidate on its own merit.
    """
    from backend.core.database import get_chunks_by_hash, get_linked_hashes

    if not chunks:
        return chunks
    seed_hashes = {str(c.get("file_hash")) for c in chunks if c.get("file_hash")}
    if not seed_hashes:
        return chunks
    conn = _open_db()
    try:
        linked = get_linked_hashes(conn, seed_hashes)
        extra_hashes: set[str] = set()
        for h in seed_hashes:
            extra_hashes.update(linked.get(h, []))
        extra_hashes -= seed_hashes
        if not extra_hashes:
            return chunks
        out = list(chunks)
        added = 0
        for h in sorted(extra_hashes):
            if added >= max_chunks:
                break
            for row in get_chunks_by_hash(conn, h):
                if added >= max_chunks:
                    break
                out.append(row)
                added += 1
        if added:
            LOG.info("link expansion: +%d chunks from %d linked documents", added, len(extra_hashes))
        return out
    finally:
        conn.close()


def _drop_toc(chunks: list[dict]) -> list[dict]:
    """Remove table-of-contents candidates — they carry titles, not rules."""
    kept = [c for c in chunks if not looks_like_toc(str(c.get("text", "")))]
    dropped = len(chunks) - len(kept)
    if dropped:
        LOG.info("dropped %d table-of-contents candidate(s)", dropped)
    return kept


def _retrieve(query: str, top_k: int) -> list[dict]:
    """Vector search with an explicit, logged lexical fallback."""
    from backend.services.qa_generator import get_or_create_collection

    col = get_or_create_collection()
    if col is None:
        LOG.warning("vector store unavailable — no chunks can be retrieved")
        return []
    try:
        from backend.services.ollama_client import get_ollama_client

        qvec = get_ollama_client().embed(query)[0]
        res = col.query(
            query_embeddings=[qvec],
            n_results=max(top_k * CANDIDATE_MULTIPLIER, 20),
            include=["documents", "metadatas", "embeddings"],
        )
        return _drop_toc(_chunks_from_query_results(res))
    except Exception:
        LOG.warning("vector query failed — falling back to full lexical scan", exc_info=True)
        try:
            res = col.get(limit=1000, include=["documents", "metadatas", "embeddings"])
            return _drop_toc(_chunks_from_get_results(res))
        except Exception:
            LOG.error("lexical fallback retrieval also failed", exc_info=True)
            return []


def _related_documents(hashes: set[str]) -> list[dict]:
    """Resolve links touching the cited hashes into UI-friendly entries."""
    from backend.core.database import get_links_for_hash

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    if not hashes:
        return out
    conn = _open_db()
    try:
        for h in sorted(hashes):
            for link in get_links_for_hash(conn, h):
                other_hash, other_name, link_type = _other_side(link, h)
                key = (other_hash, link_type)
                if other_name and key not in seen:
                    seen.add(key)
                    out.append({"file_name": other_name, "file_hash": other_hash, "link_type": link_type})
    finally:
        conn.close()
    return out


def _other_side(link: dict, known_hash: str) -> tuple[str, str, str]:
    if link["source_hash"] == known_hash:
        return link["target_hash"], link.get("target_name") or "", link["link_type"]
    return link["source_hash"], link.get("source_name") or "", link["link_type"]


@router.post("")
def chat(req: ChatRequest):
    cfg = get_config()
    start = time.time()
    timings: dict[str, int] = {}

    # 1. approved cache
    t0 = time.time()
    cached = get_cached_answer(req.query)
    timings["cache_ms"] = int((time.time() - t0) * 1000)
    if cached:
        return {
            "answer": cached["answer"],
            "citations": [],
            "pdf_links": [],
            "source_hashes": cached.get("source_hashes", []),
            "cached": True,
            "timings": timings,
        }

    # 2. retrieve candidates (vector search, logged lexical fallback)
    t0 = time.time()
    chunks = _retrieve(req.query, cfg.retrieval.top_k)
    chunks = _resolve_question_chunks(chunks)
    if cfg.retrieval.link_expansion:
        chunks = _expand_with_linked_chunks(chunks, cfg.retrieval.link_expansion_max_chunks)
    chunks = _drop_toc(chunks)
    chunks = _attach_embeddings(chunks)
    timings["retrieve_ms"] = int((time.time() - t0) * 1000)

    # 3. hybrid fusion + dedup
    t0 = time.time()
    ranked = hybrid_search(req.query, chunks) if chunks else []
    timings["hybrid_ms"] = int((time.time() - t0) * 1000)

    # 4. stage 3 refinement — bounded so a slow model can never consume the
    # generation budget: reserve room for generation, skip the LLM step when
    # what remains is too small to be useful.
    t0 = time.time()
    if ranked:
        elapsed_before_refine = time.time() - start
        gen_reserve = max(35.0, cfg.performance.chat_timeout_seconds * 0.3)
        stage3_budget = cfg.performance.chat_timeout_seconds - elapsed_before_refine - gen_reserve
        stage3_budget = max(0.0, min(15.0, stage3_budget))
        ranked = maybe_refine(
            req.query,
            ranked,
            chunks,
            stage3_budget_s=stage3_budget,
            retriever=lambda q: _attach_embeddings(_retrieve(q, cfg.retrieval.top_k)),
        )
    timings["stage3_ms"] = int((time.time() - t0) * 1000)

    # 5. answer generation with remaining budget enforced on the call itself
    elapsed = time.time() - start
    remaining = cfg.performance.chat_timeout_seconds - elapsed
    if remaining <= 0:
        raise HTTPException(status_code=503, detail="chat timeout exceeded before generation")
    t0 = time.time()
    try:
        result = generate_answer(req.query, ranked[: cfg.retrieval.top_k], timeout_s=remaining)
    except Exception as exc:
        LOG.error("chat failed for query=%r: %s", req.query[:100], exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"answer generation failed: {exc}") from exc
    timings["generate_ms"] = int((time.time() - t0) * 1000)
    timings["total_ms"] = int((time.time() - start) * 1000)

    cited = ranked[:3] if ranked else []
    if result.get("refusal"):
        # a refusal must stand alone: no document buttons, no related docs
        cited, source_hashes = [], []
    pdf_links = [
        f"/data/{quote(str(c.get('file_name', '')))}#page={c.get('page_number', 1)}"
        for c in cited
        if c.get("file_name")
    ]
    source_hashes = sorted({str(c.get("file_hash")) for c in cited if c.get("file_hash")})

    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "pdf_links": pdf_links,
        "source_hashes": source_hashes,
        "related_documents": _related_documents(set(source_hashes)),
        "cached": False,
        "refusal": result.get("refusal", False),
        "timings": timings,
    }
