"""Approved-QA cache with semantic extension.

Two lookup tiers, both guarded by hash-validity auto-invalidation:
1. exact query_text match (original behavior);
2. semantic match: the approved query embedding vs the current query embedding
   (bge-m3 cosine ≥ performance.cache_semantic_threshold) — a later question
   that is very similar and addresses the same topic is served instantly.

Embedding failures never break the chat flow: the answer is cached
exact-match only (logged, never silently swallowed).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path

from backend.core.config import get_config
from backend.core.database import (
    add_answer_feedback,
    add_approved_answer,
    delete_approved_answer,
    get_all_valid_hashes,
    get_connection,
    init_db,
    list_approved_answers_for,
    list_approved_semantic_candidates,
)

LOG = logging.getLogger(__name__)


def _ollama_client():
    from backend.services.ollama_client import get_ollama_client

    return get_ollama_client()


def approve_answer(
    query_text: str,
    approved_response: str,
    source_hashes: list[str],
    db_path: Path | str | None = None,
) -> int:
    """Cache an approved answer; also embed the query for semantic lookup."""
    init_db(db_path)
    conn = get_connection(db_path)
    emb_json: str | None = None
    try:
        try:
            vec = _ollama_client().embed(query_text)[0]
            emb_json = json.dumps(vec, ensure_ascii=False)
        except Exception:
            LOG.error("approve: query embedding failed — answer cached exact-match only", exc_info=True)
        return add_approved_answer(
            conn,
            query_text=query_text,
            approved_response=approved_response,
            source_hashes=source_hashes,
            query_embedding=emb_json,
        )
    finally:
        conn.close()


def _payload(row: dict) -> dict:
    return {
        "answer": row["approved_response"],
        "source_hashes": json.loads(row["source_hashes"]),
        "approved_at": row["approved_at"],
        "cached": True,
    }


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_lookup(
    conn: sqlite3.Connection,
    query_text: str,
    valid_hashes: set[str],
    ollama_client=None,
) -> dict | None:
    threshold = get_config().performance.cache_semantic_threshold
    candidates = list_approved_semantic_candidates(conn, valid_hashes=valid_hashes)
    if not candidates:
        return None
    if ollama_client is None:
        try:
            ollama_client = _ollama_client()
        except Exception:
            LOG.warning("semantic cache: ollama unavailable — treating as cache miss", exc_info=True)
            return None
    try:
        qvec = ollama_client.embed(query_text)[0]
    except Exception:
        LOG.warning("semantic cache: query embedding failed — treating as cache miss", exc_info=True)
        return None
    best, best_sim = None, threshold
    for row in candidates:
        sim = _cosine(qvec, row["query_embedding_vec"])
        if sim >= best_sim:
            best, best_sim = row, sim
    if best is None:
        LOG.info("semantic cache miss for query=%r (%d candidates, threshold %.2f)", query_text[:80], len(candidates), threshold)
        return None
    LOG.info("semantic cache hit similarity=%.3f for query=%r", best_sim, query_text[:80])
    return _payload(best)


def get_cached_answer(query_text: str, db_path: Path | str | None = None) -> dict | None:
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        valid = get_all_valid_hashes(conn)
        rows = list_approved_answers_for(conn, query_text, valid_hashes=valid)
        if rows:
            return _payload(rows[0])
        return _semantic_lookup(conn, query_text, valid)
    finally:
        conn.close()


def record_incorrect(
    query_text: str,
    response_text: str,
    source_hashes: list[str],
    comment: str = "",
    db_path: Path | str | None = None,
    ollama_client=None,
) -> dict:
    """Record an 'incorrect' verdict and retire every cache row that serves it.

    Learning rule (no heuristics, mirrors the serving paths exactly):
    * exact query_text matches always serve this answer → retired;
    * semantic candidates serve their own stored response → retired only when
      it equals the flagged response AND similarity is above the serving
      threshold (i.e. rows that would have served this same wrong answer).
    The next identical/similar query therefore regenerates fresh instead of
    re-serving the flagged answer. The verdict row itself is the audit trail.

    Embedding failures only skip the semantic tier (logged); exact retirement
    never depends on Ollama.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        valid = get_all_valid_hashes(conn)
        retired: list[int] = []
        for row in list_approved_answers_for(conn, query_text, valid_hashes=valid):
            retired.append(int(row["id"]))
        semantic_ids = _semantic_match_ids(conn, query_text, response_text, valid, ollama_client)
        retired.extend(i for i in semantic_ids if i not in retired)
        for row_id in retired:
            delete_approved_answer(conn, row_id)
        if retired:
            LOG.warning("incorrect verdict retired %d cached answer(s) for query=%r", len(retired), query_text[:80])
        fid = add_answer_feedback(
            conn,
            query_text=query_text,
            response_text=response_text,
            source_hashes=source_hashes,
            verdict="incorrect",
            comment=comment,
            retired_answer_ids=retired,
        )
        return {"feedback_id": fid, "retired_answer_ids": sorted(retired)}
    finally:
        conn.close()


def _semantic_match_ids(
    conn: sqlite3.Connection,
    query_text: str,
    response_text: str,
    valid_hashes: set[str],
    ollama_client=None,
) -> list[int]:
    """Cache row ids that would serve ``response_text`` for a similar query."""
    threshold = get_config().performance.cache_semantic_threshold
    candidates = list_approved_semantic_candidates(conn, valid_hashes=valid_hashes)
    if not candidates:
        return []
    if ollama_client is None:
        try:
            ollama_client = _ollama_client()
        except Exception:
            LOG.warning("incorrect verdict: ollama unavailable — semantic tier not retired", exc_info=True)
            return []
    try:
        qvec = ollama_client.embed(query_text)[0]
    except Exception:
        LOG.warning("incorrect verdict: query embedding failed — semantic tier not retired", exc_info=True)
        return []
    return [
        int(row["id"])
        for row in candidates
        if row.get("approved_response") == response_text
        and _cosine(qvec, row["query_embedding_vec"]) >= threshold
    ]
