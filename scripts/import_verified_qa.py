#!/usr/bin/env python3
"""Ingest HITL-approved synthetic Q&A pairs into both retrieval targets.

Input: approved JSONL exported from frontend/review_qa.py
(status in {approved, edited}; edited rows use edited_answer).

Target A — recall: embed the approved *question* into the Chroma
``regurag_chunks`` collection as kind='question' (id = qa_id, metadata maps
back via chunk_id). Answers never become retrievable documents — the chat
path resolves question hits to real chunk text before generation.

Target B — instant answers: insert into SQLite ``approved_answers`` with the
approved response + source_hashes, reusing the same query embedding
(exact-match + semantic cache per backend/services/approved_answers.py).

Usage:
    python -m scripts.import_verified_qa --dry-run
    python -m scripts.import_verified_qa --input artifacts/synthetic_qa_pilot.approved.jsonl
    python -m scripts.import_verified_qa --targets cache   # semantic cache only

Idempotent by default: qa_ids already in Chroma and exact (query, response)
rows already cached are skipped (use --no-resume to force re-write).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger("import_qa")

APPROVED_STATUSES = ("approved", "edited")


def load_approved(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                LOG.warning("line %d: unreadable JSON — skipped", lineno)
                continue
            if r.get("status") not in APPROVED_STATUSES:
                LOG.warning("line %d: status %r not approved — skipped", lineno, r.get("status"))
                continue
            if not r.get("question") or not pick_response(r):
                LOG.warning("line %d: empty question/answer — skipped", lineno)
                continue
            rows.append(r)
    return rows


def pick_response(record: dict) -> str:
    """Edited answer wins when present, else the generated answer."""
    edited = (record.get("edited_answer") or "").strip()
    return edited if edited else (record.get("answer") or "").strip()


def vector_item(record: dict) -> tuple[str, str, dict]:
    """(id, document, metadata) for the Chroma question row."""
    return (
        record["qa_id"],
        record["question"].strip(),
        {
            "chunk_id": record["chunk_id"],
            "file_hash": record["file_hash"],
            "file_name": record["file_name"],
            "page_number": record["page_number"],
            "section_number": record.get("section_number", ""),
            "kind": "question",
            "verified": True,
        },
    )


def import_vectors(records: list[dict], client, persist_dir, *, resume: bool) -> dict:
    from backend.services.qa_generator import CHROMA_FLUSH_BATCH, get_or_create_collection

    col = get_or_create_collection(persist_dir=persist_dir)
    stats = {"written": 0, "skipped": 0, "failed": 0}
    if col is None:
        LOG.error("no Chroma collection available — vector import aborted")
        stats["failed"] = len(records)
        return stats
    items = [vector_item(r) for r in records]
    if resume:
        try:
            existing = set(col.get(ids=[i[0] for i in items], include=[])["ids"] or [])
        except Exception:
            LOG.warning("chroma pre-check failed — writing all rows", exc_info=True)
            existing = set()
        fresh, skipped = [i for i in items if i[0] not in existing], [i for i in items if i[0] in existing]
        stats["skipped"] = len(skipped)
        items = fresh
    for start in range(0, len(items), CHROMA_FLUSH_BATCH):
        batch = items[start : start + CHROMA_FLUSH_BATCH]
        try:
            vecs = client.embed([doc for _, doc, _ in batch])
        except Exception:
            LOG.error("embedding failed for batch of %d — skipped", len(batch), exc_info=True)
            stats["failed"] += len(batch)
            continue
        try:
            col.upsert(
                ids=[i for i, _, _ in batch],
                documents=[d for _, d, _ in batch],
                metadatas=[m for _, _, m in batch],  # type: ignore[arg-type]
                embeddings=vecs,  # type: ignore[arg-type]
            )
            stats["written"] += len(batch)
        except Exception:
            LOG.error("chroma upsert failed for batch of %d", len(batch), exc_info=True)
            stats["failed"] += len(batch)
    return stats


def import_cache(records: list[dict], client, db_path, *, resume: bool) -> dict:
    import json as _json

    from backend.core.database import add_approved_answer, get_connection, init_db, list_approved_answers_for

    stats = {"written": 0, "skipped": 0, "exact_only": 0}
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        for r in records:
            question = r["question"].strip()
            response = pick_response(r)
            if resume:
                dupes = [d for d in list_approved_answers_for(conn, question) if d["approved_response"] == response]
                if dupes:
                    stats["skipped"] += 1
                    continue
            try:
                vec = client.embed(question)[0]
                emb_json: str | None = _json.dumps(vec, ensure_ascii=False)
            except Exception:
                LOG.error("query embedding failed for %s — cached exact-match only", r["qa_id"], exc_info=True)
                emb_json = None
                stats["exact_only"] += 1
            add_approved_answer(
                conn,
                query_text=question,
                approved_response=response,
                source_hashes=[r["file_hash"]],
                query_embedding=emb_json,
            )
            stats["written"] += 1
    finally:
        conn.close()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest approved synthetic Q&A pairs (vectors + semantic cache)")
    parser.add_argument("--input", default="artifacts/synthetic_qa_pilot.approved.jsonl")
    parser.add_argument("--targets", choices=("both", "vectors", "cache"), default="both")
    parser.add_argument("--persist-dir", default="./data/vectordb")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="re-write rows that are already ingested")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        LOG.error("input not found: %s (export it from the review dashboard first)", path)
        return 1
    records = load_approved(path)
    LOG.info("%d approved records in %s", len(records), path)
    if not records:
        return 0
    if args.dry_run:
        for r in records[:20]:
            LOG.info("[dry-run] %s | %.60s... → cache + vector", r["qa_id"], r["question"])
        if len(records) > 20:
            LOG.info("[dry-run] ... and %d more", len(records) - 20)
        return 0

    from backend.services.ollama_client import get_ollama_client

    try:
        client = get_ollama_client()
    except Exception:
        LOG.error("ollama client unavailable — aborting", exc_info=True)
        return 1
    resume = not args.no_resume
    if args.targets in ("both", "vectors"):
        stats = import_vectors(records, client, args.persist_dir, resume=resume)
        LOG.info("vectors: %s", stats)
    if args.targets in ("both", "cache"):
        stats = import_cache(records, client, Path(args.db_path) if args.db_path else None, resume=resume)
        LOG.info("cache: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
