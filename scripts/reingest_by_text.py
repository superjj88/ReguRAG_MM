#!/usr/bin/env python3
"""Force-reingest documents whose chunk text contains a given substring.

The document scanner keys on file hashes, so extraction-logic fixes do not
re-trigger processing of unchanged PDFs. This script locates documents whose
chunk rows match a substring (e.g. a misdecoded glyph), purges their old
vectors + chunk rows (no orphans), re-chunks via pdf_parser, persists rows
and re-embeds — mirroring ingestion_pipeline._run_pipeline per document.

Usage:
    python -m scripts.reingest_by_text "ð"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.core.config import get_config
from backend.core.database import (
    delete_chunks_by_hash,
    get_connection,
    init_db,
    insert_chunks,
    log_ingestion,
    upsert_document_by_hash,
)
from backend.services.pdf_parser import chunk_document
from backend.services.qa_generator import QAGenerator, delete_doc_vectors

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger("reingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Force-reingest documents whose chunk text matches a substring")
    parser.add_argument("substring", help="chunk-text substring to match for re-ingest")
    parser.add_argument("--commit", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    cfg = get_config()
    db_path = init_db()

    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT DISTINCT file_hash, file_name FROM chunks WHERE text LIKE ?",
        (f"%{args.substring}%",),
    ).fetchall()
    targets = [(r["file_hash"], r["file_name"]) for r in rows]
    if not targets:
        LOG.info("no chunks matching %r — nothing to do", args.substring)
        conn.close()
        return 0

    LOG.info("re-ingesting %d documents containing %r", len(targets), args.substring)

    gen = QAGenerator(batch_size=cfg.performance.ingestion_batch_size) if args.commit else None
    for file_hash, file_name in targets:
        doc = conn.execute(
            "SELECT * FROM documents WHERE file_hash_sha256=?", (file_hash,)
        ).fetchone()
        if doc is None:
            LOG.warning("no documents row for hash %s — skipping %s", file_hash[:12], file_name)
            continue
        path = Path(doc["path"])

        chunks = chunk_document(path, file_hash=file_hash)
        if not chunks:
            LOG.warning("no chunks extracted for %s — skipping", file_name)
            continue
        changed = sum(1 for c in chunks if args.substring in c["text"])

        if not args.commit:
            LOG.info(
                "[dry-run] %s (%s): %d chunks, %d still containing %r",
                file_name, path, len(chunks), changed, args.substring,
            )
            continue

        deleted_vecs = delete_doc_vectors(file_hash)
        deleted_rows = delete_chunks_by_hash(conn, file_hash)
        insert_chunks(conn, chunks)
        upsert_document_by_hash(
            conn,
            file_name=file_name,
            path=str(path),
            file_hash_sha256=file_hash,
            page_count=len(set(c["page_number"] for c in chunks)),
        )
        log_ingestion(
            conn,
            file_hash=file_hash,
            file_name=file_name,
            action="reingested",
            detail=f"{len(chunks)} chunks (purged {deleted_rows} rows, {deleted_vecs} vectors)",
        )
        progress = gen.generate_for_chunks(chunks) if gen else {}
        LOG.info(
            "re-ingested %s (%s): %d chunks, %d still containing %r — %s",
            file_name, path, len(chunks), changed, args.substring,
            progress.get("state", "n/a"),
        )

    conn.close()
    LOG.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
