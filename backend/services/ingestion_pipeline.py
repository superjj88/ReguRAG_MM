"""Ingestion orchestrator.

One vector writer per run (previously a fresh Chroma client per
document — resource churn and a known Chroma anti-pattern); failures logged.
Link-expansion decision: old vectors AND chunk rows are purged when a
document changes or is deleted (no vectordb orphans).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from backend.core.config import get_config
from backend.core.database import (
    delete_chunks_by_hash,
    get_connection,
    get_document_by_hash,
    init_db,
    insert_chunks,
    log_ingestion,
    upsert_document_by_hash,
)
from backend.services.document_scanner import scan_documents
from backend.services.pdf_parser import chunk_document

LOG = logging.getLogger(__name__)

_state: dict = {"state": "idle", "processed": 0, "total": 0, "eta_seconds": None, "last_error": None, "_thread": None}
_lock = threading.RLock()


def get_status() -> dict:
    with _lock:
        return {k: v for k, v in _state.items() if not k.startswith("_")}


def _purge_document(conn, file_hash: str, file_name: str) -> None:
    """Delete old vectors + chunk rows for a document (re-ingest or deletion)."""
    try:
        from backend.services.qa_generator import delete_doc_vectors

        deleted = delete_doc_vectors(file_hash)
        LOG.info("purged %d old vectors for %s", deleted, file_name)
    except Exception:
        LOG.error("vector purge failed for %s — aborting document update", file_name, exc_info=True)
        raise
    deleted_rows = delete_chunks_by_hash(conn, file_hash)
    if deleted_rows:
        LOG.info("purged %d old chunk rows for %s", deleted_rows, file_name)


def _handle_deletions(conn, deleted: list[dict]) -> None:
    for d in deleted:
        _purge_document(conn, d["hash"], d["file_name"])
        conn.execute("DELETE FROM documents WHERE path=?", (d["path"],))
        log_ingestion(conn, file_hash=d["hash"], file_name=d["file_name"], action="deleted")


def _scan_contradictions(conn, processed_hashes: set[str], db_path) -> int:
    """Cross-document contradiction scan for newly ingested documents.

    Every chunk is embedded exactly once (cache keyed by file_hash); the
    shared similarity matrix keeps the scan O(new × others) without network
    traffic per pair. LLM judgment + persistence happen inside
    detect_contradictions_for_document.
    """
    from backend.core.database import get_all_chunks
    from backend.services.contradiction_detector import detect_contradictions_for_document
    from backend.services.ollama_client import get_ollama_client

    if not processed_hashes:
        return 0
    try:
        client = get_ollama_client()
    except Exception:
        LOG.warning("embeddings unavailable — skipping contradiction scan", exc_info=True)
        return 0

    by_hash: dict[str, list[dict]] = {}
    for c in get_all_chunks(conn):
        by_hash.setdefault(c["file_hash"], []).append(c)

    vec_cache: dict[str, dict[int, list[float]]] = {}
    for h, rows in by_hash.items():
        try:
            vecs = client.embed([r["text"] for r in rows])
            vec_cache[h] = dict(enumerate(vecs))
        except Exception:
            LOG.warning("embedding failed for %s — doc excluded from conflict scan", h[:12], exc_info=True)

    total = 0
    for h in sorted(processed_hashes):
        new_rows = by_hash.get(h)
        if not new_rows or h not in vec_cache:
            continue
        others: list[dict] = []
        other_vecs: list[list[float]] = []
        for hh, rows in by_hash.items():
            if hh == h or hh not in vec_cache:
                continue
            others.extend(rows)
            other_vecs.extend(vec_cache[hh][i] for i in range(len(rows)))
        conflicts = detect_contradictions_for_document(
            new_rows,
            others,
            ollama_client=client,
            db_path=db_path,
            new_vectors=vec_cache[h],
            existing_vectors=dict(enumerate(other_vecs)),
        )
        if conflicts:
            LOG.warning("contradiction scan: %d conflicts for %s", len(conflicts), h[:12])
        total += len(conflicts)
    if total:
        LOG.warning("contradiction scan complete: %d conflicts across %d new documents", total, len(processed_hashes))
    return total


def _run_pipeline(documents_path: Path | str | None = None, db_path: Path | str | None = None):
    cfg = get_config()
    docs_path = Path(documents_path) if documents_path else Path(cfg.system.documents_path)
    try:
        init_db(db_path)
        conn = get_connection(db_path)
        scan = scan_documents(docs_path, db_path=db_path, conn=conn)
        to_process = scan["new"] + scan["changed"]
        with _lock:
            _state["total"] = len(to_process)
            _state["processed"] = 0
            _state["state"] = "processing" if to_process else "done"
            _state["last_error"] = None
        if not to_process:
            _handle_deletions(conn, scan["deleted"])
            conn.close()
            with _lock:
                _state["state"] = "done"
                _state["eta_seconds"] = 0
            return

        # vector writer: one instance (and thus one Chroma client) per run
        from backend.services.qa_generator import QAGenerator

        gen = QAGenerator(batch_size=cfg.performance.ingestion_batch_size)
        processed_hashes: set[str] = set()

        # process each PDF
        start = time.time()
        for idx, item in enumerate(to_process):
            try:
                # changed documents: purge old vectors/chunks BEFORE re-ingest
                if "old_hash" in item and item["old_hash"] != item["hash"]:
                    _purge_document(conn, item["old_hash"], item["file_name"])
                chunks = chunk_document(item["path"], file_hash=item["hash"], use_ocr=cfg.ocr.enabled)
                existing = get_document_by_hash(conn, item["hash"])
                if existing is not None and existing["path"] != item["path"]:
                    # duplicate content under a second path — vectors already
                    # stored under this hash; record and skip
                    LOG.warning("duplicate content skipped: %s == %s", item["file_name"], existing["file_name"])
                    log_ingestion(
                        conn,
                        file_hash=item["hash"],
                        file_name=item["file_name"],
                        action="duplicate",
                        detail=f"same content as {existing['file_name']}",
                    )
                    continue
                insert_chunks(conn, chunks)
                upsert_document_by_hash(
                    conn,
                    file_name=item["file_name"],
                    path=item["path"],
                    file_hash_sha256=item["hash"],
                    page_count=len(set(c["page_number"] for c in chunks)),
                )
                log_ingestion(conn, file_hash=item["hash"], file_name=item["file_name"], action="ingested", detail=f"{len(chunks)} chunks")
                processed_hashes.add(item["hash"])
                # vector write — best effort, batched
                try:
                    gen.generate_for_chunks(chunks)
                except Exception as e:
                    LOG.error("vector write failed for %s", item["file_name"], exc_info=True)
                    log_ingestion(conn, file_hash=item["hash"], file_name=item["file_name"], action="vector_error", detail=str(e))
            except Exception as e:
                with _lock:
                    _state["last_error"] = str(e)
                LOG.error("ingestion failed for %s", item.get("file_name"), exc_info=True)
                log_ingestion(conn, file_hash=item.get("hash"), file_name=item.get("file_name"), action="error", detail=str(e))
            with _lock:
                _state["processed"] = idx + 1
                elapsed = time.time() - start
                per = elapsed / (idx + 1) if idx + 1 else 0
                remaining = len(to_process) - (idx + 1)
                _state["eta_seconds"] = int(per * remaining) if per else None

        # deletions
        _handle_deletions(conn, scan["deleted"])
        processed_hashes -= {d["hash"] for d in scan["deleted"]}
        # cross-document links (annex/parent/sibling) — idempotent rebuild
        try:
            from backend.services.document_links import rebuild_document_links

            links = rebuild_document_links(conn)
            log_ingestion(conn, file_hash=None, file_name=None, action="links_rebuilt", detail=f"{links} edges")
        except Exception:
            LOG.error("document link rebuild failed", exc_info=True)
        # contradiction scan over the newly ingested documents
        try:
            _scan_contradictions(conn, processed_hashes, db_path)
        except Exception:
            LOG.error("contradiction scan failed", exc_info=True)
        conn.commit()
        conn.close()
        with _lock:
            _state["state"] = "done"
            _state["eta_seconds"] = 0
    except Exception as e:
        LOG.error("ingestion pipeline crashed", exc_info=True)
        with _lock:
            _state["state"] = "error"
            _state["last_error"] = str(e)
            _state["eta_seconds"] = None


def start_refresh(documents_path: Path | str | None = None, db_path: Path | str | None = None) -> dict:
    with _lock:
        if _state["state"] == "processing":
            return get_status()
        _state.update({"state": "processing", "processed": 0, "total": 0, "eta_seconds": None, "last_error": None})
        t = threading.Thread(target=_run_pipeline, args=(documents_path, db_path), daemon=True)
        _state["_thread"] = t
        t.start()
        return get_status()
