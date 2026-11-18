"""Folder scanner — detects new/changed/deleted PDFs via SHA-256."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.core.database import get_connection, init_db, list_documents
from backend.services.pdf_parser import compute_file_hash


def scan_documents(documents_path: Path | str, db_path: Path | str | None = None, conn: sqlite3.Connection | None = None) -> dict:
    """Compare filesystem PDFs against DB.

    Returns dict with keys: new, changed, unchanged, deleted — each a list
    of dicts {path, file_name, hash, page_count?}.
    """
    docs_path = Path(documents_path)
    owned_conn = False
    if conn is None:
        init_db(db_path)
        conn = get_connection(db_path)
        owned_conn = True

    try:
        # existing by path
        existing = {r["path"]: r for r in list_documents(conn)}
        existing_by_hash = {r["file_hash_sha256"]: r for r in list_documents(conn)}

        fs_pdfs = list(docs_path.rglob("*.pdf")) if docs_path.exists() else []

        new, changed, unchanged = [], [], []
        seen_paths: set[str] = set()

        for pdf in fs_pdfs:
            path_str = str(pdf)
            seen_paths.add(path_str)
            fhash = compute_file_hash(pdf)
            if path_str not in existing:
                # new path — check if hash existed elsewhere (moved/renamed)
                if fhash in existing_by_hash:
                    # treat as unchanged if hash already known (moved file)
                    unchanged.append({"path": path_str, "file_name": pdf.name, "hash": fhash})
                else:
                    new.append({"path": path_str, "file_name": pdf.name, "hash": fhash})
            else:
                old = existing[path_str]
                if old["file_hash_sha256"] != fhash:
                    changed.append({"path": path_str, "file_name": pdf.name, "hash": fhash, "old_hash": old["file_hash_sha256"]})
                else:
                    unchanged.append({"path": path_str, "file_name": pdf.name, "hash": fhash})

        deleted = []
        for path_str, row in existing.items():
            if path_str not in seen_paths:
                deleted.append({"path": path_str, "file_name": row["file_name"], "hash": row["file_hash_sha256"]})

        return {"new": new, "changed": changed, "unchanged": unchanged, "deleted": deleted}
    finally:
        if owned_conn:
            conn.close()
