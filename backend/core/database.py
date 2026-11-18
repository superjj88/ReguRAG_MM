"""SQLite data layer.

Single file DB at ``./data/regulator.db`` (configurable). No concurrency
concerns (single-user). Four tables:

* documents — SHA-256 dedup per PDF
* approved_answers — exact schema
* answer_feedback — user verdicts on served answers (incorrect flags + cache retirements)
* conflicts — contradiction detection output
* ingestion_log — audit trail
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from backend.core.config import AppConfig, get_config

LOG = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/regulator.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    file_name TEXT NOT NULL,
    path TEXT NOT NULL PRIMARY KEY,
    file_hash_sha256 TEXT NOT NULL UNIQUE,
    page_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS approved_answers (
    id INTEGER PRIMARY KEY,
    query_text TEXT NOT NULL,
    approved_response TEXT NOT NULL,
    source_hashes TEXT NOT NULL,
    query_embedding TEXT,
    approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS answer_feedback (
    id INTEGER PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    source_hashes TEXT NOT NULL,
    verdict TEXT NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    retired_answer_ids TEXT NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY,
    new_doc_hash TEXT NOT NULL,
    other_doc_hash TEXT NOT NULL,
    topic_summary TEXT NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolution TEXT,
    resolved_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY,
    file_hash TEXT,
    file_name TEXT,
    action TEXT NOT NULL,
    detail TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    file_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    section_number TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS document_links (
    id INTEGER PRIMARY KEY,
    source_hash TEXT NOT NULL,
    target_hash TEXT NOT NULL,
    link_type TEXT NOT NULL,
    basis TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_hash, target_hash, link_type)
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash_sha256);
CREATE INDEX IF NOT EXISTS idx_conflicts_new_hash ON conflicts(new_doc_hash);
CREATE INDEX IF NOT EXISTS idx_conflicts_resolved ON conflicts(resolved_at);
CREATE INDEX IF NOT EXISTS idx_approved_query ON approved_answers(query_text);
CREATE INDEX IF NOT EXISTS idx_feedback_query ON answer_feedback(query_text);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(file_hash);
CREATE INDEX IF NOT EXISTS idx_links_source ON document_links(source_hash);
CREATE INDEX IF NOT EXISTS idx_links_target ON document_links(target_hash);
"""


def get_db_path(config: AppConfig | None = None) -> Path:
    """Resolve DB path. Uses config if provided, else default.

    Rule: <documents_path parent>/regulator.db so that each topic's data
    lives next to its documents folder. Fallback: ./data/regulator.db.
    """
    if config is None:
        try:
            config = get_config()
        except Exception:
            return DEFAULT_DB_PATH
    # documents_path e.g. "./data/housing_committee" -> parent is "./data"
    docs = Path(config.system.documents_path)
    parent = docs.parent if str(docs.parent) != "" else Path("data")
    return parent / "regulator.db"


def get_connection(db_path: Path | str | None = None, config: AppConfig | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else get_db_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path | str | None = None, config: AppConfig | None = None) -> Path:
    path = Path(db_path) if db_path is not None else get_db_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(path)
    try:
        conn.executescript(SCHEMA)
        _migrate_schema(conn)
        conn.commit()
    finally:
        conn.close()
    return path


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Idempotent in-place migration for DBs created before the semantic cache.

    Older ``approved_answers`` tables lack the ``query_embedding`` column.
    """
    try:
        conn.execute("ALTER TABLE approved_answers ADD COLUMN query_embedding TEXT")
        conn.commit()
        LOG.info("migration: approved_answers.query_embedding column added")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return  # already migrated — expected on warm DBs
        raise


def backup_database(src: Path | str, dest: Path | str) -> Path:
    """Consistent backup of a live (WAL) SQLite database.

    Uses the sqlite3 online backup API instead of a raw file copy, which can
    capture torn state when journal_mode=WAL is active.
    """
    src_conn = sqlite3.connect(str(src))
    try:
        dest_conn = sqlite3.connect(str(dest))
        try:
            src_conn.backup(dest_conn)
            dest_conn.commit()
        finally:
            dest_conn.close()
    finally:
        src_conn.close()
    return Path(dest)


# ---- documents ----
def upsert_document_by_hash(
    conn: sqlite3.Connection,
    *,
    file_name: str,
    path: str,
    file_hash_sha256: str,
    page_count: int = 0,
    status: str = "ingested",
) -> None:
    """Insert or replace document row keyed by file_hash_sha256 + path.

    Uses INSERT ... ON CONFLICT to update hash/path mapping.
    """
    conn.execute(
        """
        INSERT INTO documents (file_name, path, file_hash_sha256, page_count, status, ingested_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(path) DO UPDATE SET
            file_name=excluded.file_name,
            file_hash_sha256=excluded.file_hash_sha256,
            page_count=excluded.page_count,
            status=excluded.status,
            ingested_at=CURRENT_TIMESTAMP
        """,
        (file_name, path, file_hash_sha256, page_count, status),
    )
    # Also handle hash uniqueness: if hash exists under different path, update that row
    # (legacy) — keep simple: the UNIQUE constraint will raise if violated with different path.
    conn.commit()


def get_document_by_hash(conn: sqlite3.Connection, file_hash_sha256: str) -> dict | None:
    cur = conn.execute("SELECT * FROM documents WHERE file_hash_sha256=?", (file_hash_sha256,))
    row = cur.fetchone()
    return dict(row) if row else None


def list_documents(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC")
    return [dict(r) for r in cur.fetchall()]


def delete_document_by_path(conn: sqlite3.Connection, path: str) -> None:
    conn.execute("DELETE FROM documents WHERE path=?", (path,))
    conn.commit()


# ---- approved_answers ----
def add_approved_answer(
    conn: sqlite3.Connection,
    *,
    query_text: str,
    approved_response: str,
    source_hashes: list[str],
    query_embedding: str | None = None,
) -> int:
    payload = json.dumps(source_hashes, ensure_ascii=False)
    cur = conn.execute(
        "INSERT INTO approved_answers (query_text, approved_response, source_hashes, query_embedding) "
        "VALUES (?, ?, ?, ?)",
        (query_text, approved_response, payload, query_embedding),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore


def list_approved_semantic_candidates(
    conn: sqlite3.Connection,
    *,
    valid_hashes: set[str] | None = None,
) -> list[dict]:
    """Approved answers that carry a query embedding (semantic-cache pool).

    Rows with a corrupt embedding are logged and skipped (never silent).
    When valid_hashes is provided, only answers whose source_hashes are a
    subset of valid hashes participate (same auto-invalidation rule as the
    exact-match path).
    """
    cur = conn.execute(
        "SELECT * FROM approved_answers WHERE query_embedding IS NOT NULL ORDER BY approved_at DESC"
    )
    out = []
    for r in cur.fetchall():
        d = dict(r)
        try:
            emb = json.loads(d["query_embedding"])
            if not isinstance(emb, list) or not emb:
                raise ValueError("empty embedding")
            d["query_embedding_vec"] = [float(x) for x in emb]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            LOG.warning("approved_answers id=%s has unusable query_embedding — skipped (%s)", d.get("id"), exc)
            continue
        if valid_hashes is not None:
            try:
                hashes = set(json.loads(d["source_hashes"]))
            except Exception as exc:
                LOG.warning("approved_answers id=%s has unreadable source_hashes — skipped (%s)", d.get("id"), exc)
                continue
            if not hashes.issubset(valid_hashes):
                continue
        out.append(d)
    return out


def list_approved_answers_for(
    conn: sqlite3.Connection,
    query_text: str,
    *,
    valid_hashes: set[str] | None = None,
) -> list[dict]:
    """Return approved answers for exact query_text, filtered by hash validity.

    If valid_hashes is provided, only answers whose source_hashes are subset
    of valid_hashes are returned (auto-invalidation).
    """
    cur = conn.execute(
        "SELECT * FROM approved_answers WHERE query_text=? ORDER BY approved_at DESC",
        (query_text,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    if valid_hashes is None:
        return rows
    out = []
    for r in rows:
        try:
            hashes = set(json.loads(r["source_hashes"]))
        except Exception:
            continue
        if hashes.issubset(valid_hashes):
            out.append(r)
    return out


def get_all_valid_hashes(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT file_hash_sha256 FROM documents")
    return {r[0] for r in cur.fetchall()}


def delete_approved_answer(conn: sqlite3.Connection, row_id: int) -> None:
    """Retire a cached answer so it can never be served again (incorrect-flag learning)."""
    conn.execute("DELETE FROM approved_answers WHERE id=?", (row_id,))
    conn.commit()


# ---- answer_feedback ----
def add_answer_feedback(
    conn: sqlite3.Connection,
    *,
    query_text: str,
    response_text: str,
    source_hashes: list[str],
    verdict: str,
    comment: str = "",
    retired_answer_ids: list[int] | None = None,
) -> int:
    """Record a user verdict on a served answer (e.g. verdict='incorrect').

    The row is the audit trail for cache retirements: ``retired_answer_ids``
    lists the approved_answers rows deleted because of this verdict.
    """
    payload = json.dumps(source_hashes, ensure_ascii=False)
    retired = json.dumps(sorted(retired_answer_ids or []))
    cur = conn.execute(
        "INSERT INTO answer_feedback "
        "(query_text, response_text, source_hashes, verdict, comment, retired_answer_ids) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (query_text, response_text, payload, verdict, comment, retired),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore


def list_answer_feedback(conn: sqlite3.Connection, *, limit: int = 50) -> list[dict]:
    """Newest user verdicts first — reviewer visibility into flagged answers."""
    cur = conn.execute(
        "SELECT * FROM answer_feedback ORDER BY created_at DESC, id DESC LIMIT ?",
        (max(1, limit),),
    )
    return [dict(r) for r in cur.fetchall()]


# ---- conflicts ----
def add_conflict(
    conn: sqlite3.Connection,
    *,
    new_doc_hash: str,
    other_doc_hash: str,
    topic_summary: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO conflicts (new_doc_hash, other_doc_hash, topic_summary) VALUES (?, ?, ?)",
        (new_doc_hash, other_doc_hash, topic_summary),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore


def list_conflicts(conn: sqlite3.Connection, *, unresolved_only: bool = False) -> list[dict]:
    if unresolved_only:
        cur = conn.execute("SELECT * FROM conflicts WHERE resolved_at IS NULL ORDER BY detected_at DESC")
    else:
        cur = conn.execute("SELECT * FROM conflicts ORDER BY detected_at DESC")
    return [dict(r) for r in cur.fetchall()]


def resolve_conflict(conn: sqlite3.Connection, conflict_id: int, resolution: str) -> None:
    conn.execute(
        "UPDATE conflicts SET resolution=?, resolved_at=CURRENT_TIMESTAMP WHERE id=?",
        (resolution, conflict_id),
    )
    conn.commit()


def is_hash_excluded(conn: sqlite3.Connection, file_hash: str) -> bool:
    """True if hash belongs to a document involved in an unresolved conflict."""
    cur = conn.execute(
        "SELECT 1 FROM conflicts WHERE resolved_at IS NULL AND (new_doc_hash=? OR other_doc_hash=?) LIMIT 1",
        (file_hash, file_hash),
    )
    return cur.fetchone() is not None


# ---- ingestion_log ----
def log_ingestion(conn: sqlite3.Connection, *, file_hash: str | None, file_name: str | None, action: str, detail: str | None = None) -> None:
    conn.execute(
        "INSERT INTO ingestion_log (file_hash, file_name, action, detail) VALUES (?, ?, ?, ?)",
        (file_hash, file_name, action, detail),
    )
    conn.commit()


# ---- chunks ----
def insert_chunks(conn: sqlite3.Connection, chunks: list[dict]) -> int:
    """Persist chunk rows (idempotent on chunk_id). Returns number of rows."""
    rows = [
        (
            c["chunk_id"],
            c["file_hash"],
            c["file_name"],
            c["page_number"],
            c.get("section_number", ""),
            c["text"],
        )
        for c in chunks
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO chunks (chunk_id, file_hash, file_name, page_number, section_number, text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def delete_chunks_by_hash(conn: sqlite3.Connection, file_hash: str) -> int:
    cur = conn.execute("DELETE FROM chunks WHERE file_hash=?", (file_hash,))
    conn.commit()
    return cur.rowcount


def get_chunks_by_hash(conn: sqlite3.Connection, file_hash: str) -> list[dict]:
    cur = conn.execute(
        "SELECT chunk_id, file_hash, file_name, page_number, section_number, text FROM chunks WHERE file_hash=?",
        (file_hash,),
    )
    return [dict(r) for r in cur.fetchall()]


def get_all_chunks(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT chunk_id, file_hash, file_name, page_number, section_number, text FROM chunks"
    )
    return [dict(r) for r in cur.fetchall()]


def get_chunks_excluding_hashes(conn: sqlite3.Connection, hashes: set[str]) -> list[dict]:
    if not hashes:
        return get_all_chunks(conn)
    placeholders = ",".join("?" for _ in hashes)
    cur = conn.execute(
        f"SELECT chunk_id, file_hash, file_name, page_number, section_number, text FROM chunks WHERE file_hash NOT IN ({placeholders})",
        tuple(hashes),
    )
    return [dict(r) for r in cur.fetchall()]


def get_chunk_text(conn: sqlite3.Connection, chunk_id: str) -> dict | None:
    cur = conn.execute(
        "SELECT chunk_id, file_hash, file_name, page_number, section_number, text FROM chunks WHERE chunk_id=?",
        (chunk_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def count_chunks(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    return int(cur.fetchone()[0])


# ---- document_links ----
def replace_links_for_source(conn: sqlite3.Connection, source_hash: str, links: list[dict]) -> int:
    """Replace all links *outgoing* from source_hash. Each link: {target_hash, link_type, basis}."""
    conn.execute("DELETE FROM document_links WHERE source_hash=?", (source_hash,))
    rows = [
        (source_hash, link["target_hash"], link["link_type"], link.get("basis", ""))
        for link in links
        if link["target_hash"] != source_hash
    ]
    conn.executemany(
        """
        INSERT OR IGNORE INTO document_links (source_hash, target_hash, link_type, basis)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def get_links_for_hash(conn: sqlite3.Connection, file_hash: str) -> list[dict]:
    """All links touching file_hash (outgoing or incoming), with doc names resolved."""
    cur = conn.execute(
        """
        SELECT l.id, l.source_hash, l.target_hash, l.link_type, l.basis,
               sf.file_name AS source_name, tf.file_name AS target_name
        FROM document_links l
        LEFT JOIN documents sf ON sf.file_hash_sha256 = l.source_hash
        LEFT JOIN documents tf ON tf.file_hash_sha256 = l.target_hash
        WHERE l.source_hash=? OR l.target_hash=?
        ORDER BY l.id
        """,
        (file_hash, file_hash),
    )
    return [dict(r) for r in cur.fetchall()]


def get_linked_hashes(conn: sqlite3.Connection, hashes: set[str]) -> dict[str, list[str]]:
    """For each known hash, return hashes of linked documents (any link type).

    Only hashes present in the documents table are returned (links pointing
    at purged documents are dropped).
    """
    if not hashes:
        return {}
    placeholders = ",".join("?" for _ in hashes)
    cur = conn.execute(
        f"""
        SELECT DISTINCT source_hash, target_hash FROM document_links
        WHERE source_hash IN ({placeholders}) OR target_hash IN ({placeholders})
        """,
        list(hashes) + list(hashes),
    )
    out: dict[str, list[str]] = {}
    for src, tgt in cur.fetchall():
        if src in hashes and tgt not in hashes:
            out.setdefault(src, []).append(tgt)
        if tgt in hashes and src not in hashes:
            out.setdefault(tgt, []).append(src)
    return out
