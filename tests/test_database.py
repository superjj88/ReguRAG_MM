import json

import pytest

from backend.core.database import (
    add_approved_answer,
    add_conflict,
    get_all_valid_hashes,
    get_connection,
    init_db,
    is_hash_excluded,
    list_approved_answers_for,
    list_conflicts,
    resolve_conflict,
    upsert_document_by_hash,
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    conn = get_connection(path)
    yield conn
    conn.close()


def test_fresh_creates_all_tables(db):
    cur = db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    names = {r[0] for r in cur.fetchall()}
    assert {"documents", "approved_answers", "conflicts", "ingestion_log"}.issubset(names)
    # exact schema check
    cur = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='approved_answers'")
    sql = cur.fetchone()[0]
    assert "query_text TEXT NOT NULL" in sql
    assert "approved_response TEXT NOT NULL" in sql
    assert "source_hashes TEXT NOT NULL" in sql
    assert "approved_at" in sql


def test_upsert_by_hash(db):
    upsert_document_by_hash(db, file_name="a.pdf", path="/tmp/a.pdf", file_hash_sha256="abc123", page_count=10)
    assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    # upsert same path with new hash
    upsert_document_by_hash(db, file_name="a.pdf", path="/tmp/a.pdf", file_hash_sha256="def456", page_count=12)
    assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    row = db.execute("SELECT file_hash_sha256, page_count FROM documents WHERE path='/tmp/a.pdf'").fetchone()
    assert row[0] == "def456"
    assert row[1] == 12

    # different path same hash would violate UNIQUE — ensure two docs with distinct hashes
    upsert_document_by_hash(db, file_name="b.pdf", path="/tmp/b.pdf", file_hash_sha256="abc123", page_count=5)
    assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 2


def test_approved_answer_lookup_with_hash_validation(db):
    h1, h2 = "hash1", "hash2"
    upsert_document_by_hash(db, file_name="a.pdf", path="/a.pdf", file_hash_sha256=h1, page_count=1)
    upsert_document_by_hash(db, file_name="b.pdf", path="/b.pdf", file_hash_sha256=h2, page_count=1)

    add_approved_answer(db, query_text="מה סעיף 1?", approved_response="תשובה", source_hashes=[h1])
    add_approved_answer(db, query_text="מה סעיף 1?", approved_response="תשובה2", source_hashes=[h2, h1])

    # valid hashes contain both -> both returned
    rows = list_approved_answers_for(db, "מה סעיף 1?", valid_hashes={h1, h2})
    assert len(rows) == 2

    # delete one document -> hash invalid -> only first remains
    db.execute("DELETE FROM documents WHERE file_hash_sha256=?", (h2,))
    db.commit()
    valid = get_all_valid_hashes(db)
    assert valid == {h1}
    rows = list_approved_answers_for(db, "מה סעיף 1?", valid_hashes=valid)
    assert len(rows) == 1
    assert json.loads(rows[0]["source_hashes"]) == [h1]

    # without filter, returns all
    assert len(list_approved_answers_for(db, "מה סעיף 1?")) == 2


def test_conflict_lifecycle(db):
    cid = add_conflict(db, new_doc_hash="new123", other_doc_hash="old456", topic_summary="סתירה בסעיף 4")
    assert cid is not None
    assert len(list_conflicts(db, unresolved_only=True)) == 1
    assert is_hash_excluded(db, "new123") is True
    assert is_hash_excluded(db, "old456") is True
    assert is_hash_excluded(db, "other") is False

    resolve_conflict(db, cid, "new wins")
    assert len(list_conflicts(db, unresolved_only=True)) == 0
    assert is_hash_excluded(db, "new123") is False
    row = list_conflicts(db)[0]
    assert row["resolution"] == "new wins"
    assert row["resolved_at"] is not None


def test_db_path_configurable(tmp_path):
    # custom path
    p = tmp_path / "custom.db"
    init_db(p)
    conn = get_connection(p)
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    conn.close()
    # ensure isolation: second init doesn't wipe
    init_db(p)
    assert p.exists()
