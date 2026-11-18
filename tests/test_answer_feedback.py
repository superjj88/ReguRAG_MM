"""Answer-feedback tests: incorrect-flag learning (cache retirement + audit trail).

Offline: tmp SQLite DBs, stub embedding client; Ollama failures must only
skip the semantic tier, never break exact retirement.
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.database import (  # noqa: E402
    add_approved_answer,
    get_all_valid_hashes,
    get_connection,
    init_db,
    list_approved_answers_for,
    upsert_document_by_hash,
)
from backend.services import approved_answers as svc  # noqa: E402
from backend.services.approved_answers import get_cached_answer, record_incorrect  # noqa: E402
from tests.test_api_layer import MINIMAL_CONFIG  # noqa: E402
from backend.core.config import get_config  # noqa: E402

VEC = [1.0, 0.0, 0.0]


class StubClient:
    def __init__(self, vec):
        self._vec = vec

    def embed(self, _text):
        return [list(self._vec)]


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "fb.db"
    init_db(path)
    conn = get_connection(path)
    upsert_document_by_hash(conn, file_name="doc.pdf", path="/tmp/doc.pdf", file_hash_sha256="h1")
    conn.close()
    return path


def _seed(path, query, response, emb=None):
    conn = get_connection(path)
    try:
        return add_approved_answer(
            conn,
            query_text=query,
            approved_response=response,
            source_hashes=["h1"],
            query_embedding=json.dumps(emb) if emb else None,
        )
    finally:
        conn.close()


def test_exact_cache_row_retired_and_answer_regenerates(db, monkeypatch):
    """Flagging a cached answer deletes it; the same query is a cache miss after."""
    _seed(db, "מה התשובה?", "תשובה שגויה")
    assert get_cached_answer("מה התשובה?", db_path=db) is not None
    monkeypatch.setattr(svc, "_ollama_client", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    res = record_incorrect("מה התשובה?", "תשובה שגויה", ["h1"], db_path=db)
    assert res["retired_answer_ids"] != []
    assert get_cached_answer("מה התשובה?", db_path=db) is None
    conn = get_connection(db)
    try:
        assert list_approved_answers_for(conn, "מה התשובה?") == []
        fb = conn.execute("SELECT * FROM answer_feedback").fetchall()
        assert len(fb) == 1 and fb[0]["verdict"] == "incorrect"
        assert json.loads(fb[0]["retired_answer_ids"]) == res["retired_answer_ids"]
    finally:
        conn.close()


def test_semantic_same_answer_retired_other_response_kept(db):
    """A semantic-tier row serving the same wrong answer is retired; others stay."""
    qid_same = _seed(db, "שאלה דומה", "תשובה שגויה", emb=VEC)
    qid_other = _seed(db, "שאלה דומה 2", "תשובה אחרת", emb=VEC)
    res = record_incorrect("שאלה קרובה", "תשובה שגויה", ["h1"], db_path=db, ollama_client=StubClient(VEC))
    assert res["retired_answer_ids"] == [qid_same]
    conn = get_connection(db)
    try:
        remaining = [r["id"] for r in conn.execute("SELECT id FROM approved_answers").fetchall()]
        assert remaining == [qid_other]
    finally:
        conn.close()


def test_no_match_still_records_feedback(db, monkeypatch):
    """Flagging a fresh (never cached) answer: nothing retired, verdict stored."""
    monkeypatch.setattr(svc, "_ollama_client", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    res = record_incorrect("שאלה חדשה", "תשובה חדשה", ["h1"], comment="לא נכון", db_path=db)
    assert res == {"feedback_id": 1, "retired_answer_ids": []}
    conn = get_connection(db)
    try:
        rows = conn.execute("SELECT * FROM answer_feedback").fetchall()
        assert len(rows) == 1 and rows[0]["comment"] == "לא נכון"
    finally:
        conn.close()


def test_semantic_tier_skipped_when_ollama_down(db, monkeypatch):
    """Ollama outage must not break exact retirement (log-and-continue)."""
    _seed(db, "q?", "bad", emb=VEC)
    monkeypatch.setattr(svc, "_ollama_client", lambda: (_ for _ in ()).throw(ConnectionError("down")))
    res = record_incorrect("q?", "bad", ["h1"], db_path=db)
    assert len(res["retired_answer_ids"]) == 1


# ---- API endpoints (isolated app config + tmp DB) ----


@pytest.fixture()
def client(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(MINIMAL_CONFIG.format(docs=docs), encoding="utf-8")
    monkeypatch.setenv("REGURAG_CONFIG", str(cfg_file))
    get_config.cache_clear()
    monkeypatch.setattr(svc, "_ollama_client", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    from backend.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, tmp_path / "regulator.db"
    monkeypatch.delenv("REGURAG_CONFIG", raising=False)
    get_config.cache_clear()


def test_feedback_endpoint_retires_and_lists(client):
    c, db_path = client
    init_db(db_path)
    conn = get_connection(db_path)
    upsert_document_by_hash(conn, file_name="d.pdf", path="/x/d.pdf", file_hash_sha256="h9")
    add_approved_answer(conn, query_text="ש?", approved_response="רעה", source_hashes=["h9"])
    conn.close()
    r = c.post(
        "/api/answers/feedback",
        json={"query_text": "ש?", "response_text": "רעה", "source_hashes": ["h9"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "recorded" and len(body["retired_answer_ids"]) == 1
    r2 = c.get("/api/answers/feedback?limit=10")
    assert r2.status_code == 200
    items = r2.json()["feedback"]
    assert len(items) == 1 and items[0]["verdict"] == "incorrect"
    assert get_all_valid_hashes(get_connection(db_path)) == {"h9"}
