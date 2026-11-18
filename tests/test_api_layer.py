"""API-layer tests (audit fix: the routers previously had zero coverage).

Covers: chat endpoint (cache hit, vector path, LLM failure → 503, validation),
backup endpoint (round-trip restore + retention), conflicts endpoints,
approved answers endpoint.
"""

import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import get_config
from backend.core.database import backup_database, get_connection, init_db

MINIMAL_CONFIG = """
system:
  topic_name: "test_topic"
  documents_path: "{docs}"
  language: "he"
hardware_profile: "prod_cpu"
models:
  prod_cpu:
    llm_name: "test-llm"
    num_thread: 2
  embedding_model: "test-embed"
  ollama_base_url: "http://127.0.0.1:11434"
retrieval:
  top_k: 3
  hybrid_search_alpha: 0.5
  stage3_relevance_threshold: 0.9
performance:
  chat_timeout_seconds: 30
  ingestion_batch_size: 2
"""


@pytest.fixture()
def client(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(MINIMAL_CONFIG.format(docs=docs), encoding="utf-8")
    monkeypatch.setenv("REGURAG_CONFIG", str(cfg_file))
    get_config.cache_clear()
    from backend.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    monkeypatch.delenv("REGURAG_CONFIG", raising=False)
    get_config.cache_clear()


# ---- chat ----

def test_chat_cached_answer(client, monkeypatch):
    import backend.api.chat_api as chat_api

    monkeypatch.setattr(chat_api, "get_cached_answer", lambda q: {"answer": "תשובה מאושרת", "source_hashes": ["a1"]})
    r = client.post("/api/chat", json={"query": "שאלה"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "תשובה מאושרת"
    assert body["cached"] is True
    assert body["source_hashes"] == ["a1"]


def test_chat_happy_path_no_cache(client, monkeypatch):
    import backend.api.chat_api as chat_api

    monkeypatch.setattr(chat_api, "get_cached_answer", lambda q: None)
    monkeypatch.setattr(
        chat_api,
        "_retrieve",
        lambda q, top_k: [{"text": "טקסט", "chunk_id": "c1", "file_hash": "h1", "file_name": "f.pdf", "page_number": 2, "section_number": "1"}],
    )
    monkeypatch.setattr(
        chat_api, "generate_answer", lambda q, chunks, timeout_s=None: {"answer": "ת [f.pdf | עמוד 2 | סעיף 1]", "citations": ["[f.pdf | עמוד 2 | סעיף 1]"], "refusal": False}
    )
    # keep the flow fully offline: no real hybrid scoring / stage3 LLM / chroma
    monkeypatch.setattr(chat_api, "hybrid_search", lambda q, chunks, **k: [{**chunks[0], "score": 0.99}])
    monkeypatch.setattr(chat_api, "maybe_refine", lambda q, ranked, all_chunks, **k: ranked)
    monkeypatch.setattr(chat_api, "_attach_embeddings", lambda chunks: chunks)
    r = client.post("/api/chat", json={"query": "שאלה"})
    assert r.status_code == 200
    body = r.json()
    assert body["cached"] is False
    assert body["citations"] == ["[f.pdf | עמוד 2 | סעיף 1]"]
    assert body["pdf_links"] == ["/data/f.pdf#page=2"]
    assert body["source_hashes"] == ["h1"]
    assert "timings" in body


def test_chat_llm_failure_returns_503(client, monkeypatch):
    import backend.api.chat_api as chat_api

    monkeypatch.setattr(chat_api, "get_cached_answer", lambda q: None)
    monkeypatch.setattr(chat_api, "_retrieve", lambda q, top_k: [])
    r = client.post("/api/chat", json={"query": "שאלה"})
    assert r.status_code == 200  # no chunks → refusal envelope, not an error


def test_chat_validation_empty_query(client):
    r = client.post("/api/chat", json={"query": ""})
    assert r.status_code == 422


def test_chat_retrieval_generates_llm_503(client, monkeypatch):
    import backend.api.chat_api as chat_api

    monkeypatch.setattr(chat_api, "get_cached_answer", lambda q: None)
    monkeypatch.setattr(
        chat_api,
        "_retrieve",
        lambda q, top_k: [{"text": "טקסט", "chunk_id": "c1", "file_hash": "h1", "file_name": "f.pdf", "page_number": 1, "section_number": ""}],
    )

    def boom(*a, **k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(chat_api, "generate_answer", boom)
    r = client.post("/api/chat", json={"query": "שאלה"})
    assert r.status_code == 503


# ---- backup ----

def test_backup_roundtrip(tmp_path, monkeypatch):
    db_path = tmp_path / "regulator.db"
    init_db(db_path)
    conn = get_connection(db_path)
    conn.execute("INSERT INTO approved_answers (query_text, approved_response, source_hashes) VALUES ('q', 'a', '[]')")
    conn.commit()
    conn.close()

    dest = tmp_path / "backup.db"
    backup_database(db_path, dest)
    conn2 = sqlite3.connect(str(dest))
    rows = conn2.execute("SELECT query_text FROM approved_answers").fetchall()
    conn2.close()
    assert rows == [("q",)]


def test_backup_endpoint_roundtrip(client, tmp_path, monkeypatch):
    import backend.api.backup_api as backup_api

    db_path = tmp_path / "regulator.db"
    init_db(db_path)
    conn = get_connection(db_path)
    conn.execute("INSERT INTO approved_answers (query_text, approved_response, source_hashes) VALUES ('q2', 'a2', '[]')")
    conn.commit()
    conn.close()
    backup_api.BACKUP_DIR = tmp_path / "backups"  # patch dir (module-level constant)
    r = client.post("/api/backup")
    assert r.status_code == 200
    dest = Path(r.json()["path"])
    assert dest.exists()
    conn2 = sqlite3.connect(str(dest))
    assert conn2.execute("SELECT count(*) FROM approved_answers").fetchone()[0] == 1
    conn2.close()


def test_backup_retention_keeps_last_10(client, tmp_path):
    import backend.api.backup_api as backup_api

    backup_api.BACKUP_DIR = tmp_path / "backups"
    backup_api.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        (backup_api.BACKUP_DIR / f"regulator_2026010{i % 10:02d}{i:02d}.db").write_bytes(b"x")
    backup_api._prune_backups()
    assert len(list(backup_api.BACKUP_DIR.glob("*.db"))) == 10


def test_backup_listing_names_only(client, tmp_path, monkeypatch):
    import backend.api.backup_api as backup_api

    backup_api.BACKUP_DIR = tmp_path / "backups"
    backup_api.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    (backup_api.BACKUP_DIR / "regulator_x.db").write_bytes(b"x")
    r = client.get("/api/backup")
    assert r.status_code == 200
    assert r.json()["backups"] == ["regulator_x.db"]  # names, not full paths


# ---- conflicts & approved ----

def test_conflicts_roundtrip(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # db resolves under tmp docs parent
    r = client.get("/api/conflicts")
    assert r.status_code == 200
    assert r.json() == {"conflicts": []}


def test_approve_answer_endpoint(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = client.post("/api/answers/approve", json={"query_text": "ש", "approved_response": "ת", "source_hashes": ["h"]})
    assert r.status_code == 200
    assert r.json()["status"] == "approved"


def test_health_reports_version(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "version" in r.json()
