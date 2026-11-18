"""Offline tests for the chunk/link/vector infrastructure.

Covers: chunks table helpers, document link matcher (annex/sibling), the
QAGenerator vector writer (MockTransport ollama + tmp Chroma), question-chunk
resolution and link expansion in the chat path, pairwise similarity math,
and the contradiction detector persistence. No network, no live DB.
"""

import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.database import (
    delete_chunks_by_hash,
    get_chunk_text,
    get_chunks_by_hash,
    get_connection,
    init_db,
    insert_chunks,
    replace_links_for_source,
    upsert_document_by_hash,
)

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
def tmp_db(monkeypatch, tmp_path):
    """Tmp DB + config env pointing at a tmp documents dir."""
    docs = tmp_path / "docs"
    docs.mkdir()
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(MINIMAL_CONFIG.format(docs=docs), encoding="utf-8")
    monkeypatch.setenv("REGURAG_CONFIG", str(cfg_file))
    from backend.core.config import get_config

    get_config.cache_clear()
    db_path = tmp_path / "regulator.db"
    init_db(db_path)
    yield tmp_path
    monkeypatch.delenv("REGURAG_CONFIG", raising=False)
    get_config.cache_clear()


def _conn(tmp_path):
    return get_connection(Path(tmp_path) / "regulator.db")


def _chunk(cid: str, h: str = "h1", text: str = "טקסט", page: int = 1, section: str = "") -> dict:
    return {
        "chunk_id": cid,
        "file_hash": h,
        "file_name": f"{h}.pdf",
        "page_number": page,
        "section_number": section,
        "text": text,
    }


def _upsert(conn, path: str, h: str) -> None:
    upsert_document_by_hash(
        conn,
        file_name=Path(path).name,
        path=path,
        file_hash_sha256=h,
    )


# ---- chunks table ----


def test_chunks_crud_roundtrip(tmp_db):
    conn = _conn(tmp_db)
    try:
        chunks = [_chunk("h1_0", "h1"), _chunk("h1_1", "h1", page=2), _chunk("h2_0", "h2")]
        assert insert_chunks(conn, chunks) == 3
        rows = get_chunks_by_hash(conn, "h1")
        assert [r["chunk_id"] for r in rows] == ["h1_0", "h1_1"]
        assert rows[0]["text"] == "טקסט"
        row = get_chunk_text(conn, "h1_1")
        assert row is not None and row["file_hash"] == "h1" and row["page_number"] == 2
        assert get_chunk_text(conn, "nope") is None
        assert delete_chunks_by_hash(conn, "h1") == 2
        assert get_chunks_by_hash(conn, "h1") == []
        assert get_chunk_text(conn, "h1_0") is None
    finally:
        conn.close()


def test_chunks_reinsert_is_idempotent(tmp_db):
    conn = _conn(tmp_db)
    try:
        insert_chunks(conn, [_chunk("h1_0", "h1", text="v1")])
        insert_chunks(conn, [_chunk("h1_0", "h1", text="v2")])
        rows = get_chunks_by_hash(conn, "h1")
        assert len(rows) == 1 and rows[0]["text"] == "v2"
    finally:
        conn.close()


# ---- document_links ----


def test_annex_fragment_extracts_topic():
    from backend.services.document_links import annex_fragment

    assert annex_fragment("נספחים לתקנון ועדת שיכון") == "ועדת שיכון"
    assert annex_fragment("נספח לתקנון קליטה") == "קליטה"
    assert annex_fragment("תוספת לתקנון פנסיה") == "פנסיה"
    assert annex_fragment("תקנון ועדת שיכון") is None
    assert annex_fragment("ללא תבנית") is None


def test_annex_link_matches_parent_not_annexes(tmp_db):
    from backend.services.document_links import build_annex_links

    conn = _conn(tmp_db)
    try:
        _upsert(conn, "data/regulations/תקנון ועדת שיכון/תקנון ועדת שיכון.pdf", "parent")
        _upsert(conn, "data/regulations/נספחים לתקנון ועדת שיכון/נספחים לתקנון ועדת שיכון.pdf", "other_annex")
        _upsert(conn, "data/regulations/נוהל בריכה/נוהל בריכה.pdf", "unrelated")
        docs = [dict(d) for d in conn.execute("SELECT * FROM documents").fetchall()]
        links = build_annex_links("ועדת שיכון", docs, "annex_self")
        assert len(links) == 1
        assert links[0]["target_hash"] == "parent"
        assert links[0]["link_type"] == "annex"
    finally:
        conn.close()


def test_sibling_links_share_folder(tmp_db):
    from backend.services.document_links import build_sibling_links

    conn = _conn(tmp_db)
    try:
        _upsert(conn, "data/regulations/תקנון ועדת חתונות/a.pdf", "sib1")
        _upsert(conn, "data/regulations/תקנון ועדת חתונות/b.pdf", "sib2")
        _upsert(conn, "data/regulations/אחר/c.pdf", "far")
        docs = [dict(d) for d in conn.execute("SELECT * FROM documents").fetchall()]
        links = build_sibling_links("data/regulations/תקנון ועדת חתונות/a.pdf", docs)
        assert len(links) == 1
        assert links[0]["target_hash"] == "sib2"
        assert links[0]["link_type"] == "sibling"
    finally:
        conn.close()


def test_rebuild_document_links_end_to_end(tmp_db):
    from backend.core.database import get_links_for_hash
    from backend.services.document_links import rebuild_document_links

    conn = _conn(tmp_db)
    try:
        _upsert(conn, "data/regulations/תקנון קליטה/תקנון קליטה.pdf", "parent")
        _upsert(conn, "data/regulations/נספח לתקנון קליטה/נספח לתקנון קליטה.pdf", "annex")
        n = rebuild_document_links(conn)
        assert n >= 1
        annex_links = get_links_for_hash(conn, "annex")
        assert any(link["link_type"] == "annex" and link["target_hash"] == "parent" for link in annex_links)
        parent_links = get_links_for_hash(conn, "parent")
        assert any(link["link_type"] == "annex" and link["source_hash"] == "annex" for link in parent_links)
        assert any(link.get("source_name") for link in annex_links)
        rebuild_document_links(conn)
        assert len(get_links_for_hash(conn, "annex")) == len(annex_links)
    finally:
        conn.close()


def test_replace_links_skips_self_link(tmp_db):
    conn = _conn(tmp_db)
    try:
        n = replace_links_for_source(
            conn,
            "h1",
            [{"target_hash": "h1", "link_type": "sibling", "basis": "folder"}],
        )
        assert n == 0
    finally:
        conn.close()


# ---- vector writer (MockTransport ollama + tmp Chroma) ----


def _embed_transport():
    def handler2(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content.decode("utf-8"))
        vecs = [[0.1, 0.2, 0.3] for _ in body["input"]]
        return httpx.Response(200, json={"embeddings": vecs})

    return handler2


def _make_transport(handler_fn):
    return httpx.MockTransport(handler_fn)


def _qa_transport():
    """Embed + chat (question generation) transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content.decode("utf-8"))
        if "input" in body:
            vecs = [[0.5, 0.5, 0.5] for _ in body["input"]]
            return httpx.Response(200, json={"embeddings": vecs})
        return httpx.Response(
            200,
            json={"model": "test-llm", "message": {"content": "מה אומר התקנון?\nהאם הוא חל?\nלמה הוא נכתב?"}},
        )

    return handler


def _write_cfg_with(tmp_path, *, docs: Path, qa_pairs_enabled: bool) -> Path:
    cfg_file = tmp_path / f"config_{int(qa_pairs_enabled)}.yaml"
    cfg_file.write_text(
        MINIMAL_CONFIG.format(docs=docs) + f"  qa_pairs_enabled: {str(qa_pairs_enabled).lower()}\n",
        encoding="utf-8",
    )
    return cfg_file


@pytest.fixture()
def qa_env(monkeypatch, tmp_db):
    """Returns a callable that (re)points REGURAG_CONFIG at a qa-flag config."""
    from backend.core.config import get_config

    def _activate(qa_pairs_enabled: bool) -> None:
        cfg_file = _write_cfg_with(tmp_db, docs=tmp_db / "docs", qa_pairs_enabled=qa_pairs_enabled)
        monkeypatch.setenv("REGURAG_CONFIG", str(cfg_file))
        get_config.cache_clear()

    yield _activate
    monkeypatch.delenv("REGURAG_CONFIG", raising=False)
    get_config.cache_clear()


def test_vector_writer_stores_chunk_text_with_metadata(tmp_db, qa_env):
    from backend.services.ollama_client import OllamaClient
    from backend.services.qa_generator import QAGenerator, get_or_create_collection

    qa_env(False)
    ollama = OllamaClient(transport=_make_transport(_embed_transport()))
    persist = str(tmp_db / "vectordb")
    gen = QAGenerator(ollama_client=ollama, batch_size=2, persist_dir=persist)
    progress = gen.generate_for_chunks([_chunk("h1_0", "h1"), _chunk("h1_1", "h1", page=2)])

    assert progress["state"] == "done"
    assert progress["processed"] == 2
    assert progress["last_error"] is None
    assert progress["skipped_no_embedding"] == 0
    col = get_or_create_collection(persist_dir=persist)
    assert col is not None
    got = col.get(include=["documents", "metadatas"])
    assert sorted(got["ids"]) == ["h1_0", "h1_1"]
    assert got["documents"][0] == "טקסט"  # chunk text, not generated questions
    meta = got["metadatas"][0]
    assert meta["kind"] == "chunk"
    assert meta["file_name"] == "h1.pdf" and meta["file_hash"] == "h1"
    assert meta["chunk_id"] in ("h1_0", "h1_1")
    ollama.close()


def test_vector_writer_qa_pairs_optional(tmp_db, qa_env):
    from backend.services.ollama_client import OllamaClient
    from backend.services.qa_generator import QAGenerator, get_or_create_collection

    ollama = OllamaClient(transport=_make_transport(_qa_transport()))
    persist = str(tmp_db / "vectordb")

    qa_env(False)
    QAGenerator(ollama_client=ollama, batch_size=2, persist_dir=persist).generate_for_chunks(
        [_chunk("h1_0", "h1")]
    )
    col = get_or_create_collection(persist_dir=persist)
    assert col is not None
    got = col.get(include=["metadatas"])
    assert {str(m["kind"]) for m in got["metadatas"]} == {"chunk"}  # default: QA off

    qa_env(True)
    QAGenerator(ollama_client=ollama, batch_size=2, persist_dir=persist).generate_for_chunks(
        [_chunk("h1_0", "h1")]
    )
    col2 = get_or_create_collection(persist_dir=persist)
    assert col2 is not None
    got2 = col2.get(include=["metadatas"])
    assert {str(m["kind"]) for m in got2["metadatas"]} == {"chunk", "question"}
    ollama.close()


def test_vector_writer_embedding_failure_is_surfaced(tmp_db, qa_env):
    from backend.services.ollama_client import OllamaClient
    from backend.services.qa_generator import QAGenerator

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    qa_env(False)
    ollama = OllamaClient(transport=httpx.MockTransport(handler))
    gen = QAGenerator(ollama_client=ollama, batch_size=2, persist_dir=str(tmp_db / "vectordb"))
    progress = gen.generate_for_chunks([_chunk("h1_0", "h1")])
    assert progress["state"] == "done"
    assert progress["last_error"] is not None
    assert progress["skipped_no_embedding"] == 1
    ollama.close()


def test_delete_doc_vectors_removes_all_kinds(tmp_db):
    from backend.services.qa_generator import delete_doc_vectors, get_or_create_collection

    persist = str(tmp_db / "vectordb")
    col = get_or_create_collection(persist_dir=persist)
    assert col is not None
    col.add(
        ids=["h1_0", "h1_0_q0", "h2_0"],
        documents=["a", "q", "b"],
        metadatas=[
            {"file_hash": "h1", "kind": "chunk"},
            {"file_hash": "h1", "kind": "question"},
            {"file_hash": "h2", "kind": "chunk"},
        ],
        embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    )
    assert delete_doc_vectors("h1", persist_dir=persist) == 2
    got = get_or_create_collection(persist_dir=persist).get()
    assert got["ids"] == ["h2_0"]


# ---- chat path: question resolution + link expansion ----


@pytest.fixture()
def chat_client(tmp_db):
    from backend.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _chat_api():
    import backend.api.chat_api as m

    return m


def test_chat_resolves_question_kind_from_chunks_table(chat_client, tmp_db, monkeypatch):
    conn = _conn(tmp_db)
    try:
        insert_chunks(conn, [_chunk("h1_0", "h1", text="הטקסט האמיתי")])
    finally:
        conn.close()
    api = _chat_api()
    monkeypatch.setattr(api, "get_cached_answer", lambda q: None)
    monkeypatch.setattr(
        api,
        "_retrieve",
        lambda q, top_k: [
            {
                "text": "מה אומר התקנון?",
                "chunk_id": "h1_0",
                "file_hash": "h1",
                "file_name": "h1.pdf",
                "page_number": 1,
                "section_number": "",
                "kind": "question",
            }
        ],
    )
    monkeypatch.setattr(
        api,
        "generate_answer",
        lambda q, chunks, timeout_s=None: {"answer": "ת [h1.pdf | עמוד 1 | סעיף 1]", "citations": [], "refusal": False},
    )
    # keep the flow fully offline: no real hybrid scoring / stage3 LLM / chroma
    monkeypatch.setattr(api, "hybrid_search", lambda q, chunks, **k: [{**chunks[0], "score": 0.99}])
    monkeypatch.setattr(api, "maybe_refine", lambda q, ranked, all_chunks, **k: ranked)
    monkeypatch.setattr(api, "_attach_embeddings", lambda chunks: chunks)
    r = chat_client.post("/api/chat", json={"query": "שאלה"})
    assert r.status_code == 200
    assert r.json()["cached"] is False


def test_question_without_chunk_row_is_dropped(chat_client, tmp_db, monkeypatch):
    api = _chat_api()
    monkeypatch.setattr(api, "get_cached_answer", lambda q: None)
    monkeypatch.setattr(api, "_retrieve", lambda q, top_k: [])
    resolved = api._resolve_question_chunks(
        [{"text": "שאלה", "chunk_id": "ghost", "file_hash": "h", "file_name": "f", "page_number": 1, "section_number": "", "kind": "question"}]
    )
    assert resolved == []


def test_link_expansion_widens_pool(tmp_db):
    from backend.services.document_links import rebuild_document_links
    import backend.api.chat_api as chat_api

    conn = _conn(tmp_db)
    try:
        _upsert(conn, "data/regulations/תקנון קליטה/תקנון קליטה.pdf", "parent")
        _upsert(conn, "data/regulations/נספח לתקנון קליטה/נספח לתקנון קליטה.pdf", "annex")
        insert_chunks(conn, [_chunk("p_0", "parent"), _chunk("a_0", "annex")])
        rebuild_document_links(conn)
    finally:
        conn.close()

    seed = [{"text": "ט", "chunk_id": "p_0", "file_hash": "parent", "file_name": "p.pdf", "page_number": 1, "section_number": ""}]
    expanded = chat_api._expand_with_linked_chunks(seed, max_chunks=5)
    assert any(c["file_hash"] == "annex" for c in expanded)
    assert len(expanded) == 2


def test_link_expansion_respects_max_chunks(tmp_db):
    from backend.services.document_links import rebuild_document_links
    import backend.api.chat_api as chat_api

    conn = _conn(tmp_db)
    try:
        _upsert(conn, "data/regulations/תקנון קליטה/תקנון קליטה.pdf", "parent")
        _upsert(conn, "data/regulations/נספח לתקנון קליטה/נספח לתקנון קליטה.pdf", "annex")
        insert_chunks(conn, [_chunk(f"a_{i}", "annex", page=i + 1) for i in range(5)])
        rebuild_document_links(conn)
    finally:
        conn.close()

    seed = [{"text": "ט", "chunk_id": "p_0", "file_hash": "parent", "file_name": "p.pdf", "page_number": 1, "section_number": ""}]
    expanded = chat_api._expand_with_linked_chunks(seed, max_chunks=2)
    assert len(expanded) == 3  # seed + 2 capped


# ---- pairwise similarities ----


def test_pairwise_similarities_matrix():
    from backend.services.vector_math import pairwise_similarities

    new_vecs = [[1.0, 0.0], [0.0, 1.0]]
    existing_vecs = [[1.0, 0.0], [0.7071067811865476, 0.7071067811865476]]
    mat = pairwise_similarities(new_vecs, existing_vecs)
    assert mat is not None
    assert len(mat) == 2 and len(mat[0]) == 2
    assert abs(mat[0][0] - 1.0) < 1e-6
    assert abs(mat[0][1] - 0.7071067811865476) < 1e-6
    assert abs(mat[1][0]) < 1e-6


def test_pairwise_similarities_empty_returns_none():
    from backend.services.vector_math import pairwise_similarities

    assert pairwise_similarities([], [[1.0]]) is None
    assert pairwise_similarities([[1.0]], []) is None


# ---- contradiction detector ----


def test_contradiction_detector_persists_conflict(tmp_db):
    from backend.core.database import list_conflicts
    import backend.services.contradiction_detector as cd

    vec = [1.0, 0.0]

    class FakeClient:
        def generate(self, prompt, **kwargs):
            class R:
                text = "כן, יש סתירה בנושא שעות"

            return R()

    conflicts = cd.detect_contradictions_for_document(
        [_chunk("h1_0", "h1", text="סעיף א קובע 3 שעות")],
        [_chunk("h2_0", "h2", text="סעיף ב קובע 3 שעות")],
        ollama_client=FakeClient(),
        db_path=tmp_db / "regulator.db",
        new_vectors={0: vec},
        existing_vectors={0: vec},
    )
    assert len(conflicts) == 1
    conn = get_connection(tmp_db / "regulator.db")
    try:
        rows = list_conflicts(conn)
        assert len(rows) == 1
        assert {rows[0]["new_doc_hash"], rows[0]["other_doc_hash"]} == {"h1", "h2"}
    finally:
        conn.close()


def test_judgment_heuristic_rejects_negation():
    import backend.services.contradiction_detector as cd

    # the model often echoes the question: "לא, אין סתירה ... כן שניהם עוסקים..."
    assert cd._judgment_is_contradiction("לא, אין סתירה בין שני הקטעים. הקטעים משלימים זה את זה.") is False
    assert cd._judgment_is_contradiction("לא, הקטעים אינם סותרים אלא משלימים.") is False
    assert cd._judgment_is_contradiction("כן, יש סתירה בנושא שעות העבודה.") is True
    assert cd._judgment_is_contradiction("הקטעים סותרים זה את זה בנושא ההחזר.") is True
    assert cd._judgment_is_contradiction("לא.") is False
