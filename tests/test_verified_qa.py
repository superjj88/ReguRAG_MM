"""Offline tests for HITL review overlay + verified-QA importer."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend.review_qa import export_approved, load_sidecar, merged, save_decision
from scripts.import_verified_qa import import_cache, import_vectors, load_approved, pick_response, vector_item


def _rec(qa_id="c_0_s0", status="pending", **kw):
    r = {
        "qa_id": qa_id,
        "chunk_id": "c_0",
        "file_hash": "h1",
        "file_name": "תקנון.pdf",
        "page_number": 1,
        "section_number": "",
        "question": "שאלה?",
        "answer": "תשובה.",
        "quote": "",
        "citation": "[תקנון.pdf | עמוד 1]",
        "qtype": "factual",
        "quote_verified": True,
        "status": status,
        "reviewer": "",
        "edited_answer": "",
        "model": "m",
        "generated_at": "t",
    }
    r.update(kw)
    return r


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- review overlay ----

def test_save_and_merge_decision(tmp_path):
    base = tmp_path / "qa.jsonl"
    _write_jsonl(base, [_rec()])
    save_decision(base, "c_0_s0", "edited", "תשובה ערוכה.", "דנה")
    decisions = load_sidecar(base)
    assert decisions["c_0_s0"]["status"] == "edited"
    rows = merged([_rec()], decisions)
    assert rows[0]["status"] == "edited" and rows[0]["edited_answer"] == "תשובה ערוכה."


def test_save_decision_rejects_bad_status(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        save_decision(tmp_path / "qa.jsonl", "x", "maybe", "", "")


def test_export_approved_only(tmp_path):
    base = tmp_path / "qa.jsonl"
    _write_jsonl(base, [_rec("a", "pending"), _rec("b", "pending")])
    save_decision(base, "a", "approved", "", "r")
    out = export_approved(base)
    assert out.name == "qa.approved.jsonl"
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["qa_id"] for r in rows] == ["a"]


# ---- importer ----

def test_pick_response_prefers_edit():
    assert pick_response(_rec(edited_answer="ערוכה")) == "ערוכה"
    assert pick_response(_rec()) == "תשובה."


def test_load_approved_filters(tmp_path):
    p = tmp_path / "in.jsonl"
    p.write_text(
        json.dumps(_rec("a", "approved"), ensure_ascii=False) + "\n"
        "not json\n"
        + json.dumps(_rec("b", "rejected"), ensure_ascii=False) + "\n"
        + json.dumps(_rec("c", "edited", edited_answer=""), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = load_approved(p)
    assert [r["qa_id"] for r in rows] == ["a", "c"]


def test_vector_item_shape():
    _id, doc, meta = vector_item(_rec())
    assert _id == "c_0_s0" and doc == "שאלה?"
    assert meta["kind"] == "question" and meta["chunk_id"] == "c_0" and meta["verified"] is True


class _FakeEmbed:
    def __init__(self, dim=4):
        self.calls = []

    def embed(self, texts, **kw):
        if isinstance(texts, str):
            texts = [texts]
        self.calls.extend(texts)
        return [[float(len(t))] * 4 for t in texts]


def test_import_vectors_roundtrip_and_resume(tmp_path):
    recs = [_rec("c_0_s0", "approved"), _rec("c_0_s1", "approved", question="שאלה שנייה?")]
    persist = str(tmp_path / "vectordb")
    s1 = import_vectors(recs, _FakeEmbed(), persist, resume=True)
    assert s1 == {"written": 2, "skipped": 0, "failed": 0}
    s2 = import_vectors(recs, _FakeEmbed(), persist, resume=True)
    assert s2["written"] == 0 and s2["skipped"] == 2

    from backend.services.qa_generator import get_or_create_collection

    col = get_or_create_collection(persist_dir=persist)
    assert col is not None
    got = col.get(include=["documents", "metadatas"])
    assert sorted(got["ids"]) == ["c_0_s0", "c_0_s1"]
    assert {str(m["kind"]) for m in got["metadatas"] or []} == {"question"}


def test_import_cache_writes_and_skips_dupes(tmp_path):
    from backend.core.database import get_connection, init_db

    db = tmp_path / "t.db"
    init_db(db)
    conn = get_connection(db)
    conn.execute(
        "INSERT INTO documents (file_name, path, file_hash_sha256, page_count, status) VALUES (?,?,?,?,?)",
        ("תקנון.pdf", "/x.pdf", "h1", 1, "ingested"),
    )
    conn.commit()
    conn.close()
    recs = [_rec("c_0_s0", "approved")]
    s1 = import_cache(recs, _FakeEmbed(), db, resume=True)
    assert s1["written"] == 1
    s2 = import_cache(recs, _FakeEmbed(), db, resume=True)
    assert s2["written"] == 0 and s2["skipped"] == 1

    from backend.services import approved_answers as ac

    hit = ac.get_cached_answer("שאלה?", db_path=db)
    assert hit is not None and hit["answer"] == "תשובה."


def test_import_cache_exact_only_on_embed_failure(tmp_path):
    from backend.core.database import get_connection, init_db

    db = tmp_path / "t.db"
    init_db(db)
    conn = get_connection(db)
    conn.execute(
        "INSERT INTO documents (file_name, path, file_hash_sha256, page_count, status) VALUES (?,?,?,?,?)",
        ("תקנון.pdf", "/x.pdf", "h1", 1, "ingested"),
    )
    conn.commit()
    conn.close()

    class _Boom:
        def embed(self, texts, **kw):
            raise RuntimeError("down")

    stats = import_cache([_rec("c_0_s0", "approved")], _Boom(), db, resume=True)
    assert stats["written"] == 1 and stats["exact_only"] == 1
