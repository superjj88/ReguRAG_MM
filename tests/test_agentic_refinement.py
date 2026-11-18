"""Offline tests: stage-3 refinement must not feed prose back into retrieval."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services import agentic_refinement as ar


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, prompt, **kw):
        self.calls.append(prompt)
        return SimpleNamespace(text=self.text)


def _ranked():
    return [
        {
            "chunk_id": "c1",
            "text": "טקסט",
            "file_hash": "h",
            "file_name": "f.pdf",
            "page_number": 1,
            "section_number": "",
            "score": 0.5,
            "dense_score": 0.1,
        }
    ]


def test_prose_answer_is_not_used_as_query(monkeypatch):
    seen = []
    monkeypatch.setattr(ar, "hybrid_search", lambda q, chunks, **k: seen.append(q) or [])
    ranked = ar.maybe_refine(
        "שאלה",
        _ranked(),
        _ranked(),
        ollama_client=FakeClient("## התשובה: התשובה לא קיימת בהקשר\n- שכר\n- שיכון"),
        stage3_budget_s=15,
    )
    assert seen == []
    assert ranked[0]["chunk_id"] == "c1"


def test_refined_query_widens_pool_and_is_reranked_on_original(monkeypatch):
    seen = {}
    fresh = {
        "chunk_id": "c2",
        "text": "קטע חדש",
        "file_hash": "h2",
        "file_name": "g.pdf",
        "page_number": 3,
        "section_number": "",
        "score": 0.4,
        "dense_score": 0.4,
    }

    def fake_hybrid(q, chunks, **k):
        seen["query"] = q
        seen["chunks"] = chunks
        return [chunks[-1]]

    monkeypatch.setattr(ar, "hybrid_search", fake_hybrid)
    retrieved = []
    out = ar.maybe_refine(
        "שאלה",
        _ranked(),
        _ranked(),
        ollama_client=FakeClient("  זכאות לחופשת לידה  "),
        stage3_budget_s=15,
        retriever=lambda q: retrieved.append(q) or [fresh],
    )
    assert retrieved == ["זכאות לחופשת לידה"]
    # relevance is decided by the ORIGINAL query over the widened pool
    assert seen["query"] == "שאלה"
    assert {c["chunk_id"] for c in seen["chunks"]} == {"c1", "c2"}
    assert out == [fresh]


def test_as_search_query_rejects_long_or_multi_line_text():
    assert ar._as_search_query("שאילתה קצרה") == "שאילתה קצרה"
    assert ar._as_search_query("שורה\nשנייה") is None
    assert ar._as_search_query("א" * (ar.MAX_REFINED_QUERY_CHARS + 1)) is None
