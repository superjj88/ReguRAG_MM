"""Regression tests for audit fixes: hybrid retrieval (no golden boosts), vector math."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.hybrid_retrieval import hybrid_search
from backend.services.vector_math import cosine_similarity


def _chunk(cid, text, embedding=None):
    return {"chunk_id": cid, "text": text, "file_hash": "h", "file_name": "f.pdf", "page_number": 1, "section_number": "", "embedding": embedding}


def test_cosine_similarity_basics():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert cosine_similarity([1, 2], [1, 2]) == pytest.approx(1.0)


def test_no_golden_boost_regression():
    """The chunk lexically matching the query must rank first even when another
    chunk contains the old hardcoded golden pattern (תכלול 2 מרכזי)."""
    query = "מספר הדיירים בבניין"
    golden = _chunk("g", "הוועדה תכלול 2 מרכזי הוועדה ונציגי ציבור")
    relevant = _chunk("r", "סה\"כ 41 איש בבניין. מספר הדיירים בבניין הוא 41")
    ranked = hybrid_search(query, [golden, relevant])
    assert ranked[0]["chunk_id"] == "r"


def test_dedup_keeps_highest_score():
    chunks = [
        {"chunk_id": "a", "text": "טקסט על דיירים ודיירים"},
        {"chunk_id": "a", "text": "טקסט על דיירים"},
    ]
    ranked = hybrid_search("דיירים", chunks, top_k=5)
    assert len([r for r in ranked if r["chunk_id"] == "a"]) == 1


def test_dense_scoring_uses_embeddings():
    # vector pointing at chunk "b" must rank it first without any lexical overlap
    vec_hit = [1.0, 0.0]
    vec_miss = [0.0, 1.0]
    chunks = [
        _chunk("a", "משפט לגמרי לא קשור כאן", embedding=vec_miss),
        _chunk("b", "משפט שונה לגמרי מהשאלה", embedding=vec_hit),
    ]
    ranked = hybrid_search("שאלה", chunks, top_k=2)
    assert ranked[0]["chunk_id"] == "b"


def test_empty_chunks_returns_empty():
    assert hybrid_search("שאלה", []) == []


class _FakeEmbed:
    def embed(self, texts, **kw):
        return [[1.0, 0.0]]


def test_dense_fusion_uses_stored_embeddings_and_exposes_cosine(monkeypatch):
    monkeypatch.setattr("backend.services.ollama_client.get_ollama_client", lambda: _FakeEmbed())
    chunks = [
        _chunk("lex", "דיירים דיירים", embedding=[1.0, 0.0]),
        _chunk("off", "דיירים", embedding=[0.0, 1.0]),
    ]
    ranked = hybrid_search("דיירים", chunks, top_k=2)
    by_id = {r["chunk_id"]: r for r in ranked}
    assert by_id["lex"]["dense_score"] == pytest.approx(1.0)
    assert by_id["off"]["dense_score"] == pytest.approx(0.0)
    # top row is 1.0 only because it leads on *both* signals, not by definition
    assert by_id["lex"]["score"] == pytest.approx(1.0)
    assert by_id["off"]["score"] < by_id["lex"]["score"]


def test_chunk_without_embedding_is_scored_lexically_and_reports_no_cosine(monkeypatch):
    monkeypatch.setattr("backend.services.ollama_client.get_ollama_client", lambda: _FakeEmbed())
    ranked = hybrid_search("טקסט", [_chunk("a", "טקסט")], top_k=1)
    assert ranked[0]["dense_score"] is None
    assert ranked[0]["score"] >= 0.0


def test_query_results_carry_embeddings_into_ranking():
    from backend.api import chat_api

    res = {
        "documents": [["טקסט"]],
        "metadatas": [[{"chunk_id": "c1", "file_hash": "h"}]],
        "embeddings": [[[0.1, 0.2, 0.3]]],
    }
    chunks = chat_api._chunks_from_query_results(res)
    assert chunks[0]["embedding"] == [0.1, 0.2, 0.3]


def test_toc_detection_uses_dot_leaders():
    from backend.services.hybrid_retrieval import looks_like_toc

    toc = 'נספח א׳ .............. 2\nנספח ב׳ .............. 3\nנספח ג׳ .............. 4\nנספח ד׳ .............. 5'
    assert looks_like_toc(toc)
    assert not looks_like_toc("נספח א': הנחיות לאכלוס דירות אירוח. 2. נספח ב': לו\"ז לאישור תכניות")
