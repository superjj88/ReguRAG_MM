"""Offline tests: answer guardrails (page-level citations + verification +
budget split) and the semantic approved-answers cache.

2026-09-16: user requirement — the system must always return a definitive
cited answer or an explicit refusal, and a later very similar question must
retrieve the approved answer instantly.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3

from backend.core.database import (
    add_approved_answer,
    get_connection,
    init_db,
    list_approved_semantic_candidates,
)
from backend.services import approved_answers as ac
from backend.services import answer_generator as ag

DOC = "נוהל בריכה מתאריך 15.4.26.pdf"
CHUNKS = [
    {"text": ".16 האחריות בבריכות הילדים הינה של ההורים בלבד.", "file_name": DOC,
     "file_hash": "h1", "chunk_id": "h1_1", "page_number": 1, "section_number": ""},
    {"text": ".10 אין להכניס ילדים לא גמולים לחלוטין לבריכת השחייה הגדולה.", "file_name": DOC,
     "file_hash": "h1", "chunk_id": "h1_2", "page_number": 1, "section_number": ""},
    {"text": "כלל כלשהו", "file_name": "תקנון פנסיה וגמלאות.pdf",
     "file_hash": "h2", "chunk_id": "h2_1", "page_number": 4, "section_number": "2.1"},
]


@dataclass
class FakeGenClient:
    responses: list[str]
    calls: list[tuple[str, float | None]] = field(default_factory=list)

    def generate(self, prompt, *, temperature=0.0, timeout_s=None, **kw):
        self.calls.append((prompt, timeout_s))
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return SimpleNamespace(text=self.responses[idx], model="fake", eval_count=None, duration_s=0.0)


@dataclass
class FakeEmbedClient:
    vectors: dict[str, list[list[float]]]
    calls: list[str] = field(default_factory=list)

    def embed(self, texts, **kw):
        if isinstance(texts, str):
            texts = [texts]
        for t in texts:
            self.calls.append(t)
        return [self.vectors[t][0] for t in texts]


# ---- answer_generator -------------------------------------------------------

def test_context_omits_empty_section():
    ctx = ag._build_context(CHUNKS)
    assert f"[{DOC} | עמוד 1]" in ctx
    assert "סעיף ]" not in ctx and "סעיף ]]" not in ctx
    assert "[תקנון פנסיה וגמלאות.pdf | עמוד 4 | סעיף 2.1]" in ctx


def test_citation_regex_accepts_both_forms():
    assert ag.CITATION_RE.search(f"[{DOC} | עמוד 1]")
    assert ag.CITATION_RE.search("[תקנון פנסיה.pdf | עמוד 4 | סעיף 2.1]")
    assert not ag.CITATION_RE.search("[מסמך | עמוד איקס]")
    assert not ag.CITATION_RE.search("רק טקסט בלי מראה מקור")


def test_verified_citations_drops_fabricated():
    answer = f"תשובה [{DOC} | עמוד 1] ועוד [מסמך אחר.pdf | עמוד 9] וגם [{DOC} | עמוד 77]"
    cites = ag._verified_citations(answer, CHUNKS)
    assert cites == [f"[{DOC} | עמוד 1]"]


def test_generate_accepts_verified_answer_first_try():
    answer = f"כן, האחריות על ההורים [{DOC} | עמוד 1]"
    client = FakeGenClient([answer])
    result = ag.generate_answer("q", CHUNKS, ollama_client=client, timeout_s=60)
    assert result["refusal"] is False
    assert result["citations"] == [f"[{DOC} | עמוד 1]"]
    assert len(client.calls) == 1


def test_generate_refuses_when_citation_unverifiable():
    bad = "תשובה [מסמך לא קיים.pdf | עמוד 3]"
    client = FakeGenClient([bad, bad])
    result = ag.generate_answer("q", CHUNKS, ollama_client=client, timeout_s=60)
    assert result["refusal"] is True
    assert result["answer"] == ag.REFUSAL
    assert len(client.calls) == 2  # retried once, then refused


def test_budget_split_across_attempts():
    client = FakeGenClient([f"ת [ {DOC} | עמוד 1 ]".replace(" ", ""), ""])
    ag.generate_answer("q", CHUNKS, ollama_client=client, timeout_s=100)
    assert client.calls[0][1] is not None and client.calls[1][1] is not None
    t1: float = client.calls[0][1]  # type: ignore[assignment]
    t2: float = client.calls[1][1]  # type: ignore[assignment]
    assert 0 < t1 < 100 and 0 < t2 < 100 and t1 + t2 <= 100.0 + 1e-6


def test_no_retry_budget_skips_second_attempt():
    client = FakeGenClient(["ללא מראה מקור"])
    result = ag.generate_answer("q", CHUNKS, ollama_client=client, timeout_s=8)
    assert result["refusal"] is True
    assert len(client.calls) == 1


def test_split_budget_none_uses_config():
    first, retry = ag._split_budget(None)
    assert first > 0 and retry > 0


# ---- semantic cache ---------------------------------------------------------

def _seed_db(tmp_path):
    db = tmp_path / "cache.db"
    init_db(db)
    conn = get_connection(db)
    conn.execute(
        "INSERT INTO documents (file_name, path, file_hash_sha256, page_count, status) VALUES (?, ?, ?, ?, ?)",
        ("d.pdf", "/d.pdf", "h1", 1, "ingested"),
    )
    conn.commit()
    return db, conn


def test_approve_stores_embedding(tmp_path):
    db, _ = _seed_db(tmp_path)
    client = FakeEmbedClient({"שאלה": [[1.0, 0.0]]})
    orig = ac._ollama_client
    ac._ollama_client = lambda: client
    try:
        aid = ac.approve_answer("שאלה", "תשובה", ["h1"], db_path=db)
        assert aid > 0
    finally:
        ac._ollama_client = orig
    conn = get_connection(db)
    rows = list_approved_semantic_candidates(conn, valid_hashes={"h1"})
    conn.close()
    assert len(rows) == 1 and rows[0]["query_embedding_vec"] == [1.0, 0.0]


def test_semantic_hit_on_paraphrase(tmp_path, monkeypatch):
    db, _ = _seed_db(tmp_path)
    vec = [0.6, 0.8]
    client = FakeEmbedClient({
        "מאיזה גיל מותר להכנס לבריכה לבד?": [vec],
        "מאיזה גיל מותר להיכנס לבריכה לבד?": [vec],  # identical embedding
    })
    monkeypatch.setattr(ac, "_ollama_client", lambda: client)
    ac.approve_answer("מאיזה גיל מותר להיכנס לבריכה לבד?", "תשובה מאושרת", ["h1"], db_path=db)
    hit = ac.get_cached_answer("מאיזה גיל מותר להכנס לבריכה לבד?", db_path=db)
    assert hit is not None and hit["answer"] == "תשובה מאושרת" and hit["cached"] is True


def test_semantic_miss_below_threshold(tmp_path, monkeypatch):
    db, _ = _seed_db(tmp_path)
    client = FakeEmbedClient({
        "שאלה א": [[1.0, 0.0]],
        "שאלה שונה לגמרי": [[0.0, 1.0]],  # cosine 0 < 0.85
    })
    monkeypatch.setattr(ac, "_ollama_client", lambda: client)
    ac.approve_answer("שאלה א", "תשובה", ["h1"], db_path=db)
    assert ac.get_cached_answer("שאלה שונה לגמרי", db_path=db) is None
    # exact still works even for a miss-shaped vector
    exact = ac.get_cached_answer("שאלה א", db_path=db)
    assert exact is not None and exact["answer"] == "תשובה"


def test_embedding_failure_caches_exact_only(tmp_path, monkeypatch):
    db, _ = _seed_db(tmp_path)

    class BoomEmbed:
        def embed(self, texts, **kw):
            raise RuntimeError("ollama down")

    monkeypatch.setattr(ac, "_ollama_client", lambda: BoomEmbed())
    aid = ac.approve_answer("שאלה", "תשובה", ["h1"], db_path=db)  # must not raise
    conn = get_connection(db)
    rows = list_approved_semantic_candidates(conn)
    conn.close()
    assert aid > 0 and rows == []  # no usable embedding → not in semantic pool
    assert ac.get_cached_answer("שאלה", db_path=db) is not None  # exact path intact


def test_corrupt_embedding_skipped_not_crash(tmp_path, monkeypatch):
    db, _ = _seed_db(tmp_path)
    ac.approve_answer("שאלה", "תשובה", ["h1"], db_path=db)
    conn = get_connection(db)
    conn.execute("UPDATE approved_answers SET query_embedding='not json' WHERE id=1")
    conn.commit()
    assert list_approved_semantic_candidates(conn) == []
    conn.close()
    client = FakeEmbedClient({"שאלה פרפרזה": [[1.0, 0.0]]})
    monkeypatch.setattr(ac, "_ollama_client", lambda: client)
    # corrupt embedding row excluded from semantic pool → paraphrase misses
    assert ac.get_cached_answer("שאלה פרפרזה", db_path=db) is None
    # exact text still matches through tier 1
    assert ac.get_cached_answer("שאלה", db_path=db) is not None


def test_validity_filter_excludes_stale_sources(tmp_path, monkeypatch):
    db, _ = _seed_db(tmp_path)
    vec = [1.0, 0.0]
    client = FakeEmbedClient({"שאלה": [vec]})
    monkeypatch.setattr(ac, "_ollama_client", lambda: client)
    ac.approve_answer("שאלה", "תשובה", ["ghost-hash"], db_path=db)
    assert ac.get_cached_answer("שאלה", db_path=db) is None


def test_legacy_db_migration_adds_column(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE approved_answers (id INTEGER PRIMARY KEY, query_text TEXT, approved_response TEXT, source_hashes TEXT, approved_at TIMESTAMP)")
    conn.execute("INSERT INTO approved_answers (query_text, approved_response, source_hashes) VALUES ('q', 'a', '[]')")
    conn.commit()
    conn.close()
    init_db(db)
    cols = {r[1] for r in sqlite3.connect(str(db)).execute("PRAGMA table_info(approved_answers)")}
    assert "query_embedding" in cols
    # pre-existing row unaffected + still exact-matchable
    assert ac.get_cached_answer("q", db_path=db) is not None


def test_add_approved_answer_direct_without_embedding(tmp_path):
    db, _ = _seed_db(tmp_path)
    conn = get_connection(db)
    aid = add_approved_answer(conn, query_text="q", approved_response="a", source_hashes=["h1"])
    conn.close()
    assert aid == 1


def test_generate_treats_stated_absence_as_refusal_even_with_citations():
    answer = f"לא נמצא בתקנונים התייחסות לנושא. [{DOC} | עמוד 1]"
    client = FakeGenClient([answer])
    result = ag.generate_answer("q", CHUNKS, ollama_client=client, timeout_s=60)
    assert result["refusal"] is True
    assert result["answer"] == ag.REFUSAL
    assert result["citations"] == []
    assert len(client.calls) == 1


def test_generate_keeps_real_answer_that_mentions_absence_later():
    padding = "התקנון קובע את סדרי העדיפויות בוועדת השיכון ובחלוקת הדירות בקיבוץ " * 3
    answer = f"{padding}עם זאת, לא נמצא אזכור למקרה חריג [{DOC} | עמוד 1]"
    client = FakeGenClient([answer])
    result = ag.generate_answer("q", CHUNKS, ollama_client=client, timeout_s=60)
    assert result["refusal"] is False
    assert result["citations"] == [f"[{DOC} | עמוד 1]"]


def test_looks_like_refusal_detects_common_phrasings():
    assert ag._looks_like_refusal(ag.REFUSAL)
    assert ag._looks_like_refusal("אין התייחסות לנושא בתקנונים")
    assert ag._looks_like_refusal("הנושא אינו מופיע בתקנון")
    assert not ag._looks_like_refusal("התקנון קובע כי הוועדה תכלול שני מרכזים")


def test_looks_like_refusal_ignores_prompt_window_complaints():
    assert not ag._looks_like_refusal("התוכן המפורט של הנוהל אינו מופיע בקונטקסט")
