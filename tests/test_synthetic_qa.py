"""Offline tests for the synthetic Q&A pilot generator (no Ollama, no network)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_synthetic_qa import (
    SYSTEM_PROMPT,
    _extract_json_array,
    build_citation,
    doc_name_keys,
    drop_recorded,
    fetch_chunks,
    generate_for_chunk,
    is_self_contained,
    load_doc_titles,
    load_done_chunk_ids,
    load_existing_qa_ids,
    make_records,
    parse_pairs,
    quote_verified,
    select_self_contained,
    title_candidate,
)

CHUNK = {
    "chunk_id": "doc_0",
    "file_hash": "h1",
    "file_name": "תקנון.pdf",
    "page_number": 3,
    "section_number": "2.1",
    "text": "האחריות בבריכות הילדים הינה של ההורים בלבד. אין להכניס ילדים לא גמולים לבריכה הגדולה.",
}


def test_build_citation_with_section():
    assert build_citation(CHUNK) == "[תקנון.pdf | עמוד 3 | סעיף 2.1]"


def test_build_citation_page_level_without_section():
    c = dict(CHUNK, section_number="")
    assert build_citation(c) == "[תקנון.pdf | עמוד 3]"
    assert "סעיף ]" not in build_citation(c)


def test_parse_pairs_json_array():
    raw = json.dumps(
        [
            {"question": "מי אחראי בבריכות הילדים?", "answer": "ההורים", "quote": "האחריות בבריכות"},
            {"question": "  ", "answer": "ריק", "quote": ""},  # dropped: empty question
        ],
        ensure_ascii=False,
    )
    pairs = parse_pairs(raw, 4)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "מי אחראי בבריכות הילדים?"


def test_parse_pairs_tolerates_fences_and_dedups():
    raw = '```json\n[{"question": "q?", "answer": "a", "quote": ""}, {"question": "q?", "answer": "a2", "quote": ""}]\n```'
    assert parse_pairs(raw, 4) == [{"question": "q?", "answer": "a", "quote": ""}]


def test_parse_pairs_rejects_non_list():
    import pytest

    with pytest.raises(ValueError):
        parse_pairs('{"question": "q"}', 4)


def test_quote_verified_anchor():
    assert quote_verified("האחריות בבריכות הילדים הינה", CHUNK["text"]) is True
    assert quote_verified("משפט שלא קיים בטקסט כלל וכלל", CHUNK["text"]) is False
    assert quote_verified("", CHUNK["text"]) is False


def test_make_records_schema_and_qtype_rotation():
    pairs = [
        {"question": f"q{i}?", "answer": f"a{i}", "quote": "האחריות בבריכות"}
        for i in range(4)
    ]
    recs = make_records(CHUNK, pairs, "fake-model")
    assert [r["qa_id"] for r in recs] == [f"doc_0_s{i}" for i in range(4)]
    assert [r["qtype"] for r in recs] == ["factual", "procedural", "conditional", "factual"]
    assert all(r["status"] == "pending" and r["citation"].endswith("| סעיף 2.1]") for r in recs)
    assert all(r["quote_verified"] for r in recs)
    assert all(r["model"] == "fake-model" for r in recs)


def test_generate_for_chunk_uses_client():
    client = SimpleNamespace(
        text=json.dumps([{"question": "מה קובע תקנון בנושא?", "answer": "a", "quote": ""}], ensure_ascii=False),
        calls=[],
    )

    class Fake:
        def generate(self, prompt, *, temperature=0.0, timeout_s=None, **kw):
            assert "תקנון.pdf" in prompt
            assert temperature == 0.7
            return SimpleNamespace(text=client.text, model="m", eval_count=None, duration_s=0.0)

    assert generate_for_chunk(CHUNK, Fake(), 4, 0.7, 10.0) == [
        {"question": "מה קובע תקנון בנושא?", "answer": "a", "quote": ""}
    ]


def test_load_done_chunk_ids_skips_bad_lines(tmp_path):
    p = tmp_path / "qa.jsonl"
    p.write_text(
        json.dumps({"chunk_id": "a"}, ensure_ascii=False) + "\nnot json\n" + json.dumps({"nope": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert load_done_chunk_ids(p) == {"a"}
    assert load_done_chunk_ids(tmp_path / "missing.jsonl") == set()


def test_system_prompt_requires_self_contained_named_questions():
    # Regression guard for HITL findings: vague questions ("מאיזה תאריך מחושב המדד?")
    # and unclear referents ("מה קורה אם לא מחזירים את הציוד?").
    assert "עצמאית" in SYSTEM_PROMPT
    assert "מקור:" in SYSTEM_PROMPT
    assert "הציוד" in SYSTEM_PROMPT  # explicit-referent rule names the bad pattern


def test_make_records_qa_prefix():
    pairs = [{"question": "q?", "answer": "a", "quote": ""}]
    recs = make_records(CHUNK, pairs, "m", qa_prefix="v2-")
    assert recs[0]["qa_id"] == "v2-doc_0_s0"
    assert recs[0]["chunk_id"] == "doc_0"  # prefix only touches qa_id


def _seed_chunks_db(db_path: Path) -> None:
    import sqlite3

    from backend.core.database import init_db

    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        for i in range(3):
            conn.execute(
                "INSERT INTO chunks (chunk_id, file_hash, file_name, page_number, section_number, text)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (f"c_{i}", "h", "f.pdf", i + 1, "", f"text {i}"),
            )
        conn.commit()
    finally:
        conn.close()


def test_fetch_chunks_by_ids(tmp_path):
    db = tmp_path / "t.db"
    _seed_chunks_db(db)
    rows = fetch_chunks(db, limit=50, offset=0, file_hash=None, chunk_ids=["c_2", "c_0"])
    assert [r["chunk_id"] for r in rows] == ["c_0", "c_2"]  # rowid order, not request order
    assert all(set(r) == {"chunk_id", "file_hash", "file_name", "page_number", "section_number", "text"} for r in rows)


def test_fetch_chunks_default_path_unchanged(tmp_path):
    db = tmp_path / "t.db"
    _seed_chunks_db(db)
    rows = fetch_chunks(db, limit=2, offset=1, file_hash=None)
    assert [r["chunk_id"] for r in rows] == ["c_1", "c_2"]


# ---- self-containment check ----

def test_doc_name_keys_full_and_date_stripped_base():
    keys = doc_name_keys("נוהל דמי עזיבה מעודכן למרץ 2026.pdf")
    assert len(keys) == 2
    assert any("2026" in k for k in keys)  # full stem kept
    assert keys[1] == "נוהל דמי עזיבה"  # date tail stripped


def test_doc_name_keys_no_tail_single_key():
    # keys are normalized (punctuation dropped, one prefix letter stripped per word)
    assert doc_name_keys("נוהל הזמנה ושימוש באולם קולנוע.pdf") == ["נוהל זמנה שימוש אולם קולנוע"]
    assert doc_name_keys("נוהל הזמנת ציוד תרבות.pd.pdf") == ["נוהל זמנת ציוד תרבות"]  # broken ext tolerated


def test_is_self_contained_accepts_name_variants():
    keys = doc_name_keys("נוהל בקרת אשראי לצעירים בהתנתקות.pdf")
    assert is_self_contained("על פי נוהל בקרת אשראי – צעירים בהתנתקות (עצמאות כלכלית), מהי יתרה גרעונית?", keys)
    assert is_self_contained("לפי 'נוהל בקרת אשראי לצעירים בהתנתקות', מה עושים בגרעון?", keys)
    keys = doc_name_keys("נוהל דמי עזיבה מעודכן למרץ 2026.pdf")
    assert is_self_contained("לפי נוהל דמי עזיבה, מי חותם על המסמך הסופי?", keys)  # tail-stripped base


def test_is_self_contained_rejects_vague_hitl_archetypes():
    keys = doc_name_keys("נוהל דמי עזיבה מעודכן למרץ 2026.pdf")
    assert not is_self_contained("מאיזה תאריך מחושב המדד?", keys)  # finding 1: unnamed regulation
    assert not is_self_contained("מה קורה אם לא מחזירים את הציוד?", keys)  # finding 3: vague referent
    keys = doc_name_keys("נוהל הזמנה ושימוש באולם קולנוע.pdf")
    assert not is_self_contained("איך יש להזמין את אולם הקולנוע של הקהילה?", keys)  # finding 2: no doc name


def test_select_self_contained_splits():
    chunk = dict(CHUNK, file_name="נוהל דמי עזיבה מעודכן למרץ 2026.pdf")
    pairs = [
        {"question": "לפי נוהל דמי עזיבה, מי חותם?", "answer": "a", "quote": ""},
        {"question": "מאיזה תאריך מחושב המדד?", "answer": "b", "quote": ""},
    ]
    kept, dropped = select_self_contained(pairs, chunk)
    assert [p["answer"] for p in kept] == ["a"]
    assert [p["answer"] for p in dropped] == ["b"]


def _pair_with(question: str) -> str:
    return json.dumps([{"question": question, "answer": "a", "quote": ""}], ensure_ascii=False)


class _ScriptedClient:
    """Fake Ollama client replaying canned responses; records prompts."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.prompts: list[str] = []

    def generate(self, prompt, *, temperature=0.0, timeout_s=None, **kw):
        self.prompts.append(prompt)
        text = self.texts[min(len(self.prompts) - 1, len(self.texts) - 1)]
        return SimpleNamespace(text=text, model="m", eval_count=None, duration_s=0.0)


def test_generate_for_chunk_retries_vague_then_keeps_good():
    client = _ScriptedClient([_pair_with("מאיזה תאריך מחושב המדד?"), _pair_with("לפי נוהל דמי עזיבה, מי חותם?")])
    chunk = dict(CHUNK, file_name="נוהל דמי עזיבה מעודכן למרץ 2026.pdf")
    out = generate_for_chunk(chunk, client, 1, 0.7, 10.0, max_attempts=3)
    assert [p["question"] for p in out] == ["לפי נוהל דמי עזיבה, מי חותם?"]
    assert len(client.prompts) == 2
    assert "תיפסל" in client.prompts[1]  # retry carries the reinforced instruction


def test_generate_for_chunk_returns_best_when_exhausted():
    client = _ScriptedClient([_pair_with("מאיזה תאריך מחושב המדד?")])
    chunk = dict(CHUNK, file_name="נוהל דמי עזיבה מעודכן למרץ 2026.pdf")
    assert generate_for_chunk(chunk, client, 1, 0.7, 10.0, max_attempts=2) == []
    assert len(client.prompts) == 2


def test_generate_for_chunk_no_selfcheck_passthrough():
    client = _ScriptedClient([_pair_with("מאיזה תאריך מחושב המדד?")])
    out = generate_for_chunk(CHUNK, client, 1, 0.7, 10.0, enforce_self_contained=False)
    assert [p["question"] for p in out] == ["מאיזה תאריך מחושב המדד?"]
    assert len(client.prompts) == 1


# ---- JSON extraction hardening (v3 failure shapes) ----

def test_extract_recovers_first_array_despite_trailing_block():
    raw = (
        '[{"question": "a?", "answer": "x", "quote": ""}]'
        ' הסבר נוסף '
        '[{"question": "b?", "answer": "y", "quote": ""}]'
    )
    assert _extract_json_array(raw) == [{"question": "a?", "answer": "x", "quote": ""}]


def test_extract_ignores_leading_and_trailing_prose():
    raw = 'הנה התשובה:\n[{"question": "a?", "answer": "x", "quote": ""}]\nסוף.'
    assert _extract_json_array(raw) == [{"question": "a?", "answer": "x", "quote": ""}]


def test_extract_repairs_stray_backslash():
    raw = '[{"question": "a\\qb?", "answer": "x", "quote": ""}]'  # \q is not a valid escape
    assert _extract_json_array(raw) == [{"question": "a\\qb?", "answer": "x", "quote": ""}]


def test_extract_still_rejects_garbage_and_non_list():
    import pytest

    with pytest.raises(ValueError):
        _extract_json_array("no json here at all")
    with pytest.raises(ValueError):
        _extract_json_array('{"question": "q"}')


# ---- printed-title keys (filename != internal title) ----

def test_title_candidate_first_line_guards():
    assert title_candidate("נוהל מסירת דירת קבע לדייר - קיבוץ הקהילה\nהמשך") == "נוהל מסירת דירת קבע לדייר - קיבוץ הקהילה"
    assert title_candidate("\n\nעמוד 1\nטקסט") == ""  # page marker rejected
    assert title_candidate("כותרת") == ""  # single word rejected
    assert title_candidate("") == ""


def test_chunk_keys_accepts_printed_title():
    chunk = dict(CHUNK, file_name="נוהל מסירת דירה לחבר.pdf")
    title = "נוהל מסירת דירת קבע לדייר - קיבוץ הקהילה"
    pairs = [{"question": "על פי נוהל מסירת דירת קבע לדייר - קיבוץ הקהילה, כיצד מתבצע תיאום מועד המסירה?", "answer": "a", "quote": ""}]
    kept, dropped = select_self_contained(pairs, chunk)
    assert dropped  # filename-only check rejects the internal title
    kept, dropped = select_self_contained(pairs, chunk, doc_title=title)
    assert len(kept) == 1 and not dropped


def test_load_doc_titles_from_first_chunk(tmp_path):
    import sqlite3

    from backend.core.database import init_db

    db = tmp_path / "t.db"
    init_db(db)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO chunks (chunk_id, file_hash, file_name, page_number, section_number, text)"
            " VALUES ('c_0', 'h', 'f.pdf', 1, '', 'נוהל בדיקה פנימי\nגוף')",
        )
        conn.execute(
            "INSERT INTO chunks (chunk_id, file_hash, file_name, page_number, section_number, text)"
            " VALUES ('c_1', 'h', 'f.pdf', 2, '', 'המשך טקסט')",
        )
        conn.commit()
    finally:
        conn.close()
    assert load_doc_titles(db, ["h"]) == {"h": "נוהל בדיקה פנימי"}


def test_load_existing_qa_ids(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(
        '{"qa_id": "v3-doc_0_s0", "chunk_id": "doc_0"}\n'
        '{"qa_id": "v3-doc_0_s1", "chunk_id": "doc_0"}\n'
        "not json\n",
        encoding="utf-8",
    )
    assert load_existing_qa_ids(p) == {"v3-doc_0_s0", "v3-doc_0_s1"}
    assert load_existing_qa_ids(tmp_path / "missing.jsonl") == set()


def test_drop_recorded_filters_rerun_dupes():
    chunk = dict(CHUNK)
    pairs = [
        {"question": "שאלה א?", "answer": "תשובה", "quote": "ההורים"},
        {"question": "שאלה ב?", "answer": "תשובה", "quote": "ההורים"},
    ]
    records = make_records(chunk, pairs, model="m", qa_prefix="v3-")
    assert [r["qa_id"] for r in records] == ["v3-doc_0_s0", "v3-doc_0_s1"]
    # rerun of the same chunk: s0 already recorded, only s1 is fresh
    fresh, dup = drop_recorded(records, {"v3-doc_0_s0"})
    assert [r["qa_id"] for r in fresh] == ["v3-doc_0_s1"]
    assert dup == 1
    # full rerun: nothing fresh
    fresh, dup = drop_recorded(records, {"v3-doc_0_s0", "v3-doc_0_s1"})
    assert fresh == [] and dup == 2
