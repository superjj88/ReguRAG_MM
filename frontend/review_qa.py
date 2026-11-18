"""HITL review dashboard for synthetic Q&A pairs (Streamlit, Hebrew RTL).

Reads the generator's base JSONL plus an optional sidecar of review
decisions, lets a reviewer Approve / Reject / Edit each pair with full
source context, and exports the approved subset for ingestion.

Run:  streamlit run frontend/review_qa.py
File: set REVIEW_QA_FILE env or use the sidebar picker (default:
      artifacts/synthetic_qa_pilot.jsonl).

Storage model (deliberate):
* base JSONL is append-only output of scripts/generate_synthetic_qa.py —
  never rewritten here (a generation run may be in flight).
* decisions live in a sidecar ``<stem>.reviews.jsonl`` mapping
  qa_id -> {status, edited_answer, reviewer, reviewed_at}.
* export merges base + sidecar and writes approved.jsonl
  (status in {approved, edited}).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_FILE = Path("artifacts/synthetic_qa_pilot.jsonl")
APPROVED_STATUSES = ("approved", "edited")
VALID_STATUSES = ("pending", "approved", "rejected", "edited")


def sidecar_path(base: Path) -> Path:
    return base.with_name(f"{base.stem}.reviews.jsonl")


def export_path(base: Path) -> Path:
    return base.with_name(f"{base.stem}.approved.jsonl")


def load_base(base: Path) -> list[dict]:
    records: list[dict] = []
    if not base.exists():
        return records
    with open(base, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_sidecar(base: Path) -> dict[str, dict]:
    path = sidecar_path(base)
    decisions: dict[str, dict] = {}
    if not path.exists():
        return decisions
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("qa_id") and d.get("status") in VALID_STATUSES:
                    decisions[d["qa_id"]] = d
            except json.JSONDecodeError:
                continue
    return decisions


def merged(records: list[dict], decisions: dict[str, dict]) -> list[dict]:
    out = []
    for r in records:
        row = dict(r)
        d = decisions.get(r.get("qa_id", ""))
        if d:
            row["status"] = d["status"]
            row["edited_answer"] = d.get("edited_answer", "")
            row["reviewer"] = d.get("reviewer", "")
        out.append(row)
    return out


def save_decision(base: Path, qa_id: str, status: str, edited_answer: str, reviewer: str) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}")
    path = sidecar_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "qa_id": qa_id,
        "status": status,
        "edited_answer": edited_answer,
        "reviewer": reviewer,
        "reviewed_at": datetime.now(UTC).isoformat(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_approved(base: Path) -> Path:
    rows = [r for r in merged(load_base(base), load_sidecar(base)) if r.get("status") in APPROVED_STATUSES]
    out = export_path(base)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def chunk_context(chunk_id: str, db_path: Path | None = None) -> str:
    """Full source chunk text for context display (read-only; '' when unavailable)."""
    try:
        if db_path is None:
            from backend.core.database import get_db_path

            db_path = get_db_path()
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT text FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
            return str(row[0]) if row else ""
        finally:
            conn.close()
    except Exception:
        return ""


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="ReguRAG_MM — סקירת Q&A", layout="wide")
    st.markdown(
        """<style>
        html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, .stSidebar {
            direction: rtl; text-align: right;
            font-family: 'Arial', 'Noto Sans Hebrew', sans-serif;
        }</style>""",
        unsafe_allow_html=True,
    )

    default = os.environ.get("REVIEW_QA_FILE", str(DEFAULT_FILE))
    base = Path(st.sidebar.text_input("קובץ JSONL", value=default))
    status_filter = st.sidebar.selectbox("סינון סטטוס", ["pending", "all", "approved", "rejected", "edited"])
    reviewer = st.sidebar.text_input("שם סוקר", value="")
    db_opt = st.sidebar.text_input("DB (ריק = ברירת מחדל)", value="")

    records = merged(load_base(base), load_sidecar(base))
    if status_filter != "all":
        records = [r for r in records if r.get("status") == status_filter]
    total_all = len(load_base(base))
    done = sum(1 for r in merged(load_base(base), load_sidecar(base)) if r.get("status") != "pending")
    st.sidebar.write(f"התקדמות: {done}/{total_all}")
    st.sidebar.progress(done / total_all if total_all else 0.0)
    if st.sidebar.button("ייצוא מאושרים"):
        out = export_approved(base)
        st.sidebar.success(f"יוצא: {out}")

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    st.session_state.idx = max(0, min(st.session_state.idx, max(len(records) - 1, 0)))
    if not records:
        st.info("אין רשומות לתצוגה בסינון הנוכחי.")
        return
    r = records[st.session_state.idx]
    st.title(f"סקירת Q&A ({st.session_state.idx + 1}/{len(records)})")
    st.caption(f"{r.get('file_name', '')} | עמוד {r.get('page_number', '')} | סעיף {r.get('section_number', '') or '—'} | {r.get('qtype', '')}")

    st.subheader("שאלה")
    st.markdown(r.get("question", ""))
    st.subheader("תשובה")
    st.markdown(r.get("answer", ""))
    edited = st.text_area("תשובה ערוכה (אופציונלי)", value=r.get("edited_answer", "") or "")
    with st.expander("הקשר: ציטוט + מראה מקום + טקסט מקור"):
        st.markdown(f"**ציטוט:** {r.get('quote', '') or '—'}")
        st.code(r.get("citation", ""), language=None)
        ctx = chunk_context(r.get("chunk_id", ""), Path(db_opt) if db_opt else None)
        st.markdown(ctx or "_טקסט מקור לא זמין_")
        if not r.get("quote_verified"):
            st.warning("הציטוט לא אומת מול הטקסט — לבדוק ידנית.")

    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("✅ אשר"):
        save_decision(base, r["qa_id"], "approved", "", reviewer)
        st.session_state.idx += 1
        st.rerun()
    if c2.button("✏️ שמור עריכה"):
        if not edited.strip():
            st.error("אין תשובה ערוכה לשמירה.")
        else:
            save_decision(base, r["qa_id"], "edited", edited.strip(), reviewer)
            st.session_state.idx += 1
            st.rerun()
    if c3.button("❌ דחה"):
        save_decision(base, r["qa_id"], "rejected", "", reviewer)
        st.session_state.idx += 1
        st.rerun()
    if c4.button("⏭️ דלג"):
        st.session_state.idx += 1
        st.rerun()
    if c5.button("⏮️ חזרה"):
        st.session_state.idx -= 1
        st.rerun()


if __name__ == "__main__":
    main()
