#!/usr/bin/env python3
"""Offline synthetic Q&A generator — pilot for the HITL benchmark pipeline.

Reads chunk rows from SQLite (``chunks`` table), calls the local Ollama LLM
once per chunk to produce 3-5 diverse Hebrew question/answer pairs, and
appends them as JSONL for human review. Citations are built
deterministically from chunk metadata (never trusted from the model) in the
same ``[doc | עמוד N | סעיף S]`` / page-level shape the answer pipeline
verifies (``backend.services.answer_generator``).

Usage:
    python -m scripts.generate_synthetic_qa --limit 50
    python -m scripts.generate_synthetic_qa --limit 50 --offset 0 --output artifacts/synthetic_qa_pilot.jsonl
    python -m scripts.generate_synthetic_qa --file-hash <sha256> --pairs 4 --dry-run
    python -m scripts.generate_synthetic_qa --chunk-ids <id1,id2> --pairs 3 --qa-prefix v2- --output artifacts/synthetic_qa_v2.jsonl

Resume: chunks already present in the output JSONL (by chunk_id) are
skipped, so a killed run can be re-invoked with identical flags.
Air-gapped: only talks to the configured Ollama endpoint; failures are
logged and counted, never silently swallowed.

Quality: every question must pass a programmatic self-containment check —
the normalized question text must explicitly contain the document name
(full stem or date-tail-stripped base). Failing questions are dropped
(logged with text) and the chunk is retried with a reinforced prompt, up
to --max-attempts; the best attempt's kept pairs are stored.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger("synthetic_qa")

SYSTEM_PROMPT = (
    "אתה מחולל שאלות ותשובות בעברית על בסיס הטקסט המצורף בלבד. "
    "חל איסור מוחלט להשתמש בידע כללי. "
    "צור {n} זוגות שאלה-תשובה מגוונים (עובדתי / פרוצדורלי / מותנה). "
    "כללים מחייבים לכל שאלה: "
    "(1) השאלה עצמאית — מובנת גם למי שלא רואה את הטקסט; "
    "לכלול בה במפורש את שם המסמך/הנוהל — כפי שהוא מופיע בשורת 'מקור:' או בכותרת המסמך עצמו. "
    "(2) עברית תקינה ורהוטה, ניסוח מלא וברור. "
    "(3) בלי כינויי רמז עמומים כגון 'הציוד', 'המדד', 'הוועדה', 'הנוהל' — "
    "תמיד לציין את השם המפורש מהטקסט (איזה ציוד, איזה מדד, איזו ועדה). "
    "לכל זוג: שאלה טבעית של משתמש, תשובה קצרה ומדויקת הנובעת מהטקסט, "
    "וציטוט מילולי קצר (עד 200 תווים) מתוך הטקסט התומך בתשובה. "
    'החזר מערך JSON בלבד בפורמט: [{{"question": "...", "answer": "...", "quote": "..."}}] '
    "ללא טקסט נוסף, ללא markdown."
)

QTYPES = ("factual", "procedural", "conditional")

# --- self-containment check: a usable standalone question must name its document ---
_PUNCT_RE = re.compile(r"[\"'“”‘’()\[\]{},.:;!?–—\-/\\|‚„«»]+")
_EXT_RE = re.compile(r"(\.(pdf|pd|docx?|txt))+\s*$", re.IGNORECASE)
_DATE_TAIL_RE = re.compile(r"\s+(?:מעודכן\s+ל)?(?:[א-ת]+\s+)?\d{4}\s*$")
_HEB_PREFIXES = "בכהלוומ"  # one-letter prefixes stripped for matching (no ש: שימוש != ימוש)

RETRY_SUFFIX = (
    "שים לב: כל שאלה חייבת לכלול במפורש את שם המסמך כפי שהוא מופיע בשורת 'מקור:'. "
    "שאלה ללא שם המסמך תיפסל."
)


def _strip_prefix(word: str) -> str:
    if len(word) > 2 and word[0] in _HEB_PREFIXES:
        return word[1:]
    return word


def normalize_hebrew(text: str) -> str:
    """Lowercase-free normalization for name matching: drop punctuation,
    strip one Hebrew prefix letter per word, collapse whitespace."""
    no_punct = _PUNCT_RE.sub(" ", text)
    return " ".join(_strip_prefix(w) for w in no_punct.split())


def doc_name_keys(file_name: str) -> list[str]:
    """Normalized accepted document-name forms: full stem plus, when it ends
    in a date/version tail, the tail-stripped base (min 2 words)."""
    stem = _EXT_RE.sub("", file_name).strip()
    full = normalize_hebrew(stem)
    keys = [full] if full else []
    base = normalize_hebrew(_DATE_TAIL_RE.sub("", stem).strip())
    if base and base != full and len(base.split()) >= 2:
        keys.append(base)
    return keys


def is_self_contained(question: str, keys: list[str]) -> bool:
    """True when the normalized question names the document explicitly."""
    norm_q = normalize_hebrew(question)
    return bool(norm_q) and any(k in norm_q for k in keys)


def title_candidate(text: str) -> str:
    """The document's printed title: first non-empty line of its first chunk.

    Guarded (2-15 words, ≥2 Hebrew words, not a page marker) so headers like
    'עמוד 1' never become accepted keys. Returns '' when unusable.
    """
    for line in text.splitlines():
        s = _norm(line)
        if not s:
            continue
        words = s.split()
        heb_words = sum(1 for w in words if re.search(r"[א-ת]", w))
        if 2 <= len(words) <= 15 and heb_words >= 2 and not re.fullmatch(r"[עמוד\s\d.\-|–—()]+", s):
            return s
        return ""
    return ""


def chunk_keys(chunk: dict, doc_title: str = "") -> list[str]:
    """Accepted name forms: filename variants plus the printed title (if any)."""
    keys = doc_name_keys(chunk.get("file_name", ""))
    if doc_title:
        norm_title = normalize_hebrew(doc_title)
        if norm_title and norm_title not in keys:
            keys.append(norm_title)
    return keys


def select_self_contained(pairs: list[dict], chunk: dict, doc_title: str = "") -> tuple[list[dict], list[dict]]:
    """Split parsed pairs into (kept, dropped) by the self-containment check."""
    keys = chunk_keys(chunk, doc_title)
    kept = [p for p in pairs if is_self_contained(p["question"], keys)]
    dropped = [p for p in pairs if not is_self_contained(p["question"], keys)]
    return kept, dropped


def load_doc_titles(db_path, file_hashes: list[str]) -> dict[str, str]:
    """Map file_hash → printed title from each document's first chunk (rowid)."""
    from backend.core.database import get_connection, init_db

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        titles = {}
        for fh in file_hashes:
            row = conn.execute(
                "SELECT text FROM chunks WHERE file_hash=? ORDER BY rowid LIMIT 1", (fh,)
            ).fetchone()
            if row:
                titles[fh] = title_candidate(row[0])
        return titles
    finally:
        conn.close()


def build_citation(chunk: dict) -> str:
    """Deterministic citation from chunk metadata (mirrors answer_generator)."""
    section = (chunk.get("section_number") or "").strip()
    if section:
        return f"[{chunk.get('file_name', '')} | עמוד {chunk.get('page_number', 1)} | סעיף {section}]"
    return f"[{chunk.get('file_name', '')} | עמוד {chunk.get('page_number', 1)}]"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_json_array(raw: str) -> list[dict]:
    """Pull the first JSON array out of raw LLM output (fences tolerated).

    Only the first JSON value is parsed, so trailing prose or extra blocks
    ("Extra data") no longer reject an otherwise valid array. A single
    repair pass escapes stray backslashes ("Invalid \\escape") before
    giving up.
    """
    cleaned = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("[")
        if start != -1:
            cleaned = cleaned[start:]
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        repaired = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", cleaned)
        try:
            data, _ = decoder.raw_decode(repaired)
        except json.JSONDecodeError:
            raise ValueError(f"LLM output is not a JSON array: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"LLM output is not a list (got {type(data).__name__})")
    return [d for d in data if isinstance(d, dict)]


def parse_pairs(raw: str, max_pairs: int) -> list[dict]:
    """Parse + validate LLM output into [{question, answer, quote}]."""
    out = []
    for item in _extract_json_array(raw)[:max_pairs]:
        q = _norm(str(item.get("question", "")))
        a = _norm(str(item.get("answer", "")))
        quote = _norm(str(item.get("quote", "")))
        if not q or not a:
            continue
        out.append({"question": q, "answer": a, "quote": quote[:500]})
    # de-dup on normalized question
    seen: set[str] = set()
    deduped = []
    for p in out:
        if p["question"] in seen:
            continue
        seen.add(p["question"])
        deduped.append(p)
    return deduped


def quote_verified(quote: str, chunk_text: str) -> bool:
    """True when a 30-char anchor of the quote appears verbatim in the chunk."""
    if not quote:
        return False
    anchor = _norm(quote)[:30]
    return bool(anchor) and anchor in _norm(chunk_text)


def make_records(chunk: dict, pairs: list[dict], model: str, qa_prefix: str = "") -> list[dict]:
    now = datetime.now(UTC).isoformat()
    records = []
    for i, p in enumerate(pairs):
        qtype = QTYPES[i % len(QTYPES)]
        records.append(
            {
                "qa_id": f"{qa_prefix}{chunk['chunk_id']}_s{i}",
                "chunk_id": chunk["chunk_id"],
                "file_hash": chunk["file_hash"],
                "file_name": chunk["file_name"],
                "page_number": chunk["page_number"],
                "section_number": chunk.get("section_number", ""),
                "question": p["question"],
                "answer": p["answer"],
                "quote": p["quote"],
                "citation": build_citation(chunk),
                "qtype": qtype,
                "quote_verified": quote_verified(p["quote"], chunk["text"]),
                "status": "pending",
                "reviewer": "",
                "edited_answer": "",
                "model": model,
                "generated_at": now,
            }
        )
    return records


def load_done_chunk_ids(output: Path) -> set[str]:
    done: set[str] = set()
    if not output.exists():
        return done
    with open(output, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError) as exc:
                LOG.warning("skipping unreadable JSONL line in %s (%s)", output, exc)
    return done


def load_existing_qa_ids(output: Path) -> set[str]:
    """qa_ids already recorded in the output JSONL (pair-level resume state)."""
    seen: set[str] = set()
    if not output.exists():
        return seen
    with open(output, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["qa_id"])
            except (json.JSONDecodeError, KeyError) as exc:
                LOG.warning("skipping unreadable JSONL line in %s (%s)", output, exc)
    return seen


def drop_recorded(records: list[dict], existing_qa: set[str]) -> tuple[list[dict], int]:
    """Split off records whose qa_id is already in the output file.

    Regenerating a partially-covered chunk (explicit --chunk-ids rerun)
    must not append duplicate qa_ids — the importer rejects them.
    """
    fresh = [r for r in records if r["qa_id"] not in existing_qa]
    return fresh, len(records) - len(fresh)


def fetch_chunks(
    db_path,
    *,
    limit: int,
    offset: int,
    file_hash: str | None,
    chunk_ids: list[str] | None = None,
) -> list[dict]:
    from backend.core.database import get_connection, init_db

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            cur = conn.execute(
                "SELECT chunk_id, file_hash, file_name, page_number, section_number, text"
                f" FROM chunks WHERE chunk_id IN ({placeholders}) ORDER BY rowid",
                tuple(chunk_ids),
            )
        elif file_hash:
            cur = conn.execute(
                "SELECT chunk_id, file_hash, file_name, page_number, section_number, text"
                " FROM chunks WHERE file_hash=? ORDER BY rowid LIMIT ? OFFSET ?",
                (file_hash, limit, offset),
            )
        else:
            cur = conn.execute(
                "SELECT chunk_id, file_hash, file_name, page_number, section_number, text"
                " FROM chunks ORDER BY rowid LIMIT ? OFFSET ?",
                (limit, offset),
            )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def generate_for_chunk(
    chunk: dict,
    client,
    pairs: int,
    temperature: float,
    timeout_s: float,
    max_attempts: int = 3,
    enforce_self_contained: bool = True,
    doc_title: str = "",
) -> list[dict]:
    """Generate pairs for one chunk, retrying when questions fail the
    self-containment check. Returns the best attempt's kept pairs (possibly
    fewer than requested); drops are logged, never silently swallowed."""
    text = chunk["text"][:2000]
    base_prompt = (
        f"{SYSTEM_PROMPT.format(n=pairs)}\n\n"
        f"מקור: {chunk['file_name']} עמוד {chunk['page_number']}\n"
        f"טקסט:\n{text}"
    )
    best_kept: list[dict] = []
    attempts = max(1, max_attempts) if enforce_self_contained else 1
    for attempt in range(1, attempts + 1):
        prompt = base_prompt if attempt == 1 else f"{base_prompt}\n\n{RETRY_SUFFIX}"
        res = client.generate(prompt, temperature=temperature, timeout_s=timeout_s, max_retries=0)
        parsed = parse_pairs(res.text, pairs)
        if not enforce_self_contained:
            return parsed
        kept, dropped = select_self_contained(parsed, chunk, doc_title)
        for d in dropped:
            LOG.warning("chunk %s: dropping non-self-contained question: %s", chunk["chunk_id"], d["question"][:120])
        if len(kept) >= len(best_kept):
            best_kept = kept
        if len(kept) >= pairs:
            return kept[:pairs]
        LOG.info(
            "chunk %s: attempt %d/%d kept %d/%d self-contained — retrying",
            chunk["chunk_id"], attempt, attempts, len(kept), len(parsed),
        )
    return best_kept[:pairs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline synthetic Q&A pilot generator (JSONL for HITL review)")
    parser.add_argument("--limit", type=int, default=50, help="max chunks to process")
    parser.add_argument("--offset", type=int, default=0, help="chunk offset (ORDER BY rowid)")
    parser.add_argument("--output", default="artifacts/synthetic_qa_pilot.jsonl", help="output JSONL path")
    parser.add_argument("--pairs", type=int, default=4, help="pairs requested per chunk (3-5)")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-chunk LLM timeout seconds")
    parser.add_argument("--max-attempts", type=int, default=3, help="retries per chunk for the self-containment check")
    parser.add_argument("--no-selfcheck", action="store_true", help="disable the self-containment filter (debug only)")
    parser.add_argument("--file-hash", default=None, help="restrict to one document hash")
    parser.add_argument(
        "--chunk-ids",
        default=None,
        help="comma-separated chunk_ids to (re)generate (ignores --limit/--offset/--file-hash)",
    )
    parser.add_argument(
        "--qa-prefix",
        default="",
        help="prefix for generated qa_id values (keeps reruns distinct, e.g. 'v2-')",
    )
    parser.add_argument("--db-path", default=None, help="SQLite path (default: resolved from config)")
    parser.add_argument("--dry-run", action="store_true", help="list target chunks without calling the LLM")
    parser.add_argument("--no-resume", action="store_true", help="reprocess chunks already in the output file")
    args = parser.parse_args()

    if not 1 <= args.pairs <= 5:
        parser.error("--pairs must be 1-5")
    if args.limit <= 0 or args.offset < 0:
        parser.error("--limit must be > 0 and --offset >= 0")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be > 0")

    from backend.core.config import get_config

    cfg = get_config()
    db_path = Path(args.db_path) if args.db_path else None
    chunk_ids = [c.strip() for c in args.chunk_ids.split(",") if c.strip()] if args.chunk_ids else None
    chunks = fetch_chunks(db_path, limit=args.limit, offset=args.offset, file_hash=args.file_hash, chunk_ids=chunk_ids)
    if not chunks:
        LOG.info("no chunks selected — nothing to do")
        return 0

    output = Path(args.output)
    if args.dry_run:
        for c in chunks:
            LOG.info("[dry-run] %s | %s p%d | %d chars", c["chunk_id"], c["file_name"], c["page_number"], len(c["text"]))
        LOG.info("[dry-run] %d chunks would be processed → %s", len(chunks), output)
        return 0

    done = set() if args.no_resume else load_done_chunk_ids(output)
    todo = [c for c in chunks if c["chunk_id"] not in done]
    LOG.info("%d chunks selected, %d already done, %d to process → %s", len(chunks), len(chunks) - len(todo), len(todo), output)
    if not todo:
        return 0
    existing_qa = set() if args.no_resume else load_existing_qa_ids(output)

    from backend.services.ollama_client import get_ollama_client

    try:
        client = get_ollama_client()
    except Exception:
        LOG.error("ollama client unavailable — aborting (no silent fallback)", exc_info=True)
        return 1
    model = cfg.models.llm_name
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = sorted({c["file_hash"] for c in todo})
    titles = {} if args.no_selfcheck else load_doc_titles(db_path, hashes)
    LOG.info("loaded %d doc titles for self-containment check", len(titles))

    stats = {"pairs": 0, "verified": 0, "failed": 0, "empty": 0, "short": 0, "dup": 0}
    start = time.time()
    with open(output, "a", encoding="utf-8") as f:
        for idx, chunk in enumerate(todo, 1):
            try:
                pairs = generate_for_chunk(
                    chunk, client, args.pairs, args.temperature, args.timeout,
                    max_attempts=args.max_attempts,
                    enforce_self_contained=not args.no_selfcheck,
                    doc_title=titles.get(chunk["file_hash"], ""),
                )
            except Exception:
                stats["failed"] += 1
                LOG.error("chunk %s (%s p%d) failed — skipped", chunk["chunk_id"], chunk["file_name"], chunk["page_number"], exc_info=True)
                continue
            if not pairs:
                stats["empty"] += 1
                LOG.warning("chunk %s returned no valid pairs — skipped", chunk["chunk_id"])
                continue
            stats["short"] += args.pairs - len(pairs)
            records = make_records(chunk, pairs, model, qa_prefix=args.qa_prefix)
            records, dup = drop_recorded(records, existing_qa)
            if dup:
                stats["dup"] += dup
                LOG.info("chunk %s: skipping %d already-recorded qa_ids", chunk["chunk_id"], dup)
            if not records:
                stats["empty"] += 1
                LOG.warning("chunk %s added no new pairs — skipped", chunk["chunk_id"])
                continue
            existing_qa.update(r["qa_id"] for r in records)
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            stats["pairs"] += len(records)
            stats["verified"] += sum(1 for r in records if r["quote_verified"])
            elapsed = time.time() - start
            eta = int(elapsed / idx * (len(todo) - idx)) if idx else 0
            LOG.info(
                "[%d/%d] %s p%d → %d pairs (%d verified) ETA %ds",
                idx, len(todo), chunk["file_name"][:40], chunk["page_number"],
                len(records), sum(1 for r in records if r["quote_verified"]), eta,
            )
    elapsed = time.time() - start
    LOG.info(
        "done: %d pairs from %d chunks (%d failed, %d empty, %d short of requested, %d dup skipped), %d quote-verified, %.0fs → %s",
        stats["pairs"], len(todo), stats["failed"], stats["empty"], stats["short"], stats["dup"], stats["verified"], elapsed, output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
