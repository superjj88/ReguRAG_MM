"""Answer generation & guardrails.

Design: removed every fabricated/canned fallback answer and the
golden-corpus chunk-selection heuristics. The pipeline is now honest:
* LLM failure → raises (caller converts to HTTP 503);
* answer without a *verified* citation → exact refusal (zero-hallucination
  guardrail).

2026-09-16 hardening (page-level fallback): chunks without a detected
section number used to render a malformed ``[doc | עמוד 1 | סעיף ]`` template;
the model imitated it verbatim and the answer was refused although the corpus
contained the rule. Now the context block omits the סעיף part, page-level
citations are accepted, and every citation is verified against the retrieved
chunks (doc + page must match) before the answer is returned — an unverifiable
answer is retried once, then explicitly refused.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from backend.core.config import get_config

LOG = logging.getLogger(__name__)

REFUSAL = "המידע אינו קיים בתקנונים שבמערכת."

# Citation may carry a section ([doc | עמוד N | סעיף S]) or be page-level
# ([doc | עמוד N]) when the chunk had no detected section number.
CITATION_RE = re.compile(r"\[[^|\[\]]+?\|\s*עמוד\s*\d+(?:\s*\|\s*סעיף\s*[^|\[\]]+?)?\]")
TOLERANT_CITATION_RE = re.compile(
    r"\[\s*([^|\[\],]+?)\s*[|,]\s*(?:עמוד|עמ')\s*(\d+)(?:\s*[|,]\s*(?:סעיף|סע')\s*([^|\[\]]+?))?\s*\]"
)


def _extract_quoted_snippets(text: str) -> list[str]:
    """Extract quoted text inside Hebrew or standard quotation marks."""
    quotes = re.findall(r'["״]([^"״\n]{8,})["״]', text)
    return [q.strip() for q in quotes if q.strip()]


def _verify_groundedness(answer: str, chunks: list[dict]) -> tuple[bool, str]:
    """Verify that factual claims, quotes, and numbers in the answer are grounded in context."""
    if not chunks:
        return False, "no chunks available"

    combined_context = " ".join(c.get("text", "") for c in chunks).lower()

    # 1. Check direct quotes if present
    quotes = _extract_quoted_snippets(answer)
    for q in quotes:
        q_clean = " ".join(q.split()).lower()
        if len(q_clean) >= 15 and q_clean not in combined_context:
            first_15 = q_clean[:15]
            if first_15 not in combined_context:
                LOG.info("grounding check: quote %r not found in context", q[:40])
                return False, f"quote not grounded: {q[:30]}"

    # 2. Check substantive numbers not present in context or citation
    # Strip bracketed citations first so file names, dates, page numbers are not treated as factual claims
    body_no_citations = re.sub(r"\[[^\]]+\]", " ", answer)
    factual_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", body_no_citations))
    if factual_numbers:
        ctx_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", combined_context))
        hallucinated = [
            n for n in factual_numbers
            if n not in ctx_numbers and int(float(n.replace(',', '').split('.')[0])) > 10
        ]
        if hallucinated:
            LOG.info("grounding check: numbers %r not found in retrieved chunks", hallucinated)
            return False, f"hallucinated numbers: {hallucinated}"

    return True, ""


# The model often says "the information is absent" in its own words instead of
# returning the exact REFUSAL string. Such an answer must be treated as a
# refusal even when it appends tangential citations (observed: "לא נמצא
# בתקנונים התייחסות ספציפית..." followed by unrelated source links).
NO_INFO_RE = re.compile(
    r"(?:אין|לא)\s+(?:כל\s+)?(?:התייחסות|מידע|אזכור)"
    r"|(?:לא|אינו|אינה)\s+(?:נמצא|נמצאה|קיים|קיימת|מופיע|מופיעה|מפורט|מפורטת|מוזכר|מוזכרת)"
)

# ...but only when the absence is asserted about the corpus itself. A model
# complaining that something "אינו מופיע בקונטקסט" is talking about the prompt
# window, not about the regulations, and must not be turned into a refusal.
CORPUS_RE = re.compile(r"תקנון|תקנונים|מסמך|מסמכים|במערכת|בנהלים|באוגדן|בחוברת")

REFUSAL_HEAD_CHARS = 200
CORPUS_WINDOW_BEFORE = 30
CORPUS_WINDOW_AFTER = 40

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "qa_system.he.txt"

DEFAULT_SYSTEM_PROMPT = "אתה עוזר משפטי. ענה רק על בסיס הקונטקסט."


def load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        LOG.warning("system prompt file missing/unreadable: %s — using default", PROMPT_PATH)
        return DEFAULT_SYSTEM_PROMPT


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        section = c.get("section_number", "")
        if section:
            meta = f"[{c.get('file_name','')} | עמוד {c.get('page_number','')} | סעיף {section}]"
        else:
            # page-level fallback citation — never render an empty סעיף field
            meta = f"[{c.get('file_name','')} | עמוד {c.get('page_number','')}]"
        parts.append(f"{meta}\n{c.get('text','')}")
    return "\n\n".join(parts)


def _norm_doc_name(name: str) -> str:
    return " ".join(str(name).replace(".pdf", "").split()).strip().lower()


def _verified_citations(answer: str, chunks: list[dict]) -> list[str]:
    """Return only citations whose (document, page) matches a retrieved chunk."""
    known = {
        (_norm_doc_name(c.get("file_name", "")), c.get("page_number"))
        for c in chunks
    }
    out = []
    for cit in CITATION_RE.findall(answer):
        body = cit.strip("[]")
        m = re.match(r"(.+?)\|\s*עמוד\s*(\d+)", body)
        if not m:
            continue
        doc, page = m.group(1), int(m.group(2))
        if (_norm_doc_name(doc), page) in known:
            out.append(cit)
    if out:
        return out

    # Tolerant fallback: normalize valid citations to canonical form
    for m in TOLERANT_CITATION_RE.finditer(answer):
        doc_raw, page_str, sec_raw = m.group(1), m.group(2), m.group(3)
        try:
            page = int(page_str)
        except ValueError:
            continue
        norm_doc = _norm_doc_name(doc_raw)
        matching_chunk = next(
            (c for c in chunks if _norm_doc_name(c.get("file_name", "")) == norm_doc and c.get("page_number") == page),
            None,
        )
        if matching_chunk:
            actual_fn = matching_chunk.get("file_name", doc_raw)
            sec = sec_raw.strip() if sec_raw else matching_chunk.get("section_number", "")
            if sec:
                canonical_cit = f"[{actual_fn} | עמוד {page} | סעיף {sec}]"
            else:
                canonical_cit = f"[{actual_fn} | עמוד {page}]"
            if canonical_cit not in out:
                out.append(canonical_cit)
    return out


def _looks_like_refusal(answer: str) -> bool:
    """True when the answer *states* the information is absent from the corpus.

    Only the opening of the answer is inspected: a real answer may well mention
    "the regulations do not specify X" halfway through and then quote the rule
    that does apply, and that must not be swallowed as a refusal. Each absence
    phrase is also required to sit next to a corpus word ("תקנון", "מסמכים",
    "במערכת", ...) so prompt-window complaints ("אינו מופיע בקונטקסט") are not
    mistaken for a statement about the regulations.
    """
    head = " ".join(answer.strip().split())[:REFUSAL_HEAD_CHARS]
    # document names may themselves contain absence words ("מסמך לא קיים.pdf")
    head = CITATION_RE.sub(" ", head)
    for match in NO_INFO_RE.finditer(head):
        window = head[max(0, match.start() - CORPUS_WINDOW_BEFORE): match.end() + CORPUS_WINDOW_AFTER]
        if CORPUS_RE.search(window):
            return True
    return False


def _split_budget(timeout_s: float | None) -> tuple[float, float]:
    """Split the generation budget across the first attempt and the retry.

    Each attempt runs as a SINGLE LLM call (no internal retries) so the total
    stays within the budget even when the model is slow; retrying is the
    caller-level concern (citation reinforcement), not a transport concern.
    """
    if timeout_s is None:
        timeout_s = float(get_config().performance.chat_timeout_seconds)
    budget = float(timeout_s)
    if budget <= 0:
        raise ValueError(f"generation budget must be positive, got {budget!r}")
    first = max(budget * 0.72, 1.0)
    retry = budget - first
    return first, retry


def generate_answer(query: str, chunks: list[dict], ollama_client=None, timeout_s: float | None = None) -> dict:
    """Generate an answer strictly grounded in the provided chunks.

    Returns {answer, citations, refusal} where citations were verified against
    the retrieved chunks. Raises OllamaError (or ConnectionError) when the LLM
    is unavailable — the caller must surface the failure, never fabricate.
    ``timeout_s`` is the TOTAL budget for generation (both attempts).
    """
    if not chunks:
        return {"answer": REFUSAL, "citations": [], "refusal": True}

    system = load_system_prompt()
    context = _build_context(chunks)
    first_budget, retry_budget = _split_budget(timeout_s)

    if ollama_client is None:
        from backend.services.ollama_client import get_ollama_client

        ollama_client = get_ollama_client()

    full_prompt = f"{system}\n\nקונטקסט:\n{context}\n\nשאלה: {query}"
    res = ollama_client.generate(full_prompt, temperature=0.0, timeout_s=first_budget, max_retries=0)
    answer = res.text.strip()
    if _looks_like_refusal(answer):
        # the model declared the information absent in its own words — no retry,
        # and no citations: tangential sources must not dress up a non-answer.
        LOG.info("model declared the information absent for query=%r; head: %.200s", query[:80], answer)
        return {"answer": REFUSAL, "citations": [], "refusal": True}
    cites = _verified_citations(answer, chunks)
    is_grounded, _ = _verify_groundedness(answer, chunks) if cites else (False, "no citation")
    if cites and is_grounded:
        return {"answer": answer, "citations": cites, "refusal": False}
    if REFUSAL in answer:
        # the model explicitly declared the information absent — no retry
        LOG.info("model refused explicitly for query=%r", query[:80])
        return {"answer": REFUSAL, "citations": [], "refusal": True}
    LOG.info("attempt 1 rejected: citations=%r, grounded=%r; answer head: %.300s", cites, is_grounded, answer)

    if retry_budget <= 5.0:
        LOG.info("answer refused: unverified citation or ungrounded and no retry budget for query=%r", query[:80])
        return {"answer": REFUSAL, "citations": [], "refusal": True}

    # retry once with reinforced citation instruction; a transport failure on
    # the retry degrades to the explicit refusal (the user must see a definitive
    # answer or an explicit 'unknown' — never a bare 503 when we already hold
    # an (unverifiable) draft).
    reinforced = (
        f"{full_prompt}\n\nחובה לכלול מראה מקור בצורה המדויקת הבאה: "
        "[שם המסמך | עמוד X | סעיף Y], וכאשר מספר הסעיף אינו מופיע בקונטקסט — "
        "[שם המסמך | עמוד X] בלבד. הקפד לדייק במספרים ובעובדות בדיוק כפי שהם מופיעים בקטעים."
    )
    try:
        res2 = ollama_client.generate(reinforced, temperature=0.0, timeout_s=retry_budget, max_retries=0)
        answer2 = res2.text.strip()
    except Exception:
        LOG.warning("citation-reinforced retry failed — falling back to explicit refusal", exc_info=True)
        return {"answer": REFUSAL, "citations": [], "refusal": True}
    if _looks_like_refusal(answer2):
        LOG.info("retry declared the information absent for query=%r; head: %.200s", query[:80], answer2)
        return {"answer": REFUSAL, "citations": [], "refusal": True}
    cites2 = _verified_citations(answer2, chunks)
    is_grounded2, _ = _verify_groundedness(answer2, chunks) if cites2 else (False, "no citation")
    if cites2 and is_grounded2:
        return {"answer": answer2, "citations": cites2, "refusal": False}

    LOG.info("attempt 2 rejected: citations=%r, grounded=%r; answer head: %.300s", cites2, is_grounded2, answer2)
    LOG.info("answer refused: no verified citation or ungrounded for query=%r (%d chunks)", query[:80], len(chunks))
    return {"answer": REFUSAL, "citations": [], "refusal": True}
