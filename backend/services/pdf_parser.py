"""Hebrew PDF parsing & chunking."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import pdfplumber

from backend.services.ocr import needs_ocr, ocr_pdf_page, tesseract_available

LOG = logging.getLogger(__name__)

# Hebrew section markers: סעיף 4.2, פרק ב', תקנה 12 etc.
SECTION_RE = re.compile(r"(סעיף|פרק|תקנה)\s+[\d\.\-א-ת\"'׳״]+", re.UNICODE)
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n", re.UNICODE)


def compute_file_hash(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Common Hebrew function/content words used to detect visually-reversed text
# (replaces the earlier corpus-specific keyword heuristic).
_HEBREW_WORDS = (
    "של", "את", "על", "שיכון", "ועדה", "ועדת", "תקנון", "תקנונים", "מרכזי",
    "חבר", "חברי", "הוועדה", "מסמך", "מסמכים", "סעיף", "סעיפים", "פרק",
    "נציג", "נציגי", "ציבור", "תכלול", "איש", "אנשים", "התקנון", "כלל",
    "אין", "יש", "לא", "עם", "בין", "או", "אם", "כל", "רק", "מן", "שיהיה",
    "קיבוץ", "נוהל", "נהלים", "מנהל", "מזכירות", "תאריך", "גוף", "מאשר",
    "גרסה", "אשכול", "בריאות", "חינוך", "חברים", "מסמכים", "התקנון",
)


def _hebrew_word_count(text: str) -> int:
    """Count occurrences of common Hebrew words as whole tokens."""
    count = 0
    for w in _HEBREW_WORDS:
        count += len(re.findall(rf"(?<![^\s\"'()\-—,\.]){re.escape(w)}(?![^\s\"'()\-—,\.])", text))
    return count


def _fix_bidi(text: str) -> str:
    """Fix visually-reversed Hebrew extracted by pdfplumber.

    Some PDFs store Hebrew glyphs in visual (reversed) order. Decide per page
    by counting a lexicon of common Hebrew words in the forward text versus
    its reversal; reverse only when the reversed form scores clearly higher.

    The correction is applied per line (not by reversing the whole page):
    raw extraction already yields lines in the correct order, whole-page
    reversal would scramble line order, and digit runs (dates, phone numbers,
    section numbers) are stored logically and must not be flipped.
    """
    if not text:
        return text
    forward_score = _hebrew_word_count(text)
    reverse_score = _hebrew_word_count(text[::-1])
    if reverse_score > forward_score and reverse_score >= 2:
        return "\n".join(_reverse_hebrew_line(line) for line in text.splitlines())
    return text


def _reverse_hebrew_line(line: str) -> str:
    """Restore logical order of a visually-ordered Hebrew line.

    Token-level transform: reverse token order and char-reverse non-digit
    tokens. Digit runs (dates, phone numbers, numbered clauses) are already
    stored logically, so digit-bearing tokens are kept intact.
    """
    tokens = line.split()
    restored = [
        tok if any(ch.isdigit() for ch in tok) else tok[::-1]
        for tok in reversed(tokens)
    ]
    return " ".join(restored)


# Legacy-Hebrew misdecode: older Hebrew tooling stored glyphs single-byte per
# ISO-8859-8 (א..ת = U+05D0..U+05EA) but ships missing/broken ToUnicode maps,
# so extractors decode byte 0xF0 (nun) as Latin-1 "ð" (U+00F0). Inside Hebrew
# text U+00F0 is always nun; no legitimate Latin word uses it there.
_LATIN1_TO_HEBREW = str.maketrans({"\u00f0": "\u05e0"})


def _fix_misencoded_hebrew(text: str) -> str:
    """Repair single-byte Hebrew leaks (ð→נ) from legacy PDF fonts.

    Applied only when the page contains Hebrew script, so genuine Latin text
    (e.g. accented Spanish pages) is left untouched.
    """
    if not text:
        return text
    if not any("\u0590" <= ch <= "\u05ff" for ch in text):
        return text
    return text.translate(_LATIN1_TO_HEBREW)


_HEBREW_LETTER_RE = re.compile(r"[א-ת]")
_SPACED_MIN_TOKENS = 20
_SPACED_SINGLE_RATIO = 0.4
_SPACED_X_TOLERANCE = 10.0


def _is_single_hebrew_or_digit(tok: str) -> bool:
    return len(tok) == 1 and ("א" <= tok <= "ת" or tok.isascii() and tok.isdigit())


def _looks_spaced(text: str) -> bool:
    """True when the text is dominated by single Hebrew letters/digits —
    the signature of over-split OCR output (wide letter tracking)."""
    toks = text.split()
    if len(toks) < _SPACED_MIN_TOKENS:
        return False
    singles = sum(1 for t in toks if _is_single_hebrew_or_digit(t))
    return singles / len(toks) >= _SPACED_SINGLE_RATIO


def _despace_line(line: str, min_run: int = 2) -> str:
    """Join runs of single Hebrew letters/digits back into words.

    Only maximal runs of >= min_run single-char tokens are joined, so
    isolated list markers (e.g. 'סעיף א') keep their spacing.
    """
    toks = line.split(" ")
    res: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= min_run:
            res.append("".join(run))
        else:
            res.extend(run)
        run.clear()

    for tok in toks:
        if _is_single_hebrew_or_digit(tok):
            run.append(tok)
        else:
            flush()
            res.append(tok)
    flush()
    return " ".join(res)


def _despace_hebrew(text: str) -> str:
    """Repair over-split OCR text line-wise (see _despace_line)."""
    return "\n".join(_despace_line(line) for line in text.split("\n"))


def _ocr_enabled_default() -> bool:
    """OCR switch: explicit caller arg wins, else config, else on (binary-gated)."""
    try:
        from backend.core.config import get_config

        return get_config().ocr.enabled
    except Exception:
        return True


def _ocr_languages_default() -> str:
    """OCR languages: config value, else heb+eng."""
    try:
        from backend.core.config import get_config

        return get_config().ocr.languages
    except Exception:
        return "heb+eng"


def extract_pages(path: Path | str, use_ocr: bool | None = None) -> list[dict[str, Any]]:
    """Return list of {page_number: int, text: str}.

    Pages that yield no extractable text but contain raster images
    (scanned pages) fall back to Tesseract OCR when enabled and the
    binary is present; otherwise the page stays empty (never fatal).
    """
    if use_ocr is None:
        use_ocr = _ocr_enabled_default()
    ocr_usable = use_ocr and tesseract_available()
    pages: list[dict] = []
    with pdfplumber.open(str(path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text and _looks_spaced(text):
                # Over-split OCR text layer (wide letter tracking): retry
                # with a wider char tolerance; fall back to run-joining.
                try:
                    wider = page.extract_text(x_tolerance=_SPACED_X_TOLERANCE) or ""
                    if wider.strip():
                        text = wider
                except Exception:
                    LOG.warning("wide-tolerance re-extract failed for %s p%d", path, idx, exc_info=True)
                if _looks_spaced(text):
                    text = _despace_hebrew(text)
            if not text.strip() and ocr_usable:
                try:
                    image_count = len(page.images)
                except Exception:
                    image_count = 0
                if needs_ocr(text, image_count):
                    try:
                        text = ocr_pdf_page(path, idx, languages=_ocr_languages_default())
                        if _looks_spaced(text):
                            text = _despace_hebrew(text)
                    except Exception:
                        LOG.error("OCR fallback failed for %s p%d — page left empty", path, idx, exc_info=True)
            text = _fix_bidi(text)
            text = _fix_misencoded_hebrew(text)
            pages.append({"page_number": idx, "text": text})
    return pages


def _detect_sections(text: str, default_section: str = "") -> list[tuple[str, str]]:
    """Split text into (section_number, content) tuples.

    If SECTION_RE matches, split on matches; else return single chunk
    with section_number=default_section.
    """
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        # fallback: paragraph split
        parts = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
        if not parts:
            return [(default_section, text.strip())] if text.strip() else []
        return [(default_section, p) for p in parts]

    segments: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        sec = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        # include marker plus following content
        if content:
            segments.append((sec, content))
        else:
            segments.append((sec, ""))
    # prepend preamble before first marker if non-empty
    preamble = text[: matches[0].start()].strip()
    if preamble:
        segments.insert(0, (default_section, preamble))
    return segments


def get_searchable_text(chunk: dict) -> str:
    """Build context-enriched text for embedding and retrieval.

    Prepends document title, page number, and section to the chunk text so
    both dense vector embeddings and BM25 lexical search have full semantic
    context even when the paragraph itself is brief or implicit.
    """
    fn = str(chunk.get("file_name", ""))
    stem = Path(fn).stem if fn else ""
    page = chunk.get("page_number", 1)
    sec = str(chunk.get("section_number", "")).strip()
    parts = []
    if stem:
        parts.append(f"תקנון: {stem}")
    if page:
        parts.append(f"עמוד: {page}")
    if sec:
        parts.append(f"סעיף: {sec}")
    header = f"[{' | '.join(parts)}]" if parts else ""
    text = chunk.get("text", "")
    if header and not text.startswith(header):
        return f"{header}\n{text}"
    return text


def chunk_document(path: Path | str, file_hash: str | None = None, use_ocr: bool | None = None) -> list[dict[str, Any]]:
    """Extract and chunk a PDF. Each chunk has file_name, page_number, section_number, chunk_id, file_hash, text."""
    p = Path(path)
    if file_hash is None:
        file_hash = compute_file_hash(p)
    pages = extract_pages(p, use_ocr=use_ocr)
    chunks: list[dict] = []
    chunk_id = 0
    active_section = ""
    for page in pages:
        page_num = page["page_number"]
        text = page["text"]
        if not text.strip():
            continue
        segments = _detect_sections(text, default_section=active_section)
        for sec_num, content in segments:
            if not content.strip():
                continue
            if sec_num:
                active_section = sec_num
            # further split long segments into ~800 char paragraphs
            if len(content) > 1200:
                paras = [c.strip() for c in PARAGRAPH_SPLIT_RE.split(content) if c.strip()]
                for para in paras:
                    if not para:
                        continue
                    chunks.append(
                        {
                            "file_name": p.name,
                            "page_number": page_num,
                            "section_number": sec_num,
                            "chunk_id": f"{p.stem}_{chunk_id}",
                            "file_hash": file_hash,
                            "text": para,
                        }
                    )
                    chunk_id += 1
            else:
                chunks.append(
                    {
                        "file_name": p.name,
                        "page_number": page_num,
                        "section_number": sec_num,
                        "chunk_id": f"{p.stem}_{chunk_id}",
                        "file_hash": file_hash,
                        "text": content,
                    }
                )
                chunk_id += 1
    return chunks
