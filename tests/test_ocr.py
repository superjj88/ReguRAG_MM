"""Offline tests for the OCR fallback (backend/services/ocr.py + pdf_parser hook).

No tesseract/pdftoppm binaries required: all subprocess and binary-probing
calls are monkeypatched. Tests must run fully offline (repo rule).
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.services.ocr as ocr_mod
from backend.services.ocr import needs_ocr, ocr_page_image, ocr_pdf_page


# ---- needs_ocr decision ----

def test_needs_ocr_empty_text_with_images():
    assert needs_ocr("", image_count=1) is True
    assert needs_ocr("   \n ", image_count=2) is True


def test_needs_ocr_blank_page_without_images_is_false():
    assert needs_ocr("", image_count=0) is False


def test_needs_ocr_real_text_is_false_even_with_images():
    assert needs_ocr("תקנון תרומות " * 10, image_count=1) is False


def test_needs_ocr_respects_min_chars():
    assert needs_ocr("abc", image_count=1, min_chars=50) is True
    assert needs_ocr("abc", image_count=1, min_chars=3) is False


# ---- ocr_page_image ----

def test_ocr_page_image_missing_binary_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr_mod.shutil, "which", lambda _: None)
    try:
        ocr_page_image(tmp_path / "p.png")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
    # restore is automatic via monkeypatch


def test_ocr_page_image_success(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr_mod.shutil, "which", lambda _: "/usr/bin/tesseract")

    class _Proc:
        returncode = 0
        stdout = "שלום עולם"
        stderr = ""
        args = ["tesseract"]

    monkeypatch.setattr(ocr_mod.subprocess, "run", lambda *a, **k: _Proc())
    assert ocr_page_image(tmp_path / "p.png") == "שלום עולם"


def test_ocr_page_image_failure_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr_mod.shutil, "which", lambda _: "/usr/bin/tesseract")

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "error"
        args = ["tesseract"]

    monkeypatch.setattr(ocr_mod.subprocess, "run", lambda *a, **k: _Proc())
    try:
        ocr_page_image(tmp_path / "p.png")
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("expected CalledProcessError")


# ---- ocr_pdf_page ----

def test_ocr_pdf_page_missing_pdftoppm_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr_mod.shutil, "which", lambda name: None if name == "pdftoppm" else "/usr/bin/tesseract")
    try:
        ocr_pdf_page(tmp_path / "doc.pdf", 1)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


# ---- pdf_parser hook ----

def test_extract_pages_skips_ocr_when_disabled(monkeypatch):
    """use_ocr=False → blank scanned-looking page stays empty, no subprocess."""
    from backend.services import pdf_parser

    class _Page:
        images = [{"x0": 0}]  # looks scanned

        def extract_text(self):
            return ""

    class _PDF:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pdf_parser.pdfplumber, "open", lambda *a, **k: _PDF())
    called = []
    monkeypatch.setattr(pdf_parser, "ocr_pdf_page", lambda *a, **k: called.append(1) or "x")
    pages = pdf_parser.extract_pages("dummy.pdf", use_ocr=False)
    assert pages == [{"page_number": 1, "text": ""}]
    assert called == []


def test_extract_pages_ocr_fallback_recovers_text(monkeypatch):
    """Scanned page + OCR enabled + binary present → OCR text is used."""
    from backend.services import pdf_parser

    class _Page:
        images = [{"x0": 0}]

        def extract_text(self):
            return ""

    class _PDF:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pdf_parser.pdfplumber, "open", lambda *a, **k: _PDF())
    monkeypatch.setattr(pdf_parser, "tesseract_available", lambda: True)
    monkeypatch.setattr(pdf_parser, "ocr_pdf_page", lambda *a, **k: "תקנון תרומות")
    pages = pdf_parser.extract_pages("dummy.pdf", use_ocr=True)
    assert pages[0]["text"] == "תקנון תרומות"


def test_extract_pages_ocr_failure_leaves_page_empty(monkeypatch):
    """OCR raising must not crash extraction (logged, page left empty)."""
    from backend.services import pdf_parser

    class _Page:
        images = [{"x0": 0}]

        def extract_text(self):
            return ""

    class _PDF:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pdf_parser.pdfplumber, "open", lambda *a, **k: _PDF())
    monkeypatch.setattr(pdf_parser, "tesseract_available", lambda: True)

    def _boom(*a, **k):
        raise RuntimeError("tesseract blew up")

    monkeypatch.setattr(pdf_parser, "ocr_pdf_page", _boom)
    pages = pdf_parser.extract_pages("dummy.pdf", use_ocr=True)
    assert pages == [{"page_number": 1, "text": ""}]


# ---- spaced OCR-text repair (_looks_spaced / _despace_hebrew) ----

def test_looks_spaced_detects_split_letters():
    from backend.services.pdf_parser import _looks_spaced

    assert _looks_spaced("ר ש א / ס ף / ג ר ס ה ת א ר י ך " * 4) is True
    assert _looks_spaced("גרסה תאריך שינוי אחרון ומהות השינוי " * 4) is False
    assert _looks_spaced("") is False
    assert _looks_spaced("קצר") is False  # too few tokens


def test_despace_hebrew_joins_runs_preserves_markers():
    from backend.services.pdf_parser import _despace_hebrew

    assert _despace_hebrew("ר א ש / ס ף") == "ראש / סף"
    assert _despace_hebrew("2 0 / 1 2") == "20 / 12"
    # isolated single letters (list markers) keep spacing
    assert _despace_hebrew("סעיף א חלוקה") == "סעיף א חלוקה"


def test_extract_pages_repairs_spaced_layer_without_ocr(monkeypatch):
    """Spaced embedded text layer is repaired via wide tolerance, no OCR call."""
    from backend.services import pdf_parser

    class _Page:
        images = []

        def extract_text(self, **kwargs):
            if kwargs.get("x_tolerance", 3) >= 10:
                return "גרסה תאריך שינוי אחרון מהות"
            return "ג ר ס ה ת א ר י ך ש י נ ו י א ח ר ו ן ו א ו מ ה ו ת"

    class _PDF:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pdf_parser.pdfplumber, "open", lambda *a, **k: _PDF())
    called = []
    monkeypatch.setattr(pdf_parser, "ocr_pdf_page", lambda *a, **k: called.append(1) or "x")
    pages = pdf_parser.extract_pages("dummy.pdf", use_ocr=True)
    assert called == []
    assert "ג ר ס ה" not in pages[0]["text"]
    assert "גרסה" in pages[0]["text"]
