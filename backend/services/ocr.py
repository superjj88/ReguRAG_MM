"""OCR fallback for scanned PDFs — Tesseract (heb+eng).

Some regulation PDFs are pure raster scans (no embedded text layer):
pdfplumber extracts 0 characters from every page and the document ends up
with 0 chunks. This module detects such pages (``needs_ocr``) and recovers
their text by rendering the page with ``pdftoppm`` (poppler) and running
``tesseract`` with Hebrew+English language data.

Zero new Python dependencies: both tools are used via subprocess.
When the binaries are absent the caller must skip OCR gracefully
(ingestion must never crash because OCR tooling is missing).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

LOG = logging.getLogger(__name__)

DEFAULT_LANGUAGES = "heb+eng"
DEFAULT_MIN_CHARS = 50
DEFAULT_DPI = 300


def tesseract_available() -> bool:
    """True when the ``tesseract`` binary is on PATH."""
    return shutil.which("tesseract") is not None


def pdftoppm_available() -> bool:
    """True when the ``pdftoppm`` binary (poppler) is on PATH."""
    return shutil.which("pdftoppm") is not None


def needs_ocr(text: str, image_count: int = 0, min_chars: int = DEFAULT_MIN_CHARS) -> bool:
    """Decide whether a page needs OCR.

    A page needs OCR when it yields (almost) no extractable text but
    contains embedded raster images — the signature of a scanned page.
    Blank pages (no text AND no images) do not need OCR.
    """
    if len((text or "").strip()) >= min_chars:
        return False
    return image_count > 0


def ocr_page_image(png_path: Path | str, languages: str = DEFAULT_LANGUAGES) -> str:
    """Run tesseract on a single PNG image, return extracted text.

    Raises:
        FileNotFoundError: tesseract binary missing.
        subprocess.CalledProcessError: tesseract failed (logged with context).
    """
    if not tesseract_available():
        raise FileNotFoundError("tesseract binary not found on PATH")
    try:
        proc = subprocess.run(
            ["tesseract", str(png_path), "stdout", "-l", languages],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except Exception:
        LOG.error("tesseract execution failed for %s", png_path, exc_info=True)
        raise
    if proc.returncode != 0:
        LOG.error("tesseract failed for %s (rc=%d): %s", png_path, proc.returncode, proc.stderr.strip()[-500:])
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
    return proc.stdout


def ocr_pdf_page(
    pdf_path: Path | str,
    page_number: int,
    languages: str = DEFAULT_LANGUAGES,
    dpi: int = DEFAULT_DPI,
) -> str:
    """Render one PDF page (1-based) to PNG and OCR it. Returns text (may be empty)."""
    if not pdftoppm_available():
        raise FileNotFoundError("pdftoppm binary not found on PATH")
    pdf_path = Path(pdf_path)
    with tempfile.TemporaryDirectory(prefix="regurag_ocr_") as tmp:
        prefix = str(Path(tmp) / "page")
        render = subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", str(page_number), "-l", str(page_number),
             str(pdf_path), prefix],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if render.returncode != 0:
            LOG.error("pdftoppm failed for %s p%d (rc=%d): %s",
                      pdf_path.name, page_number, render.returncode, render.stderr.strip()[-500:])
            raise subprocess.CalledProcessError(render.returncode, render.args, stderr=render.stderr)
        pngs = sorted(Path(tmp).glob("*.png"))
        if not pngs:
            LOG.error("pdftoppm produced no PNG for %s p%d", pdf_path.name, page_number)
            raise RuntimeError(f"pdftoppm produced no image for {pdf_path.name} page {page_number}")
        return ocr_page_image(pngs[0], languages=languages)
