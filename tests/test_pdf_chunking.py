from unittest.mock import MagicMock, patch


from backend.core.database import get_connection, init_db, upsert_document_by_hash
from backend.services.document_scanner import scan_documents
from backend.services.pdf_parser import chunk_document, compute_file_hash


def _mock_pdf_pages(texts_by_page):
    """Helper to mock pdfplumber.open to return pages with given texts."""
    mock_pdf = MagicMock()
    pages = []
    for t in texts_by_page:
        m = MagicMock()
        m.extract_text.return_value = t
        pages.append(m)
    mock_pdf.pages = pages
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    return mock_pdf


def test_chunking_hebrew_sections(tmp_path):
    dummy = tmp_path / "reg.pdf"
    dummy.write_bytes(b"%PDF dummy")

    pages = [
        "סעיף 1.1 זהו תוכן סעיף ראשון.\n\nסעיף 1.2 תוכן שני בפרק.",
        "פרק ב' תוכן פרק ב.\n\nתקנה 12 תוכן תקנה.",
        "זה פסקה ללא סימון סעיף אבל בעברית מלאה.",
    ]
    with patch("backend.services.pdf_parser.pdfplumber.open", return_value=_mock_pdf_pages(pages)):
        chunks = chunk_document(dummy, file_hash="hash123")

    assert len(chunks) >= 4
    for c in chunks:
        assert c["file_name"] == "reg.pdf"
        assert "page_number" in c and c["page_number"] in (1, 2, 3)
        assert "section_number" in c
        assert "chunk_id" in c and c["chunk_id"].startswith("reg_")
        assert c["file_hash"] == "hash123"
        assert c["text"].strip()
        # Hebrew presence
        assert any("\u0590" <= ch <= "\u05FF" for ch in c["text"])


def test_chunking_fallback_paragraph(tmp_path):
    dummy = tmp_path / "nop.pdf"
    dummy.write_bytes(b"%PDF")
    pages = ["פסקה ראשונה.\n\nפסקה שנייה.\n\nפסקה שלישית."]
    with patch("backend.services.pdf_parser.pdfplumber.open", return_value=_mock_pdf_pages(pages)):
        chunks = chunk_document(dummy, file_hash="h2")
    # fallback splits into 3 paragraphs
    assert len(chunks) == 3
    assert all(c["section_number"] == "" for c in chunks)


def test_scanner_detects_new_changed_deleted(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    db_path = tmp_path / "scan.db"
    init_db(db_path)

    # create two pdfs
    a = docs / "a.pdf"
    b = docs / "b.pdf"
    a.write_bytes(b"content a v1")
    b.write_bytes(b"content b v1")

    conn = get_connection(db_path)
    # first scan -> both new
    res = scan_documents(docs, db_path=db_path, conn=conn)
    assert len(res["new"]) == 2 and len(res["unchanged"]) == 0

    # upsert them to DB
    for item in res["new"]:
        upsert_document_by_hash(conn, file_name=item["file_name"], path=item["path"], file_hash_sha256=item["hash"], page_count=1)
    # second scan without changes -> unchanged
    res2 = scan_documents(docs, db_path=db_path, conn=conn)
    assert len(res2["unchanged"]) == 2 and len(res2["new"]) == 0

    # modify a.pdf
    a.write_bytes(b"content a v2 different")
    res3 = scan_documents(docs, db_path=db_path, conn=conn)
    assert len(res3["changed"]) == 1 and res3["changed"][0]["file_name"] == "a.pdf"
    assert len(res3["unchanged"]) == 1

    # delete b.pdf
    b.unlink()
    res4 = scan_documents(docs, db_path=db_path, conn=conn)
    assert len(res4["deleted"]) == 1 and res4["deleted"][0]["file_name"] == "b.pdf"

    conn.close()


def test_compute_hash(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    h = compute_file_hash(p)
    assert len(h) == 64
    assert h == compute_file_hash(p)


def test_fix_misencoded_hebrew_resolves_nun():
    from backend.services.pdf_parser import _fix_misencoded_hebrew

    raw = "ההðהלה לא תעלה על 11 חברים — מðהל עðף הרכב, ðציג תאגידים"
    fixed = _fix_misencoded_hebrew(raw)
    assert "\u00f0" not in fixed
    assert "ההנהלה" in fixed and "מנהל ענף" in fixed and "נציג תאגידים" in fixed
    assert "נ" in fixed


def test_fix_misencoded_hebrew_ignores_latin_only():
    from backend.services.pdf_parser import _fix_misencoded_hebrew

    latin = "olralgerra ratnetni on ,enoicnuf on/odacsata étse rodavele"
    assert _fix_misencoded_hebrew(latin) == latin


def test_chunking_repairs_misencoded_hebrew(tmp_path):
    dummy = tmp_path / "reg.pdf"
    dummy.write_bytes(b"%PDF dummy")

    pages = ["סעיף 3.1 ההðהלה תפעל לפי ðוהל זה.\n\nסעיף 3.2 מðהל התחום אחראי."]
    with patch("backend.services.pdf_parser.pdfplumber.open", return_value=_mock_pdf_pages(pages)):
        chunks = chunk_document(dummy, file_hash="h3")

    texts = " ".join(c["text"] for c in chunks)
    assert "\u00f0" not in texts
    assert "ההנהלה" in texts and "מנהל" in texts
