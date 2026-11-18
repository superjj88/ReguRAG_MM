"""Unit tests for the 3 reliability and accuracy improvements:
1. Contextual chunk enrichment (get_searchable_text + section continuity)
2. True Okapi BM25 with IDF and cross-relevance reranking
3. Tolerant citation parsing and factual grounding verification
"""

from backend.services.answer_generator import (
    _verified_citations,
    _verify_groundedness,
)
from backend.services.hybrid_retrieval import (
    BM25Okapi,
    _bm25_scores,
    hybrid_search,
)
from backend.services.pdf_parser import _detect_sections, get_searchable_text


# ---- 1. Contextual Chunk Enrichment ----


def test_get_searchable_text_formats_full_context():
    chunk = {
        "file_name": "תקנון שיוך דירות 2024.pdf",
        "page_number": 3,
        "section_number": "סעיף 12.2",
        "text": "על החבר להשלים 5 שנות חברות רצופות.",
    }
    enriched = get_searchable_text(chunk)
    assert "[תקנון: תקנון שיוך דירות 2024 | עמוד: 3 | סעיף: סעיף 12.2]" in enriched
    assert "על החבר להשלים 5 שנות חברות רצופות." in enriched


def test_get_searchable_text_without_section():
    chunk = {
        "file_name": "נוהל בריכה.pdf",
        "page_number": 1,
        "section_number": "",
        "text": "אין להכניס ילדים לא גמולים.",
    }
    enriched = get_searchable_text(chunk)
    assert "[תקנון: נוהל בריכה | עמוד: 1]" in enriched
    assert "סעיף:" not in enriched


def test_detect_sections_carries_forward_active_section():
    text_page2 = "המשך הוראות בנושא זכאות לתשלום.\n\nסעיף 2.2 הוראה חדשה."
    segments = _detect_sections(text_page2, default_section="סעיף 2.1")
    assert len(segments) == 2
    assert segments[0][0] == "סעיף 2.1"
    assert "המשך הוראות" in segments[0][1]
    assert segments[1][0] == "סעיף 2.2"


# ---- 2. True Okapi BM25 & Reranking ----


def test_bm25_okapi_idf_rewards_rare_terms():
    # Corpus where "תקנון" appears in all 3 docs, but "פנסיה" appears in only 1
    doc1 = ["תקנון", "כללי", "חברים"]
    doc2 = ["תקנון", "פנסיה", "חישוב"]
    doc3 = ["תקנון", "ועדה", "החלטות"]
    bm25 = BM25Okapi([doc1, doc2, doc3])

    # "פנסיה" should have significantly higher IDF than ubiquitous "תקנון"
    assert bm25.idf["פנסיה"] > bm25.idf["תקנון"]

    scores_pension = bm25.get_scores(["פנסיה"])
    assert scores_pension[1] > scores_pension[0]
    assert scores_pension[0] == 0.0


def test_bm25_hebrew_prefix_matching():
    chunks = [
        {"chunk_id": "c1", "text": "הוועדה תתכנס אחת לחודש", "file_name": "נהלים.pdf", "page_number": 1},
        {"chunk_id": "c2", "text": "טקסט אחר לגמרי", "file_name": "אחר.pdf", "page_number": 1},
    ]
    # Query without prefix "ועדה" should match "הוועדה"
    scores = _bm25_scores("ועדה", chunks)
    assert scores[0] > scores[1]
    assert scores[1] == 0.0


def test_hybrid_search_title_and_coverage_reranking():
    chunks = [
        {
            "chunk_id": "generic",
            "text": "החבר יקבל דמי עזיבה לפי שנות הוותק",
            "file_name": "תקנון כללי.pdf",
            "page_number": 1,
            "section_number": "1",
        },
        {
            "chunk_id": "specific",
            "text": "דמי עזיבה יחושבו על ידי חשב המזכירות לפי שנות הוותק המוכרות",
            "file_name": "נוהל דמי עזיבה 2026.pdf",
            "page_number": 2,
            "section_number": "4",
        },
    ]
    # Query matches both chunks, but specific chunk has filename title congruence
    ranked = hybrid_search("נוהל דמי עזיבה וחשב המזכירות", chunks, top_k=2)
    assert ranked[0]["chunk_id"] == "specific"


# ---- 3. Tolerant Citations & Grounding Verification ----


def test_tolerant_citation_regex_and_normalization():
    chunks = [
        {"file_name": "תקנון קליטה 2025.pdf", "page_number": 6, "section_number": "3.3.5", "text": "תנאי קליטה."},
    ]
    # Citation with comma and עמ' instead of pipe and עמוד
    answer = "התנאים מפורטים בנספח [תקנון קליטה 2025 | עמ' 6 | סע' 3.3.5]"
    cites = _verified_citations(answer, chunks)
    assert len(cites) == 1
    assert cites[0] == "[תקנון קליטה 2025.pdf | עמוד 6 | סעיף 3.3.5]"


def test_grounding_verification_detects_hallucinated_numbers():
    chunks = [
        {"file_name": "תקנון.pdf", "page_number": 1, "text": "שיעור התשלום הינו 250 ש״ח לחודש."},
    ]
    valid_answer = "הסכום לתשלום הוא 250 ש״ח [תקנון.pdf | עמוד 1]"
    grounded, _ = _verify_groundedness(valid_answer, chunks)
    assert grounded is True

    # Hallucinated number 990 not present in context
    hallucinated_answer = "הסכום לתשלום הוא 990 ש״ח [תקנון.pdf | עמוד 1]"
    grounded_bad, reason = _verify_groundedness(hallucinated_answer, chunks)
    assert grounded_bad is False
    assert "990" in reason
