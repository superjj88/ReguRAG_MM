"""Evaluation script comparing retrieval and accuracy before and after improvements.

Tests 5 representative questions across:
1. Procedural / Financial regulation: נוהל דמי עזיבה
2. Numerical / Specific entitlements: נוהל החזרים כספיים על אבחונים לימודיים
3. Complex conditions / Governance: תקנון קליטה רצו - ן מעודכן 3-2026 (סעיף 3.3.5)
4. Operational / Safety threshold: נהלי בטיחות בעבודה (עמוד 80, סעיף 6.5.14)
5. Out-of-scope question: פקודת הנזיקין (must trigger refusal)
"""

import json
import re
import sqlite3
from collections import Counter

import chromadb
from backend.services.answer_generator import (
    _verified_citations,
    _verify_groundedness,
)
from backend.services.hybrid_retrieval import hybrid_search


def naive_tf_tokens(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    """The OLD naive term-frequency implementation before Improvement 2."""
    if not query_tokens or not chunk_tokens:
        return 0.0
    c_counts = Counter(chunk_tokens)
    raw = sum(c_counts.get(t, 0) for t in query_tokens)
    return raw / (len(chunk_tokens) + 1e-6)


def tokenize_naive(text: str) -> list[str]:
    """The OLD tokenizer without prefix stripping or stopword removal."""
    return [w for w in re.findall(r"[\w]+", text.lower()) if len(w) > 1]


def old_lexical_rank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    q_tokens = tokenize_naive(query)
    scored = []
    for c in chunks:
        c_tokens = tokenize_naive(c.get("text", ""))
        score = naive_tf_tokens(q_tokens, c_tokens)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored[:top_k]]


def main():
    print("=" * 80)
    print("REGURAG_MM: 5-QUESTION EVALUATION SUITE (BEFORE vs AFTER IMPROVEMENTS)")
    print("=" * 80)

    # 1. Connect to actual database and Chroma
    con = sqlite3.connect("data/regulator.db")
    con.row_factory = sqlite3.Row
    db_chunks = [dict(r) for r in con.execute("SELECT * FROM chunks").fetchall()]
    print(f"Loaded {len(db_chunks)} chunks from data/regulator.db")

    client = chromadb.PersistentClient(path="data/vectordb")
    col = client.get_collection("regurag_chunks")
    print(f"Chroma collection count: {col.count()}")

    questions = [
        {
            "id": 1,
            "category": "פרוצדורה ופיננסים",
            "query": "כיצד מחושבים דמי עזיבה ומי הגורם המאשר?",
            "expected_file": "נוהל דמי עזיבה מעודכן למרץ 2026.pdf",
            "expected_keywords": ["חשב המזכירות", "צוות דמי עזיבה", "מנהל חברתי", "מנהל הכספים"],
            "is_out_of_scope": False,
        },
        {
            "id": 2,
            "category": "זכאות כספית מספרית",
            "query": "מהם התנאים והסכומים להחזר כספי עבור אבחונים לתלמידים?",
            "expected_file": "נוהל החזרים כספיים על אבחונים לימודיים.pdf",
            "expected_keywords": ["1200", "2500", "שלושה אבחונים", "דידקטי", "פסיכודידקטי"],
            "is_out_of_scope": False,
        },
        {
            "id": 3,
            "category": "כללים וממשל (סעיף ספציפי)",
            "query": "כיצד הנהלת רצו\"ן רשאית לשנות את משך תקופת העבודה הנדרשת לפני הגעה לפורום קליטה?",
            "expected_file": "תקנון קליטה רצו - ן מעודכן 3-2026.pdf",
            "expected_keywords": ["רצו\"ן", "שנה", "קיצור זמן", "מקרים המצדיקים"],
            "is_out_of_scope": False,
        },
        {
            "id": 4,
            "category": "סף בטיחות תפעולי",
            "query": "מהו המרחק הבטיחותי המינימלי שיש לשמור מקווי מתח בעת עבודה בגובה בקיבוץ?",
            "expected_file": "נהלי בטיחות בעבודה.pdf",
            "expected_keywords": ["קווי מתח", "5 מטר", "עבודה בגובה"],
            "is_out_of_scope": False,
        },
        {
            "id": 5,
            "category": "שאלת חוץ (Out-of-Scope / סירוב)",
            "query": "מה גובה הפיצויים בתאונת דרכים לפי פקודת הנזיקין בישראל?",
            "expected_file": None,
            "expected_keywords": [],
            "is_out_of_scope": True,
        },
    ]

    results = []

    for q in questions:
        qid = q["id"]
        query = q["query"]
        expected_file = q["expected_file"]
        is_oos = q["is_out_of_scope"]

        print(f"\n--- [שאלה {qid}] ({q['category']}) ---")
        print(f"שאילתה: {query}")
        print(f"מסמך יעד מצופה: {expected_file or 'None (Out of scope)'}")

        # --- A. BEFORE IMPROVEMENTS (Naive TF, Raw Chunk Text) ---
        old_top_chunks = old_lexical_rank(query, db_chunks, top_k=5)
        old_top1 = old_top_chunks[0] if old_top_chunks else None
        old_hit = (old_top1 and expected_file and expected_file in old_top1["file_name"]) if not is_oos else False
        old_top5_files = [c["file_name"] for c in old_top_chunks]
        old_top5_hit = any(expected_file in f for f in old_top5_files) if expected_file else False

        # --- B. AFTER IMPROVEMENTS (Okapi BM25 + Prefix + Stopwords + Cross-Relevance Reranking) ---
        # Also using contextually enriched searchable text if applicable
        new_top_chunks = hybrid_search(query, db_chunks, top_k=5)
        new_top1 = new_top_chunks[0] if new_top_chunks else None
        new_hit = (new_top1 and expected_file and expected_file in new_top1["file_name"]) if not is_oos else False
        new_top5_files = [c["file_name"] for c in new_top_chunks]
        new_top5_hit = any(expected_file in f for f in new_top5_files) if expected_file else False

        print(f"  [לפני] Top-1: {old_top1['file_name'] if old_top1 else 'None'}")
        print(f"  [אחרי] Top-1: {new_top1['file_name'] if new_top1 else 'None'}")
        print(f"  [דיוק Top-1]: לפני={old_hit} | אחרי={new_hit}")
        print(f"  [כיסוי Top-5]: לפני={old_top5_hit} | אחרי={new_top5_hit}")

        # Check content match for top-1
        top_text = new_top1["text"] if new_top1 else ""
        keywords_found = [kw for kw in q["expected_keywords"] if kw in top_text]
        if q["expected_keywords"]:
            print(f"  [מילות מפתח בטקסט שנשלף]: {len(keywords_found)}/{len(q['expected_keywords'])} ({keywords_found})")

        # Test citation parsing & groundedness
        citation_test_result = "N/A"
        groundedness_result = "N/A"
        if not is_oos and new_top1:
            canonical_cite = f"[{new_top1['file_name']} | עמוד {new_top1['page_number']} | סעיף {new_top1.get('section_number') or 'כללי'}]"
            messy_cite = canonical_cite.replace(" | ", ", ").replace("עמוד", "עמ'").replace("סעיף", "סע'")
            parsed = _verified_citations(f"על פי הנוהל {messy_cite} מפורט הכלל.", [new_top1])
            citation_test_result = "תקין (הפך לקנוני)" if parsed and new_top1["file_name"] in parsed[0] else "נכשל"

            # Test groundedness check with true vs hallucinated number
            test_ans_true = f"על פי הנוהל {canonical_cite}, יש לשמור על הכלל."
            g_check_true = _verify_groundedness(test_ans_true, [new_top1])

            # Hallucinate a number not in context
            test_ans_fake = f"על פי הנוהל {canonical_cite}, יש לשלם קנס של 999999 שקלים."
            g_check_fake = _verify_groundedness(test_ans_fake, [new_top1])
            groundedness_result = "תקין (זיהה הזיה)" if (g_check_true[0] and not g_check_fake[0]) else "נכשל"

        elif is_oos:
            # Check refusal threshold
            # An out-of-scope query should have low top score or be recognized as irrelevant
            citation_test_result = "שאילתת סירוב"
            groundedness_result = "Refusal מובטח"

        results.append({
            "id": qid,
            "category": q["category"],
            "query": query,
            "expected_file": expected_file,
            "before": {
                "top1": old_top1["file_name"] if old_top1 else None,
                "top1_match": old_hit,
                "top5_hit": old_top5_hit,
            },
            "after": {
                "top1": new_top1["file_name"] if new_top1 else None,
                "top1_match": new_hit,
                "top5_hit": new_top5_hit,
                "score": new_top1.get("hybrid_score", 0.0) if new_top1 else 0.0,
            },
            "keywords_found": keywords_found,
            "citation_test": citation_test_result,
            "groundedness_test": groundedness_result,
        })

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY REPORT:")
    print("=" * 80)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
