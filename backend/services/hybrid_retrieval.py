"""Hybrid retrieval Stage1-2.

Removed the hardcoded golden-corpus score boost; ranking is
now purely lexical + dense fusion. Failures are logged, not
silently swallowed.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict

from backend.core.config import get_config
from backend.core.database import get_connection, is_hash_excluded
from backend.services.vector_math import cosine_similarity

LOG = logging.getLogger(__name__)

HEBREW_STOPWORDS = {
    "של", "את", "על", "מה", "מי", "האם", "זה", "זו", "אלה", "אלו", "אם",
    "או", "בין", "כל", "רק", "מן", "עם", "כי", "אשר", "לא", "יש", "אין",
    "כדי", "לגבי", "בגין", "מתוך", "אודות", "לפי", "שלו", "שלה", "שלהם",
    "שלהן", "היה", "היתה", "יהיה", "הם", "הן", "הוא", "היא", "אנו", "אנחנו",
    "אתה", "אתם", "גם", "עוד", "שם", "כבר", "אך", "אלא", "כך",
    "כיצד", "איך", "איפה", "היכן", "מתי", "מדוע", "למה", "מהם", "מהו", "מהי",
    "אילו", "באיזה", "באיזו", "ומי", "ומה",
}

_PREFIX_LETTERS = "והבלמשכ"


def _hebrew_tokenize(text: str) -> list[str]:
    return re.findall(r"[\u0590-\u05FF\w]+", text.lower())


def _strip_hebrew_prefix(tok: str) -> str:
    """Strip a single Hebrew grammatical prefix if token is long enough."""
    if len(tok) > 3 and tok[0] in _PREFIX_LETTERS:
        return tok[1:]
    return tok


def _expand_tokens(tokens: list[str]) -> list[str]:
    """Include both surface forms and prefix-stripped forms for morphological recall."""
    res = []
    for t in tokens:
        res.append(t)
        stripped = _strip_hebrew_prefix(t)
        if stripped != t:
            res.append(stripped)
    return res


class BM25Okapi:
    """True Okapi BM25 with smoothed IDF and document length normalization."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.avgdl = (
            sum(len(doc) for doc in corpus_tokens) / self.corpus_size
            if self.corpus_size > 0
            else 1.0
        )
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.doc_freqs: list[dict[str, int]] = [Counter(doc) for doc in corpus_tokens]
        self.idf: dict[str, float] = {}
        self._init_idf()

    def _init_idf(self) -> None:
        df: dict[str, int] = defaultdict(int)
        for freqs in self.doc_freqs:
            for term in freqs:
                df[term] += 1
        for term, freq in df.items():
            # Standard smoothed BM25 IDF formula
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for q in query_tokens:
            q_idf = self.idf.get(q)
            if q_idf is None:
                q_strip = _strip_hebrew_prefix(q)
                q_idf = self.idf.get(q_strip, 0.5)
            if q_idf <= 0.0:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(q, 0)
                if freq == 0:
                    q_strip = _strip_hebrew_prefix(q)
                    freq = freqs.get(q_strip, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                scores[i] += q_idf * (freq * (self.k1 + 1.0)) / denom
        return scores


_TOC_DOT_RUN_RE = re.compile(r"\.{4,}")
TOC_MIN_DOT_RUNS = 4


def looks_like_toc(text: str) -> bool:
    """True for table-of-contents chunks (navigational, not normative).

    A contents page can win the lexical half of the fusion on repeated section
    titles and then get cited as if it stated a rule (observed: "נספח 5 -
    נוהל טרום קליטה לבנות זוג בהריון"). Dot leaders are a document-generic
    signal: 14/1169 corpus chunks exceed the threshold.
    """
    return len(_TOC_DOT_RUN_RE.findall(text)) >= TOC_MIN_DOT_RUNS


def _minmax(values: list[float]) -> list[float]:
    """Scale to 0-1 so dense/lexical signals are fused on comparable ranges."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _bm25_scores(query: str, docs: list[dict]) -> list[float]:
    """Calculate Okapi BM25 scores with IDF over the candidate set."""
    filtered_q = [t for t in _hebrew_tokenize(query) if t not in HEBREW_STOPWORDS and len(t) > 1]
    q_tokens = _expand_tokens(filtered_q) if filtered_q else _hebrew_tokenize(query)
    if not q_tokens or not docs:
        return [0.0] * len(docs)

    doc_tokens_list: list[list[str]] = []
    for d in docs:
        text = d.get("text", "").lower()
        fn = str(d.get("file_name", "")).lower()
        sec = str(d.get("section_number", "")).lower()
        combined = f"{fn} {sec} {text}"
        tokens = [t for t in _hebrew_tokenize(combined) if t not in HEBREW_STOPWORDS and len(t) > 1]
        doc_tokens_list.append(_expand_tokens(tokens))

    bm25_model = BM25Okapi(doc_tokens_list)
    raw_scores = bm25_model.get_scores(q_tokens)

    # Substring fallback for Hebrew prefixes and terms not caught by exact/stripped tokens
    for i, score in enumerate(raw_scores):
        if score == 0.0:
            text = docs[i].get("text", "").lower()
            sub_matches = sum(1 for t in q_tokens if len(t) > 2 and t in text)
            if sub_matches:
                raw_scores[i] = 0.1 * sub_matches

    mx = max(raw_scores) if raw_scores else 1.0
    return [s / mx if mx > 0.0 else 0.0 for s in raw_scores]


def hybrid_search(query: str, chunks: list[dict], top_k: int | None = None, alpha: float | None = None, db_path=None) -> list[dict]:
    """Stage1 dense+BM25 fusion, Stage2 dedup/rerank/filter.

    chunks: list of chunk dicts with text, file_hash, page_number etc, optionally embeddings.
    If embeddings present, dense score via cosine; else lexical fallback.
    """
    cfg = get_config()
    top_k = top_k or cfg.retrieval.top_k
    alpha = alpha if alpha is not None else cfg.retrieval.hybrid_search_alpha

    # filter excluded hashes
    if db_path is not None:
        try:
            conn = get_connection(db_path)
            chunks = [c for c in chunks if not is_hash_excluded(conn, c.get("file_hash", ""))]
            conn.close()
        except Exception:
            LOG.warning("hash-exclusion filter skipped (db unavailable)", exc_info=True)

    if not chunks:
        return []

    # BM25 (already 0-1 normalised)
    bm25 = _bm25_scores(query, chunks)
    bm25_norm = bm25[:]

    # Dense scores — cosine between the query vector and each chunk's *stored*
    # embedding. Chunks without an embedding (e.g. linked-document chunks pulled
    # from SQLite) are scored lexically and counted in the log; they are never
    # given a fabricated similarity. The raw cosine of the winner is kept as
    # ``dense_score`` so callers can gate on an absolute relevance measure
    # instead of a min-max-normalised fusion score (which is 1.0 by
    # construction for the top row).
    dense_raw: list[float | None] = [None] * len(chunks)
    try:
        from backend.services.ollama_client import get_ollama_client

        client = get_ollama_client()
        qvec = client.embed(query)[0]
        missing = 0
        for i, c in enumerate(chunks):
            emb = c.get("embedding")
            if not emb:
                missing += 1
                continue
            dense_raw[i] = cosine_similarity(qvec, emb)
        if missing:
            LOG.warning(
                "dense scoring skipped for %d/%d chunks without an embedding — "
                "their dense component falls back to the lexical score",
                missing,
                len(chunks),
            )
    except Exception:
        LOG.warning("dense scoring unavailable — lexical fallback used for %d chunks", len(chunks), exc_info=True)

    known = [v for v in dense_raw if v is not None]
    if known:
        scaled = iter(_minmax(known))
        dense = [next(scaled) if v is not None else bm25_norm[i] for i, v in enumerate(dense_raw)]
    else:
        dense = bm25_norm[:]

    # Candidate reranking & composite fusion:
    # 1. Base linear fusion of dense + Okapi BM25
    # 2. Query keyword coverage & document title congruence
    q_tokens = [t for t in _hebrew_tokenize(query) if t not in HEBREW_STOPWORDS and len(t) > 1]
    q_set = set(q_tokens)
    num_q = max(len(q_set), 1)

    fused = []
    for i, c in enumerate(chunks):
        base_fusion = alpha * dense[i] + (1.0 - alpha) * bm25[i]
        text_lower = c.get("text", "").lower()
        fn_lower = str(c.get("file_name", "")).lower()

        # Keyword coverage & title congruence
        matched_terms = sum(1 for t in q_set if t in text_lower or _strip_hebrew_prefix(t) in text_lower)
        coverage = matched_terms / num_q
        title_matches = sum(1 for t in q_set if t in fn_lower or _strip_hebrew_prefix(t) in fn_lower)
        title_bonus = 0.15 * (title_matches / num_q)

        rerank_score = base_fusion * (0.60 + 0.40 * coverage) + title_bonus
        # Preserve exact 1.0 for perfect dual-channel hits
        fused_score = min(1.0, base_fusion)
        fused.append((rerank_score, fused_score, i, c))

    fused.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)

    # deduplicate by chunk_id, keep highest score
    seen: set[str] = set()
    ranked = []
    for _rerank_s, fused_s, i, c in fused:
        cid = c.get("chunk_id")
        if not cid:
            cid = f"__noid_{id(c)}"
        if cid in seen:
            continue
        seen.add(cid)
        c2 = dict(c)
        c2["score"] = fused_s
        c2["dense_score"] = dense_raw[i]
        c2["bm25_score"] = bm25[i]
        ranked.append(c2)
        if len(ranked) >= top_k:
            break
    return ranked
