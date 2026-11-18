"""Stage 3 agentic refinement.

Failures are logged, not silently swallowed. The LLM-driven
query expansion is best-effort with a deterministic window-expansion fallback.
"""

from __future__ import annotations

import logging

from backend.core.config import get_config
from backend.services.hybrid_retrieval import hybrid_search

LOG = logging.getLogger(__name__)

MAX_REFINED_QUERY_CHARS = 120

# A small model returns prose ("## התשובה: ... לא קיימת בהקשר") instead of a
# search query. Feeding that back into hybrid_search poisons retrieval, so only
# a short single-line query is accepted; anything else falls back to window
# expansion.
_PROSE_MARKERS = ("התשובה", "ההקשר", "לא קיימת", "לא נמצא", "אינו מופיע")


def _as_search_query(text: str) -> str | None:
    """Return a usable reformulated query, or None when the model wrote prose."""
    raw = text.strip()
    if not raw or "\n" in raw or "##" in raw or "**" in raw or raw.lstrip().startswith(("-", "•", "*")):
        return None
    query = " ".join(raw.split()).strip(" \"'`״׳")
    if not 2 <= len(query) <= MAX_REFINED_QUERY_CHARS:
        return None
    if any(marker in query for marker in _PROSE_MARKERS):
        return None
    return query


def maybe_refine(
    query: str,
    ranked: list[dict],
    all_chunks: list[dict],
    ollama_client=None,
    stage3_budget_s: float | None = None,
    retriever=None,
) -> list[dict]:
    """Refine low-confidence retrievals.

    ``stage3_budget_s`` bounds the LLM expansion: below ~10s the LLM step is
    skipped entirely (window-expansion fallback only), and the LLM call runs
    as a single attempt (no retries) so the chat budget is never poisoned by
    retry storms — a slow/large model degrades to the fallback, it does not
    consume the whole generation budget.

    ``retriever`` (optional) runs a *fresh* vector search for the reformulated
    query. Without it the refined query can only reorder the already-truncated
    candidate pool, so documents missing from the first pass can never surface.
    Fresh candidates are merged into the pool and the pool is re-ranked against
    the ORIGINAL query — the reformulation widens recall, it does not get to
    redefine relevance.
    """
    cfg = get_config()
    threshold = cfg.retrieval.stage3_relevance_threshold
    if not ranked:
        return ranked
    best = ranked[0].get("dense_score")
    if best is None:
        # no embedding available (offline / lexical-only path): fall back to the
        # fused score, which is min-max scaled and therefore always 1.0 for the
        # top row — refinement then always runs, which is the safe default.
        best = ranked[0].get("score", 0)
    if best >= threshold:
        return ranked  # no refinement needed

    llm_allowed = stage3_budget_s is None or stage3_budget_s >= 10.0

    if llm_allowed and ollama_client is None:
        try:
            from backend.services.ollama_client import get_ollama_client

            ollama_client = get_ollama_client()
        except Exception:
            LOG.warning("stage3 refinement: LLM unavailable — window-expansion fallback", exc_info=True)
            ollama_client = None

    if llm_allowed and ollama_client is not None:
        try:
            ctx = "\n".join(c.get("text", "")[:400] for c in ranked[:3])
            prompt = (
                f"השאלה: {query}\nהקשר (קטעים):\n{ctx}\n\n"
                "המשימה: נסח מחדש את השאלה כשאילתת חיפוש קצרה בעברית (עד 12 מילים) "
                "שתמצא את הקטעים הרלוונטיים במסמכי התקנון. "
                "השב אך ורק בשאילתה עצמה — בלי הסברים, בלי תשובה לשאלה, בלי מרכאות."
            )
            res = ollama_client.generate(
                prompt,
                temperature=0.0,
                timeout_s=stage3_budget_s if stage3_budget_s is not None else 15.0,
                max_retries=0,
            )
            refined = _as_search_query(res.text)
            if refined and refined != query:
                LOG.info("stage3 refined query: %.200s", refined)
                # widen the pool with a fresh vector search, then rank the whole
                # pool against the original question
                pool = list(all_chunks)
                known = {str(c.get("chunk_id")) for c in pool if c.get("chunk_id")}
                if retriever is not None:
                    try:
                        for c in retriever(refined):
                            cid = str(c.get("chunk_id"))
                            if cid and cid not in known:
                                pool.append(c)
                                known.add(cid)
                    except Exception:
                        LOG.warning("stage3 refinement: re-retrieval failed — pool unchanged", exc_info=True)
                fresh_ranked = hybrid_search(query, pool)
                if fresh_ranked:
                    return fresh_ranked[: cfg.retrieval.top_k]
                return ranked[: cfg.retrieval.top_k]
        except Exception:
            LOG.warning("stage3 refinement: LLM expansion failed — window-expansion fallback", exc_info=True)

    # fallback window expansion without LLM
    expanded = list(ranked)
    return _add_neighbor_pages(expanded, ranked, all_chunks)[: cfg.retrieval.top_k]


def _add_neighbor_pages(expanded: list[dict], ranked: list[dict], all_chunks: list[dict]) -> list[dict]:
    """Append chunks from pages adjacent (±1) to the current ranked pages."""
    ranked_ids = {r.get("chunk_id") for r in expanded if r.get("chunk_id")}
    page_nums: set[int] = set()
    for c in ranked:
        pn = c.get("page_number")
        if isinstance(pn, int):
            page_nums.add(pn)
    neighbors = {p + 1 for p in page_nums} | {p - 1 for p in page_nums}
    for c in all_chunks:
        if c.get("page_number") in neighbors and c.get("chunk_id") not in ranked_ids:
            expanded.append(c)
            ranked_ids.add(c.get("chunk_id"))
    return expanded
