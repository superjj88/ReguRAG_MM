"""Contradiction detection. BGE-M3 similarity + LLM judgment.

Design notes:
* embeddings are fetched in batches (previously 2 HTTP calls per chunk pair —
  O(n²) network traffic);
* vector math shared with hybrid retrieval;
* failures logged, not silently swallowed.

Cross-document scan design:
* similarity gate: cosine ≥ SIMILARITY_THRESHOLD on chunk pairs (numpy
  matrix when available, per-pair fallback otherwise);
* one LLM judgment per DOCUMENT pair — the most-similar passing chunk pair
  represents the pair (bounded LLM cost for large corpora); conflicts are
  persisted canonicalized (smaller hash first) so a pair is never stored
  twice under swapped orderings.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.core.database import add_conflict, get_connection, init_db
from backend.services.vector_math import pairwise_similarities

LOG = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.75  # cosine similarity threshold for candidate pairs

_NEGATION_MARKERS = ("אין סתירה", "אינם סותרים", "אינן סותרות", "אינה סתירה", "אין התנגשות", "אין צורך")


def _judgment_is_contradiction(txt: str) -> bool:
    """Pure heuristic over the LLM's yes/no + summary answer.

    The model may answer "לא, אין סתירה ... כן הקטעים עוסקים באותו נושא"
    or similar — a bare substring match on "כן"/"סתירה" misclassifies the
    negation. A judgment counts as a contradiction only when no negation
    marker appears and an affirmative marker is present.
    """
    lowered = txt.lower()
    if any(marker in lowered for marker in _NEGATION_MARKERS):
        return False
    return "יש סתירה" in lowered or "סותרים" in lowered or "yes" in lowered


def detect_contradictions_for_document(
    chunks: list[dict],
    existing_chunks: list[dict],
    ollama_client=None,
    db_path=None,
    new_vectors: dict[int, list[float]] | None = None,
    existing_vectors: dict[int, list[float]] | None = None,
) -> list[dict]:
    """Compare new chunks vs existing. Requires BOTH similarity and LLM confirmation.

    ``new_vectors``/``existing_vectors`` may carry precomputed embeddings
    (index → vector) so a full-corpus scan can embed every chunk exactly once;
    when omitted the vectors are embedded here (backward compatible).
    """
    client = None
    try:
        from backend.services.ollama_client import get_ollama_client

        client = ollama_client or get_ollama_client()
    except Exception:
        LOG.warning("LLM/embedding client unavailable — heuristic gate only", exc_info=True)

    new_vecs: dict[int, list[float]] = dict(new_vectors) if new_vectors else {}
    existing_vecs: dict[int, list[float]] = dict(existing_vectors) if existing_vectors else {}
    matrix: list[list[float]] | None = None
    if len(new_vecs) == len(chunks) and len(existing_vecs) == len(existing_chunks) and chunks and existing_chunks:
        matrix = pairwise_similarities(
            [new_vecs[i] for i in range(len(chunks))],
            [existing_vecs[i] for i in range(len(existing_chunks))],
        )
    if matrix is None and client is not None and chunks and existing_chunks:
        try:
            vecs = client.embed([c["text"] for c in chunks])
            new_vecs = dict(enumerate(vecs))
            vecs = client.embed([c["text"] for c in existing_chunks])
            existing_vecs = dict(enumerate(vecs))
            matrix = pairwise_similarities(
                [new_vecs[i] for i in range(len(chunks))],
                [existing_vecs[i] for i in range(len(existing_chunks))],
            )
        except Exception:
            LOG.warning("batch embedding failed — falling back to lexical heuristics", exc_info=True)
            new_vecs, existing_vecs = {}, {}

    # ---- candidate pairs, grouped per document pair ----
    candidates: list[tuple[float, int, int]] = []  # (sim, ni, ei)
    for ni, nc in enumerate(chunks):
        for ei, ec in enumerate(existing_chunks):
            if matrix is not None:
                sim = matrix[ni][ei]
            elif client is None:
                # lexical fallback heuristic (no vectors, no LLM)
                sim = 0.8 if nc["text"][:30].lower() == ec["text"][:30].lower() else 0.1
            else:
                sim = 0.0
            if sim >= SIMILARITY_THRESHOLD:
                candidates.append((sim, ni, ei))

    # best chunk-pair per document pair (deterministic: sim desc, index asc)
    best_per_pair: dict[tuple[str, str], tuple[float, int, int]] = {}
    for sim, ni, ei in sorted(candidates, key=lambda x: (-x[0], x[1], x[2])):
        nc, ec = chunks[ni], existing_chunks[ei]
        h1, h2 = str(nc["file_hash"]), str(ec["file_hash"])
        if h1 == h2:
            continue
        key = (h1, h2) if h1 <= h2 else (h2, h1)
        if key not in best_per_pair:
            best_per_pair[key] = (sim, ni, ei)

    conflicts: list[dict[str, Any]] = []
    for pair, (sim, ni, ei) in best_per_pair.items():
        nc, ec = chunks[ni], existing_chunks[ei]
        is_contradiction = False
        topic = ""
        if client is not None:
            try:
                prompt = f"האם יש סתירה בין שני הקטעים הבאים? ענה כן/לא וסכם נושא בעברית:\n1) {nc['text'][:500]}\n2) {ec['text'][:500]}"
                res = client.generate(prompt, temperature=0.0, timeout_s=15)
                txt = res.text.strip()
                is_contradiction = _judgment_is_contradiction(txt)
                topic = txt[:200] if is_contradiction else ""
            except Exception:
                LOG.warning("LLM contradiction judgment failed", exc_info=True)
                is_contradiction = False
        else:
            # without LLM, require high similarity + differing numbers as heuristic
            is_contradiction = sim > 0.85 and nc["text"] != ec["text"]
            topic = f"סתירה אפשרית בין {nc.get('section_number','')} ל-{ec.get('section_number','')}"

        if is_contradiction:
            conflicts.append(
                {"pair": pair, "new_chunk": nc, "existing_chunk": ec, "similarity": sim, "topic_summary": topic or "סתירה בין מסמכים"}
            )

    # persist to DB (canonical order: smaller hash first, from the pair key)
    if conflicts:
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            for c in conflicts:
                add_conflict(
                    conn,
                    new_doc_hash=c["pair"][0],
                    other_doc_hash=c["pair"][1],
                    topic_summary=c["topic_summary"],
                )
        finally:
            conn.close()
    return conflicts
