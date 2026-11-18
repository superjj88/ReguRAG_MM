"""Shared vector math utilities (single cosine implementation)."""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length vectors; 0.0 on zero vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def pairwise_similarities(
    new_vecs: list[list[float]], existing_vecs: list[list[float]]
) -> list[list[float]] | None:
    """Full n×m cosine matrix (rows=new, cols=existing); None without numpy.

    Same math as cosine_similarity, vectorized for large cross-document
    scans (hundreds of chunk pairs)."""
    try:
        import numpy as np
    except ImportError:
        return None
    if not new_vecs or not existing_vecs:
        return None
    a = np.asarray(new_vecs, dtype=np.float64)
    b = np.asarray(existing_vecs, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return (a @ b.T).tolist()
