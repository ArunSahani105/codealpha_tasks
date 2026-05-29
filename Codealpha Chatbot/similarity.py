"""Cosine-similarity based best-match finder."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def best_match(query_vec: csr_matrix, faq_matrix: csr_matrix) -> Tuple[int, float]:
    """Return (index, score) of the FAQ most similar to the query."""
    if faq_matrix.shape[0] == 0:
        return -1, 0.0
    sims = cosine_similarity(query_vec, faq_matrix).flatten()
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


def top_k_matches(query_vec: csr_matrix, faq_matrix: csr_matrix, k: int = 3):
    """Return the top-k (index, score) matches sorted by similarity."""
    sims = cosine_similarity(query_vec, faq_matrix).flatten()
    k = min(k, len(sims))
    idxs = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idxs]


__all__ = ["best_match", "top_k_matches"]