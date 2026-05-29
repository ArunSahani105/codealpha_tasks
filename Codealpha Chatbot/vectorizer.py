"""TF-IDF vectorizer wrapper for FAQ questions."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class FaqVectorizer:
    """Fit a TF-IDF vectorizer on preprocessed FAQ questions."""

    def __init__(self, ngram_range=(1, 2), min_df: int = 1) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.matrix: csr_matrix | None = None

    def fit(self, documents: Iterable[str]) -> "FaqVectorizer":
        docs: List[str] = [d if d else " " for d in documents]
        self.matrix = self.vectorizer.fit_transform(docs)
        return self

    def transform(self, documents: Iterable[str]) -> csr_matrix:
        if self.matrix is None:
            raise RuntimeError("Vectorizer must be fitted before transform().")
        docs = [d if d else " " for d in documents]
        return self.vectorizer.transform(docs)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_) if self.matrix is not None else 0

    def to_dense(self) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("Vectorizer must be fitted first.")
        return self.matrix.toarray()


__all__ = ["FaqVectorizer"]