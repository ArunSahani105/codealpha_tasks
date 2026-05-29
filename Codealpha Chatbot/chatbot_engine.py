"""
Core chatbot engine: loads the FAQ dataset, builds the TF-IDF index,
and answers user queries with confidence scoring + simple analytics.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .preprocessing import preprocess_text
from .similarity import best_match, top_k_matches
from .vectorizer import FaqVectorizer

UNKNOWN_RESPONSE = "Sorry, I could not understand your question."
EMPTY_RESPONSE = "Please type a question so I can help you."


@dataclass
class ChatResult:
    """Structured response returned by the engine."""

    answer: str
    confidence: float
    matched_question: Optional[str]
    timestamp: str
    top_matches: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "matched_question": self.matched_question,
            "timestamp": self.timestamp,
            "top_matches": [
                {"question": q, "score": round(s, 4)} for q, s in self.top_matches
            ],
        }


class FAQChatbot:
    """TF-IDF + cosine similarity FAQ chatbot."""

    def __init__(
        self,
        dataset_path: str,
        threshold: float = 0.30,
        log_path: Optional[str] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.threshold = threshold
        self.log_path = log_path
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.processed: List[str] = []
        self.vectorizer = FaqVectorizer()
        self._query_counter: Counter = Counter()
        self._load_dataset()
        self._build_index()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _load_dataset(self) -> None:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"FAQ dataset not found: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        cols = {c.lower(): c for c in df.columns}
        if "question" not in cols or "answer" not in cols:
            raise ValueError("CSV must contain 'question' and 'answer' columns.")
        df = df.dropna(subset=[cols["question"], cols["answer"]])
        self.questions = df[cols["question"]].astype(str).tolist()
        self.answers = df[cols["answer"]].astype(str).tolist()

    def _build_index(self) -> None:
        self.processed = [preprocess_text(q) for q in self.questions]
        self.vectorizer.fit(self.processed)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def ask(self, query: str) -> ChatResult:
        """Return a ChatResult for the given user query."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if query is None or not query.strip():
            return ChatResult(EMPTY_RESPONSE, 0.0, None, ts)

        processed = preprocess_text(query)
        if not processed:
            return ChatResult(UNKNOWN_RESPONSE, 0.0, None, ts)

        query_vec = self.vectorizer.transform([processed])
        idx, score = best_match(query_vec, self.vectorizer.matrix)
        top = top_k_matches(query_vec, self.vectorizer.matrix, k=3)
        top_pairs = [(self.questions[i], s) for i, s in top if i >= 0]

        if idx == -1 or score < self.threshold:
            result = ChatResult(UNKNOWN_RESPONSE, score, None, ts, top_pairs)
        else:
            result = ChatResult(
                answer=self.answers[idx],
                confidence=score,
                matched_question=self.questions[idx],
                timestamp=ts,
                top_matches=top_pairs,
            )

        self._record(query, result)
        return result

    # ------------------------------------------------------------------ #
    # Analytics + logging
    # ------------------------------------------------------------------ #
    def _record(self, query: str, result: ChatResult) -> None:
        self._query_counter[query.strip().lower()] += 1
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["timestamp", "query", "matched_question", "confidence", "answer"]
                )
            writer.writerow(
                [
                    result.timestamp,
                    query,
                    result.matched_question or "",
                    f"{result.confidence:.4f}",
                    result.answer,
                ]
            )

    def analytics(self, top_n: int = 5) -> Dict:
        most_common = self._query_counter.most_common(top_n)
        return {
            "total_queries": sum(self._query_counter.values()),
            "unique_queries": len(self._query_counter),
            "most_asked": [{"query": q, "count": c} for q, c in most_common],
        }

    def analytics_json(self, top_n: int = 5) -> str:
        return json.dumps(self.analytics(top_n), indent=2)


__all__ = ["FAQChatbot", "ChatResult", "UNKNOWN_RESPONSE"]