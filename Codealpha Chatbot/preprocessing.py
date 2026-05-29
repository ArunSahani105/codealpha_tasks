"""
NLP preprocessing utilities for the FAQ Chatbot.

Pipeline:
    lowercasing -> punctuation/regex cleaning -> tokenization
    -> stopword removal -> lemmatization

Uses NLTK for tokenization/stopwords and spaCy for lemmatization
(falls back to NLTK's WordNetLemmatizer if spaCy model is unavailable).
"""

from __future__ import annotations

import re
import string
from typing import List

import nltk


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources on first use."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


_ensure_nltk_resources()

from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

# Try to load spaCy for high-quality lemmatization.
_NLP = None
try:
    import spacy

    try:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        # Model not installed; we'll fall back to WordNet lemmatizer.
        _NLP = None
except ImportError:
    _NLP = None

_WN_LEMMATIZER = WordNetLemmatizer()

try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/numbers/punctuation, and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    """Split a cleaned string into tokens using NLTK."""
    try:
        return word_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common English stopwords."""
    return [t for t in tokens if t and t not in _STOPWORDS]


def lemmatize_text(tokens: List[str]) -> List[str]:
    """Lemmatize tokens with spaCy (preferred) or WordNet fallback."""
    if not tokens:
        return []
    if _NLP is not None:
        doc = _NLP(" ".join(tokens))
        return [t.lemma_ for t in doc if t.lemma_.strip()]
    return [_WN_LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline returning a single normalized string."""
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    return " ".join(tokens)


__all__ = [
    "clean_text",
    "tokenize_text",
    "remove_stopwords",
    "lemmatize_text",
    "preprocess_text",
]