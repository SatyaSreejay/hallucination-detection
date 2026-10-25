"""Passage chunking and lexical ranking shared by all evidence sources."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but is are was were be been being "
    "it its this that these those as which who whom whose what when where why how has have "
    "had do does did not no than then there their they he she his her him we you your our "
    "i me my also into about over after before during can could would should will may".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case content tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _stems(text: str) -> set[str]:
    # 4-character prefixes: crude, but lets "boils" match "Boiling point"
    return {t[:4] for t in tokenize(text) if len(t) >= 3}


def subject_related(title: str, text: str) -> bool:
    """True if a page title shares a content word with ``text`` (ignoring "(film)" etc.).

    Used to stop passages about a different subject (e.g. "Machu Picchu" for a claim
    about the Taj Mahal) from deciding a verdict.
    """
    stems = _stems(re.sub(r"\(.*?\)", " ", title))
    return not stems or bool(stems & _stems(text))


def chunk_text(text: str, max_words: int = 90) -> list[str]:
    """Split source text into paragraph-based passages of at most ~``max_words`` words.

    Sentences are kept whole, so a passage may slightly exceed the limit.
    """
    passages: list[str] = []
    for para in re.split(r"\n\s*\n|\n(?==+ )|\n", text):
        para = para.strip()
        if not para or para.startswith("=="):  # skip section headings
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", para)
        current: list[str] = []
        count = 0
        for sent in sentences:
            n = len(sent.split())
            if current and count + n > max_words:
                passages.append(" ".join(current))
                current, count = [], 0
            current.append(sent)
            count += n
        if current:
            passages.append(" ".join(current))
    return [p for p in passages if len(p.split()) >= 5]


def rank_passages(query: str, passages: list[str], top_k: int = 3) -> list[tuple[int, float]]:
    """Rank passages by a BM25-style overlap with ``query``.

    Returns ``(index, score)`` pairs, best first, excluding zero-overlap passages.
    Scores are only for ordering; they say nothing about truth.
    """
    q_tokens = set(tokenize(query))
    if not q_tokens or not passages:
        return []
    docs = [tokenize(p) for p in passages]
    n_docs = len(docs)
    avg_len = sum(len(d) for d in docs) / n_docs or 1.0
    df = Counter(t for d in docs for t in set(d))
    k1, b = 1.5, 0.75
    scored: list[tuple[int, float]] = []
    for i, doc in enumerate(docs):
        tf = Counter(doc)
        score = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            idf = math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avg_len))
        if score > 0:
            scored.append((i, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k]
