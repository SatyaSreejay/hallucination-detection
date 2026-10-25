"""Evidence "retrieval" from a supplied source document (used for RAGTruth).

In RAG settings the reference is the context the model was given, not Wikipedia, so the
same verifier is run against passages of that context instead.
"""

from __future__ import annotations

from src.retrieval.passages import chunk_text, rank_passages
from src.schemas import EvidencePassage, RetrievalResult, RetrievalStatus


class ContextRetriever:
    """Rank passages of a fixed context document against each claim."""

    source_name = "provided-context"

    def __init__(self, context: str, title: str = "Source context", url: str = "", top_k: int = 3):
        self.title = title
        self.url = url
        self.top_k = top_k
        self.passages = chunk_text(context, max_words=80)

    def retrieve(self, claim_text: str) -> RetrievalResult:
        if not self.passages:
            return RetrievalResult(claim_text, RetrievalStatus.ERROR, error="Context is empty")
        ranked = rank_passages(claim_text, self.passages, self.top_k)
        passages = [
            EvidencePassage(
                self.title, self.url, self.passages[i], self.source_name, None, i, round(s, 4)
            )
            for i, s in ranked
        ]
        status = RetrievalStatus.OK if passages else RetrievalStatus.NO_RESULTS
        return RetrievalResult(claim_text, status, passages)
