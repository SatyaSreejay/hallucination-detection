"""Verifier interface and shared helpers.

A verifier only ever receives the claim text and evidence passages. It never sees
expected labels, mutation metadata or dataset annotations; this is what keeps the
controlled experiment and dataset evaluations independent.
"""

from __future__ import annotations

from typing import Protocol

from src.schemas import EvidencePassage, Label, RetrievalResult, RetrievalStatus, Verdict


class Verifier(Protocol):
    """Anything that maps (claim text, evidence passages) to a :class:`Verdict`."""

    @property
    def method(self) -> str: ...

    def verify(self, claim_text: str, passages: list[EvidencePassage]) -> Verdict: ...


class VerifierUnavailableError(RuntimeError):
    """The requested verifier cannot be used (missing package, weights, or server)."""


def verdict_without_evidence(retrieval: RetrievalResult, method: str) -> Verdict:
    """NOT_ENOUGH_INFORMATION verdict for a claim with no usable evidence."""
    if retrieval.status is RetrievalStatus.ERROR:
        explanation = (
            f"Evidence retrieval failed ({retrieval.error}). Nothing can be concluded about "
            "this claim."
        )
    else:
        explanation = (
            "No relevant evidence was found. Absence of a search result is not evidence that "
            "the claim is false."
        )
    return Verdict(Label.NOT_ENOUGH_INFORMATION, method, explanation)


def format_premise(passage: EvidencePassage) -> str:
    """Premise text for a passage; the page title helps resolve pronouns."""
    return f"{passage.title}: {passage.text}" if passage.title else passage.text
