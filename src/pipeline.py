"""End-to-end pipeline: question -> answer -> claims -> evidence -> verdicts -> summary."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from src.claims.extractor import ExtractionResult, extract_claims
from src.llm.base import LLMClient
from src.retrieval.passages import subject_related
from src.schemas import (
    Answer,
    ClaimResult,
    Label,
    PipelineResult,
    RetrievalResult,
    RetrievalStatus,
    Verdict,
    summarize,
)
from src.verification.base import Verifier, verdict_without_evidence

if TYPE_CHECKING:
    from src.config import Settings
    from src.verification.factory import VerifierChoice

logger = logging.getLogger(__name__)

ANSWER_SYSTEM = (
    "Answer the user's question factually in one short paragraph (3-6 sentences). "
    "State concrete facts such as names, dates, places and numbers."
)


class Retriever(Protocol):
    def retrieve(self, claim_text: str) -> RetrievalResult: ...


class HallucinationPipeline:
    """Wires the independent components together. Each one can be swapped or mocked."""

    def __init__(
        self,
        client: LLMClient | None,
        retriever: Retriever,
        verifier: Verifier,
        extractor_method: str = "spacy",
        max_claims: int = 10,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.verifier = verifier
        self.extractor_method = extractor_method
        self.max_claims = max_claims

    def generate_answer(self, question: str) -> Answer:
        """Ask the configured LLM. Raises :class:`LLMError` subclasses on failure."""
        if self.client is None:
            raise RuntimeError("No generation client configured")
        text = self.client.generate(question.strip(), system=ANSWER_SYSTEM)
        return Answer(question=question, text=text.strip(), model=self.client.model,
                      provider=getattr(self.client, "provider", ""))

    def extract(self, answer_text: str) -> ExtractionResult:
        return extract_claims(answer_text, self.extractor_method, self.client, self.max_claims)

    def verify_claim_text(
        self, claim_text: str, retriever: Retriever | None = None, topic: str | None = None
    ) -> tuple[RetrievalResult, Verdict]:
        """Retrieve evidence for and verify a single claim given **only its text**.

        This is the detector entry point used by the live app, the controlled experiment
        and dataset evaluation, so none of them can leak labels or metadata into it.
        ``topic`` (the answer's subject, a Wikipedia title) only steers retrieval; it is
        never shown to the verifier.
        """
        retriever = retriever or self.retriever
        open_domain = getattr(retriever, "accepts_topic", False)
        try:
            retrieval = (retriever.retrieve(claim_text, topic=topic) if open_domain
                         else retriever.retrieve(claim_text))
        except Exception as exc:  # noqa: BLE001 - retrieval must never crash the run
            logger.exception("Retriever crashed")
            retrieval = RetrievalResult(claim_text, RetrievalStatus.ERROR, error=str(exc)[:300])
        if retrieval.status is not RetrievalStatus.OK or not retrieval.passages:
            return retrieval, verdict_without_evidence(retrieval, self.verifier.method)
        passages = retrieval.passages
        if open_domain:
            # Search can return pages about other subjects (a claim about the Taj Mahal's
            # UNESCO listing matched "Machu Picchu"). Such passages must not decide the verdict.
            subject = f"{claim_text} {topic or ''}"
            passages = [p for p in retrieval.passages if subject_related(p.title, subject)]
            if not passages:
                titles = ", ".join(dict.fromkeys(p.title for p in retrieval.passages))
                return retrieval, Verdict(
                    Label.NOT_ENOUGH_INFORMATION, self.verifier.method,
                    f"The retrieved pages ({titles}) are about other subjects, so they were not used. "
                    "No relevant evidence was found; this does not mean the claim is false.",
                    raw_scores={"off_subject_pages": titles},
                )
        try:
            verdict = self.verifier.verify(claim_text, passages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Verifier crashed")
            verdict = verdict_without_evidence(retrieval, self.verifier.method)
            verdict.explanation = "The verifier failed on this claim; no verdict was made."
            verdict.error = f"{type(exc).__name__}: {exc}"[:300]
        return retrieval, verdict

    @staticmethod
    def find_topic(answer: Answer, extraction: ExtractionResult, retriever: Retriever) -> str | None:
        """The answer's subject as a Wikipedia title, from the question (or the first claim)."""
        if not getattr(retriever, "accepts_topic", False):
            return None
        question = answer.question.strip()
        text = question if question and question != "(none)" else (
            extraction.claims[0].text if extraction.claims else "")
        return retriever.top_title(text) if text else None

    def analyze(
        self,
        answer: Answer,
        progress: Callable[[int, int, str], None] | None = None,
        retriever: Retriever | None = None,
    ) -> PipelineResult:
        """Extract and verify every claim in ``answer`` (which is never modified)."""
        extraction = self.extract(answer.text)
        topic = self.find_topic(answer, extraction, retriever or self.retriever)
        results: list[ClaimResult] = []
        for i, claim in enumerate(extraction.claims):
            if progress:
                progress(i, len(extraction.claims), claim.text)
            retrieval, verdict = self.verify_claim_text(claim.text, retriever, topic)
            results.append(ClaimResult(claim, retrieval, verdict))
        warnings = list(extraction.warnings)
        if not extraction.claims:
            warnings.append("No verifiable factual claims were found in the answer.")
        return PipelineResult(
            answer=answer,
            claims=results,
            summary=summarize(results),
            extraction_method=extraction.method,
            verifier_method=self.verifier.method,
            warnings=warnings,
            sentences=extraction.sentences,
        )


def build_pipeline(settings: Settings) -> tuple[HallucinationPipeline, VerifierChoice]:
    """Build the live pipeline (generation backend + Wikipedia + verifier) from settings."""
    from src.llm.factory import make_llm_client
    from src.retrieval.wikipedia import WikipediaRetriever
    from src.verification.factory import build_verifier

    client = make_llm_client(settings)
    retriever = WikipediaRetriever(
        settings.wiki_api_url, settings.wiki_user_agent, settings.wiki_timeout_s,
        settings.wiki_max_retries, settings.wiki_search_results,
        settings.wiki_passages_per_claim, settings.wiki_min_interval_s, settings.wiki_cache_dir,
    )
    choice = build_verifier(settings, client)
    pipeline = HallucinationPipeline(
        client, retriever, choice.verifier, settings.extractor, settings.max_claims
    )
    return pipeline, choice
