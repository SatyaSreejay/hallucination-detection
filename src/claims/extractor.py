"""Candidate-claim extraction from an answer.

Default (``spacy``): spaCy's rule-based Sentencizer splits the answer into sentences.
Every sentence gets an id ("S1", "S2", ...) and its character span, in the original
order. Sentences that are not factual assertions (questions, greetings, opinions, advice,
fragments) are kept but marked with a ``skip_reason`` and not verified. Sentences that
probably contain several assertions are **flagged**, not decomposed: sentence splitting is
not atomic-fact extraction.

Optional (``llm``): the same sentences are decomposed into sub-claims by an LLM JSON
prompt ("S2.1", "S2.2"). Output is validated; sub-claims that cannot be traced back to a
sentence of the answer are dropped with a warning.

Extraction never verifies anything.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.llm.base import LLMClient, LLMError, parse_json_object
from src.schemas import Claim, Sentence

logger = logging.getLogger(__name__)


class ClaimExtractionError(RuntimeError):
    """Raised when claims cannot be extracted (bad input or malformed LLM JSON)."""


@dataclass
class ExtractionResult:
    claims: list[Claim]
    method: str
    warnings: list[str] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)


# ------------------------------------------------------------- sentence split
@lru_cache(maxsize=1)
def _nlp():
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into ``(sentence, start, end)`` with spaCy's Sentencizer.

    Each line is processed separately so list items without final punctuation do not
    merge; bullet markers are excluded from the span.
    """
    spans: list[tuple[str, int, int]] = []
    for line in re.finditer(r"[^\n]+", text):
        offset = line.start()
        body = line.group(0)
        bullet = _BULLET_RE.match(body)
        if bullet:
            offset += bullet.end()
            body = body[bullet.end():]
        for sent in _nlp()(body).sents:
            stripped = sent.text.strip()
            if stripped:
                start = offset + sent.start_char + (len(sent.text) - len(sent.text.lstrip()))
                spans.append((stripped, start, start + len(stripped)))
    return spans


_SKIP_PATTERNS: list[tuple[str, str]] = [
    (r"^(hi|hello|hey|sure|certainly|of course|absolutely|great question|good question)\b", "greeting"),
    (r"^(i hope|hope this|let me know|feel free|happy to help|thanks|thank you)\b", "meta comment"),
    (r"^(here (is|are)|below (is|are)|in summary|to summari[sz]e)\b[^.]*:$", "meta comment"),
    (r"\b(i think|i believe|i feel|in my opinion|in my view|personally|i would say)\b", "opinion"),
    (r"\b(you should|you might|you may want|i recommend|we recommend)\b", "advice"),
    (r"^(as an ai|i am an ai|i'm an ai|i don't have|i do not have|i cannot|i can't)\b", "meta comment"),
]
_SKIP_RE = [(re.compile(p, re.IGNORECASE), reason) for p, reason in _SKIP_PATTERNS]

_MULTI_PATTERNS: list[tuple[str, str]] = [
    (r";", "semicolon joins clauses"),
    (r",\s+(and|but|while|whereas|although)\s+\w+", "coordinated clauses"),
    (r",\s+(which|who|where)\s+", "relative clause"),
    (r"\(", "parenthetical"),
]
_MULTI_RE = [(re.compile(p, re.IGNORECASE), reason) for p, reason in _MULTI_PATTERNS]
_VALUE_RE = re.compile(r"\b\d[\d,.]*\b")


def skip_reason(sentence: str) -> str | None:
    """Why a sentence is not treated as a factual claim, or ``None`` if it is one."""
    s = sentence.strip()
    if s.endswith("?"):
        return "question"
    if len(s.split()) < 4:
        return "fragment (fewer than 4 words)"
    for pattern, reason in _SKIP_RE:
        if pattern.search(s):
            return reason
    return None


def multi_assertion_flags(sentence: str) -> tuple[str, ...]:
    """Heuristic signs that a sentence makes more than one factual assertion."""
    flags = [reason for pattern, reason in _MULTI_RE if pattern.search(sentence)]
    if len(set(_VALUE_RE.findall(sentence))) >= 2:
        flags.append("several numbers/dates")
    return tuple(dict.fromkeys(flags))


def segment_answer(answer: str) -> list[Sentence]:
    """All sentences of ``answer`` with ids, spans, skip reasons and multi-assertion flags."""
    out = []
    for i, (text, start, end) in enumerate(split_sentences(answer), start=1):
        flags = multi_assertion_flags(text)
        out.append(Sentence(f"S{i}", text, start, end, skip_reason(text), bool(flags), flags))
    return out


def _check_answer(answer: object) -> str:
    if not isinstance(answer, str):
        raise ClaimExtractionError(f"Answer must be text, got {type(answer).__name__}")
    return answer


# ------------------------------------------------------------ spaCy extractor
class SpacySentenceExtractor:
    """One candidate claim per factual sentence (spaCy Sentencizer)."""

    method = "spacy"

    def __init__(self, max_claims: int = 10) -> None:
        self.max_claims = max_claims

    def extract(self, answer: str) -> ExtractionResult:
        answer = _check_answer(answer)
        sentences = segment_answer(answer)
        warnings: list[str] = []
        if not answer.strip():
            warnings.append("The answer is empty, so there is nothing to verify.")
        claims = [
            Claim(s.id, s.text, s.text, s.start, s.end, self.method, s.multi_assertion, s.flags)
            for s in sentences
            if s.skip_reason is None
        ]
        if any(c.multi_assertion for c in claims):
            warnings.append(
                "Some sentences may contain several assertions (flagged). They are verified as a "
                "whole; a sentence is only SUPPORTED if the evidence supports all of it."
            )
        if len(claims) > self.max_claims:
            warnings.append(
                f"{len(claims)} candidate claims; only the first {self.max_claims} are verified (MAX_CLAIMS)."
            )
            claims = claims[: self.max_claims]
        return ExtractionResult(claims, self.method, warnings, sentences)


# ----------------------------------------------------------------- LLM extractor
EXTRACTION_SYSTEM = (
    "You decompose sentences into atomic factual claims. You never add facts that are not "
    "in the text and you never judge whether claims are true."
)
EXTRACTION_PROMPT = """Decompose each numbered SENTENCE below into atomic factual claims.

Rules:
- Each claim states exactly one checkable fact (who/what/when/where/how many).
- A sentence may yield one or several claims.
- Make each claim self-contained: replace pronouns with the entity they refer to.
- Do not add information that is not in the sentence.

Return only JSON of the form:
{{"claims": [{{"sentence_id": "S1", "claim": "..."}}]}}

SENTENCES:
{sentences}"""


class LLMClaimExtractor:
    """spaCy sentences decomposed into sub-claims by the LLM, with validation."""

    method = "llm"

    def __init__(self, client: LLMClient, max_claims: int = 10, seed: int = 13) -> None:
        self.client = client
        self.max_claims = max_claims
        self.seed = seed

    def extract(self, answer: str) -> ExtractionResult:
        answer = _check_answer(answer)
        sentences = segment_answer(answer)
        candidates = [s for s in sentences if s.skip_reason is None]
        if not candidates:
            return ExtractionResult([], self.method, ["No factual sentences to decompose."], sentences)
        listing = "\n".join(f"{s.id}: {s.text}" for s in candidates)
        raw = self.client.generate(
            EXTRACTION_PROMPT.format(sentences=listing),
            system=EXTRACTION_SYSTEM,
            json_mode=True,
            temperature=0.0,
            seed=self.seed,
        )
        return self.parse(raw, sentences)

    def parse(self, raw: str, sentences: list[Sentence]) -> ExtractionResult:
        """Validate LLM output and attach each sub-claim to its source sentence."""
        try:
            data = parse_json_object(raw)
        except ValueError as exc:
            raise ClaimExtractionError(f"Claim extractor returned invalid JSON: {exc}") from exc
        items = data.get("claims")
        if not isinstance(items, list):
            raise ClaimExtractionError("Claim extractor JSON has no 'claims' list")

        by_id = {s.id: s for s in sentences if s.skip_reason is None}
        counters: dict[str, int] = {}
        claims: list[Claim] = []
        warnings: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("claim"), str):
                warnings.append(f"Skipped malformed claim entry: {str(item)[:80]}")
                continue
            text = " ".join(item["claim"].split())
            if len(text.split()) < 3 or text.lower() in seen:
                continue
            sent = by_id.get(str(item.get("sentence_id", "")).strip())
            if sent is None or not _grounded(text, sent.text):
                sent = _best_sentence(text, list(by_id.values()))
            if sent is None:
                warnings.append(
                    f"Dropped claim not traceable to the answer (possible extraction hallucination): '{text[:80]}'"
                )
                continue
            seen.add(text.lower())
            counters[sent.id] = counters.get(sent.id, 0) + 1
            claims.append(
                Claim(f"{sent.id}.{counters[sent.id]}", text, sent.text, sent.start, sent.end, self.method)
            )
        claims.sort(key=lambda c: (int(c.id[1:].split(".")[0]), int(c.id.split(".")[1])))
        if len(claims) > self.max_claims:
            warnings.append(f"Only the first {self.max_claims} of {len(claims)} claims are verified.")
            claims = claims[: self.max_claims]
        return ExtractionResult(claims, self.method, warnings, sentences)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 or w.isdigit()}


def _grounded(claim: str, sentence: str) -> bool:
    """At least half of the claim's content words appear in the sentence."""
    words = _content_words(claim)
    return not words or len(words & _content_words(sentence)) / len(words) >= 0.5


def _best_sentence(claim: str, sentences: list[Sentence]) -> Sentence | None:
    best, best_score = None, 0.0
    for s in sentences:
        score = difflib.SequenceMatcher(None, claim.lower(), s.text.lower()).ratio()
        if _grounded(claim, s.text) and score > best_score:
            best, best_score = s, score
    return best


def extract_claims(
    answer: str,
    method: str,
    client: LLMClient | None = None,
    max_claims: int = 10,
    allow_fallback: bool = True,
) -> ExtractionResult:
    """Extract candidate claims with ``method`` ("spacy" or "llm").

    If LLM decomposition fails and ``allow_fallback`` is set, spaCy sentences are used and
    a warning says so (never silent).
    """
    spacy_extractor = SpacySentenceExtractor(max_claims)
    if method != "llm" or client is None:
        return spacy_extractor.extract(answer)
    try:
        return LLMClaimExtractor(client, max_claims).extract(answer)
    except (ClaimExtractionError, LLMError) as exc:
        if not allow_fallback:
            raise
        logger.warning("LLM claim decomposition failed, using spaCy sentences: %s", exc)
        result = spacy_extractor.extract(answer)
        result.warnings.insert(
            0, f"LLM claim decomposition failed ({exc}). spaCy sentences were used instead."
        )
        return result
