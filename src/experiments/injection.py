"""Controlled Hallucination Test: inject one known-false claim into a *copy* of an answer.

The original :class:`~src.schemas.Answer` is frozen and never changed. The test copy is a
new ``Answer`` with ``source="controlled-test"`` so it can never be mistaken for model
output. Both copies go through the same extraction and verification pipeline, and the
verdicts for the injected claim are reported as they come out (detection is not assumed).

False claims come from:
* manually reviewed examples in ``data/controlled_examples.json`` (inserted into the copy), or
* a sentence of the answer that the user edits, optionally starting from a rule-based
  suggestion (date / number / negation change), which the user reviews before running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.config import DATA_DIR
from src.experiments.mutation import ReferenceItem, RuleBasedMutator
from src.pipeline import HallucinationPipeline
from src.schemas import Answer, ClaimResult, Label, PipelineResult, Sentence

EXAMPLES_PATH = DATA_DIR / "controlled_examples.json"


class InjectionError(ValueError):
    """Invalid injection request (unknown sentence, unchanged text, empty claim)."""


@dataclass(frozen=True)
class ReviewedExample:
    id: str
    topic: str
    true_claim: str
    false_claim: str
    transformation: str
    reference_url: str


@dataclass(frozen=True)
class Injection:
    """Record of exactly what was changed in the test copy."""

    original_answer: str
    modified_answer: str
    mode: str  # "insert" or "substitute"
    injected_text: str
    replaced_text: str  # "" for insert
    target_sentence_id: str | None
    transformation: str
    origin: str  # e.g. "reviewed example 'malaria'" or "user-edited sentence"
    injected_start: int
    injected_end: int


@dataclass
class ControlledTestResult:
    injection: Injection
    original: PipelineResult
    modified: PipelineResult
    injected_claims: list[ClaimResult] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        """True if at least one injected claim was labelled CONTRADICTED."""
        return any(c.verdict.label is Label.CONTRADICTED for c in self.injected_claims)


def load_reviewed_examples(path: Path = EXAMPLES_PATH) -> list[ReviewedExample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ReviewedExample(e["id"], e["topic"], e["true_claim"], e["false_claim"],
                        e["transformation"], e.get("reference_url", ""))
        for e in data["examples"]
    ]


def insert_claim(answer: Answer, claim: str, transformation: str, origin: str,
                 after_sentence: Sentence | None = None) -> Injection:
    """Insert ``claim`` after ``after_sentence`` (or at the end) of a copy of the answer."""
    claim = claim.strip()
    if not claim:
        raise InjectionError("The claim to inject is empty.")
    text = answer.text
    pos = after_sentence.end if after_sentence else len(text.rstrip())
    prefix = text[:pos]
    sep = "" if not prefix or prefix.endswith((" ", "\n")) else " "
    start = len(prefix) + len(sep)
    modified = prefix + sep + claim + ("" if text[pos:].startswith((" ", "\n")) or not text[pos:] else " ") + text[pos:]
    return Injection(text, modified, "insert", claim, "", None, transformation, origin,
                     start, start + len(claim))


def substitute_sentence(answer: Answer, sentence: Sentence, new_text: str, transformation: str,
                        origin: str) -> Injection:
    """Replace one sentence in a copy of the answer with ``new_text``."""
    new_text = new_text.strip()
    if not new_text:
        raise InjectionError("The replacement sentence is empty.")
    if answer.text[sentence.start:sentence.end] != sentence.text:
        raise InjectionError("Sentence span does not match the answer.")
    if " ".join(new_text.split()) == " ".join(sentence.text.split()):
        raise InjectionError("The replacement is identical to the original sentence.")
    modified = answer.text[:sentence.start] + new_text + answer.text[sentence.end:]
    return Injection(answer.text, modified, "substitute", new_text, sentence.text, sentence.id,
                     transformation, origin, sentence.start, sentence.start + len(new_text))


def suggest_replacements(sentence: str, seed: int = 13) -> list[tuple[str, str]]:
    """Rule-based starting points (date, number, negation) for the user to review/edit."""
    item = ReferenceItem(sentence, sentence)
    return [
        (c.mutated_claim, f"{c.mutation_type.value}: '{c.replaced}' -> '{c.replacement.strip()}'"
         if c.replaced else f"{c.mutation_type.value}: inserted '{c.replacement.strip()}'")
        for c in RuleBasedMutator(seed).mutate(item)
    ]


def run_controlled_test(
    pipeline: HallucinationPipeline,
    answer: Answer,
    injection: Injection,
    original_result: PipelineResult | None = None,
) -> ControlledTestResult:
    """Verify the original (if not already done) and the modified copy with one pipeline."""
    if injection.original_answer != answer.text:
        raise InjectionError("Injection was built from a different answer.")
    original = original_result if original_result is not None else pipeline.analyze(answer)
    test_copy = Answer(answer.question, injection.modified_answer, answer.model, source="controlled-test",
                       provider=answer.provider)
    modified = pipeline.analyze(test_copy)
    injected = [
        c for c in modified.claims
        if c.claim.span_start is not None
        and c.claim.span_start < injection.injected_end and injection.injected_start < c.claim.span_end
    ]
    return ControlledTestResult(injection, original, modified, injected)
