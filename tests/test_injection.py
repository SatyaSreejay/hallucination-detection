"""Controlled Hallucination Test: original stays unchanged, copy is labelled, results are real."""

import dataclasses

import pytest

from src.experiments.injection import (
    InjectionError,
    insert_claim,
    load_reviewed_examples,
    run_controlled_test,
    substitute_sentence,
    suggest_replacements,
)
from src.pipeline import HallucinationPipeline
from src.schemas import Answer, Label, Outcome
from tests.conftest import FakeRetriever, KeywordVerifier

ANSWER = Answer("q", "The Eiffel Tower is in Paris. It was completed in 1889.", "llama3.2")
EVIDENCE = {
    "The Eiffel Tower is in Paris.": "the eiffel tower is in paris.",
    "It was completed in 1889.": "it was completed in 1889.",
    "It was completed in 1925.": "it was completed in 1889.",
    "Malaria is caused by bacteria.": "malaria is caused by plasmodium parasites.",
}


def pipeline():
    return HallucinationPipeline(None, FakeRetriever(EVIDENCE), KeywordVerifier(), "spacy")


def test_reviewed_examples_load_and_are_distinct():
    examples = load_reviewed_examples()
    assert len(examples) >= 5
    for e in examples:
        assert e.true_claim != e.false_claim and e.transformation and e.reference_url.startswith("https://")


def test_insert_keeps_original_and_records_transformation():
    inj = insert_claim(ANSWER, "Malaria is caused by bacteria.", "cause changed", "reviewed example 'malaria'")
    assert ANSWER.text == "The Eiffel Tower is in Paris. It was completed in 1889."
    assert inj.original_answer == ANSWER.text
    assert inj.modified_answer == ANSWER.text + " Malaria is caused by bacteria."
    assert inj.modified_answer[inj.injected_start:inj.injected_end] == "Malaria is caused by bacteria."
    assert inj.mode == "insert" and inj.transformation == "cause changed"


def test_substitute_and_validation_errors():
    result = pipeline().analyze(ANSWER)
    s2 = result.sentences[1]
    inj = substitute_sentence(ANSWER, s2, "It was completed in 1925.", "date 1889 -> 1925", "user edit")
    assert inj.modified_answer == "The Eiffel Tower is in Paris. It was completed in 1925."
    assert inj.replaced_text == "It was completed in 1889." and inj.target_sentence_id == "S2"
    with pytest.raises(InjectionError):
        substitute_sentence(ANSWER, s2, "It was completed in 1889.", "none", "user edit")
    with pytest.raises(InjectionError):
        insert_claim(ANSWER, "  ", "x", "y")


def test_controlled_test_runs_both_answers_and_reports_actual_verdict():
    p = pipeline()
    original = p.analyze(ANSWER)
    inj = substitute_sentence(ANSWER, original.sentences[1], "It was completed in 1925.", "date", "user edit")
    res = run_controlled_test(p, ANSWER, inj, original)
    assert res.original is original and res.original.answer.text == ANSWER.text
    assert res.modified.answer.source == "controlled-test"  # never presented as model output
    assert [c.claim.text for c in res.injected_claims] == ["It was completed in 1925."]
    assert res.injected_claims[0].outcome is Outcome.CONTRADICTED and res.detected
    assert [c.verdict.label for c in res.original.claims] == [Label.SUPPORTED, Label.SUPPORTED]
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.modified.answer.text = "x"


def test_undetected_injection_is_reported_as_such():
    p = HallucinationPipeline(None, FakeRetriever(EVIDENCE), KeywordVerifier(), "spacy")
    inj = insert_claim(ANSWER, "Zorblax is a famous scientist from Mars.", "invented", "user")
    res = run_controlled_test(p, ANSWER, inj)
    assert res.injected_claims and not res.detected
    assert res.injected_claims[0].outcome is Outcome.INSUFFICIENT_EVIDENCE


def test_suggestions_are_reviewable_rule_edits():
    sugg = suggest_replacements("Neil Armstrong walked on the Moon in 1969.", seed=1)
    texts = [t for t, _ in sugg]
    assert any("1969" not in t for t in texts)
    assert all(d for _, d in sugg)
