"""Controlled mutation experiment: generation, validation, label hiding, metrics."""

import inspect
import json

from src.experiments.mutation import (
    LLMMutator,
    MutationCandidate,
    MutationType,
    MutationValidator,
    ReferenceItem,
    RuleBasedMutator,
    ValidationStatus,
    build_cases,
    generate_candidates,
    run_experiment,
)
from src.schemas import EvidencePassage, Label, RetrievalResult, RetrievalStatus, Verdict
from src.verification.base import Verifier
from tests.conftest import FakeOllama, KeywordVerifier

ITEM = ReferenceItem(
    "The Eiffel Tower was completed in 1889 and is 330 metres tall.",
    "The Eiffel Tower in Paris was completed in 1889 for the World's Fair. It is 330 metres tall "
    "and was designed by the company of Gustave Eiffel.",
    "Eiffel Tower",
)


def test_rule_mutator_generates_generic_mutations_deterministically():
    a = RuleBasedMutator(seed=1, entity_swaps=True).mutate(ITEM)
    b = RuleBasedMutator(seed=1, entity_swaps=True).mutate(ITEM)
    assert [c.mutated_claim for c in a] == [c.mutated_claim for c in b]
    types = {c.mutation_type for c in a}
    assert {MutationType.DATE, MutationType.NUMBER, MutationType.ENTITY, MutationType.NEGATION} <= types
    for c in a:
        assert c.original_claim == ITEM.claim and c.mutated_claim != ITEM.claim
        assert c.reference_evidence == ITEM.evidence
        assert c.expected_label is Label.CONTRADICTED
        assert c.validation_status is ValidationStatus.UNVERIFIED  # not assumed false
    entity = next(c for c in a if c.mutation_type is MutationType.ENTITY)
    assert entity.replacement in ITEM.evidence  # replacement drawn from evidence, not hardcoded


def test_rule_mutator_works_on_other_claims_without_specific_rules():
    item = ReferenceItem("Marie Curie won 2 Nobel Prizes.",
                         "Marie Curie was a physicist from Warsaw. She won 2 Nobel Prizes.")
    muts = RuleBasedMutator(seed=3, entity_swaps=True).mutate(item)
    assert any(m.mutation_type is MutationType.NUMBER for m in muts)
    assert any(m.mutation_type is MutationType.ENTITY and "Warsaw" in m.mutated_claim for m in muts)


def test_entity_swaps_are_off_by_default():
    assert all(m.mutation_type is not MutationType.ENTITY for m in RuleBasedMutator(seed=1).mutate(ITEM))


def cand(mutated, mtype=MutationType.DATE):
    return MutationCandidate(ITEM.claim, mutated, mtype, "rule", ITEM.evidence)


def test_validator_deterministic_value_check():
    v = MutationValidator()
    ok = v.validate(cand("The Eiffel Tower was completed in 1901 and is 330 metres tall."))
    assert ok.validation_status is ValidationStatus.VALIDATED and "1889" in ok.validation_reason
    # new value present in evidence -> may be true -> rejected
    rej = v.validate(cand("The Eiffel Tower was completed in 330 and is 330 metres tall."))
    assert rej.validation_status is ValidationStatus.REJECTED
    # unchanged -> rejected
    assert v.validate(cand(ITEM.claim)).validation_status is ValidationStatus.REJECTED
    # entity swap has no deterministic check and no verifier -> unverified, excluded
    un = v.validate(cand("The Gustave Eiffel was completed in 1889 and is 330 metres tall.",
                         MutationType.ENTITY))
    assert un.validation_status is ValidationStatus.UNVERIFIED


def test_validator_with_verifier_rejects_supported_mutation():
    class AlwaysSupported:
        method = "x"

        def verify(self, claim_text, passages):
            return Verdict(Label.SUPPORTED, "x", "")
    mutated = cand("The Eiffel Tower was completed in 1901 and is 330 metres tall.")
    c = MutationValidator(AlwaysSupported()).validate(mutated)
    assert c.validation_status is ValidationStatus.REJECTED


def test_llm_mutator_output_is_validated_not_trusted():
    raw = json.dumps({"mutations": [
        {"mutated_claim": "The Eiffel Tower was completed in 1889 and is 330 metres tall.", "mutation_type": "date"},
        {"mutated_claim": "The Eiffel Tower was completed in 1925 and is 330 metres tall.", "mutation_type": "date"},
    ]})
    cands, warnings = generate_candidates([ITEM], None, LLMMutator(FakeOllama([raw])), MutationValidator())
    statuses = [c.validation_status for c in cands]
    assert statuses == [ValidationStatus.REJECTED, ValidationStatus.VALIDATED]
    assert all(c.method == "llm" for c in cands)
    assert LLMMutator(FakeOllama(["not json"])).mutate(ITEM) == []


def test_detector_receives_only_claim_text():
    """The detect function and verifiers never see labels or mutation metadata."""
    candidates, _ = generate_candidates([ITEM], RuleBasedMutator(seed=1), None, MutationValidator())
    received = []

    def detect(text):
        received.append(text)
        return RetrievalResult(text, RetrievalStatus.NO_RESULTS), Verdict(Label.NOT_ENOUGH_INFORMATION, "m", "")

    run_experiment([ITEM], candidates, detect)
    assert all(type(x) is str for x in received)
    validated = {c.mutated_claim for c in candidates if c.validation_status is ValidationStatus.VALIDATED}
    assert set(received) == validated | {ITEM.claim}
    forbidden = ["CONTRADICTED", "SUPPORTED", "mutation", "date", "number", "expected"]
    assert not any(f in x for x in received for f in forbidden)
    # the verifier interface itself only accepts claim text and passages
    assert list(inspect.signature(Verifier.verify).parameters) == ["self", "claim_text", "passages"]


def test_cases_exclude_unvalidated_and_are_shuffled_deterministically():
    candidates, _ = generate_candidates([ITEM], RuleBasedMutator(seed=1), None, MutationValidator())
    cases = build_cases([ITEM], candidates, seed=5)
    assert all(c.candidate is None or c.candidate.validation_status is ValidationStatus.VALIDATED for c in cases)
    assert [c.claim_text for c in cases] == [c.claim_text for c in build_cases([ITEM], candidates, seed=5)]


def test_experiment_metrics_from_known_predictions():
    candidates, _ = generate_candidates([ITEM], RuleBasedMutator(seed=1), None, MutationValidator())
    verifier = KeywordVerifier()

    def detect(text):
        retrieval = RetrievalResult(text, RetrievalStatus.OK,
                                    [EvidencePassage("Eiffel Tower", "u", ITEM.evidence)])
        return retrieval, verifier.verify(text, retrieval.passages)

    result = run_experiment([ITEM], candidates, detect, {"validator_same_as_detector": True})
    n_mut = sum(o.kind == "mutation" for o in result.outcomes)
    assert result.report.n == n_mut + 1
    assert result.detection_rate == 1.0
    assert result.report.true_distribution["SUPPORTED"] == 1
    assert any("synthetic errors" in w for w in result.warnings)
    assert any("same as the detector" in w for w in result.warnings)
