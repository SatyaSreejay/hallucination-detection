"""End-to-end pipeline with mocked Ollama, retriever and verifier."""

import dataclasses
import json

import pytest

from src.dataset_eval import evaluate_fever, evaluate_ragtruth
from src.datasets.fever_loader import load_fever_jsonl
from src.datasets.ragtruth_loader import load_ragtruth
from src.llm.ollama_client import OllamaConnectionError, OllamaModelNotFoundError
from src.pipeline import HallucinationPipeline
from src.schemas import Answer, Label, RetrievalStatus
from tests.conftest import FIXTURES, FakeOllama, FakeRetriever, KeywordVerifier

ANSWER = "The Eiffel Tower is in Paris. It was completed in 1925. It hosts a famous bakery."
EVIDENCE = {
    "The Eiffel Tower is in Paris.": "the eiffel tower is in paris, france.",
    "It was completed in 1925.": "it was completed in 1889.",
}


def make(ollama=None, retriever=None):
    return HallucinationPipeline(ollama or FakeOllama([ANSWER]), retriever or FakeRetriever(EVIDENCE),
                                 KeywordVerifier(), extractor_method="spacy")


def test_full_pipeline_with_mocks():
    p = make()
    answer = p.generate_answer("Tell me about the Eiffel Tower")
    assert answer.text == ANSWER and answer.model == "llama3.2"
    result = p.analyze(answer)
    labels = [c.verdict.label for c in result.claims]
    assert labels == [Label.SUPPORTED, Label.CONTRADICTED, Label.NOT_ENOUGH_INFORMATION]
    s = result.summary
    assert (s.total_claims, s.supported, s.contradicted, s.not_enough_information) == (3, 1, 1, 1)
    assert s.retrieval_failures == 0 and s.claims_with_evidence == 2
    assert s.contradicted_share_of_evidenced == 0.5
    assert result.answer.text == ANSWER  # original untouched
    json.dumps(result.to_dict())  # serialisable


def test_answer_is_immutable():
    a = Answer("q", "text", "m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.text = "changed"


def test_retrieval_failures_are_counted_separately():
    p = make(retriever=FakeRetriever(status=RetrievalStatus.ERROR))
    result = p.analyze(Answer("q", ANSWER, "m"))
    assert result.summary.retrieval_failures == 3
    assert result.summary.claims_with_evidence == 0
    assert result.summary.contradicted_share_of_evidenced is None
    assert all(c.verdict.label is Label.NOT_ENOUGH_INFORMATION for c in result.claims)
    assert all("retrieval failed" in c.verdict.explanation for c in result.claims)


def test_retriever_or_verifier_crash_does_not_crash_pipeline():
    class Crashing:
        def retrieve(self, text):
            raise RuntimeError("boom")
    p = make(retriever=Crashing())
    r, v = p.verify_claim_text("Some claim here.")
    assert r.status is RetrievalStatus.ERROR and v.label is Label.NOT_ENOUGH_INFORMATION

    class BadVerifier(KeywordVerifier):
        def verify(self, claim_text, passages):
            raise ValueError("nan")
    p = HallucinationPipeline(None, FakeRetriever(default="x y z"), BadVerifier(), "spacy")
    _, v = p.verify_claim_text("Claim")
    assert v.label is Label.NOT_ENOUGH_INFORMATION and "nan" in v.error


@pytest.mark.parametrize("err", [OllamaConnectionError("down"), OllamaModelNotFoundError("missing")])
def test_generation_errors_propagate_instead_of_fake_answer(err):
    with pytest.raises(type(err)):
        make(ollama=FakeOllama(error=err)).generate_answer("q")


def test_no_claims_warning():
    result = make().analyze(Answer("q", "Hello! Hope this helps.", "m"))
    assert result.claims == [] and any("No verifiable" in w for w in result.warnings)


def test_fever_evaluation_with_mocked_retrieval():
    examples, report = load_fever_jsonl(FIXTURES / "fever_synthetic.jsonl")
    retriever = FakeRetriever({
        "Synthetic Town is located in Example Country.": "synthetic town is located in example country.",
        "Synthetic Town was founded in 1999.": "synthetic town was founded in 1850.",
        "Flattened format synthetic claim.": "flattened format synthetic claim.",
    })
    p = HallucinationPipeline(None, retriever, KeywordVerifier(), "spacy")
    run = evaluate_fever(examples, p, n=10, seed=0, split=report.split)
    assert run.report.n == 4  # only labelled rows with a valid label
    assert run.report.config["split"] == "fever_synthetic"
    assert set(retriever.calls) <= {e.claim for e in examples}  # detector saw claims only
    assert any("unlabelled" in w for w in run.warnings)


def test_ragtruth_evaluation_uses_context_not_wikipedia():
    examples, _ = load_ragtruth(FIXTURES / "ragtruth_synthetic" / "response.jsonl")
    wiki = FakeRetriever(status=RetrievalStatus.ERROR)

    class ContextVerifier(KeywordVerifier):
        pass
    p = HallucinationPipeline(None, wiki, ContextVerifier(), "spacy")
    run = evaluate_ragtruth(examples, p, n=10, seed=0, split_filter="test", tasks=None)
    assert wiki.calls == []
    gold = {r["id"]: r["gold"] for r in run.rows}
    assert gold == {"r1": "HALLUCINATED", "r2": "NOT_HALLUCINATED"}
    # sentence level: r1 has 2 sentences (1 hallucinated), r2 has 1
    assert run.report.n == 3 and run.report.config["experiment"] == "C: sentence-level, strict"
    assert run.report.true_distribution == {"HALLUCINATED": 1, "NOT_HALLUCINATED": 2}
    assert set(run.extra_reports) == {"sentence_lenient", "response_strict", "response_lenient"}
    assert run.extra_reports["response_strict"].n == 2
