"""spaCy sentence extraction, multi-assertion flags, LLM decomposition and malformed output."""

import pytest

from src.claims.extractor import (
    ClaimExtractionError,
    LLMClaimExtractor,
    SpacySentenceExtractor,
    extract_claims,
    multi_assertion_flags,
    segment_answer,
    split_sentences,
)
from src.llm.ollama_client import OllamaConnectionError
from tests.conftest import FakeOllama

ANSWER = (
    "Sure! The Eiffel Tower is located in Paris. It was completed in 1889; it is 330 metres tall. "
    "I think it is the most beautiful building in the world. Would you like to know more?"
)


def test_split_sentences_uses_spacy_and_spans_point_into_text():
    text = "Dr. Smith lived in the U.S. for 3.5 years. He moved to Rome in 1990! Done?"
    sents = split_sentences(text)
    assert [s for s, _, _ in sents] == [
        "Dr. Smith lived in the U.S. for 3.5 years.", "He moved to Rome in 1990!", "Done?"]
    for s, start, end in sents:
        assert text[start:end] == s


def test_sentences_have_ids_order_and_skip_reasons():
    sents = segment_answer(ANSWER)
    assert [s.id for s in sents] == ["S1", "S2", "S3", "S4", "S5"]
    assert [s.skip_reason for s in sents] == [
        "fragment (fewer than 4 words)", None, None, "opinion", "question"]
    for s in sents:
        assert ANSWER[s.start:s.end] == s.text


def test_spacy_extractor_keeps_sentences_whole_and_flags_multi_assertions():
    result = SpacySentenceExtractor().extract(ANSWER)
    assert result.method == "spacy"
    assert [c.id for c in result.claims] == ["S2", "S3"]
    assert result.claims[1].text == "It was completed in 1889; it is 330 metres tall."
    assert result.claims[1].multi_assertion and "semicolon joins clauses" in result.claims[1].flags
    assert not result.claims[0].multi_assertion
    assert any("several assertions" in w for w in result.warnings)
    assert len(result.sentences) == 5  # skipped sentences are kept for display


@pytest.mark.parametrize("sentence,flag", [
    ("Paris is the capital of France, and Berlin is the capital of Germany.", "coordinated clauses"),
    ("The tower, which was built in 1889, is in Paris.", "relative clause"),
    ("It opened in 1889 and is 330 metres tall.", "several numbers/dates"),
])
def test_multi_assertion_flags(sentence, flag):
    assert flag in multi_assertion_flags(sentence)


def test_bullets_empty_and_max_claims():
    text = "- Paris is the capital of France\n- Berlin is the capital of Germany\n- Rome is in Italy."
    result = SpacySentenceExtractor(max_claims=2).extract(text)
    assert [c.text for c in result.claims] == ["Paris is the capital of France", "Berlin is the capital of Germany"]
    assert any("only the first 2" in w for w in result.warnings)
    empty = SpacySentenceExtractor().extract("   ")
    assert empty.claims == [] and "empty" in empty.warnings[0]


def test_malformed_input_raises():
    with pytest.raises(ClaimExtractionError):
        SpacySentenceExtractor().extract(None)  # type: ignore[arg-type]


def test_llm_decomposition_validates_and_grounds_claims():
    raw = ('{"claims": [{"sentence_id": "S3", "claim": "The Eiffel Tower was completed in 1889."}, '
           '{"sentence_id": "S3", "claim": "The Eiffel Tower is 330 metres tall."}, '
           '{"sentence_id": "S2", "claim": "The Eiffel Tower is located in Paris."}, '
           '{"sentence_id": "S2", "claim": "The Louvre has 35,000 works of art."}, 42, {"claim": ["x"]}]}')
    client = FakeOllama([raw])
    result = LLMClaimExtractor(client).extract(ANSWER)
    assert [c.id for c in result.claims] == ["S2.1", "S3.1", "S3.2"]
    assert result.claims[1].source_sentence == "It was completed in 1889; it is 330 metres tall."
    assert any("not traceable" in w for w in result.warnings)
    assert any("malformed" in w for w in result.warnings)
    assert "S4" not in client.prompts[0]  # opinion sentence not sent for decomposition


@pytest.mark.parametrize("raw", ["not json at all", '{"facts": []}', '{"claims": "oops"}'])
def test_llm_extractor_rejects_malformed_output(raw):
    with pytest.raises(ClaimExtractionError):
        LLMClaimExtractor(FakeOllama([raw])).extract(ANSWER)


def test_extract_claims_falls_back_visibly():
    result = extract_claims(ANSWER, "llm", FakeOllama(["garbage"]))
    assert result.method == "spacy"
    assert "spaCy sentences were used instead" in result.warnings[0]
    with pytest.raises(OllamaConnectionError):
        extract_claims(ANSWER, "llm", FakeOllama(error=OllamaConnectionError("down")), allow_fallback=False)


def test_greeting_only_answer_has_no_claims():
    assert SpacySentenceExtractor().extract("Hello! I hope this helps. Let me know.").claims == []
