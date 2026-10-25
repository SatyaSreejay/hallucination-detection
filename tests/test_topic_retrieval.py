"""Answer-subject handling: topic-steered search and ignoring pages about other subjects."""

from __future__ import annotations

from src.pipeline import HallucinationPipeline
from src.retrieval.passages import subject_related
from src.schemas import Answer, EvidencePassage, Label, RetrievalResult, RetrievalStatus
from tests.conftest import KeywordVerifier
from tests.test_retrieval import make, wiki_handler


def test_subject_related():
    assert subject_related("Taj Mahal", "The Taj Mahal is in Agra.")
    assert subject_related("Mughal architecture", "commissioned by Mughal emperor Shah Jahan")
    assert subject_related("Boiling point", "Water boils at 100 degrees Celsius.")
    assert subject_related("Hamlet (play)", "Hamlet was written by Shakespeare.")
    assert not subject_related("Machu Picchu", "It was designated a UNESCO World Heritage Site in 1983. Taj Mahal")
    assert not subject_related("Baguio", "The monument covers about 42 hectares. Taj Mahal")


def test_retriever_adds_topic_only_when_claim_lacks_it():
    r = make(wiki_handler())
    r.retrieve("It was completed in 1889.", topic="Eiffel Tower")
    r.retrieve("The Eiffel Tower was completed in 1889.", topic="Eiffel Tower")
    searches = [k["params"]["srsearch"] for _, _, k in r.session.calls if k["params"].get("list") == "search"]
    claim_searches = [s for s in searches if s != "Eiffel Tower"]
    assert claim_searches[0] == "Eiffel Tower: It was completed in 1889."
    assert claim_searches[1] == "The Eiffel Tower was completed in 1889."
    assert "Eiffel Tower" in searches  # the topic's own article is always looked up
    assert r.top_title("Where is the Eiffel Tower?") == "Eiffel Tower"


class OpenDomainRetriever:
    """Topic-aware double: pronoun claims hit the wrong page unless the topic is passed."""

    accepts_topic = True

    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    def top_title(self, text):
        return "Taj Mahal"

    def retrieve(self, claim_text, topic=None):
        self.calls.append((claim_text, topic))
        if "UNESCO" in claim_text:  # pretend search went to the wrong article
            page, text = "Machu Picchu", "Machu Picchu became a UNESCO World Heritage Site in 1983 in Peru."
        else:
            page, text = "Taj Mahal", "The Taj Mahal is in Agra and was completed in 1653."
        return RetrievalResult(claim_text, RetrievalStatus.OK, [EvidencePassage(page, "https://example.org", text)])


def test_pipeline_passes_topic_and_ignores_off_subject_pages():
    retriever = OpenDomainRetriever()
    pipe = HallucinationPipeline(None, retriever, KeywordVerifier())
    answer = Answer("Where is Taj Mahal located?",
                    "The Taj Mahal is in Agra and was completed in 1653. "
                    "It was designated a UNESCO World Heritage Site in 1983.", "m")
    res = pipe.analyze(answer)
    assert all(topic == "Taj Mahal" for _, topic in retriever.calls)
    s1, s2 = res.claims
    assert s1.verdict.label is Label.SUPPORTED
    assert s2.verdict.label is Label.NOT_ENOUGH_INFORMATION
    assert "Machu Picchu" in s2.verdict.explanation and "other subjects" in s2.verdict.explanation


def test_closed_retrievers_are_unchanged():
    from tests.conftest import FakeRetriever

    pipe = HallucinationPipeline(None, FakeRetriever(default="The tower was completed in 1653."), KeywordVerifier())
    res = pipe.analyze(Answer("q", "The tower was completed in 1653.", "m"))
    assert res.claims[0].verdict.label is Label.SUPPORTED  # "Fake Page" title is not filtered
