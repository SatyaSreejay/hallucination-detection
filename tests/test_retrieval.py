"""Wikipedia retriever with a fake session: statuses, retries, caching, user agent."""

import requests

from src.retrieval.context import ContextRetriever
from src.retrieval.passages import chunk_text, rank_passages
from src.retrieval.wikipedia import WikipediaRetriever
from src.schemas import RetrievalStatus
from tests.conftest import FakeResponse, FakeSession

EXTRACT = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is named after "
    "Gustave Eiffel.\n\n== History ==\nThe tower was completed in 1889 as the entrance to the "
    "World's Fair. It was the tallest man-made structure in the world until 1930."
)


def wiki_handler(search_hits=True, fail_times=0):
    state = {"fails": 0}

    def handler(method, url, kwargs):
        params = kwargs["params"]
        if state["fails"] < fail_times:
            state["fails"] += 1
            return FakeResponse(503, text="busy")
        if params.get("list") == "search":
            hits = [{"title": "Eiffel Tower", "pageid": 9232}] if search_hits else []
            return FakeResponse(200, {"query": {"search": hits}})
        if params.get("prop"):
            return FakeResponse(200, {"query": {"pages": [{
                "pageid": 9232, "title": "Eiffel Tower", "extract": EXTRACT,
                "fullurl": "https://en.wikipedia.org/wiki/Eiffel_Tower"}]}})
        return FakeResponse(200, {"query": {"general": {}}})
    return handler


def make(handler, **kw):
    return WikipediaRetriever(session=FakeSession(handler), sleep=lambda s: None,
                              user_agent="TestAgent/1.0 (test@example.org)", **kw)


def test_retrieve_returns_ranked_passages_with_real_urls():
    r = make(wiki_handler())
    res = r.retrieve("The Eiffel Tower was completed in 1889.")
    assert res.status is RetrievalStatus.OK
    top = res.passages[0]
    assert "1889" in top.text
    assert top.url == "https://en.wikipedia.org/wiki/Eiffel_Tower"
    assert top.title == "Eiffel Tower" and top.source == "wikipedia"
    assert r.session.headers["User-Agent"].startswith("TestAgent")
    for _, _, kwargs in r.session.calls:
        assert kwargs["timeout"] == r.timeout_s


def test_no_search_results_is_explicit_status_not_falsity():
    res = make(wiki_handler(search_hits=False)).retrieve("Zorblax the Great invented soup.")
    assert res.status is RetrievalStatus.NO_RESULTS and res.passages == []


def test_retries_then_succeeds():
    r = make(wiki_handler(fail_times=2), max_retries=3)
    assert r.retrieve("Eiffel Tower completed 1889").status is RetrievalStatus.OK


def test_network_failure_returns_error_status():
    def down(*_):
        raise requests.ConnectionError("no route")
    res = make(down, max_retries=2).retrieve("Eiffel Tower completed 1889")
    assert res.status is RetrievalStatus.ERROR and "2 attempts" in res.error


def test_non_retryable_http_error_and_malformed_payload():
    assert make(lambda *_: FakeResponse(403, text="forbidden")).retrieve("x y z").status is RetrievalStatus.ERROR
    assert make(lambda *_: FakeResponse(200, text="<html>")).retrieve("x y z").status is RetrievalStatus.ERROR


def test_cache_prevents_repeated_requests(tmp_path):
    r = make(wiki_handler(), cache_dir=tmp_path)
    r.retrieve("The Eiffel Tower was completed in 1889.")
    n = r.request_count
    again = r.retrieve("The Eiffel Tower was completed in 1889.")
    assert r.request_count == n and again.from_cache
    # disk cache is reused by a fresh retriever
    r2 = make(lambda *_: (_ for _ in ()).throw(AssertionError("should hit cache")), cache_dir=tmp_path)
    assert r2.retrieve("The Eiffel Tower was completed in 1889.").status is RetrievalStatus.OK


def test_chunk_and_rank():
    chunks = chunk_text(EXTRACT)
    assert not any(c.startswith("==") for c in chunks)
    ranked = rank_passages("completed 1889", chunks)
    assert "1889" in chunks[ranked[0][0]]
    assert rank_passages("unrelated zebra", chunks) == []


def test_context_retriever():
    ctx = ContextRetriever("Synthetic Town was founded in 1850. It has 3000 residents.\n\nThe river is long.")
    assert ctx.retrieve("Synthetic Town founded 1850").status is RetrievalStatus.OK
    assert ctx.retrieve("quantum chromodynamics").status is RetrievalStatus.NO_RESULTS
    assert ContextRetriever("").retrieve("x").status is RetrievalStatus.ERROR
