"""Shared test doubles. No test touches the network or a real Ollama server."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import requests

from src.schemas import EvidencePassage, Label, RetrievalResult, RetrievalStatus, Verdict

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, status_code: int = 200, data: Any = None, text: str | None = None,
                 headers: dict | None = None):
        self.status_code = status_code
        self._data = data
        self.text = text if text is not None else json.dumps(data)
        self.headers = headers or {}

    def json(self) -> Any:
        if self._data is None:
            return json.loads(self.text)
        return self._data


class FakeSession:
    """Stands in for ``requests.Session``; ``handler(method, url, kwargs)`` returns a response
    or raises."""

    def __init__(self, handler: Callable[[str, str, dict], FakeResponse]):
        self.handler = handler
        self.calls: list[tuple[str, str, dict]] = []
        self.headers: dict[str, str] = {}

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.handler("GET", url, kwargs)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.handler("POST", url, kwargs)

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.handler(method, url, kwargs)


class FakeOllama:
    """Returns scripted responses in order (or via a function of the prompt)."""

    def __init__(self, responses: list[str] | Callable[[str], str] | None = None,
                 model: str = "llama3.2", error: Exception | None = None):
        self.responses = responses if responses is not None else []
        self.model = model
        self.provider = "fake"
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        if callable(self.responses):
            return self.responses(prompt)
        return self.responses.pop(0)


class FakeRetriever:
    """Returns a passage containing ``evidence[claim]`` or a configured status."""

    def __init__(self, evidence: dict[str, str] | None = None, status: RetrievalStatus | None = None,
                 default: str | None = None):
        self.evidence = evidence or {}
        self.status = status
        self.default = default
        self.calls: list[Any] = []

    def retrieve(self, claim_text):
        self.calls.append(claim_text)
        if self.status is RetrievalStatus.ERROR:
            return RetrievalResult(claim_text, RetrievalStatus.ERROR, error="simulated outage")
        text = self.evidence.get(claim_text, self.default)
        if text is None:
            return RetrievalResult(claim_text, RetrievalStatus.NO_RESULTS)
        return RetrievalResult(claim_text, RetrievalStatus.OK,
                               [EvidencePassage("Fake Page", "https://example.org/Fake", text)])


class KeywordVerifier:
    """Deterministic verifier double: CONTRADICTED if the claim has a word the evidence
    lacks among numbers, SUPPORTED if all claim numbers/words appear, else NEI."""

    method = "keyword test double"

    def __init__(self):
        self.calls: list[tuple[Any, Any]] = []

    def verify(self, claim_text, passages):
        self.calls.append((claim_text, passages))
        evidence = " ".join(p.text for p in passages).lower()
        words = [w.strip(".,").lower() for w in claim_text.split()]
        numbers = [w for w in words if w.isdigit()]
        if any(n not in evidence for n in numbers) or " not " in f" {claim_text.lower()} ":
            return Verdict(Label.CONTRADICTED, self.method, "number/negation mismatch", list(passages))
        if all(w in evidence for w in words):
            return Verdict(Label.SUPPORTED, self.method, "all words present", list(passages))
        return Verdict(Label.NOT_ENOUGH_INFORMATION, self.method, "unclear", list(passages))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test tries to use the real network through requests."""

    def blocked(*_a, **_k):
        raise AssertionError("Real network access attempted in a unit test")

    monkeypatch.setattr(requests.Session, "request", blocked)
