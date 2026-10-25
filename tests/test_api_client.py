"""Hosted OpenAI-compatible backend and provider factory (no network: FakeSession)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import requests

from src.config import Settings
from src.llm.api_client import OpenAICompatibleClient
from src.llm.base import (
    LLMAuthError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from src.llm.factory import PROVIDERS, api_key_for, generation_model, make_llm_client
from src.llm.ollama_client import OllamaClient
from tests.conftest import FakeResponse, FakeSession

KEY = "sk-test-secret-123"


def chat(text: str) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})


def client(handler, **kw) -> tuple[OpenAICompatibleClient, FakeSession, list[float]]:
    session = FakeSession(handler)
    sleeps: list[float] = []
    c = OpenAICompatibleClient("https://api.example.com/v1/", kw.pop("key", KEY), "llama-test", "groq",
                               session=session, sleep=sleeps.append, **kw)
    return c, session, sleeps


def test_generate_sends_chat_request():
    c, session, _ = client(lambda m, u, k: chat("Paris is the capital of France."))
    out = c.generate("Capital of France?", system="Be brief.", json_mode=True, temperature=0.0, seed=7)
    assert out == "Paris is the capital of France."
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "https://api.example.com/v1/chat/completions")
    assert kwargs["headers"]["Authorization"] == f"Bearer {KEY}"
    body = kwargs["json"]
    assert body["model"] == "llama-test" and body["seed"] == 7 and body["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert body["response_format"] == {"type": "json_object"}


def test_missing_key_fails_without_request():
    c, session, _ = client(lambda m, u, k: chat("x"), key="")
    with pytest.raises(LLMAuthError):
        c.generate("hi")
    assert session.calls == []


@pytest.mark.parametrize(("code", "body", "error"), [
    (401, {"error": {"message": "Invalid API Key"}}, LLMAuthError),
    (404, {"error": {"message": "not found"}}, LLMModelNotFoundError),
    (400, {"error": {"message": "The model `x` does not exist or is not available"}}, LLMModelNotFoundError),
    (500, {"error": {"message": "boom"}}, LLMResponseError),
])
def test_http_errors_are_typed(code, body, error):
    c, _, _ = client(lambda m, u, k: FakeResponse(code, body))
    with pytest.raises(error):
        c.generate("hi")


def test_key_never_in_errors_or_repr():
    c, _, _ = client(lambda m, u, k: FakeResponse(401, {"error": {"message": "bad"}}))
    with pytest.raises(LLMAuthError) as exc:
        c.generate("hi")
    assert KEY not in str(exc.value) and KEY not in repr(c)


def test_rate_limit_retries_then_succeeds():
    responses = [FakeResponse(429, {}, headers={"Retry-After": "1.5"}), chat("ok")]
    c, session, sleeps = client(lambda m, u, k: responses.pop(0))
    assert c.generate("hi") == "ok"
    assert sleeps == [1.5] and len(session.calls) == 2


def test_rate_limit_gives_up():
    c, session, sleeps = client(lambda m, u, k: FakeResponse(429, {}), max_retries=2)
    with pytest.raises(LLMRateLimitError):
        c.generate("hi")
    assert len(session.calls) == 3 and sleeps == [2.0, 4.0]


@pytest.mark.parametrize(("exc", "error"), [(requests.Timeout(), LLMTimeoutError),
                                            (requests.ConnectionError(), LLMConnectionError)])
def test_network_errors(exc, error):
    def handler(m, u, k):
        raise exc
    c, _, _ = client(handler)
    with pytest.raises(error):
        c.generate("hi")


def test_malformed_payload():
    c, _, _ = client(lambda m, u, k: FakeResponse(200, {"choices": []}))
    with pytest.raises(LLMResponseError):
        c.generate("hi")


def test_status_lists_models():
    models = {"data": [{"id": "llama-test"}, {"id": "other"}]}
    c, _, _ = client(lambda m, u, k: FakeResponse(200, models))
    s = c.status()
    assert s.reachable and s.model_available and s.models == ["llama-test", "other"]
    c.model = "missing"
    s = c.status()
    assert s.reachable and not s.model_available and "missing" in s.error


def test_status_bad_key_and_unreachable():
    c, _, _ = client(lambda m, u, k: FakeResponse(401, {}))
    assert not c.status().model_available
    def down(m, u, k):
        raise requests.ConnectionError()
    c, _, _ = client(down)
    assert not c.status().reachable


def test_factory_builds_each_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    base = Settings()
    assert base.llm_provider == "groq"
    assert isinstance(make_llm_client(replace(base, llm_provider="ollama")), OllamaClient)
    groq = make_llm_client(replace(base, llm_provider="groq"))
    assert isinstance(groq, OpenAICompatibleClient)
    assert groq.base_url == PROVIDERS["groq"].base_url and groq.model == PROVIDERS["groq"].default_model
    other = make_llm_client(replace(base, llm_provider="openrouter", llm_model="m"))
    assert other.model == "m" and other.provider == "openrouter"
    with pytest.raises(ValueError):
        make_llm_client(replace(base, llm_provider="custom"))
    custom = make_llm_client(replace(base, llm_provider="custom", llm_base_url="http://x/v1", llm_model="y"))
    assert custom.base_url == "http://x/v1"
    with pytest.raises(ValueError):
        make_llm_client(replace(base, llm_provider="nope"))


def test_api_key_lookup(monkeypatch):
    for var in ("GROQ_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert api_key_for("groq") == ""
    monkeypatch.setenv("LLM_API_KEY", "generic")
    assert api_key_for("groq") == "generic" and api_key_for("custom") == "generic"
    monkeypatch.setenv("GROQ_API_KEY", "specific")
    assert api_key_for("groq") == "specific"


def test_generation_model():
    base = Settings()
    assert generation_model(replace(base, llm_provider="ollama")) == base.ollama_model
    assert generation_model(replace(base, llm_provider="groq")) == PROVIDERS["groq"].default_model
    assert generation_model(replace(base, llm_provider="groq", llm_model="z")) == "z"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("LLM_TIMEOUT", "30")
    s = Settings.from_env()
    assert (s.llm_provider, s.llm_model, s.llm_timeout_s) == ("groq", "llama-3.3-70b-versatile", 30.0)
