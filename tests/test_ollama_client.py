"""Ollama wrapper error handling (requirement: clear errors when Ollama/model unavailable)."""

import httpx
import ollama
import pytest

from src.llm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    parse_json_object,
)


class FakeBackend:
    def __init__(self, generate=None, listing=None, error=None):
        self._generate, self._listing, self.error = generate, listing, error
        self.calls = []

    def list(self):
        if self.error:
            raise self.error
        return self._listing

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self._generate


def client(**kw):
    return OllamaClient("http://ollama.test", "llama3.2", timeout_s=1, backend=FakeBackend(**kw))


def test_generate_uses_python_client_and_returns_text():
    c = client(generate=ollama.GenerateResponse(model="llama3.2", response="Paris."))
    assert c.generate("Capital of France?", system="be brief", json_mode=True, seed=3) == "Paris."
    call = c.backend.calls[0]
    assert call["model"] == "llama3.2" and call["stream"] is False and call["format"] == "json"
    assert call["options"]["seed"] == 3 and call["system"] == "be brief"


@pytest.mark.parametrize("error,expected", [
    (ConnectionError("Failed to connect to Ollama"), OllamaConnectionError),
    (httpx.ConnectError("refused"), OllamaConnectionError),
    (httpx.ReadTimeout("slow"), OllamaTimeoutError),
    (ollama.ResponseError("model 'llama3.2' not found", 404), OllamaModelNotFoundError),
    (ollama.ResponseError("internal", 500), OllamaResponseError),
])
def test_errors_are_translated(error, expected):
    with pytest.raises(expected):
        client(error=error).generate("hi")


def test_missing_model_message_tells_user_to_pull():
    with pytest.raises(OllamaModelNotFoundError, match="ollama pull llama3.2"):
        client(error=ollama.ResponseError("model not found", 404)).generate("hi")


@pytest.mark.parametrize("resp", [None, {"done": True}, ollama.GenerateResponse(model="m")])
def test_malformed_responses(resp):
    with pytest.raises(OllamaResponseError):
        client(generate=resp).generate("hi")


def test_status_reports_missing_model_without_pulling():
    listing = ollama.ListResponse(models=[ollama.ListResponse.Model(model="mistral:latest")])
    st = client(listing=listing).status()
    assert st.reachable and not st.model_available and "ollama pull llama3.2" in st.error
    ok = client(listing={"models": [{"name": "llama3.2:latest"}]}).status()
    assert ok.model_available and ok.error is None
    down = client(error=ConnectionError("x")).status()
    assert not down.reachable and "Cannot reach Ollama" in down.error
    assert not client(listing="garbage").status().reachable


def test_parse_json_object_handles_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json_object("no json here")
    with pytest.raises(ValueError):
        parse_json_object("[1, 2, 3]")
