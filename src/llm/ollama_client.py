"""Wrapper around the official Python Ollama client (https://github.com/ollama/ollama-python).

Every failure mode is raised as a specific :class:`OllamaError` subclass so callers can
show an actionable message instead of substituting a fake answer. The underlying
``ollama.Client`` can be replaced by any object with ``list()`` and ``generate()`` (tests
use a fake), so nothing here needs a running server to be tested.
"""

from __future__ import annotations

import logging
from typing import Any

from src.llm.base import (
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMResponseError,
    LLMStatus,
    LLMTimeoutError,
    parse_json_object,  # noqa: F401 - re-exported for existing imports
)

logger = logging.getLogger(__name__)


# Backwards-compatible names for the shared error types.
OllamaError = LLMError
OllamaConnectionError = LLMConnectionError
OllamaModelNotFoundError = LLMModelNotFoundError
OllamaTimeoutError = LLMTimeoutError
OllamaResponseError = LLMResponseError
OllamaStatus = LLMStatus


def _model_matches(requested: str, installed: str) -> bool:
    """``llama3.2`` matches ``llama3.2:latest``; explicit tags must match exactly."""
    if requested == installed:
        return True
    if ":" not in requested:
        return installed == f"{requested}:latest"
    return False


def _default_backend(host: str, timeout_s: float) -> Any:
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise OllamaConnectionError(
            "The Python 'ollama' package is not installed (pip install -r requirements.txt)."
        ) from exc
    return ollama.Client(host=host, timeout=timeout_s)


class OllamaClient:
    """Local Ollama backend: model listing and non-streaming generation."""

    provider = "ollama"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout_s: float = 120.0,
        temperature: float = 0.2,
        backend: Any | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self._backend = backend

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = _default_backend(self.host, self.timeout_s)
        return self._backend

    def _translate(self, exc: Exception) -> OllamaError:
        """Map exceptions from the ollama package / httpx to our error types."""
        try:
            import httpx

            timeout_types: tuple[type, ...] = (httpx.TimeoutException, TimeoutError)
            http_types: tuple[type, ...] = (httpx.HTTPError,)
        except ImportError:  # pragma: no cover
            timeout_types, http_types = (TimeoutError,), ()
        try:
            from ollama import ResponseError
        except ImportError:  # pragma: no cover
            ResponseError = ()  # type: ignore[assignment]  # noqa: N806
        if isinstance(exc, OllamaError):
            return exc
        if isinstance(exc, timeout_types):
            return OllamaTimeoutError(
                f"Ollama did not answer within {self.timeout_s:.0f}s. Increase OLLAMA_TIMEOUT "
                "or use a smaller model."
            )
        if ResponseError and isinstance(exc, ResponseError):
            status = getattr(exc, "status_code", -1)
            if status == 404 or "not found" in str(exc).lower():
                return OllamaModelNotFoundError(
                    f"Model '{self.model}' was not found on the Ollama server. "
                    f"Run `ollama pull {self.model}` first."
                )
            return OllamaResponseError(f"Ollama returned an error (HTTP {status}): {str(exc)[:200]}")
        if isinstance(exc, ConnectionError) or isinstance(exc, http_types):
            return OllamaConnectionError(
                f"Cannot reach Ollama at {self.host}. Is it installed and running "
                f"(`ollama serve`)? ({type(exc).__name__})"
            )
        return OllamaResponseError(f"Unexpected Ollama failure: {type(exc).__name__}: {str(exc)[:200]}")

    # ------------------------------------------------------------------ status
    def list_models(self) -> list[str]:
        """Return names of locally available models. Never downloads anything."""
        try:
            listing = self.backend.list()
        except Exception as exc:  # noqa: BLE001 - translated below
            raise self._translate(exc) from exc
        models = getattr(listing, "models", None)
        if models is None and isinstance(listing, dict):
            models = listing.get("models")
        if not isinstance(models, list):
            raise OllamaResponseError("Malformed response from Ollama model listing")
        names = []
        for m in models:
            name = getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
            if isinstance(name, str):
                names.append(name)
        return names

    def status(self) -> OllamaStatus:
        """Check reachability and whether the configured model is installed."""
        try:
            models = self.list_models()
        except OllamaError as exc:
            return OllamaStatus(False, [], False, str(exc))
        available = any(_model_matches(self.model, m) for m in models)
        error = None
        if not available:
            error = (
                f"Model '{self.model}' is not installed. Pull it manually with "
                f"`ollama pull {self.model}` (Llama 3.2 3B is about 2 GB)."
            )
        return OllamaStatus(True, models, available, error)

    # ---------------------------------------------------------------- generate
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Run a non-streaming completion and return the response text."""
        options: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature
        }
        if seed is not None:
            options["seed"] = seed
        kwargs: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False,
                                  "options": options}
        if system:
            kwargs["system"] = system
        if json_mode:
            kwargs["format"] = "json"
        try:
            resp = self.backend.generate(**kwargs)
        except Exception as exc:  # noqa: BLE001 - translated below
            raise self._translate(exc) from exc
        text = getattr(resp, "response", None)
        if text is None and isinstance(resp, dict):
            text = resp.get("response")
        if not isinstance(text, str):
            raise OllamaResponseError("Ollama response is missing the 'response' text")
        return text
