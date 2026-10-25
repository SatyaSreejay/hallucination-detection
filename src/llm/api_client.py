"""Hosted answer generation through an OpenAI-compatible Chat Completions API.

One client covers Groq (free tier, hosted open-weight models), OpenRouter (free Llama variants) and
OpenAI, because all three expose ``POST {base_url}/chat/completions`` and
``GET {base_url}/models``. The API key is read from the environment and is never logged,
shown in the UI, or included in error messages.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from src.llm.base import (
    LLMAuthError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMStatus,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """Chat Completions backend with timeouts, rate-limit retries and typed errors."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "api",
        timeout_s: float = 60.0,
        temperature: float = 0.2,
        max_retries: int = 2,
        session: requests.Session | None = None,
        sleep=time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.provider = provider
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._sleep = sleep

    def __repr__(self) -> str:  # never expose the key
        return f"OpenAICompatibleClient(provider={self.provider!r}, model={self.model!r})"

    @property
    def host(self) -> str:
        return self.base_url

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise LLMAuthError(
                f"No API key configured for provider '{self.provider}'. Put it in .env "
                "(see .env.example); keys are never stored in the code."
            )
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.request(method, url, headers=self._headers(), json=payload,
                                            timeout=self.timeout_s)
            except requests.Timeout as exc:
                raise LLMTimeoutError(
                    f"{self.provider} did not answer within {self.timeout_s:.0f}s (LLM_TIMEOUT)."
                ) from exc
            except requests.RequestException as exc:
                raise LLMConnectionError(
                    f"Cannot reach the {self.provider} API at {self.base_url} ({type(exc).__name__})."
                ) from exc
            if resp.status_code == 429 and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After", "")
                wait = min(float(retry_after), 20.0) if retry_after.replace(".", "", 1).isdigit() else 2.0 * 2**attempt
                logger.warning("%s rate limit hit; retrying in %.1fs", self.provider, wait)
                self._sleep(wait)
                continue
            return self._check(resp)
        raise AssertionError("unreachable")  # pragma: no cover

    def _check(self, resp: requests.Response) -> dict[str, Any]:
        code = resp.status_code
        detail = _error_detail(resp)
        if code in (401, 403):
            raise LLMAuthError(f"The {self.provider} API rejected the API key (HTTP {code}). Check the key in .env.")
        if code == 404 or (code == 400 and "model" in detail.lower() and "not" in detail.lower()):
            raise LLMModelNotFoundError(
                f"Model '{self.model}' is not available on {self.provider} (HTTP {code}). "
                "Check LLM_MODEL against the provider's model list."
            )
        if code == 429:
            raise LLMRateLimitError(
                f"{self.provider} rate limit or quota reached (HTTP 429). Wait and retry, or lower the "
                "sample size; free tiers have per-minute and per-day limits."
            )
        if code >= 400:
            raise LLMResponseError(f"{self.provider} returned HTTP {code}: {detail[:200]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMResponseError(f"{self.provider} returned a non-JSON response") from exc
        if not isinstance(data, dict):
            raise LLMResponseError(f"{self.provider} returned an unexpected payload")
        return data

    def list_models(self) -> list[str]:
        data = self._request("GET", "/models")
        items = data.get("data")
        if not isinstance(items, list):
            raise LLMResponseError(f"Malformed model list from {self.provider}")
        return [m["id"] for m in items if isinstance(m, dict) and isinstance(m.get("id"), str)]

    def status(self) -> LLMStatus:
        try:
            models = self.list_models()
        except LLMAuthError as exc:
            return LLMStatus(True, [], False, str(exc))
        except (LLMConnectionError, LLMTimeoutError, LLMResponseError, LLMRateLimitError) as exc:
            return LLMStatus(False, [], False, str(exc))
        available = self.model in models
        error = None if available else (
            f"Model '{self.model}' is not in the {self.provider} model list for this key. Set LLM_MODEL "
            "to one of the listed models."
        )
        return LLMStatus(True, models, available, error)

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 temperature: float | None = None, seed: int | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._request("POST", "/chat/completions", payload)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"{self.provider} response has no message content") from exc
        if not isinstance(text, str):
            raise LLMResponseError(f"{self.provider} response content is not text")
        return text


def _error_detail(resp: requests.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            return str(err.get("message", ""))
        return str(err or "")
    except ValueError:
        return resp.text[:300] if isinstance(resp.text, str) else ""
