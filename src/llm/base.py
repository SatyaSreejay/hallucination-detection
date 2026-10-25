"""Interface shared by all answer-generation backends (local Ollama or a hosted API).

Every backend exposes ``model``, ``generate(...)`` and ``status()`` and raises only the
:class:`LLMError` subclasses below, so the rest of the pipeline never depends on which
backend produced an answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Base class for generation-backend failures."""


class LLMConnectionError(LLMError):
    """The server/API could not be reached."""


class LLMAuthError(LLMError):
    """The API rejected the key (missing, invalid or without access)."""


class LLMModelNotFoundError(LLMError):
    """The requested model is not available on the backend."""


class LLMTimeoutError(LLMError):
    """The request exceeded the configured timeout."""


class LLMRateLimitError(LLMError):
    """The API's rate limit or quota was hit (common on free tiers)."""


class LLMResponseError(LLMError):
    """The backend answered with an error or a malformed payload."""


@dataclass
class LLMStatus:
    reachable: bool
    models: list[str]
    model_available: bool
    error: str | None = None


class LLMClient(Protocol):
    """What the pipeline needs from a generation backend."""

    model: str
    provider: str

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 temperature: float | None = None, seed: int | None = None) -> str: ...

    def status(self) -> LLMStatus: ...


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in LLM output (tolerates code fences / prose).

    Raises ``ValueError`` if no valid JSON object can be found.
    """
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE_RE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("No valid JSON object found in model output")
