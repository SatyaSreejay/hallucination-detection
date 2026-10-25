"""Choose and build the configured verifier, reporting exactly which one is in use."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings
from src.llm.base import LLMClient
from src.verification.base import Verifier, VerifierUnavailableError
from src.verification.llm_verifier import LLMVerifier
from src.verification.nli_verifier import NLIVerifier


@dataclass
class VerifierChoice:
    verifier: Verifier
    kind: str  # "nli" or "llm"
    notes: list[str]


def build_verifier(settings: Settings, client: LLMClient) -> VerifierChoice:
    """Build the verifier for ``settings.verifier`` ("nli", "llm" or "auto").

    ``nli`` raises :class:`VerifierUnavailableError` if the model cannot be loaded.
    ``auto`` falls back to the LLM verifier and records why in ``notes``.
    """
    mode = settings.verifier
    if mode not in {"nli", "llm", "auto"}:
        raise ValueError(f"Unknown VERIFIER '{mode}' (expected nli, llm or auto)")
    if mode == "llm":
        return VerifierChoice(LLMVerifier(client, settings.random_seed), "llm", [])

    nli = NLIVerifier(
        settings.nli_model,
        settings.nli_support_threshold,
        settings.nli_contradiction_threshold,
        settings.model_cache_dir,
    )
    try:
        nli.load()
        return VerifierChoice(nli, "nli", [])
    except VerifierUnavailableError as exc:
        if mode == "nli":
            raise
        return VerifierChoice(
            LLMVerifier(client, settings.random_seed),
            "llm",
            [f"NLI verifier unavailable ({exc}). Using the LLM-judgment fallback, which is "
             "not a trained classifier."],
        )
