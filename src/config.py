"""Central configuration.

All tunable settings live here. Values come from environment variables (optionally
loaded from a ``.env`` file in the project root) and can be overridden at runtime,
e.g. from the Streamlit sidebar, with :func:`dataclasses.replace`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def _load_dotenv(path: Path) -> None:
    """Minimal ``.env`` loader (KEY=VALUE lines). Non-empty existing env vars win.

    Copes with files saved on Windows: UTF-8 with or without BOM, UTF-16 (Notepad / PowerShell),
    CRLF line endings, ``export`` prefixes and trailing ``# comments`` on unquoted values.
    """
    if not path.is_file():
        return
    raw = path.read_bytes()
    encoding = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    for line in raw.decode(encoding).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in ('"', "'") and value[0] in value[1:]:
            value = value[1:value.index(value[0], 1)]
        else:
            value = "" if value.startswith("#") else value.split(" #", 1)[0].strip()
        if not os.environ.get(key):
            os.environ[key] = value


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_opt_float(name: str, default: float | None) -> float | None:
    """Float or ``None`` (unset, empty or "none")."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for every component."""

    # --- Answer generation backend: "ollama" (local) or a hosted OpenAI-compatible API
    #     ("groq", "openrouter", "openai", "custom"). API keys are read from the environment
    #     by src/llm/factory.py and are never stored in Settings.
    llm_provider: str = "groq"
    llm_model: str = ""  # empty = provider default
    llm_base_url: str = ""  # empty = provider default
    llm_timeout_s: float = 60.0

    # --- Ollama (answer generation, LLM claim extraction, LLM verifier fallback)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout_s: float = 120.0
    ollama_temperature: float = 0.2

    # --- Claim extraction: "spacy" (Sentencizer, default) or "llm" (spaCy + Ollama decomposition)
    extractor: str = "spacy"
    max_claims: int = 10

    # --- Wikipedia retrieval
    wiki_api_url: str = "https://en.wikipedia.org/w/api.php"
    wiki_user_agent: str = (
        "HallucinationDetectionPrototype/0.1 (research prototype; set WIKI_USER_AGENT with your contact)"
    )
    wiki_timeout_s: float = 10.0
    wiki_max_retries: int = 3
    wiki_search_results: int = 3
    wiki_passages_per_claim: int = 3
    wiki_min_interval_s: float = 0.2
    wiki_cache_dir: Path = field(default_factory=lambda: DATA_DIR / "cache" / "wikipedia")

    # --- Verification: "nli", "llm" or "auto" (NLI if it loads, otherwise LLM; the UI says which)
    verifier: str = "auto"
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    # None = argmax decision. Only set these from a validation threshold search.
    nli_support_threshold: float | None = None
    nli_contradiction_threshold: float | None = None
    model_cache_dir: Path = field(default_factory=lambda: MODELS_DIR / "hf-cache")

    # --- Datasets / outputs
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    random_seed: int = 13

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables (and ``.env`` if present)."""
        _load_dotenv(PROJECT_ROOT / ".env")
        d = cls()
        return cls(
            llm_provider=_env("LLM_PROVIDER", d.llm_provider).lower(),
            llm_model=_env("LLM_MODEL", d.llm_model),
            llm_base_url=_env("LLM_BASE_URL", d.llm_base_url),
            llm_timeout_s=_env_float("LLM_TIMEOUT", d.llm_timeout_s),
            ollama_host=_env("OLLAMA_HOST", d.ollama_host).rstrip("/"),
            ollama_model=_env("OLLAMA_MODEL", d.ollama_model),
            ollama_timeout_s=_env_float("OLLAMA_TIMEOUT", d.ollama_timeout_s),
            ollama_temperature=_env_float("OLLAMA_TEMPERATURE", d.ollama_temperature),
            extractor=_env("CLAIM_EXTRACTOR", d.extractor).lower(),
            max_claims=_env_int("MAX_CLAIMS", d.max_claims),
            wiki_api_url=_env("WIKI_API_URL", d.wiki_api_url),
            wiki_user_agent=_env("WIKI_USER_AGENT", d.wiki_user_agent),
            wiki_timeout_s=_env_float("WIKI_TIMEOUT", d.wiki_timeout_s),
            wiki_max_retries=_env_int("WIKI_MAX_RETRIES", d.wiki_max_retries),
            wiki_search_results=_env_int("WIKI_SEARCH_RESULTS", d.wiki_search_results),
            wiki_passages_per_claim=_env_int("WIKI_PASSAGES_PER_CLAIM", d.wiki_passages_per_claim),
            verifier=_env("VERIFIER", d.verifier).lower(),
            nli_model=_env("NLI_MODEL", d.nli_model),
            nli_support_threshold=_env_opt_float("NLI_SUPPORT_THRESHOLD", d.nli_support_threshold),
            nli_contradiction_threshold=_env_opt_float(
                "NLI_CONTRADICTION_THRESHOLD", d.nli_contradiction_threshold
            ),
            random_seed=_env_int("RANDOM_SEED", d.random_seed),
        )
