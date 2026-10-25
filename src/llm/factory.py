"""Build the configured generation backend (local Ollama or a hosted OpenAI-compatible API)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.config import Settings
from src.llm.api_client import OpenAICompatibleClient
from src.llm.base import LLMClient
from src.llm.ollama_client import OllamaClient


@dataclass(frozen=True)
class Provider:
    base_url: str
    default_model: str
    key_env: str
    note: str


# Model ids change over time; check the provider's model list if one is retired.
PROVIDERS: dict[str, Provider] = {
    "groq": Provider("https://api.groq.com/openai/v1", "openai/gpt-oss-20b", "GROQ_API_KEY",
                     "Free tier with rate limits; open-weight models (Llama is not offered on every account; "
                     "`python -m src.cli status` lists yours). Key: https://console.groq.com/keys"),
    "openrouter": Provider("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free",
                           "OPENROUTER_API_KEY",
                           "Free ':free' models with daily limits. Key: https://openrouter.ai/keys"),
    "openai": Provider("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY",
                       "Paid; not a Llama model. Key: https://platform.openai.com/api-keys"),
}
PROVIDER_NAMES = ["ollama", *PROVIDERS, "custom"]


def api_key_for(provider: str) -> str:
    """Key from the provider-specific variable, else the generic LLM_API_KEY."""
    specific = PROVIDERS[provider].key_env if provider in PROVIDERS else ""
    return os.environ.get(specific, "") or os.environ.get("LLM_API_KEY", "")


def make_llm_client(settings: Settings) -> LLMClient:
    """Return the generation client for ``settings.llm_provider``."""
    provider = settings.llm_provider
    if provider == "ollama":
        return OllamaClient(settings.ollama_host, settings.ollama_model, settings.ollama_timeout_s,
                            settings.ollama_temperature)
    if provider in PROVIDERS:
        preset = PROVIDERS[provider]
        base_url = settings.llm_base_url or preset.base_url
        model = settings.llm_model or preset.default_model
    elif provider == "custom":
        if not settings.llm_base_url or not settings.llm_model:
            raise ValueError("LLM_PROVIDER=custom needs LLM_BASE_URL and LLM_MODEL")
        base_url, model = settings.llm_base_url, settings.llm_model
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (expected one of {PROVIDER_NAMES})")
    return OpenAICompatibleClient(base_url, api_key_for(provider), model, provider,
                                  settings.llm_timeout_s, settings.ollama_temperature)


def generation_model(settings: Settings) -> str:
    """Model name that will generate answers under the current settings."""
    if settings.llm_provider == "ollama":
        return settings.ollama_model
    preset = PROVIDERS.get(settings.llm_provider)
    return settings.llm_model or (preset.default_model if preset else "")
