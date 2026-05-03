"""Provider factory and exports for LLM judge providers."""

from app.core.config import settings
from app.evaluation.providers.anthropic import AnthropicProvider
from app.evaluation.providers.base import JudgeProvider
from app.evaluation.providers.gemini import GeminiProvider
from app.evaluation.providers.ollama import OllamaProvider
from app.evaluation.providers.openai import OpenAIProvider

__all__ = [
    "JudgeProvider",
    "GeminiProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_judge",
]


def create_judge() -> JudgeProvider:
    """
    Factory function to create the appropriate judge provider.

    Returns:
        JudgeProvider: An instance of the configured provider

    Raises:
        ValueError: If the configured provider is unknown
    """
    provider = settings.judge_provider.lower()

    if provider == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            base_url=settings.gemini_base_url,
        )
    elif provider == "ollama":
        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )
    elif provider == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
        )
    elif provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            base_url=settings.anthropic_base_url,
        )
    else:
        raise ValueError(f"Unknown judge provider: {provider}")
