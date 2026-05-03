import logging

from pydantic import model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Provider selection
    judge_provider: str = "gemini"

    # Gemini configuration
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Ollama configuration (local)
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI configuration
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"

    # Anthropic configuration
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    anthropic_base_url: str = "https://api.anthropic.com"

    # Shared settings
    similarity_threshold: float = 0.8

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @model_validator(mode="after")
    def validate_provider_config(self):
        """Validate that required fields are set for the active provider."""
        provider = self.judge_provider.lower()

        if provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when JUDGE_PROVIDER=gemini"
                )
        elif provider == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when JUDGE_PROVIDER=openai"
                )
        elif provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when JUDGE_PROVIDER=anthropic"
                )
        elif provider == "ollama":
            # No API key required for Ollama
            pass
        else:
            raise ValueError(
                f"Invalid JUDGE_PROVIDER: {provider}. "
                f"Must be one of: gemini, ollama, openai, anthropic"
            )

        return self


settings = Settings()
logger.info(
    "Settings loaded: provider=%s model=%s api_key=%s",
    settings.judge_provider,
    getattr(settings, f"{settings.judge_provider}_model", "N/A"),
    (
        "configured"
        if getattr(settings, f"{settings.judge_provider}_api_key", None)
        else "NOT SET (ollama or missing)"
    ),
)
