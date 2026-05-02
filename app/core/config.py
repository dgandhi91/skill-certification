import logging

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-2.5-flash"
    similarity_threshold: float = 0.8

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
logger.info(
    "Settings loaded: model=%s threshold=%.2f api_key=%s",
    settings.gemini_model,
    settings.similarity_threshold,
    "configured" if settings.gemini_api_key else "NOT SET",
)
