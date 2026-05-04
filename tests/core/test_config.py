import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_gemini_requires_api_key():
    """Test that Gemini provider requires API key."""
    with pytest.raises(ValidationError, match="GEMINI_API_KEY is required"):
        Settings(judge_provider="gemini", gemini_api_key=None)


def test_openai_requires_api_key():
    """Test that OpenAI provider requires API key."""
    with pytest.raises(ValidationError, match="OPENAI_API_KEY is required"):
        Settings(judge_provider="openai", openai_api_key=None)


def test_anthropic_requires_api_key():
    """Test that Anthropic provider requires API key."""
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY is required"):
        Settings(judge_provider="anthropic", anthropic_api_key=None)


def test_ollama_no_api_key_required():
    """Test that Ollama provider does not require API key."""
    # Should not raise
    settings = Settings(judge_provider="ollama")
    assert settings.judge_provider == "ollama"


def test_invalid_provider_raises_error():
    """Test that invalid provider raises error."""
    with pytest.raises(ValidationError, match="Invalid JUDGE_PROVIDER"):
        Settings(judge_provider="invalid")


def test_gemini_defaults():
    """Test Gemini provider defaults."""
    settings = Settings(judge_provider="gemini", gemini_api_key="test-key")
    assert settings.gemini_model == "gemini-2.5-flash"
    assert (
        settings.gemini_base_url == "https://generativelanguage.googleapis.com/v1beta"
    )


def test_ollama_defaults():
    """Test Ollama provider defaults."""
    settings = Settings(judge_provider="ollama")
    assert settings.ollama_model == "llama3.2"
    assert settings.ollama_base_url == "http://localhost:11434"
