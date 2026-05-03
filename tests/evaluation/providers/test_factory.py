"""Tests for the provider factory function."""

from unittest.mock import Mock, patch

import pytest

from app.evaluation.providers import (
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
    create_judge,
)


class TestCreateJudge:
    """Tests for the create_judge factory function."""

    @patch("app.evaluation.providers.settings")
    def test_create_gemini_provider(self, mock_settings):
        """Test factory creates GeminiProvider when configured."""
        mock_settings.judge_provider = "gemini"
        mock_settings.gemini_api_key = "test-key"
        mock_settings.gemini_model = "gemini-2.5-flash"
        mock_settings.gemini_base_url = "https://test.com"

        judge = create_judge()

        assert isinstance(judge, GeminiProvider)

    @patch("app.evaluation.providers.settings")
    def test_create_ollama_provider(self, mock_settings):
        """Test factory creates OllamaProvider when configured."""
        mock_settings.judge_provider = "ollama"
        mock_settings.ollama_model = "llama3.2"
        mock_settings.ollama_base_url = "http://localhost:11434"

        judge = create_judge()

        assert isinstance(judge, OllamaProvider)

    @patch("app.evaluation.providers.settings")
    def test_create_openai_provider(self, mock_settings):
        """Test factory creates OpenAIProvider when configured."""
        mock_settings.judge_provider = "openai"
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_model = "gpt-4o"
        mock_settings.openai_base_url = "https://api.openai.com/v1"

        judge = create_judge()

        assert isinstance(judge, OpenAIProvider)

    @patch("app.evaluation.providers.settings")
    def test_create_anthropic_provider(self, mock_settings):
        """Test factory creates AnthropicProvider when configured."""
        mock_settings.judge_provider = "anthropic"
        mock_settings.anthropic_api_key = "test-key"
        mock_settings.anthropic_model = "claude-sonnet-4-5"
        mock_settings.anthropic_base_url = "https://api.anthropic.com"

        judge = create_judge()

        assert isinstance(judge, AnthropicProvider)

    @patch("app.evaluation.providers.settings")
    def test_create_judge_unknown_provider(self, mock_settings):
        """Test factory raises ValueError for unknown provider."""
        mock_settings.judge_provider = "unknown"

        with pytest.raises(ValueError, match="Unknown judge provider: unknown"):
            create_judge()

    @patch("app.evaluation.providers.settings")
    def test_create_judge_case_insensitive(self, mock_settings):
        """Test factory handles provider names case-insensitively."""
        mock_settings.judge_provider = "GEMINI"
        mock_settings.gemini_api_key = "test-key"
        mock_settings.gemini_model = "gemini-2.5-flash"
        mock_settings.gemini_base_url = "https://test.com"

        judge = create_judge()

        assert isinstance(judge, GeminiProvider)
