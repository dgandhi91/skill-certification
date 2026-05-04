"""Integration tests for multi-provider support."""

import importlib

import pytest

from app.evaluation.providers.anthropic import AnthropicProvider
from app.evaluation.providers.gemini import GeminiProvider
from app.evaluation.providers.ollama import OllamaProvider
from app.evaluation.providers.openai import OpenAIProvider


@pytest.mark.parametrize(
    "provider,expected_class",
    [
        ("gemini", GeminiProvider),
        ("ollama", OllamaProvider),
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
    ],
)
def test_create_judge_returns_correct_provider(provider, expected_class, monkeypatch):
    """Test that create_judge returns the correct provider type."""
    monkeypatch.setenv("JUDGE_PROVIDER", provider)

    # Set required API keys
    if provider == "gemini":
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    elif provider == "openai":
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    elif provider == "anthropic":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Reload settings and factory to pick up the new environment variables
    import app.core.config
    import app.evaluation.providers

    importlib.reload(app.core.config)
    importlib.reload(app.evaluation.providers)

    from app.evaluation.providers import create_judge

    judge = create_judge()
    assert isinstance(judge, expected_class)


def test_all_providers_implement_interface(monkeypatch):
    """Test that all providers implement the full JudgeProvider interface."""
    providers = [
        ("gemini", "GEMINI_API_KEY"),
        ("ollama", None),
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
    ]

    required_methods = [
        "score_relevance",
        "score_faithfulness",
        "grade_assertion",
        "scan_injection",
        "scan_ambiguity",
        "scan_red_flags",
        "scan_dangerous_tools",
        "close",
    ]

    for provider_name, api_key_var in providers:
        monkeypatch.setenv("JUDGE_PROVIDER", provider_name)
        if api_key_var:
            monkeypatch.setenv(api_key_var, "test-key")

        # Reload settings and factory to pick up the new environment variables
        import app.core.config
        import app.evaluation.providers

        importlib.reload(app.core.config)
        importlib.reload(app.evaluation.providers)

        from app.evaluation.providers import create_judge

        judge = create_judge()

        # Verify all required methods exist and are callable
        for method in required_methods:
            assert hasattr(judge, method), f"{provider_name} missing {method}"
            assert callable(
                getattr(judge, method)
            ), f"{provider_name}.{method} not callable"
