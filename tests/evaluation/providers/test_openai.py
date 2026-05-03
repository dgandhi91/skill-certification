import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.evaluation.providers.openai import OpenAIProvider


@pytest.fixture
def openai_provider():
    """Create an OpenAIProvider instance for testing."""
    return OpenAIProvider(
        api_key="test-key", base_url="https://api.openai.com/v1", model="gpt-4o"
    )


@pytest.mark.anyio
async def test_score_relevance(openai_provider):
    """Test relevance scoring with OpenAI."""
    mock_choice = MagicMock()
    mock_choice.message.content = '{"score": 0.9, "reasoning": "Good"}'

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(
        openai_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response

        result = await openai_provider.score_relevance("question", "answer")

        assert result["score"] == 0.9
        assert result["reasoning"] == "Good"


@pytest.mark.anyio
async def test_score_faithfulness(openai_provider):
    """Test faithfulness scoring with OpenAI."""
    mock_choice = MagicMock()
    mock_choice.message.content = '{"score": 0.85, "reasoning": "Close"}'

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(
        openai_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response

        result = await openai_provider.score_faithfulness("expected", "actual")

        assert result["score"] == 0.85
        assert result["reasoning"] == "Close"


@pytest.mark.anyio
async def test_grade_assertion(openai_provider):
    """Test assertion grading with OpenAI."""
    mock_choice = MagicMock()
    mock_choice.message.content = '{"passed": true, "evidence": "Checks out"}'

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(
        openai_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response

        result = await openai_provider.grade_assertion(
            "prompt", "expected", "actual", "assertion"
        )

        assert result["passed"] is True
        assert result["evidence"] == "Checks out"


@pytest.mark.anyio
async def test_api_failure_returns_safe_default(openai_provider):
    """Test that API failures return safe defaults."""
    with patch.object(
        openai_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = Exception("API Error")

        result = await openai_provider.score_relevance("question", "answer")

        assert result["score"] == 0.0
        assert "Judge call failed" in result["reasoning"]


@pytest.mark.anyio
async def test_close_cleanup(openai_provider):
    """Test that close() cleans up resources."""
    with patch.object(
        openai_provider._client, "close", new_callable=AsyncMock
    ) as mock_close:
        await openai_provider.close()
        mock_close.assert_called_once()
