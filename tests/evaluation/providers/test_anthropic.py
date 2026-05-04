import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.evaluation.providers.anthropic import AnthropicProvider


@pytest.fixture
def anthropic_provider():
    """Create an AnthropicProvider instance for testing."""
    return AnthropicProvider(
        api_key="test-key",
        model="claude-3-5-sonnet-20241022",
    )


@pytest.mark.anyio
async def test_score_relevance(anthropic_provider):
    """Test relevance scoring with Anthropic."""
    mock_content_block = MagicMock()
    mock_content_block.text = '{"score": 0.9, "reasoning": "Good"}'

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    with patch.object(
        anthropic_provider._client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response

        result = await anthropic_provider.score_relevance("question", "answer")

        assert result["score"] == 0.9
        assert result["reasoning"] == "Good"


@pytest.mark.anyio
async def test_score_faithfulness(anthropic_provider):
    """Test faithfulness scoring with Anthropic."""
    mock_content_block = MagicMock()
    mock_content_block.text = '{"score": 0.85, "reasoning": "Close"}'

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    with patch.object(
        anthropic_provider._client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response

        result = await anthropic_provider.score_faithfulness("expected", "actual")

        assert result["score"] == 0.85
        assert result["reasoning"] == "Close"


@pytest.mark.anyio
async def test_grade_assertion(anthropic_provider):
    """Test assertion grading with Anthropic."""
    mock_content_block = MagicMock()
    mock_content_block.text = '{"passed": true, "evidence": "Checks out"}'

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    with patch.object(
        anthropic_provider._client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response

        result = await anthropic_provider.grade_assertion(
            "prompt", "expected", "actual", "assertion"
        )

        assert result["passed"] is True
        assert result["evidence"] == "Checks out"


@pytest.mark.anyio
async def test_api_failure_returns_safe_default(anthropic_provider):
    """Test that API failures return safe defaults."""
    with patch.object(
        anthropic_provider._client.messages, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = Exception("API Error")

        result = await anthropic_provider.score_relevance("question", "answer")

        assert result["score"] == 0.0
        assert "Judge call failed" in result["reasoning"]


@pytest.mark.anyio
async def test_close_cleanup(anthropic_provider):
    """Test that close() cleans up resources."""
    with patch.object(
        anthropic_provider._client, "close", new_callable=AsyncMock
    ) as mock_close:
        await anthropic_provider.close()
        mock_close.assert_called_once()
