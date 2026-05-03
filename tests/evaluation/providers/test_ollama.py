import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.evaluation.providers.ollama import OllamaProvider


@pytest.fixture
def ollama_provider():
    """Create an OllamaProvider instance for testing."""
    return OllamaProvider(base_url="http://localhost:11434", model="llama3.2")


@pytest.mark.anyio
async def test_score_relevance(ollama_provider):
    """Test relevance scoring with Ollama."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": '{"score": 0.9, "reasoning": "Good"}'
    }

    with patch.object(
        ollama_provider._client, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = mock_response

        result = await ollama_provider.score_relevance("question", "answer")

        assert result["score"] == 0.9
        assert result["reasoning"] == "Good"


@pytest.mark.anyio
async def test_double_json_parse(ollama_provider):
    """Test that Ollama responses are double-parsed."""
    mock_response = MagicMock()
    # Ollama wraps JSON in another JSON response
    mock_response.json.return_value = {"response": '{"score": 0.75, "reasoning": "OK"}'}

    with patch.object(
        ollama_provider._client, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.return_value = mock_response

        result = await ollama_provider.score_faithfulness("expected", "actual")

        assert result["score"] == 0.75


@pytest.mark.anyio
async def test_api_failure_returns_safe_default(ollama_provider):
    """Test that API failures return safe defaults."""
    with patch.object(
        ollama_provider._client, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.side_effect = Exception("Ollama not running")

        result = await ollama_provider.score_relevance("question", "answer")

        assert result["score"] == 0.0
        assert "Judge call failed" in result["reasoning"]
