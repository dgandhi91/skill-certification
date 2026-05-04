import json
from unittest.mock import MagicMock, patch

import pytest

from app.evaluation.providers.gemini import GeminiProvider


@pytest.fixture
def gemini_provider():
    """Create a GeminiProvider instance for testing."""
    with patch("app.evaluation.providers.gemini.genai.Client"):
        return GeminiProvider(
            api_key="test-key", base_url="https://test.api", model="test-model"
        )


@pytest.mark.anyio
async def test_score_relevance(gemini_provider):
    """Test relevance scoring."""
    mock_response = MagicMock()
    mock_response.text = '{"score": 0.9, "reasoning": "Good"}'

    with patch.object(
        gemini_provider._client.models, "generate_content"
    ) as mock_generate:
        mock_generate.return_value = mock_response

        result = await gemini_provider.score_relevance("question", "answer")

        assert result["score"] == 0.9
        assert result["reasoning"] == "Good"


@pytest.mark.anyio
async def test_score_faithfulness(gemini_provider):
    """Test faithfulness scoring."""
    mock_response = MagicMock()
    mock_response.text = '{"score": 0.85, "reasoning": "Close"}'

    with patch.object(
        gemini_provider._client.models, "generate_content"
    ) as mock_generate:
        mock_generate.return_value = mock_response

        result = await gemini_provider.score_faithfulness("expected", "actual")

        assert result["score"] == 0.85
        assert result["reasoning"] == "Close"


@pytest.mark.anyio
async def test_grade_assertion(gemini_provider):
    """Test assertion grading."""
    mock_response = MagicMock()
    mock_response.text = '{"passed": true, "evidence": "Checks out"}'

    with patch.object(
        gemini_provider._client.models, "generate_content"
    ) as mock_generate:
        mock_generate.return_value = mock_response

        result = await gemini_provider.grade_assertion(
            "prompt", "expected", "actual", "assertion"
        )

        assert result["passed"] is True
        assert result["evidence"] == "Checks out"


@pytest.mark.anyio
async def test_api_failure_returns_safe_default(gemini_provider):
    """Test that API failures return safe defaults."""
    with patch.object(
        gemini_provider._client.models, "generate_content"
    ) as mock_generate:
        mock_generate.side_effect = Exception("API Error")

        result = await gemini_provider.score_relevance("question", "answer")

        assert result["score"] == 0.0
        assert "Judge call failed" in result["reasoning"]


@pytest.mark.anyio
async def test_close_cleanup(gemini_provider):
    """Test that close() completes without error."""
    # SDK-based implementation has no cleanup needed
    await gemini_provider.close()
    # Just verify it doesn't raise an exception
