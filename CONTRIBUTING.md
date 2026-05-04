# Contributing to Skill Certification Pipeline

Thank you for your interest in contributing to the Skill Certification Pipeline! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Adding a New Provider](#adding-a-new-provider)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)

## Code of Conduct

Be respectful, inclusive, and professional in all interactions. We're here to build great software together.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/skill-certification.git
   cd skill-certification
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip or poetry for dependency management
- Git

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run tests to verify setup**:
   ```bash
   pytest tests/ -v
   ```

## Project Structure

```
skill-certification/
├── app/
│   ├── core/               # Configuration, models, utilities
│   │   ├── config.py       # Multi-provider settings
│   │   ├── models.py       # Pydantic data models
│   │   └── embeddings.py   # Text embeddings for similarity
│   ├── evaluation/         # LLM judge providers and evaluation logic
│   │   ├── providers/      # Provider implementations
│   │   │   ├── base.py     # Abstract JudgeProvider base class
│   │   │   ├── gemini.py   # Gemini provider
│   │   │   ├── ollama.py   # Ollama provider
│   │   │   ├── openai.py   # OpenAI provider
│   │   │   └── anthropic.py # Anthropic provider
│   │   ├── judge.py        # Backward compatibility wrapper
│   │   ├── prompts.py      # Evaluation prompts
│   │   ├── metrics.py      # Scoring metrics
│   │   └── validation.py   # Contract validation
│   └── pipeline/           # Orchestration
│       ├── pipeline.py     # Main certification pipeline
│       ├── loader.py       # Workspace directory parser
│       └── registry.py     # Skill registry
├── tests/                  # Test suite
│   ├── core/              # Core module tests
│   ├── evaluation/        # Evaluation module tests
│   │   └── providers/     # Provider-specific tests
│   └── integration/       # Integration tests
├── skills/                # Sample skills and workspaces
├── run.py                 # CLI entrypoint
└── streamlit_app.py       # Dashboard UI
```

## Adding a New Provider

To add support for a new LLM provider (e.g., Cohere, Mistral, etc.):

### 1. Add Dependencies

Update `requirements.txt`:
```
your-provider-sdk>=1.0.0,<2.0.0
```

### 2. Update Configuration

Add provider settings to `app/core/config.py`:

```python
class Settings(BaseSettings):
    # ... existing providers ...
    
    # Your provider configuration
    yourprovider_api_key: str | None = None
    yourprovider_model: str = "default-model"
    yourprovider_base_url: str = "https://api.yourprovider.com"
    
    @model_validator(mode="after")
    def validate_provider_config(self):
        provider = self.judge_provider.lower()
        
        # ... existing validation ...
        
        elif provider == "yourprovider":
            if not self.yourprovider_api_key:
                raise ValueError("YOURPROVIDER_API_KEY is required when JUDGE_PROVIDER=yourprovider")
        # ...
```

### 3. Implement Provider Class

Create `app/evaluation/providers/yourprovider.py`:

```python
from __future__ import annotations

import json
import logging
from your_provider_sdk import AsyncYourProviderClient

from app.core.config import settings
from app.evaluation.prompts import (
    ASSERTION_GRADING_PROMPT,
    FAITHFULNESS_PROMPT,
    RELEVANCE_PROMPT,
    # ... other prompts
)
from app.evaluation.providers.base import JudgeProvider

logger = logging.getLogger(__name__)


class YourProviderProvider(JudgeProvider):
    """Your Provider LLM judge provider."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.yourprovider_api_key
        self.base_url = base_url or settings.yourprovider_base_url
        self.model = model or settings.yourprovider_model
        self._client = AsyncYourProviderClient(api_key=self.api_key)

    async def score_relevance(self, question: str, answer: str) -> dict:
        logger.info("Scoring relevance via YourProvider %s", self.model)
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        try:
            # Implement provider-specific API call
            result = await self._call_provider(prompt)
            logger.info("Relevance score: %.2f", result.get("score", 0.0))
            return result
        except Exception as e:
            logger.warning("Relevance scoring failed: %s", e)
            return {"score": 0.0, "reasoning": "Judge call failed"}

    # Implement all other abstract methods...
    # - score_faithfulness
    # - grade_assertion
    # - scan_injection
    # - scan_ambiguity
    # - scan_red_flags
    # - scan_dangerous_tools
    # - close

    async def close(self) -> None:
        await self._client.close()
```

### 4. Write Tests

Create `tests/evaluation/providers/test_yourprovider.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.evaluation.providers.yourprovider import YourProviderProvider


@pytest.fixture
def yourprovider_provider():
    """Create a YourProviderProvider instance for testing."""
    return YourProviderProvider(
        api_key="test-key",
        base_url="https://api.yourprovider.com",
        model="test-model"
    )


@pytest.mark.anyio
async def test_score_relevance(yourprovider_provider):
    """Test relevance scoring with YourProvider."""
    mock_response = MagicMock()
    mock_response.score = 0.9
    mock_response.reasoning = "Good"

    with patch.object(
        yourprovider_provider._client, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = mock_response

        result = await yourprovider_provider.score_relevance("question", "answer")

        assert result["score"] == 0.9
        assert result["reasoning"] == "Good"


@pytest.mark.anyio
async def test_api_failure_returns_safe_default(yourprovider_provider):
    """Test that API failures return safe defaults."""
    with patch.object(
        yourprovider_provider._client, "generate", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = Exception("API Error")

        result = await yourprovider_provider.score_relevance("question", "answer")

        assert result["score"] == 0.0
        assert "Judge call failed" in result["reasoning"]

# Add tests for all other methods...
```

### 5. Update Factory Function

Add to `app/evaluation/providers/__init__.py`:

```python
from app.evaluation.providers.yourprovider import YourProviderProvider

__all__ = [
    # ... existing exports ...
    "YourProviderProvider",
    "create_judge",
]

def create_judge() -> JudgeProvider:
    """Create a judge provider based on the JUDGE_PROVIDER setting."""
    provider = settings.judge_provider.lower()

    # ... existing providers ...
    
    elif provider == "yourprovider":
        return YourProviderProvider(
            api_key=settings.yourprovider_api_key,
            base_url=settings.yourprovider_base_url,
            model=settings.yourprovider_model,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Must be one of: gemini, ollama, openai, anthropic, yourprovider"
        )
```

### 6. Update Documentation

- Add provider configuration to `.env.example`
- Update `README.md` with provider setup instructions
- Run all tests: `pytest tests/ -v`

## Code Style

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for all function parameters and return values
- Maximum line length: **88 characters** (Black formatter default)
- Use **docstrings** for all classes and public methods

### Formatting

We use **Black** for code formatting:

```bash
# Install Black
pip install black

# Format code
black app/ tests/
```

### Import Order

Follow this import order (enforced by isort):

1. Standard library imports
2. Third-party imports
3. Local application imports

Example:
```python
from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.core.config import settings
from app.evaluation.providers.base import JudgeProvider
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `GeminiProvider`)
- **Functions/Methods**: `snake_case` (e.g., `score_relevance`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `RELEVANCE_PROMPT`)
- **Private methods**: Prefix with `_` (e.g., `_call_gemini`)

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/evaluation/providers/test_gemini.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing

# Run only integration tests
pytest tests/integration/ -v
```

### Test Requirements

All contributions must include tests:

- **Unit tests** for new functions/classes
- **Integration tests** for new providers
- **Minimum 80% code coverage** for new code
- All tests must pass before merging

### Writing Tests

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.anyio  # Use for async tests
async def test_your_feature():
    """Test description in docstring."""
    # Arrange
    expected = {"score": 0.9}
    
    # Act
    result = await your_function()
    
    # Assert
    assert result == expected
```

### Test File Structure

```
tests/
├── conftest.py              # Shared fixtures
├── core/
│   ├── __init__.py
│   └── test_config.py       # Config tests
├── evaluation/
│   ├── __init__.py
│   └── providers/
│       ├── __init__.py
│       ├── test_gemini.py   # Provider tests
│       └── test_factory.py  # Factory tests
└── integration/
    ├── __init__.py
    └── test_multi_provider.py
```

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest main:
   ```bash
   git checkout main
   git pull upstream main
   git checkout feature/your-feature
   git rebase main
   ```

2. **Run all tests**:
   ```bash
   pytest tests/ -v
   ```

3. **Format your code**:
   ```bash
   black app/ tests/
   ```

4. **Update documentation** if needed

### Submitting a PR

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature
   ```

2. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to any related issues
   - Screenshots (if UI changes)

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] All tests pass
   - [ ] Added new tests for new functionality
   - [ ] Updated documentation
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-reviewed code
   - [ ] Commented complex code sections
   - [ ] Updated documentation
   - [ ] No new warnings generated
   ```

### Review Process

- At least one maintainer review required
- All CI checks must pass
- Address review comments
- Keep commits clean and well-documented

### Commit Messages

Follow conventional commits format:

```
type(scope): subject

body (optional)

footer (optional)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `style`: Code style changes (formatting)
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(providers): add Cohere provider support"
git commit -m "fix(gemini): handle timeout errors gracefully"
git commit -m "docs(readme): update provider configuration examples"
```

## Documentation

### Code Documentation

- **Docstrings** for all public classes and methods
- **Inline comments** for complex logic
- **Type hints** for all function parameters and returns

Example:
```python
async def score_relevance(self, question: str, answer: str) -> dict:
    """
    Score how relevant the answer is to the question.

    Args:
        question: The input question/prompt
        answer: The generated answer

    Returns:
        dict: {"score": float (0.0-1.0), "reasoning": str}
    
    Raises:
        ValueError: If question or answer is empty
    """
    pass
```

### README Updates

Update `README.md` when:
- Adding new features
- Changing configuration options
- Modifying setup instructions
- Adding new providers

### .env.example Updates

Update `.env.example` when:
- Adding new configuration variables
- Changing default values
- Adding new providers

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues and PRs first

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (check LICENSE file).

---

Thank you for contributing! 🎉
