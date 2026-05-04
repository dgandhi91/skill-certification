import pytest


@pytest.fixture
def anyio_backend():
    """Configure anyio to use asyncio backend only."""
    return "asyncio"
