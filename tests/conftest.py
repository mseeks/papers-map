"""Pytest configuration for papers-mcp tests."""

from collections.abc import AsyncGenerator

import pytest

from papers_mcp import service

# Retry config used by default in tests: a single attempt with no delay, so
# rate-limit/timeout paths fail fast instead of sleeping.
TEST_RETRY_CONFIG = service.RetryConfig(
    max_retries=1, base_delay=0.0, max_delay=0.0, jitter=0.0
)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable real retries/delays for all tests by default."""
    monkeypatch.setattr(service, "DEFAULT_RETRY_CONFIG", TEST_RETRY_CONFIG)


@pytest.fixture(autouse=True)
async def reset_http_state() -> AsyncGenerator[None, None]:
    """Reset shared HTTP resources between tests to avoid event-loop reuse."""
    yield
    await service.reset_client()
    service.reset_semaphore()
