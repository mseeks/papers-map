"""Pytest configuration for papers-mcp tests."""

from collections.abc import Generator

import pytest

from papers_mcp.service import set_testing_mode


@pytest.fixture(autouse=True)
def enable_testing_mode() -> Generator[None, None, None]:
    """Enable testing mode for all tests to disable retries and delays."""
    set_testing_mode(True)
    yield
    set_testing_mode(False)
