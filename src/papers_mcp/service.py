"""Paper search service using Semantic Scholar API.

This module provides functions for searching and retrieving papers from the
Semantic Scholar API following functional programming principles.

Retry behaviour (exponential backoff for rate limiting and timeouts) is
configured via :class:`RetryConfig`, which callers may inject — there is no
hidden global state. A single shared :class:`httpx.AsyncClient` is reused
across requests for connection pooling.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from papers_mcp.domain import (
    SEMANTIC_SCHOLAR_API_BASE,
    Author,
    Paper,
    PaperDetails,
    SearchQuery,
    SearchResult,
    ServiceError,
)

logger = logging.getLogger(__name__)

# Awaitable sleep function, injectable so tests can avoid real delays.
SleepFn = Callable[[float], Awaitable[None]]

# Per-request HTTP timeout in seconds.
REQUEST_TIMEOUT: float = 30.0

# Limit parallel requests so we don't hammer the API.
MAX_CONCURRENT_REQUESTS: int = 3


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Configuration for HTTP retries with exponential backoff.

    Attributes:
        max_retries: Maximum number of attempts before giving up.
        base_delay: Initial delay in seconds (doubles each attempt).
        max_delay: Upper bound on the delay between attempts.
        jitter: Fraction of the delay (0-1) added as random jitter.
    """

    max_retries: int = 20
    base_delay: float = 5.0
    max_delay: float = 60.0
    jitter: float = 0.5

    def delay_for(self, attempt: int) -> float:
        """Compute the backoff delay for a retry attempt.

        Args:
            attempt: Current retry attempt (0-indexed).

        Returns:
            Delay in seconds: exponential backoff capped at ``max_delay``,
            plus up to ``jitter`` fraction of random jitter.
        """
        exponential_delay = self.base_delay * (2.0**attempt)
        delay = min(exponential_delay, self.max_delay)
        return float(delay + delay * random.uniform(0, self.jitter))


# Default retry configuration. Read at call time, so tests may monkeypatch it.
DEFAULT_RETRY_CONFIG: RetryConfig = RetryConfig()

# Lazily-created shared resources, bound to the running event loop on first use.
_semaphore: asyncio.Semaphore | None = None
_client: httpx.AsyncClient | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the global semaphore for concurrency control."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _semaphore


def _get_client() -> httpx.AsyncClient:
    """Get or create the shared HTTP client (enables connection pooling)."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    return _client


def reset_semaphore() -> None:
    """Discard the shared semaphore (used between tests to avoid loop reuse)."""
    global _semaphore
    _semaphore = None


async def reset_client() -> None:
    """Close and discard the shared client (used between tests)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


# Fields to request from the API for basic paper info
PAPER_FIELDS = (
    "paperId",
    "title",
    "abstract",
    "authors",
    "year",
    "venue",
    "citationCount",
    "url",
    "openAccessPdf",
)

# Additional fields for detailed paper info
DETAIL_FIELDS = (
    *PAPER_FIELDS,
    "tldr",
    "fieldsOfStudy",
    "publicationTypes",
    "referenceCount",
    "influentialCitationCount",
    "externalIds",
)


def parse_author(data: dict[str, Any]) -> Author:
    """Parse an author from API response data.

    Args:
        data: Raw author data from API.

    Returns:
        Parsed Author domain object.
    """
    return Author(
        author_id=data.get("authorId"),
        name=data.get("name", "Unknown"),
    )


def parse_paper(data: dict[str, Any]) -> Paper:
    """Parse a paper from API response data.

    Args:
        data: Raw paper data from API.

    Returns:
        Parsed Paper domain object.
    """
    open_access_pdf = None
    if pdf_data := data.get("openAccessPdf"):
        open_access_pdf = pdf_data.get("url")

    return Paper(
        paper_id=data.get("paperId", ""),
        title=data.get("title", ""),
        abstract=data.get("abstract"),
        authors=tuple(parse_author(a) for a in data.get("authors", [])),
        year=data.get("year"),
        venue=data.get("venue"),
        citation_count=data.get("citationCount", 0),
        url=data.get("url", ""),
        open_access_pdf=open_access_pdf,
    )


def parse_search_result(data: dict[str, Any], query: SearchQuery) -> SearchResult:
    """Parse search results from API response data.

    Args:
        data: Raw search response from API.
        query: Original search query.

    Returns:
        Parsed SearchResult domain object.
    """
    papers = tuple(parse_paper(p) for p in data.get("data", []))
    return SearchResult(
        query=query.query,
        total=data.get("total", 0),
        papers=papers,
        offset=data.get("offset", 0),
        next_offset=data.get("next"),
    )


def parse_paper_details(data: dict[str, Any]) -> PaperDetails:
    """Parse detailed paper info from API response data.

    Args:
        data: Raw paper details from API.

    Returns:
        Parsed PaperDetails domain object.
    """
    paper = parse_paper(data)

    tldr = None
    if tldr_data := data.get("tldr"):
        tldr = tldr_data.get("text")

    return PaperDetails(
        paper=paper,
        tldr=tldr,
        fields_of_study=tuple(data.get("fieldsOfStudy") or []),
        publication_types=tuple(data.get("publicationTypes") or []),
        references_count=data.get("referenceCount", 0),
        influential_citation_count=data.get("influentialCitationCount", 0),
        external_ids=dict(data.get("externalIds") or {}),
    )


def _year_param(year_start: int | None, year_end: int | None) -> str | None:
    """Build the Semantic Scholar ``year`` filter value, if any.

    Args:
        year_start: Inclusive lower bound, if set.
        year_end: Inclusive upper bound, if set.

    Returns:
        A ``year`` parameter value (e.g. ``"2020-2024"``, ``"2020-"``,
        ``"-2024"``) or None when no bounds are given.
    """
    if year_start is not None and year_end is not None:
        return f"{year_start}-{year_end}"
    if year_start is not None:
        return f"{year_start}-"
    if year_end is not None:
        return f"-{year_end}"
    return None


def _build_search_url(query: SearchQuery) -> str:
    """Build the search API URL with query parameters.

    Args:
        query: Search query with parameters.

    Returns:
        Fully formed API URL.
    """
    params: dict[str, str] = {
        "query": query.query,
        "offset": str(query.offset),
        "limit": str(query.limit),
        "fields": ",".join(PAPER_FIELDS),
    }
    if (year := _year_param(query.year_start, query.year_end)) is not None:
        params["year"] = year
    if query.fields_of_study:
        params["fieldsOfStudy"] = ",".join(query.fields_of_study)

    return f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search?{urlencode(params)}"


def _build_details_url(paper_id: str) -> str:
    """Build the paper details API URL.

    Args:
        paper_id: Semantic Scholar paper ID or external ID.

    Returns:
        Fully formed API URL.
    """
    params = {"fields": ",".join(DETAIL_FIELDS)}
    return f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}?{urlencode(params)}"


async def _make_request_with_retry(
    url: str,
    headers: dict[str, str],
    operation_name: str,
    retry_config: RetryConfig,
    sleep: SleepFn = asyncio.sleep,
) -> httpx.Response | ServiceError:
    """Make an HTTP GET request, retrying on rate limiting and timeouts.

    Args:
        url: The URL to request.
        headers: HTTP headers to include.
        operation_name: Name of the operation, for logging.
        retry_config: Retry/backoff configuration.
        sleep: Awaitable sleep function (injectable for testing).

    Returns:
        The HTTP response on success, or ServiceError on permanent failure
        or exhausted retries.
    """
    semaphore = _get_semaphore()
    client = _get_client()
    max_retries = max(1, retry_config.max_retries)

    async with semaphore:
        for attempt in range(max_retries):
            is_last_attempt = attempt + 1 >= max_retries
            try:
                response = await client.get(url, headers=headers)

                if response.status_code == 200:
                    return response

                if response.status_code == 429:
                    if is_last_attempt:
                        return ServiceError(
                            code="rate_limit",
                            message=f"Rate limit exceeded after {max_retries} attempts",
                        )
                    delay = retry_config.delay_for(attempt)
                    logger.info(
                        "%s: Rate limited, waiting %.1fs (attempt %d/%d)",
                        operation_name,
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await sleep(delay)
                    continue

                if response.status_code == 404:
                    return ServiceError(code="not_found", message="Paper not found")
                if response.status_code >= 500:
                    return ServiceError(
                        code="server_error", message="Semantic Scholar API error"
                    )
                return ServiceError(
                    code="api_error",
                    message=f"API returned status {response.status_code}",
                )

            except httpx.TimeoutException:
                if is_last_attempt:
                    return ServiceError(code="timeout", message="Request timed out")
                delay = retry_config.delay_for(attempt)
                logger.info(
                    "%s: Timeout, waiting %.1fs (attempt %d/%d)",
                    operation_name,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await sleep(delay)
                continue

            except httpx.RequestError as e:
                return ServiceError(code="network_error", message=str(e))

        # Unreachable in practice (the last attempt always returns), but keeps
        # the type checker happy and guards against max_retries == 0.
        return ServiceError(  # pragma: no cover
            code="rate_limit",
            message=f"Rate limit exceeded after {max_retries} attempts",
        )


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Build request headers, including the API key when provided."""
    return {"x-api-key": api_key} if api_key else {}


async def search_papers(
    query: SearchQuery,
    api_key: str | None = None,
    retry_config: RetryConfig | None = None,
    sleep: SleepFn = asyncio.sleep,
) -> SearchResult | ServiceError:
    """Search for papers using the Semantic Scholar API.

    Automatically retries on rate limiting with exponential backoff.

    Args:
        query: Search query with parameters.
        api_key: Optional API key for higher rate limits.
        retry_config: Retry configuration (defaults to DEFAULT_RETRY_CONFIG).
        sleep: Awaitable sleep function (injectable for testing).

    Returns:
        SearchResult on success, ServiceError on failure.
    """
    result = await _make_request_with_retry(
        _build_search_url(query),
        _auth_headers(api_key),
        f"search_papers({query.query[:30]}...)",
        retry_config or DEFAULT_RETRY_CONFIG,
        sleep,
    )
    if isinstance(result, ServiceError):
        return result
    return parse_search_result(result.json(), query)


async def get_paper_details(
    paper_id: str,
    api_key: str | None = None,
    retry_config: RetryConfig | None = None,
    sleep: SleepFn = asyncio.sleep,
) -> PaperDetails | ServiceError:
    """Get detailed information about a paper.

    Automatically retries on rate limiting with exponential backoff.

    Args:
        paper_id: Semantic Scholar paper ID or external ID (e.g., DOI:xxx, ArXiv:xxx).
        api_key: Optional API key for higher rate limits.
        retry_config: Retry configuration (defaults to DEFAULT_RETRY_CONFIG).
        sleep: Awaitable sleep function (injectable for testing).

    Returns:
        PaperDetails on success, ServiceError on failure.
    """
    result = await _make_request_with_retry(
        _build_details_url(paper_id),
        _auth_headers(api_key),
        f"get_paper_details({paper_id[:20]}...)",
        retry_config or DEFAULT_RETRY_CONFIG,
        sleep,
    )
    if isinstance(result, ServiceError):
        return result
    return parse_paper_details(result.json())
