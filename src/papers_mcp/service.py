"""Paper search service using Semantic Scholar API.

This module provides pure functions for searching and retrieving papers
from the Semantic Scholar API following functional programming principles.

Includes automatic retry logic with exponential backoff for rate limiting.
"""

import asyncio
import logging
import random
from typing import Any

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

# Configure logging
logger = logging.getLogger(__name__)

# Retry configuration (can be overridden for testing)
MAX_RETRIES = 20  # Keep trying for a while
BASE_DELAY = 5.0  # Start with 5 second delay
MAX_DELAY = 60.0  # Cap at 60 seconds between retries
JITTER = 0.5  # Random jitter factor (0-50% of delay)

# Testing mode - set to True to disable retries and delays
_testing_mode = False


def set_testing_mode(enabled: bool) -> None:
    """Enable or disable testing mode (disables retries and delays)."""
    global _testing_mode
    _testing_mode = enabled


# Concurrency control - limit parallel requests to avoid hammering the API
_semaphore: asyncio.Semaphore | None = None
MAX_CONCURRENT_REQUESTS = 3


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the global semaphore for concurrency control."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _semaphore


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


def _build_search_url(query: SearchQuery) -> str:
    """Build the search API URL with query parameters.

    Args:
        query: Search query with parameters.

    Returns:
        Fully formed API URL.
    """
    from urllib.parse import quote_plus

    base = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
    params = [
        f"query={quote_plus(query.query)}",
        f"offset={query.offset}",
        f"limit={query.limit}",
        f"fields={quote_plus(','.join(PAPER_FIELDS))}",
    ]

    if query.year_start is not None and query.year_end is not None:
        params.append(f"year={query.year_start}-{query.year_end}")
    elif query.year_start is not None:
        params.append(f"year={query.year_start}-")
    elif query.year_end is not None:
        params.append(f"year=-{query.year_end}")

    if query.fields_of_study:
        params.append(f"fieldsOfStudy={quote_plus(query.fields_of_study[0])}")

    return f"{base}?{'&'.join(params)}"


def _build_details_url(paper_id: str) -> str:
    """Build the paper details API URL.

    Args:
        paper_id: Semantic Scholar paper ID or external ID.

    Returns:
        Fully formed API URL.
    """
    from urllib.parse import quote_plus

    base = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
    fields = quote_plus(",".join(DETAIL_FIELDS))
    return f"{base}?fields={fields}"


def _calculate_delay(attempt: int) -> float:
    """Calculate delay with exponential backoff and jitter.

    Args:
        attempt: Current retry attempt (0-indexed).

    Returns:
        Delay in seconds.
    """
    # Exponential backoff: 5, 10, 20, 40, 60, 60, 60...
    exponential_delay = BASE_DELAY * (2.0**attempt)
    delay = min(exponential_delay, MAX_DELAY)
    # Add random jitter
    jitter = delay * random.uniform(0, JITTER)
    return float(delay + jitter)


async def _make_request_with_retry(
    url: str,
    headers: dict[str, str],
    operation_name: str,
) -> httpx.Response | ServiceError:
    """Make an HTTP request with automatic retry on rate limiting.

    Args:
        url: The URL to request.
        headers: HTTP headers to include.
        operation_name: Name of the operation for logging.

    Returns:
        The HTTP response on success, or ServiceError on permanent failure.
    """
    semaphore = _get_semaphore()
    max_retries = 1 if _testing_mode else MAX_RETRIES

    async with semaphore, httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(url, headers=headers, timeout=30.0)

                # Success
                if response.status_code == 200:
                    return response

                # Rate limited - retry with backoff (unless testing)
                if response.status_code == 429:
                    if _testing_mode:
                        return ServiceError(
                            code="rate_limit", message="Rate limit exceeded"
                        )
                    delay = _calculate_delay(attempt)
                    logger.info(
                        f"{operation_name}: Rate limited, "
                        f"waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue

                # Other errors - don't retry
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
                if _testing_mode:
                    return ServiceError(code="timeout", message="Request timed out")
                # Timeout - retry with backoff
                delay = _calculate_delay(attempt)
                logger.info(
                    f"{operation_name}: Timeout, "
                    f"waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
                continue

            except httpx.RequestError as e:
                return ServiceError(code="network_error", message=str(e))

        # Exhausted all retries
        return ServiceError(
            code="rate_limit",
            message=f"Rate limit exceeded after {max_retries} retries",
        )


async def search_papers(
    query: SearchQuery,
    api_key: str | None = None,
) -> SearchResult | ServiceError:
    """Search for papers using the Semantic Scholar API.

    Automatically retries on rate limiting with exponential backoff.

    Args:
        query: Search query with parameters.
        api_key: Optional API key for higher rate limits.

    Returns:
        SearchResult on success, ServiceError on failure.
    """
    url = _build_search_url(query)
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    result = await _make_request_with_retry(
        url, headers, f"search_papers({query.query[:30]}...)"
    )

    if isinstance(result, ServiceError):
        return result

    data = result.json()
    return parse_search_result(data, query)


async def get_paper_details(
    paper_id: str,
    api_key: str | None = None,
) -> PaperDetails | ServiceError:
    """Get detailed information about a paper.

    Automatically retries on rate limiting with exponential backoff.

    Args:
        paper_id: Semantic Scholar paper ID or external ID (e.g., DOI:xxx, ArXiv:xxx).
        api_key: Optional API key for higher rate limits.

    Returns:
        PaperDetails on success, ServiceError on failure.
    """
    url = _build_details_url(paper_id)
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    result = await _make_request_with_retry(
        url, headers, f"get_paper_details({paper_id[:20]}...)"
    )

    if isinstance(result, ServiceError):
        return result

    data = result.json()
    return parse_paper_details(data)
