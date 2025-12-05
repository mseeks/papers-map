"""Paper search service using Semantic Scholar API.

This module provides pure functions for searching and retrieving papers
from the Semantic Scholar API following functional programming principles.
"""

from typing import Any
from urllib.parse import quote_plus

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
    base = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/{paper_id}"
    fields = quote_plus(",".join(DETAIL_FIELDS))
    return f"{base}?fields={fields}"


def _handle_error_response(response: httpx.Response) -> ServiceError:
    """Convert HTTP error response to ServiceError.

    Args:
        response: HTTP response with error status.

    Returns:
        ServiceError with appropriate code and message.
    """
    if response.status_code == 404:
        return ServiceError(code="not_found", message="Paper not found")
    if response.status_code == 429:
        return ServiceError(code="rate_limit", message="Rate limit exceeded")
    if response.status_code >= 500:
        return ServiceError(code="server_error", message="Semantic Scholar API error")
    return ServiceError(
        code="api_error",
        message=f"API returned status {response.status_code}",
    )


async def search_papers(
    query: SearchQuery,
    api_key: str | None = None,
) -> SearchResult | ServiceError:
    """Search for papers using the Semantic Scholar API.

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

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
        except httpx.TimeoutException:
            return ServiceError(code="timeout", message="Request timed out")
        except httpx.RequestError as e:
            return ServiceError(code="network_error", message=str(e))

        if response.status_code != 200:
            return _handle_error_response(response)

        data = response.json()
        return parse_search_result(data, query)


async def get_paper_details(
    paper_id: str,
    api_key: str | None = None,
) -> PaperDetails | ServiceError:
    """Get detailed information about a paper.

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

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
        except httpx.TimeoutException:
            return ServiceError(code="timeout", message="Request timed out")
        except httpx.RequestError as e:
            return ServiceError(code="network_error", message=str(e))

        if response.status_code != 200:
            return _handle_error_response(response)

        data = response.json()
        return parse_paper_details(data)
