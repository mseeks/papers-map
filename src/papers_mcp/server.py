"""MCP server for academic paper search.

This module exposes paper search and retrieval tools via the Model Context Protocol.
Tool names and descriptions are optimized for AI agent usage.
"""

import re
from typing import Annotated

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from papers_mcp.domain import (
    Paper,
    PaperDetails,
    SearchQuery,
    ServiceError,
)
from papers_mcp.service import (
    get_paper_details,
    search_papers,
)

# =============================================================================
# Response Models with Field Annotations
# =============================================================================


class AuthorResponse(BaseModel):
    """Author information."""

    author_id: str | None = Field(
        description="Unique Semantic Scholar author identifier."
    )
    name: str = Field(description="Author's display name.")


class PaperSummary(BaseModel):
    """Summary of a paper for search results."""

    paper_id: str = Field(
        description="Semantic Scholar paper ID. Use with get_paper for full details."
    )
    title: str = Field(description="Title of the paper.")
    authors: list[AuthorResponse] = Field(description="List of paper authors.")
    year: int | None = Field(description="Publication year.")
    venue: str | None = Field(description="Publication venue (journal/conference).")
    citation_count: int = Field(description="Total number of citations.")
    abstract_snippet: str | None = Field(
        description="First 300 characters of the abstract."
    )
    pdf_url: str | None = Field(
        description="Direct URL to download the paper PDF (open access)."
    )


class SearchResponse(BaseModel):
    """Response from paper search."""

    query: str = Field(description="The search query that was executed.")
    total: int = Field(description="Total number of matching papers.")
    papers: list[PaperSummary] = Field(description="List of matching papers.")
    offset: int = Field(description="Current pagination offset.")
    next_offset: int | None = Field(
        description="Offset for next page of results. None if no more results."
    )


class PaperDetailsResponse(BaseModel):
    """Detailed information about a paper."""

    paper_id: str = Field(description="Semantic Scholar paper ID.")
    title: str = Field(description="Title of the paper.")
    authors: list[AuthorResponse] = Field(description="List of all paper authors.")
    year: int | None = Field(description="Publication year.")
    venue: str | None = Field(description="Publication venue (journal/conference).")
    citation_count: int = Field(description="Total number of citations.")
    influential_citation_count: int = Field(
        description="Number of influential citations (highly relevant citations)."
    )
    references_count: int = Field(description="Number of references in the paper.")
    abstract: str | None = Field(description="Full abstract text.")
    tldr: str | None = Field(
        description="AI-generated one-sentence summary of the paper."
    )
    fields_of_study: list[str] = Field(
        description="Academic fields this paper belongs to."
    )
    publication_types: list[str] = Field(
        description="Types of publication (e.g., Journal Article, Conference)."
    )
    external_ids: dict[str, str] = Field(
        description="External identifiers: DOI, ArXiv, PubMed, etc."
    )
    url: str = Field(description="Semantic Scholar URL for this paper.")
    pdf_url: str | None = Field(
        description="Direct URL to download the paper PDF (open access)."
    )


class ErrorResponse(BaseModel):
    """Error response from the API."""

    error: bool = Field(default=True, description="Indicates this is an error.")
    code: str = Field(description="Error code (e.g., 'rate_limit', 'not_found').")
    message: str = Field(description="Human-readable error message.")


# =============================================================================
# Initialize MCP Server
# =============================================================================

mcp = FastMCP(
    "papers",
    instructions=(
        "Academic paper search and retrieval using Semantic Scholar. "
        "Use search_papers to find papers by topic, then get_paper for details. "
        "Paper IDs from search results can be used directly with get_paper."
    ),
)


# =============================================================================
# Helper Functions
# =============================================================================


def detect_id_type(paper_id: str) -> str:
    """Detect the type of paper ID and add appropriate prefix.

    Args:
        paper_id: Raw paper ID input.

    Returns:
        Paper ID with appropriate prefix for Semantic Scholar API.
    """
    # Already has a prefix
    if ":" in paper_id and paper_id.split(":")[0].upper() in {
        "DOI",
        "ARXIV",
        "MAG",
        "ACL",
        "PMID",
        "PMCID",
        "CORPUSID",
    }:
        return paper_id

    # Looks like a DOI (contains slash after prefix)
    if "/" in paper_id and re.match(r"^10\.\d{4,}", paper_id):
        return f"DOI:{paper_id}"

    # Looks like an ArXiv ID (YYMM.NNNNN format)
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", paper_id):
        return f"ARXIV:{paper_id}"

    # Assume it's a Semantic Scholar ID
    return paper_id


def paper_to_summary(paper: Paper) -> PaperSummary:
    """Convert domain Paper to PaperSummary response.

    Args:
        paper: Domain paper object.

    Returns:
        PaperSummary response model.
    """
    abstract_snippet = None
    if paper.abstract:
        abstract_snippet = paper.abstract[:300]
        if len(paper.abstract) > 300:
            abstract_snippet += "..."

    return PaperSummary(
        paper_id=paper.paper_id,
        title=paper.title,
        authors=[
            AuthorResponse(author_id=a.author_id, name=a.name) for a in paper.authors
        ],
        year=paper.year,
        venue=paper.venue,
        citation_count=paper.citation_count,
        abstract_snippet=abstract_snippet,
        pdf_url=paper.open_access_pdf,
    )


def details_to_response(details: PaperDetails) -> PaperDetailsResponse:
    """Convert domain PaperDetails to PaperDetailsResponse.

    Args:
        details: Domain paper details object.

    Returns:
        PaperDetailsResponse response model.
    """
    paper = details.paper
    return PaperDetailsResponse(
        paper_id=paper.paper_id,
        title=paper.title,
        authors=[
            AuthorResponse(author_id=a.author_id, name=a.name) for a in paper.authors
        ],
        year=paper.year,
        venue=paper.venue,
        citation_count=paper.citation_count,
        influential_citation_count=details.influential_citation_count,
        references_count=details.references_count,
        abstract=paper.abstract,
        tldr=details.tldr,
        fields_of_study=list(details.fields_of_study),
        publication_types=list(details.publication_types),
        external_ids=dict(details.external_ids),
        url=paper.url,
        pdf_url=paper.open_access_pdf,
    )


# =============================================================================
# Tool Implementation Functions (for testing)
# =============================================================================


async def search_papers_impl(
    query: str,
    limit: int = 10,
    year_start: int | None = None,
    year_end: int | None = None,
    offset: int = 0,
) -> SearchResponse | ErrorResponse:
    """Search academic papers by relevance.

    Args:
        query: Search query string.
        limit: Maximum papers to return (1-100).
        year_start: Filter papers from this year onwards.
        year_end: Filter papers up to this year.
        offset: Pagination offset.

    Returns:
        SearchResponse on success, ErrorResponse on failure.
    """
    search_query = SearchQuery(
        query=query,
        limit=min(max(1, limit), 100),
        offset=offset,
        year_start=year_start,
        year_end=year_end,
    )

    result = await search_papers(search_query)

    if isinstance(result, ServiceError):
        return ErrorResponse(code=result.code, message=result.message)

    return SearchResponse(
        query=query,
        total=result.total,
        papers=[paper_to_summary(p) for p in result.papers],
        offset=result.offset,
        next_offset=result.next_offset,
    )


async def get_paper_impl(paper_id: str) -> PaperDetailsResponse | ErrorResponse:
    """Get detailed information about a specific paper.

    Args:
        paper_id: Paper identifier (Semantic Scholar ID, DOI, ArXiv ID, etc.).

    Returns:
        PaperDetailsResponse on success, ErrorResponse on failure.
    """
    normalized_id = detect_id_type(paper_id)
    result = await get_paper_details(normalized_id)

    if isinstance(result, ServiceError):
        return ErrorResponse(code=result.code, message=result.message)

    return details_to_response(result)


# =============================================================================
# MCP Tool Registrations
# =============================================================================


@mcp.tool(
    annotations={
        "title": "Search Papers",
        "readOnlyHint": True,
        "openWorldHint": True,
    }
)
async def search_papers_mcp(
    query: Annotated[
        str,
        Field(
            description=(
                "Search query for finding relevant papers. Use specific terms, "
                "paper titles, author names, or research topics. Boolean operators "
                'supported: +required -excluded "exact phrase".'
            ),
            min_length=1,
            max_length=500,
        ),
    ],
    limit: Annotated[
        int,
        Field(
            description="Maximum papers to return. Default 10.",
            ge=1,
            le=100,
        ),
    ] = 10,
    year_start: Annotated[
        int | None,
        Field(
            description="Filter papers published from this year onwards.",
            ge=1900,
            le=2100,
        ),
    ] = None,
    year_end: Annotated[
        int | None,
        Field(
            description="Filter papers published up to this year.",
            ge=1900,
            le=2100,
        ),
    ] = None,
    offset: Annotated[
        int,
        Field(
            description=(
                "Pagination offset. Use next_offset from previous results "
                "to get more papers."
            ),
            ge=0,
        ),
    ] = 0,
) -> SearchResponse | ErrorResponse:
    """Search academic papers by relevance.

    Returns papers matching the query ranked by relevance. Each result includes
    title, authors, year, citation count, abstract preview, and paper ID.
    Use the paper ID with get_paper for full details.
    """
    return await search_papers_impl(query, limit, year_start, year_end, offset)


@mcp.tool(
    annotations={
        "title": "Get Paper Details",
        "readOnlyHint": True,
        "openWorldHint": True,
    }
)
async def get_paper_mcp(
    paper_id: Annotated[
        str,
        Field(
            description=(
                "Paper identifier. Accepts: Semantic Scholar ID (from search results), "
                "DOI (e.g., 10.1000/xyz), ArXiv ID (e.g., 2301.00001), or prefixed IDs "
                "(DOI:xxx, ARXIV:xxx, PMID:xxx)."
            ),
            min_length=1,
            max_length=200,
        ),
    ],
) -> PaperDetailsResponse | ErrorResponse:
    """Get detailed information about a specific paper.

    Returns comprehensive paper details including full abstract, TLDR summary,
    citation metrics, fields of study, external IDs (DOI, ArXiv), and PDF link
    if available.
    """
    return await get_paper_impl(paper_id)


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
