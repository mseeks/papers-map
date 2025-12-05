"""MCP server for academic paper search.

This module exposes paper search and retrieval tools via the Model Context Protocol.
Tool names and descriptions are optimized for AI agent usage.
"""

import re
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

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

# Initialize the MCP server
mcp = FastMCP(
    "papers",
    instructions=(
        "Academic paper search and retrieval using Semantic Scholar. "
        "Use search_papers to find papers by topic, then get_paper for details. "
        "Paper IDs from search results can be used directly with get_paper."
    ),
)


def format_paper_summary(paper: Paper) -> str:
    """Format a paper as a concise summary for search results.

    Args:
        paper: Paper to format.

    Returns:
        Formatted paper summary string.
    """
    lines = [f"**{paper.title}**"]

    if paper.authors:
        author_names = ", ".join(a.name for a in paper.authors[:3])
        if len(paper.authors) > 3:
            author_names += f" et al. ({len(paper.authors)} authors)"
        lines.append(f"Authors: {author_names}")

    meta = []
    if paper.year:
        meta.append(str(paper.year))
    if paper.venue:
        meta.append(paper.venue)
    meta.append(f"{paper.citation_count} citations")
    lines.append(" | ".join(meta))

    if paper.abstract:
        abstract = paper.abstract[:300]
        if len(paper.abstract) > 300:
            abstract += "..."
        lines.append(f"Abstract: {abstract}")

    lines.append(f"ID: {paper.paper_id}")
    if paper.open_access_pdf:
        lines.append(f"PDF: {paper.open_access_pdf}")

    return "\n".join(lines)


def format_paper_details(details: PaperDetails) -> str:
    """Format detailed paper information.

    Args:
        details: Paper details to format.

    Returns:
        Formatted paper details string.
    """
    paper = details.paper
    lines = [f"# {paper.title}"]

    if paper.authors:
        author_names = ", ".join(a.name for a in paper.authors)
        lines.append(f"\n**Authors:** {author_names}")

    meta = []
    if paper.year:
        meta.append(f"Year: {paper.year}")
    if paper.venue:
        meta.append(f"Venue: {paper.venue}")
    meta.append(f"Citations: {paper.citation_count}")
    meta.append(f"Influential citations: {details.influential_citation_count}")
    meta.append(f"References: {details.references_count}")
    if meta:
        lines.append(" | ".join(meta))

    if details.tldr:
        lines.append(f"\n**TLDR:** {details.tldr}")

    if paper.abstract:
        lines.append(f"\n**Abstract:**\n{paper.abstract}")

    if details.fields_of_study:
        lines.append(f"\n**Fields:** {', '.join(details.fields_of_study)}")

    if details.publication_types:
        lines.append(f"**Type:** {', '.join(details.publication_types)}")

    if details.external_ids:
        ids = [f"{k}: {v}" for k, v in details.external_ids.items()]
        lines.append("\n**External IDs:**\n" + "\n".join(ids))

    lines.append(f"\n**Semantic Scholar URL:** {paper.url}")
    if paper.open_access_pdf:
        lines.append(f"**Open Access PDF:** {paper.open_access_pdf}")

    return "\n".join(lines)


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


async def search_papers_tool(
    query: str,
    limit: int = 10,
    year_start: int | None = None,
    year_end: int | None = None,
    offset: int = 0,
) -> str:
    """Search academic papers by relevance.

    Args:
        query: Search query string.
        limit: Maximum papers to return (1-100).
        year_start: Filter papers from this year onwards.
        year_end: Filter papers up to this year.
        offset: Pagination offset.

    Returns:
        Formatted search results string.
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
        return f"Error ({result.code}): {result.message}"

    if result.total == 0:
        return f"No papers found for query: {query}"

    lines = [f'Found {result.total} papers for "{query}":\n']

    for i, paper in enumerate(result.papers, 1):
        lines.append(f"## {i}. {format_paper_summary(paper)}\n")

    if result.next_offset:
        lines.append(
            f"---\nShowing {len(result.papers)} of {result.total}. "
            f"Use offset={result.next_offset} for more results."
        )

    return "\n".join(lines)


async def get_paper_tool(paper_id: str) -> str:
    """Get detailed information about a specific paper.

    Args:
        paper_id: Paper identifier (Semantic Scholar ID, DOI, ArXiv ID, etc.).

    Returns:
        Formatted paper details string.
    """
    normalized_id = detect_id_type(paper_id)
    result = await get_paper_details(normalized_id)

    if isinstance(result, ServiceError):
        return f"Error ({result.code}): {result.message}"

    return format_paper_details(result)


# MCP tool registrations with optimized signatures for AI agents
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
) -> str:
    """Search academic papers by relevance.

    Returns papers matching the query ranked by relevance. Each result includes
    title, authors, year, citation count, abstract preview, and paper ID.
    Use the paper ID with get_paper for full details.
    """
    return await search_papers_tool(query, limit, year_start, year_end, offset)


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
) -> str:
    """Get detailed information about a specific paper.

    Returns comprehensive paper details including full abstract, TLDR summary,
    citation metrics, fields of study, external IDs (DOI, ArXiv), and PDF link
    if available.
    """
    return await get_paper_tool(paper_id)


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
