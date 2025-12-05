"""Domain types for the papers MCP server.

This module defines immutable domain types following functional DDD principles.
All types are frozen dataclasses to ensure immutability.
"""

from dataclasses import dataclass
from typing import Final

# Semantic Scholar API constants
SEMANTIC_SCHOLAR_API_BASE: Final[str] = "https://api.semanticscholar.org/graph/v1"
DEFAULT_SEARCH_LIMIT: Final[int] = 10
MAX_SEARCH_LIMIT: Final[int] = 100


@dataclass(frozen=True, slots=True)
class Author:
    """Represents a paper author.

    Attributes:
        author_id: Unique identifier for the author.
        name: Display name of the author.
    """

    author_id: str | None
    name: str


@dataclass(frozen=True, slots=True)
class Paper:
    """Represents an academic paper.

    Attributes:
        paper_id: Semantic Scholar paper ID.
        title: Title of the paper.
        abstract: Abstract text, if available.
        authors: List of paper authors.
        year: Publication year, if known.
        venue: Publication venue, if known.
        citation_count: Number of citations.
        url: URL to the paper on Semantic Scholar.
        open_access_pdf: URL to open access PDF, if available.
    """

    paper_id: str
    title: str
    abstract: str | None
    authors: tuple[Author, ...]
    year: int | None
    venue: str | None
    citation_count: int
    url: str
    open_access_pdf: str | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Represents the result of a paper search.

    Attributes:
        query: The original search query.
        total: Total number of matching papers.
        papers: List of papers returned.
        offset: Offset used for pagination.
        next_offset: Offset for next page, None if no more results.
    """

    query: str
    total: int
    papers: tuple[Paper, ...]
    offset: int
    next_offset: int | None


@dataclass(frozen=True, slots=True)
class PaperDetails:
    """Extended paper details including references and citations.

    Attributes:
        paper: The base paper information.
        tldr: AI-generated TLDR summary, if available.
        fields_of_study: List of academic fields this paper belongs to.
        publication_types: Types of publication (e.g., Journal Article).
        references_count: Number of references.
        influential_citation_count: Number of influential citations.
        external_ids: External identifiers (DOI, ArXiv, etc.).
    """

    paper: Paper
    tldr: str | None
    fields_of_study: tuple[str, ...]
    publication_types: tuple[str, ...]
    references_count: int
    influential_citation_count: int
    external_ids: dict[str, str]


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """Represents a search query with parameters.

    Attributes:
        query: The search query string.
        limit: Maximum number of results to return.
        offset: Offset for pagination.
        year_start: Filter papers from this year onwards.
        year_end: Filter papers up to this year.
        fields_of_study: Filter by specific fields of study.
    """

    query: str
    limit: int = DEFAULT_SEARCH_LIMIT
    offset: int = 0
    year_start: int | None = None
    year_end: int | None = None
    fields_of_study: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ServiceError:
    """Represents a service error.

    Attributes:
        code: Error code.
        message: Human-readable error message.
    """

    code: str
    message: str
