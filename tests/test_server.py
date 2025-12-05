"""Tests for the MCP server tools.

Tests the MCP tool functions for searching and retrieving papers.
"""

from unittest.mock import AsyncMock, patch

import pytest

from papers_mcp.domain import (
    Author,
    Paper,
    PaperDetails,
    SearchResult,
    ServiceError,
)
from papers_mcp.server import (
    format_paper_details,
    format_paper_summary,
    get_paper_tool,
    search_papers_tool,
)


@pytest.fixture
def sample_paper() -> Paper:
    """Create a sample paper for testing."""
    return Paper(
        paper_id="abc123",
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models...",
        authors=(
            Author(author_id="1", name="Ashish Vaswani"),
            Author(author_id="2", name="Noam Shazeer"),
        ),
        year=2017,
        venue="NeurIPS",
        citation_count=50000,
        url="https://www.semanticscholar.org/paper/abc123",
        open_access_pdf="https://arxiv.org/pdf/1706.03762.pdf",
    )


@pytest.fixture
def sample_search_result(sample_paper: Paper) -> SearchResult:
    """Create a sample search result for testing."""
    return SearchResult(
        query="transformer",
        total=100,
        papers=(sample_paper,),
        offset=0,
        next_offset=10,
    )


@pytest.fixture
def sample_paper_details(sample_paper: Paper) -> PaperDetails:
    """Create sample paper details for testing."""
    return PaperDetails(
        paper=sample_paper,
        tldr="This paper introduces the Transformer architecture.",
        fields_of_study=("Computer Science", "Artificial Intelligence"),
        publication_types=("Conference",),
        references_count=50,
        influential_citation_count=5000,
        external_ids={"DOI": "10.5555/3295222.3295349", "ArXiv": "1706.03762"},
    )


class TestFormatPaperSummary:
    """Tests for format_paper_summary function."""

    def test_format_complete_paper(self, sample_paper: Paper) -> None:
        """Format a paper with all fields."""
        result = format_paper_summary(sample_paper)

        assert "Attention Is All You Need" in result
        assert "2017" in result
        assert "Ashish Vaswani" in result
        assert "50000 citations" in result
        assert "abc123" in result

    def test_format_paper_missing_fields(self) -> None:
        """Format a paper with missing optional fields."""
        paper = Paper(
            paper_id="xyz789",
            title="Minimal Paper",
            abstract=None,
            authors=(),
            year=None,
            venue=None,
            citation_count=0,
            url="https://example.com",
            open_access_pdf=None,
        )
        result = format_paper_summary(paper)

        assert "Minimal Paper" in result
        assert "xyz789" in result


class TestFormatPaperDetails:
    """Tests for format_paper_details function."""

    def test_format_full_details(self, sample_paper_details: PaperDetails) -> None:
        """Format complete paper details."""
        result = format_paper_details(sample_paper_details)

        assert "Attention Is All You Need" in result
        assert "TLDR:" in result
        assert "Transformer architecture" in result
        assert "Computer Science" in result
        assert "DOI:" in result
        assert "ArXiv:" in result

    def test_format_details_no_tldr(self, sample_paper: Paper) -> None:
        """Format details without TLDR."""
        details = PaperDetails(
            paper=sample_paper,
            tldr=None,
            fields_of_study=(),
            publication_types=(),
            references_count=0,
            influential_citation_count=0,
            external_ids={},
        )
        result = format_paper_details(details)

        assert "Attention Is All You Need" in result
        assert "TLDR:" not in result


class TestSearchPapersTool:
    """Tests for the search_papers MCP tool."""

    @pytest.mark.asyncio
    async def test_search_papers_success(
        self,
        sample_search_result: SearchResult,
    ) -> None:
        """Successfully search for papers."""
        with patch(
            "papers_mcp.server.search_papers",
            new_callable=AsyncMock,
            return_value=sample_search_result,
        ):
            result = await search_papers_tool("transformer")

            assert "Found 100 papers" in result
            assert "Attention Is All You Need" in result

    @pytest.mark.asyncio
    async def test_search_papers_with_parameters(
        self,
        sample_search_result: SearchResult,
    ) -> None:
        """Search with custom parameters."""
        with patch(
            "papers_mcp.server.search_papers",
            new_callable=AsyncMock,
            return_value=sample_search_result,
        ) as mock_search:
            result = await search_papers_tool(
                "llm",
                limit=5,
                year_start=2020,
                year_end=2024,
            )

            assert "Found 100 papers" in result
            # Verify the search was called with correct parameters
            call_args = mock_search.call_args[0][0]
            assert call_args.query == "llm"
            assert call_args.limit == 5
            assert call_args.year_start == 2020
            assert call_args.year_end == 2024

    @pytest.mark.asyncio
    async def test_search_papers_no_results(self) -> None:
        """Handle search with no results."""
        empty_result = SearchResult(
            query="nonexistent",
            total=0,
            papers=(),
            offset=0,
            next_offset=None,
        )
        with patch(
            "papers_mcp.server.search_papers",
            new_callable=AsyncMock,
            return_value=empty_result,
        ):
            result = await search_papers_tool("nonexistent")

            assert "No papers found" in result

    @pytest.mark.asyncio
    async def test_search_papers_error(self) -> None:
        """Handle search error."""
        error = ServiceError(code="rate_limit", message="Rate limit exceeded")
        with patch(
            "papers_mcp.server.search_papers",
            new_callable=AsyncMock,
            return_value=error,
        ):
            result = await search_papers_tool("test")

            assert "Error" in result
            assert "Rate limit exceeded" in result


class TestGetPaperTool:
    """Tests for the get_paper MCP tool."""

    @pytest.mark.asyncio
    async def test_get_paper_success(
        self,
        sample_paper_details: PaperDetails,
    ) -> None:
        """Successfully get paper details."""
        with patch(
            "papers_mcp.server.get_paper_details",
            new_callable=AsyncMock,
            return_value=sample_paper_details,
        ):
            result = await get_paper_tool("abc123")

            assert "Attention Is All You Need" in result
            assert "TLDR:" in result

    @pytest.mark.asyncio
    async def test_get_paper_by_doi(
        self,
        sample_paper_details: PaperDetails,
    ) -> None:
        """Get paper by DOI."""
        with patch(
            "papers_mcp.server.get_paper_details",
            new_callable=AsyncMock,
            return_value=sample_paper_details,
        ) as mock_get:
            _ = await get_paper_tool("10.5555/3295222.3295349")

            # Should add DOI: prefix
            call_args = mock_get.call_args[0][0]
            assert call_args == "DOI:10.5555/3295222.3295349"

    @pytest.mark.asyncio
    async def test_get_paper_by_arxiv(
        self,
        sample_paper_details: PaperDetails,
    ) -> None:
        """Get paper by ArXiv ID."""
        with patch(
            "papers_mcp.server.get_paper_details",
            new_callable=AsyncMock,
            return_value=sample_paper_details,
        ) as mock_get:
            _ = await get_paper_tool("1706.03762")

            # Should add ARXIV: prefix
            call_args = mock_get.call_args[0][0]
            assert call_args == "ARXIV:1706.03762"

    @pytest.mark.asyncio
    async def test_get_paper_not_found(self) -> None:
        """Handle paper not found."""
        error = ServiceError(code="not_found", message="Paper not found")
        with patch(
            "papers_mcp.server.get_paper_details",
            new_callable=AsyncMock,
            return_value=error,
        ):
            result = await get_paper_tool("notfound")

            assert "Error" in result
            assert "Paper not found" in result
