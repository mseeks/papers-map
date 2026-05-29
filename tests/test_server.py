"""Tests for the MCP server tools.

Tests the MCP tool functions for searching and retrieving papers.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client

from papers_mcp.domain import (
    Author,
    Paper,
    PaperDetails,
    SearchResult,
    ServiceError,
)
from papers_mcp.server import (
    ErrorResponse,
    PaperDetailsResponse,
    PaperSummary,
    SearchResponse,
    details_to_response,
    detect_id_type,
    get_paper_impl,
    main,
    mcp,
    paper_to_summary,
    search_papers_impl,
)


@pytest.fixture
def sample_paper() -> Paper:
    """Create a sample paper for testing."""
    return Paper(
        paper_id="abc123",
        title="Attention Is All You Need",
        abstract=(
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks."
        ),
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


class TestPaperToSummary:
    """Tests for paper_to_summary function."""

    def test_converts_paper_to_summary(self, sample_paper: Paper) -> None:
        """Convert a paper with all fields."""
        result = paper_to_summary(sample_paper)

        assert isinstance(result, PaperSummary)
        assert result.paper_id == "abc123"
        assert result.title == "Attention Is All You Need"
        assert len(result.authors) == 2
        assert result.authors[0].name == "Ashish Vaswani"
        assert result.year == 2017
        assert result.venue == "NeurIPS"
        assert result.citation_count == 50000
        assert result.pdf_url == "https://arxiv.org/pdf/1706.03762.pdf"

    def test_truncates_long_abstract(self) -> None:
        """Truncate abstract longer than 300 chars."""
        long_abstract = "A" * 500
        paper = Paper(
            paper_id="xyz",
            title="Test",
            abstract=long_abstract,
            authors=(),
            year=2020,
            venue=None,
            citation_count=0,
            url="https://example.com",
            open_access_pdf=None,
        )
        result = paper_to_summary(paper)

        assert result.abstract_snippet is not None
        assert len(result.abstract_snippet) == 303  # 300 + "..."
        assert result.abstract_snippet.endswith("...")

    def test_handles_missing_fields(self) -> None:
        """Handle paper with missing optional fields."""
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
        result = paper_to_summary(paper)

        assert result.paper_id == "xyz789"
        assert result.abstract_snippet is None
        assert result.pdf_url is None
        assert result.year is None


class TestDetailsToResponse:
    """Tests for details_to_response function."""

    def test_converts_details_to_response(
        self, sample_paper_details: PaperDetails
    ) -> None:
        """Convert paper details to response model."""
        result = details_to_response(sample_paper_details)

        assert isinstance(result, PaperDetailsResponse)
        assert result.paper_id == "abc123"
        assert result.title == "Attention Is All You Need"
        assert result.tldr == "This paper introduces the Transformer architecture."
        assert "Computer Science" in result.fields_of_study
        assert result.influential_citation_count == 5000
        assert result.external_ids["DOI"] == "10.5555/3295222.3295349"
        assert result.pdf_url == "https://arxiv.org/pdf/1706.03762.pdf"


class TestDetectIdType:
    """Tests for detect_id_type function."""

    def test_detects_doi(self) -> None:
        """Detect DOI format."""
        result = detect_id_type("10.5555/3295222.3295349")
        assert result == "DOI:10.5555/3295222.3295349"

    def test_detects_arxiv(self) -> None:
        """Detect ArXiv ID format."""
        result = detect_id_type("1706.03762")
        assert result == "ARXIV:1706.03762"

    def test_detects_arxiv_with_version(self) -> None:
        """Detect ArXiv ID with version."""
        result = detect_id_type("2301.00001v2")
        assert result == "ARXIV:2301.00001v2"

    def test_preserves_prefixed_id(self) -> None:
        """Preserve already prefixed IDs."""
        result = detect_id_type("DOI:10.1234/test")
        assert result == "DOI:10.1234/test"

    def test_assumes_semantic_scholar_id(self) -> None:
        """Assume unprefixed hex string is Semantic Scholar ID."""
        result = detect_id_type("abc123def456")
        assert result == "abc123def456"


class TestSearchPapersImpl:
    """Tests for search_papers_impl function."""

    @pytest.mark.asyncio
    async def test_returns_search_response(
        self,
        sample_search_result: SearchResult,
    ) -> None:
        """Return SearchResponse on success."""
        with patch(
            "papers_mcp.service.search_papers",
            new_callable=AsyncMock,
            return_value=sample_search_result,
        ):
            result = await search_papers_impl("transformer")

            assert isinstance(result, SearchResponse)
            assert result.total == 100
            assert result.query == "transformer"
            assert len(result.papers) == 1
            assert result.papers[0].title == "Attention Is All You Need"
            assert result.papers[0].pdf_url == "https://arxiv.org/pdf/1706.03762.pdf"

    @pytest.mark.asyncio
    async def test_returns_error_response_on_failure(self) -> None:
        """Return ErrorResponse on API error."""
        error = ServiceError(code="rate_limit", message="Rate limit exceeded")
        with patch(
            "papers_mcp.service.search_papers",
            new_callable=AsyncMock,
            return_value=error,
        ):
            result = await search_papers_impl("test")

            assert isinstance(result, ErrorResponse)
            assert result.error is True
            assert result.code == "rate_limit"
            assert result.message == "Rate limit exceeded"

    @pytest.mark.asyncio
    async def test_passes_parameters_correctly(
        self,
        sample_search_result: SearchResult,
    ) -> None:
        """Pass all parameters to search function."""
        with patch(
            "papers_mcp.service.search_papers",
            new_callable=AsyncMock,
            return_value=sample_search_result,
        ) as mock_search:
            await search_papers_impl(
                "llm",
                limit=5,
                year_start=2020,
                year_end=2024,
                offset=10,
            )

            call_args = mock_search.call_args[0][0]
            assert call_args.query == "llm"
            assert call_args.limit == 5
            assert call_args.year_start == 2020
            assert call_args.year_end == 2024
            assert call_args.offset == 10

    @pytest.mark.asyncio
    async def test_returns_empty_search_response(self) -> None:
        """Return SearchResponse with empty papers on no results."""
        empty_result = SearchResult(
            query="nonexistent",
            total=0,
            papers=(),
            offset=0,
            next_offset=None,
        )
        with patch(
            "papers_mcp.service.search_papers",
            new_callable=AsyncMock,
            return_value=empty_result,
        ):
            result = await search_papers_impl("nonexistent")

            assert isinstance(result, SearchResponse)
            assert result.total == 0
            assert result.papers == []


class TestGetPaperImpl:
    """Tests for get_paper_impl function."""

    @pytest.mark.asyncio
    async def test_returns_paper_details_response(
        self,
        sample_paper_details: PaperDetails,
    ) -> None:
        """Return PaperDetailsResponse on success."""
        with patch(
            "papers_mcp.service.get_paper_details",
            new_callable=AsyncMock,
            return_value=sample_paper_details,
        ):
            result = await get_paper_impl("abc123")

            assert isinstance(result, PaperDetailsResponse)
            assert result.paper_id == "abc123"
            assert result.title == "Attention Is All You Need"
            assert result.tldr == "This paper introduces the Transformer architecture."
            assert result.pdf_url == "https://arxiv.org/pdf/1706.03762.pdf"

    @pytest.mark.asyncio
    async def test_returns_error_response_on_not_found(self) -> None:
        """Return ErrorResponse when paper not found."""
        error = ServiceError(code="not_found", message="Paper not found")
        with patch(
            "papers_mcp.service.get_paper_details",
            new_callable=AsyncMock,
            return_value=error,
        ):
            result = await get_paper_impl("notfound")

            assert isinstance(result, ErrorResponse)
            assert result.code == "not_found"
            assert result.message == "Paper not found"

    @pytest.mark.asyncio
    async def test_normalizes_doi(
        self,
        sample_paper_details: PaperDetails,
    ) -> None:
        """Normalize DOI by adding prefix."""
        with patch(
            "papers_mcp.service.get_paper_details",
            new_callable=AsyncMock,
            return_value=sample_paper_details,
        ) as mock_get:
            await get_paper_impl("10.5555/3295222.3295349")

            call_args = mock_get.call_args[0][0]
            assert call_args == "DOI:10.5555/3295222.3295349"

    @pytest.mark.asyncio
    async def test_normalizes_arxiv(
        self,
        sample_paper_details: PaperDetails,
    ) -> None:
        """Normalize ArXiv ID by adding prefix."""
        with patch(
            "papers_mcp.service.get_paper_details",
            new_callable=AsyncMock,
            return_value=sample_paper_details,
        ) as mock_get:
            await get_paper_impl("1706.03762")

            call_args = mock_get.call_args[0][0]
            assert call_args == "ARXIV:1706.03762"


class TestMcpServer:
    """End-to-end smoke tests driving the server through an in-memory client."""

    @pytest.mark.asyncio
    async def test_tools_registered_with_expected_names(self) -> None:
        """Both tools register under the bare (no _mcp suffix) names."""
        async with Client(mcp) as client:
            tools = await client.list_tools()

        by_name = {tool.name: tool for tool in tools}
        assert set(by_name) == {"search_papers", "get_paper"}

        search = by_name["search_papers"]
        assert search.annotations is not None
        assert search.annotations.readOnlyHint is True
        assert set(search.inputSchema["properties"]) >= {
            "query",
            "limit",
            "year_start",
            "year_end",
            "offset",
        }

    @pytest.mark.asyncio
    async def test_search_tool_returns_structured_data(
        self,
        sample_search_result: SearchResult,
    ) -> None:
        """Calling search_papers returns structured paper data."""
        with patch(
            "papers_mcp.service.search_papers",
            new_callable=AsyncMock,
            return_value=sample_search_result,
        ):
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "search_papers", {"query": "transformer"}
                )

        assert result.is_error is False
        first = result.data["papers"][0]
        assert result.data["total"] == 100
        assert first["title"] == "Attention Is All You Need"
        assert first["pdf_url"] == "https://arxiv.org/pdf/1706.03762.pdf"

    @pytest.mark.asyncio
    async def test_get_paper_tool_returns_structured_data(
        self,
        sample_paper_details: PaperDetails,
    ) -> None:
        """Calling get_paper returns full structured details."""
        with patch(
            "papers_mcp.service.get_paper_details",
            new_callable=AsyncMock,
            return_value=sample_paper_details,
        ):
            async with Client(mcp) as client:
                result = await client.call_tool("get_paper", {"paper_id": "abc123"})

        assert result.is_error is False
        assert result.data["paper_id"] == "abc123"
        expected_tldr = "This paper introduces the Transformer architecture."
        assert result.data["tldr"] == expected_tldr

    @pytest.mark.asyncio
    async def test_search_tool_surfaces_service_errors(self) -> None:
        """Service errors are returned as a structured error payload."""
        error = ServiceError(code="rate_limit", message="Rate limit exceeded")
        with patch(
            "papers_mcp.service.search_papers",
            new_callable=AsyncMock,
            return_value=error,
        ):
            async with Client(mcp) as client:
                result = await client.call_tool("search_papers", {"query": "x"})

        assert result.data["error"] is True
        assert result.data["code"] == "rate_limit"


class TestMain:
    """Tests for the CLI entry point and --debug flag."""

    def test_debug_flag_enables_debug_logging(self) -> None:
        """--debug configures debug-level logging before starting the server."""
        with (
            patch("papers_mcp.server.mcp.run") as mock_run,
            patch("papers_mcp.server.logging.basicConfig") as mock_basic,
            patch("sys.argv", ["papers-mcp", "--debug"]),
        ):
            main()

        mock_run.assert_called_once_with()
        assert mock_basic.call_args.kwargs["level"] == logging.DEBUG

    def test_default_uses_info_logging(self) -> None:
        """Without --debug the server starts with info-level logging."""
        with (
            patch("papers_mcp.server.mcp.run") as mock_run,
            patch("papers_mcp.server.logging.basicConfig") as mock_basic,
            patch("sys.argv", ["papers-mcp"]),
        ):
            main()

        mock_run.assert_called_once_with()
        assert mock_basic.call_args.kwargs["level"] == logging.INFO
