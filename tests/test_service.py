"""Tests for the paper search service.

These tests follow TDD principles - written before implementation.
Uses pytest-httpx for mocking HTTP requests.
"""

import httpx
import pytest

from papers_mcp.domain import (
    Paper,
    PaperDetails,
    SearchQuery,
    SearchResult,
    ServiceError,
)
from papers_mcp.service import (
    RetryConfig,
    _build_search_url,
    get_paper_details,
    parse_paper,
    parse_paper_details,
    parse_search_result,
    search_papers,
)

# Sample API responses for testing
SAMPLE_PAPER_RESPONSE = {
    "paperId": "abc123",
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models...",
    "authors": [
        {"authorId": "1", "name": "Ashish Vaswani"},
        {"authorId": "2", "name": "Noam Shazeer"},
    ],
    "year": 2017,
    "venue": "NeurIPS",
    "citationCount": 50000,
    "url": "https://www.semanticscholar.org/paper/abc123",
    "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762.pdf"},
}

SAMPLE_SEARCH_RESPONSE = {
    "total": 100,
    "offset": 0,
    "next": 10,
    "data": [SAMPLE_PAPER_RESPONSE],
}

SAMPLE_DETAILS_RESPONSE = {
    **SAMPLE_PAPER_RESPONSE,
    "tldr": {"text": "This paper introduces the Transformer architecture."},
    "fieldsOfStudy": ["Computer Science", "Artificial Intelligence"],
    "publicationTypes": ["Conference"],
    "referenceCount": 50,
    "influentialCitationCount": 5000,
    "externalIds": {"DOI": "10.5555/3295222.3295349", "ArXiv": "1706.03762"},
}


class TestParsePaper:
    """Tests for parse_paper function."""

    def test_parse_paper_with_all_fields(self) -> None:
        """Parse a paper with all fields present."""
        result = parse_paper(SAMPLE_PAPER_RESPONSE)

        assert isinstance(result, Paper)
        assert result.paper_id == "abc123"
        assert result.title == "Attention Is All You Need"
        assert result.abstract == "The dominant sequence transduction models..."
        assert len(result.authors) == 2
        assert result.authors[0].name == "Ashish Vaswani"
        assert result.year == 2017
        assert result.venue == "NeurIPS"
        assert result.citation_count == 50000
        assert result.url == "https://www.semanticscholar.org/paper/abc123"
        assert result.open_access_pdf == "https://arxiv.org/pdf/1706.03762.pdf"

    def test_parse_paper_with_missing_optional_fields(self) -> None:
        """Parse a paper with optional fields missing."""
        minimal_response = {
            "paperId": "def456",
            "title": "Some Paper",
            "authors": [],
            "citationCount": 0,
            "url": "https://www.semanticscholar.org/paper/def456",
        }
        result = parse_paper(minimal_response)

        assert result.paper_id == "def456"
        assert result.title == "Some Paper"
        assert result.abstract is None
        assert result.authors == ()
        assert result.year is None
        assert result.venue is None
        assert result.open_access_pdf is None

    def test_parse_paper_with_null_open_access_pdf(self) -> None:
        """Parse a paper where openAccessPdf is null."""
        response = {**SAMPLE_PAPER_RESPONSE, "openAccessPdf": None}
        result = parse_paper(response)

        assert result.open_access_pdf is None


class TestParseSearchResult:
    """Tests for parse_search_result function."""

    def test_parse_search_result(self) -> None:
        """Parse a search result with papers."""
        query = SearchQuery(query="transformer")
        result = parse_search_result(SAMPLE_SEARCH_RESPONSE, query)

        assert isinstance(result, SearchResult)
        assert result.query == "transformer"
        assert result.total == 100
        assert result.offset == 0
        assert result.next_offset == 10
        assert len(result.papers) == 1
        assert result.papers[0].title == "Attention Is All You Need"

    def test_parse_search_result_no_next(self) -> None:
        """Parse a search result without next page."""
        response = {**SAMPLE_SEARCH_RESPONSE, "next": None}
        query = SearchQuery(query="test")
        result = parse_search_result(response, query)

        assert result.next_offset is None

    def test_parse_search_result_empty(self) -> None:
        """Parse an empty search result."""
        response = {"total": 0, "offset": 0, "data": []}
        query = SearchQuery(query="nonexistent")
        result = parse_search_result(response, query)

        assert result.total == 0
        assert result.papers == ()


class TestParsePaperDetails:
    """Tests for parse_paper_details function."""

    def test_parse_paper_details(self) -> None:
        """Parse full paper details."""
        result = parse_paper_details(SAMPLE_DETAILS_RESPONSE)

        assert isinstance(result, PaperDetails)
        assert result.paper.paper_id == "abc123"
        assert result.tldr == "This paper introduces the Transformer architecture."
        assert "Computer Science" in result.fields_of_study
        assert "Conference" in result.publication_types
        assert result.references_count == 50
        assert result.influential_citation_count == 5000
        assert result.external_ids["DOI"] == "10.5555/3295222.3295349"

    def test_parse_paper_details_missing_tldr(self) -> None:
        """Parse paper details without TLDR."""
        response = {**SAMPLE_DETAILS_RESPONSE, "tldr": None}
        result = parse_paper_details(response)

        assert result.tldr is None


class TestSearchPapers:
    """Integration tests for search_papers function."""

    @pytest.mark.asyncio
    async def test_search_papers_success(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Successfully search for papers."""
        httpx_mock.add_response(
            url="https://api.semanticscholar.org/graph/v1/paper/search?query=transformer&offset=0&limit=10&fields=paperId%2Ctitle%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CcitationCount%2Curl%2CopenAccessPdf",
            json=SAMPLE_SEARCH_RESPONSE,
        )

        query = SearchQuery(query="transformer")
        result = await search_papers(query)

        assert isinstance(result, SearchResult)
        assert result.total == 100
        assert len(result.papers) == 1

    @pytest.mark.asyncio
    async def test_search_papers_with_filters(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Search with year and field filters."""
        httpx_mock.add_response(
            url="https://api.semanticscholar.org/graph/v1/paper/search?query=llm&offset=0&limit=5&fields=paperId%2Ctitle%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CcitationCount%2Curl%2CopenAccessPdf&year=2020-2024&fieldsOfStudy=Computer+Science",
            json=SAMPLE_SEARCH_RESPONSE,
        )

        query = SearchQuery(
            query="llm",
            limit=5,
            year_start=2020,
            year_end=2024,
            fields_of_study=("Computer Science",),
        )
        result = await search_papers(query)

        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_papers_api_error(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Handle API error response."""
        httpx_mock.add_response(
            url="https://api.semanticscholar.org/graph/v1/paper/search?query=test&offset=0&limit=10&fields=paperId%2Ctitle%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CcitationCount%2Curl%2CopenAccessPdf",
            status_code=429,
            json={"error": "Rate limit exceeded"},
        )

        query = SearchQuery(query="test")
        result = await search_papers(query)

        assert isinstance(result, ServiceError)
        assert result.code == "rate_limit"


class TestGetPaperDetails:
    """Integration tests for get_paper_details function."""

    @pytest.mark.asyncio
    async def test_get_paper_details_success(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Successfully get paper details."""
        httpx_mock.add_response(
            url="https://api.semanticscholar.org/graph/v1/paper/abc123?fields=paperId%2Ctitle%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CcitationCount%2Curl%2CopenAccessPdf%2Ctldr%2CfieldsOfStudy%2CpublicationTypes%2CreferenceCount%2CinfluentialCitationCount%2CexternalIds",
            json=SAMPLE_DETAILS_RESPONSE,
        )

        result = await get_paper_details("abc123")

        assert isinstance(result, PaperDetails)
        assert result.paper.paper_id == "abc123"

    @pytest.mark.asyncio
    async def test_get_paper_details_not_found(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Handle paper not found."""
        httpx_mock.add_response(
            url="https://api.semanticscholar.org/graph/v1/paper/notfound?fields=paperId%2Ctitle%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CcitationCount%2Curl%2CopenAccessPdf%2Ctldr%2CfieldsOfStudy%2CpublicationTypes%2CreferenceCount%2CinfluentialCitationCount%2CexternalIds",
            status_code=404,
            json={"error": "Paper not found"},
        )

        result = await get_paper_details("notfound")

        assert isinstance(result, ServiceError)
        assert result.code == "not_found"

    @pytest.mark.asyncio
    async def test_get_paper_details_with_doi(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Get paper details using DOI."""
        httpx_mock.add_response(
            url="https://api.semanticscholar.org/graph/v1/paper/DOI:10.5555/3295222.3295349?fields=paperId%2Ctitle%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CcitationCount%2Curl%2CopenAccessPdf%2Ctldr%2CfieldsOfStudy%2CpublicationTypes%2CreferenceCount%2CinfluentialCitationCount%2CexternalIds",
            json=SAMPLE_DETAILS_RESPONSE,
        )

        result = await get_paper_details("DOI:10.5555/3295222.3295349")

        assert isinstance(result, PaperDetails)


class TestRetryConfig:
    """Tests for RetryConfig.delay_for backoff math."""

    def test_exponential_backoff_without_jitter(self) -> None:
        """Delay doubles each attempt when jitter is disabled."""
        cfg = RetryConfig(base_delay=5.0, max_delay=60.0, jitter=0.0)

        assert cfg.delay_for(0) == 5.0
        assert cfg.delay_for(1) == 10.0
        assert cfg.delay_for(2) == 20.0
        assert cfg.delay_for(3) == 40.0

    def test_delay_caps_at_max_delay(self) -> None:
        """Delay never exceeds max_delay regardless of attempt."""
        cfg = RetryConfig(base_delay=5.0, max_delay=60.0, jitter=0.0)

        assert cfg.delay_for(4) == 60.0  # 80 capped to 60
        assert cfg.delay_for(20) == 60.0

    def test_jitter_stays_within_bounds(self) -> None:
        """Jitter only ever adds 0-50% on top of the base delay."""
        cfg = RetryConfig(base_delay=10.0, max_delay=60.0, jitter=0.5)

        for attempt in range(6):
            base = min(10.0 * 2.0**attempt, 60.0)
            delay = cfg.delay_for(attempt)
            assert base <= delay <= base * 1.5


class TestRetryBehavior:
    """Tests for the retry/backoff loop in _make_request_with_retry."""

    @pytest.mark.asyncio
    async def test_retries_then_succeeds_on_rate_limit(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Retry on 429 and return the result once the API recovers."""
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(json=SAMPLE_SEARCH_RESPONSE)

        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        cfg = RetryConfig(max_retries=5, base_delay=1.0, max_delay=4.0, jitter=0.0)
        result = await search_papers(
            SearchQuery(query="x"), retry_config=cfg, sleep=fake_sleep
        )

        assert isinstance(result, SearchResult)
        assert result.total == 100
        assert delays == [1.0, 2.0]  # slept before each of the two retries

    @pytest.mark.asyncio
    async def test_exhausts_retries_on_persistent_rate_limit(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Give up with a rate_limit error after exhausting attempts."""
        for _ in range(3):
            httpx_mock.add_response(status_code=429)

        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        cfg = RetryConfig(max_retries=3, base_delay=1.0, max_delay=10.0, jitter=0.0)
        result = await search_papers(
            SearchQuery(query="x"), retry_config=cfg, sleep=fake_sleep
        )

        assert isinstance(result, ServiceError)
        assert result.code == "rate_limit"
        assert delays == [1.0, 2.0]  # no sleep after the final attempt

    @pytest.mark.asyncio
    async def test_timeout_retries_then_fails(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Retry on timeout and return a timeout error when it persists."""
        httpx_mock.add_exception(httpx.TimeoutException("timeout"))
        httpx_mock.add_exception(httpx.TimeoutException("timeout"))

        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        cfg = RetryConfig(max_retries=2, base_delay=1.0, max_delay=10.0, jitter=0.0)
        result = await search_papers(
            SearchQuery(query="x"), retry_config=cfg, sleep=fake_sleep
        )

        assert isinstance(result, ServiceError)
        assert result.code == "timeout"
        assert delays == [1.0]

    @pytest.mark.asyncio
    async def test_timeout_then_success(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Recover and return a result when a timeout is followed by success."""
        httpx_mock.add_exception(httpx.TimeoutException("timeout"))
        httpx_mock.add_response(json=SAMPLE_SEARCH_RESPONSE)

        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        cfg = RetryConfig(max_retries=3, base_delay=1.0, max_delay=10.0, jitter=0.0)
        result = await search_papers(
            SearchQuery(query="x"), retry_config=cfg, sleep=fake_sleep
        )

        assert isinstance(result, SearchResult)
        assert delays == [1.0]

    @pytest.mark.asyncio
    async def test_server_error_is_not_retried(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """5xx responses fail fast without consuming retries."""
        httpx_mock.add_response(status_code=503)

        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        cfg = RetryConfig(max_retries=5, base_delay=1.0, jitter=0.0)
        result = await search_papers(
            SearchQuery(query="x"), retry_config=cfg, sleep=fake_sleep
        )

        assert isinstance(result, ServiceError)
        assert result.code == "server_error"
        assert delays == []  # no retries for server errors

    @pytest.mark.asyncio
    async def test_network_error_is_not_retried(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Connection errors surface immediately as network_error."""
        httpx_mock.add_exception(httpx.ConnectError("boom"))

        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        cfg = RetryConfig(max_retries=5, base_delay=1.0, jitter=0.0)
        result = await search_papers(
            SearchQuery(query="x"), retry_config=cfg, sleep=fake_sleep
        )

        assert isinstance(result, ServiceError)
        assert result.code == "network_error"
        assert delays == []

    @pytest.mark.asyncio
    async def test_unexpected_status_maps_to_api_error(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        """Unhandled status codes (e.g. 400) map to a generic api_error."""
        httpx_mock.add_response(status_code=400)

        cfg = RetryConfig(max_retries=2, base_delay=1.0, jitter=0.0)
        result = await search_papers(SearchQuery(query="x"), retry_config=cfg)

        assert isinstance(result, ServiceError)
        assert result.code == "api_error"


class TestBuildSearchUrl:
    """Tests for search URL construction (encoding and filters)."""

    def test_includes_core_params_and_fields(self) -> None:
        """Core query/pagination params and requested fields are encoded."""
        url = _build_search_url(SearchQuery(query="deep learning", limit=5, offset=10))

        assert "query=deep+learning" in url
        assert "limit=5" in url
        assert "offset=10" in url
        assert "fields=paperId%2Ctitle" in url

    def test_year_start_only(self) -> None:
        """A lower-bound-only year filter is encoded as ``2020-``."""
        url = _build_search_url(SearchQuery(query="x", year_start=2020))

        assert "year=2020-" in url

    def test_year_end_only(self) -> None:
        """An upper-bound-only year filter is encoded as ``-2024``."""
        url = _build_search_url(SearchQuery(query="x", year_end=2024))

        assert "year=-2024" in url

    def test_multiple_fields_of_study_all_included(self) -> None:
        """All fields_of_study are sent, not just the first (regression test)."""
        url = _build_search_url(
            SearchQuery(query="x", fields_of_study=("Computer Science", "Biology"))
        )

        assert "fieldsOfStudy=Computer+Science%2CBiology" in url
