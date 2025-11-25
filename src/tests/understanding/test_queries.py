"""Tests for query handlers."""

import pytest


class TestKnowledgeQueryHandler:
    """Tests for KnowledgeQueryHandler."""

    @pytest.fixture
    def handler(self):
        """Create query handler."""
        from src.understanding.knowledge import KnowledgeStore
        from src.understanding.queries import KnowledgeQueryHandler

        store = KnowledgeStore()
        return KnowledgeQueryHandler(store)

    def test_detect_query_type_file(self, handler):
        """Test file query detection."""
        assert handler._detect_query_type("What's in main.py?") == "file"
        assert handler._detect_query_type("Tell me about utils.ts") == "file"

    def test_detect_query_type_symbol(self, handler):
        """Test symbol query detection."""
        assert handler._detect_query_type("function calculate_total") == "symbol"
        assert handler._detect_query_type("what is module.class.method") == "symbol"

    def test_detect_query_type_relationship(self, handler):
        """Test relationship query detection."""
        # "calls" keyword triggers relationship detection
        assert handler._detect_query_type("what calls this") == "relationship"
        assert handler._detect_query_type("what uses the module") == "relationship"
        assert handler._detect_query_type("show imports for module") == "relationship"

    def test_extract_symbol_name_quoted(self, handler):
        """Test extracting quoted symbol names."""
        assert handler._extract_symbol_name("tell me about `foo`") == "foo"
        assert handler._extract_symbol_name("what is 'bar'") == "bar"

    def test_extract_symbol_name_qualified(self, handler):
        """Test extracting qualified names."""
        assert handler._extract_symbol_name("info about module.class.method") == "module.class.method"


class TestGapQueryHandler:
    """Tests for GapQueryHandler."""

    @pytest.fixture
    def handler(self):
        """Create gap query handler."""
        from src.understanding.knowledge import KnowledgeStore, GapDetector
        from src.understanding.queries import GapQueryHandler

        store = KnowledgeStore()
        detector = GapDetector(store)
        return GapQueryHandler(store, detector)

    @pytest.mark.asyncio
    async def test_get_coverage_report_empty(self, handler):
        """Test coverage report with empty store."""
        report = await handler.get_coverage_report()

        assert "total_symbols" in report
        assert report["total_symbols"] == 0

    @pytest.mark.asyncio
    async def test_get_unanalyzed_areas(self, handler):
        """Test getting unanalyzed areas."""
        result = await handler.get_unanalyzed_areas()

        assert "unanalyzed_files" in result
        assert "coverage" in result


class TestSimilaritySearchHandler:
    """Tests for SimilaritySearchHandler."""

    @pytest.fixture
    def handler(self):
        """Create similarity search handler."""
        from src.understanding.knowledge import KnowledgeStore
        from src.understanding.queries import SimilaritySearchHandler

        store = KnowledgeStore()
        return SimilaritySearchHandler(store)

    @pytest.mark.asyncio
    async def test_find_similar_functions_empty(self, handler):
        """Test finding similar functions with empty store."""
        results = await handler.find_similar_functions("calculate total price")

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_related_nonexistent(self, handler):
        """Test finding related for nonexistent symbol."""
        result = await handler.find_related("nonexistent_symbol")

        assert "error" in result
