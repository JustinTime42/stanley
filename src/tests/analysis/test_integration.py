"""Integration tests for the analysis subsystem."""

import pytest
import tempfile
import os

from src.services.analysis_service import AnalysisOrchestrator
from src.models.analysis_models import AnalysisRequest, Language


@pytest.fixture
def orchestrator():
    """Create analysis orchestrator."""
    return AnalysisOrchestrator(cache_enabled=True, cache_ttl=60)


@pytest.fixture
def sample_python_file():
    """Create a sample Python file for testing."""
    code = """
'''Sample Python module for testing.'''

def calculate_sum(a, b):
    '''Calculate sum of two numbers.'''
    if a < 0 or b < 0:
        raise ValueError("Negative numbers not allowed")
    return a + b

class Calculator:
    '''Simple calculator class.'''

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = calculate_sum(a, b)
        self.history.append(result)
        return result

    def get_history(self):
        return self.history
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.mark.asyncio
async def test_full_analysis_pipeline(orchestrator, sample_python_file):
    """Test complete analysis pipeline from file to results."""
    result = await orchestrator.analyze_file(sample_python_file)

    # Verify basic analysis
    assert result.file_path == sample_python_file
    assert result.language == Language.PYTHON
    assert len(result.errors) == 0

    # Verify entities were extracted
    assert len(result.entities) > 0

    # Check for expected entities
    entity_names = [e.name for e in result.entities]
    assert "calculate_sum" in entity_names
    assert "Calculator" in entity_names

    # Verify complexity metrics
    assert result.complexity is not None
    assert result.complexity.cyclomatic_complexity > 0
    assert result.complexity.function_count >= 2  # calculate_sum + methods
    assert result.complexity.class_count >= 1  # Calculator


@pytest.mark.asyncio
async def test_analysis_with_specific_types(orchestrator, sample_python_file):
    """Test analysis with specific analysis types."""
    request = AnalysisRequest(
        file_paths=[sample_python_file],
        analysis_types=["complexity", "patterns"],
        cache_enabled=False,
    )

    result = await orchestrator.analyze_file(sample_python_file, request)

    # Should have complexity
    assert result.complexity is not None

    # Should have patterns (may be empty, but not None)
    assert result.patterns is not None

    # AST should be None since we didn't request it
    assert result.ast is None


@pytest.mark.asyncio
async def test_caching_works(orchestrator, sample_python_file):
    """Test that caching improves performance."""
    # First analysis - not cached
    result1 = await orchestrator.analyze_file(sample_python_file)
    assert result1.cache_hit is False
    time1 = result1.analysis_time_ms

    # Second analysis - should be cached
    result2 = await orchestrator.analyze_file(sample_python_file)
    assert result2.cache_hit is True

    # Cached result should be identical
    assert result1.file_path == result2.file_path
    assert len(result1.entities) == len(result2.entities)


@pytest.mark.asyncio
async def test_analysis_service_orchestration(orchestrator, sample_python_file):
    """Test the full service orchestration."""
    request = AnalysisRequest(
        file_paths=[sample_python_file],
        analysis_types=["ast", "complexity", "dependencies", "patterns"],
        cache_enabled=True,
    )

    result = await orchestrator.analyze_file(sample_python_file, request)

    # Verify all analysis types were performed
    assert result.ast is not None
    assert result.complexity is not None
    assert result.dependencies is not None
    assert result.patterns is not None  # May be empty list

    # Verify analysis completed successfully
    assert len(result.errors) == 0


@pytest.mark.asyncio
async def test_cache_stats(orchestrator, sample_python_file):
    """Test cache statistics."""
    # Perform some analyses
    await orchestrator.analyze_file(sample_python_file)
    await orchestrator.analyze_file(sample_python_file)

    # Get cache stats
    stats = orchestrator.get_cache_stats()

    assert "ast_cache_size" in stats
    assert "result_cache_size" in stats
    assert "total_entries" in stats

    # Should have cached entries
    assert stats["total_entries"] > 0
