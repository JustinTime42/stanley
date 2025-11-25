"""Tests for complexity analyzer."""

import pytest

from src.analysis.ast_parser import ASTParser
from src.analysis.complexity_analyzer import ComplexityAnalyzer
from src.models.analysis_models import Language


@pytest.fixture
def parser():
    """Create AST parser instance."""
    return ASTParser()


@pytest.fixture
def analyzer():
    """Create complexity analyzer instance."""
    return ComplexityAnalyzer()


@pytest.fixture
def simple_code():
    """Simple Python code with known complexity."""
    return b"""
def simple_function(x):
    return x + 1
"""


@pytest.fixture
def complex_code():
    """Complex Python code with multiple branches."""
    return b"""
def complex_function(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x - y
    elif x < 0:
        if y > 0:
            return y - x
        else:
            return -(x + y)
    else:
        return y
"""


@pytest.mark.asyncio
async def test_calculate_cyclomatic_simple(parser, analyzer, simple_code):
    """Test cyclomatic complexity for simple code."""
    ast = await parser.parse_code(simple_code, Language.PYTHON)
    complexity = analyzer.calculate_cyclomatic(ast)

    # Simple linear code should have complexity of 1
    assert complexity == 1


@pytest.mark.asyncio
async def test_calculate_cyclomatic_complex(parser, analyzer, complex_code):
    """Test cyclomatic complexity for complex code."""
    ast = await parser.parse_code(complex_code, Language.PYTHON)
    complexity = analyzer.calculate_cyclomatic(ast)

    # Code with multiple if/elif branches should have higher complexity
    assert complexity > 1
    assert complexity >= 4  # Multiple decision points


@pytest.mark.asyncio
async def test_calculate_cognitive(parser, analyzer, complex_code):
    """Test cognitive complexity calculation."""
    ast = await parser.parse_code(complex_code, Language.PYTHON)
    complexity = analyzer.calculate_cognitive(ast)

    # Nested structures increase cognitive complexity
    assert complexity > 0


@pytest.mark.asyncio
async def test_calculate_halstead(parser, analyzer, simple_code):
    """Test Halstead metrics calculation."""
    ast = await parser.parse_code(simple_code, Language.PYTHON)
    metrics = analyzer.calculate_halstead(ast)

    # Should have basic Halstead metrics
    assert "vocabulary" in metrics
    assert "length" in metrics
    assert "volume" in metrics
    assert "difficulty" in metrics
    assert "effort" in metrics


@pytest.mark.asyncio
async def test_analyze_complete(parser, analyzer, complex_code):
    """Test complete analysis."""
    ast = await parser.parse_code(complex_code, Language.PYTHON)
    source = complex_code.decode("utf-8")

    result = await analyzer.analyze("test.py", ast, Language.PYTHON, source)

    # Check all metrics are present
    assert result.cyclomatic_complexity > 0
    assert result.cognitive_complexity >= 0
    assert result.lines_of_code > 0
    assert result.function_count >= 1


@pytest.mark.asyncio
async def test_count_functions(parser, analyzer):
    """Test function counting."""
    code = b"""
def func1():
    pass

def func2():
    pass

class MyClass:
    def method1(self):
        pass
"""
    ast = await parser.parse_code(code, Language.PYTHON)
    result = await analyzer.analyze("test.py", ast, Language.PYTHON, code.decode("utf-8"))

    # Should count all 3 functions/methods
    assert result.function_count == 3


@pytest.mark.asyncio
async def test_count_classes(parser, analyzer):
    """Test class counting."""
    code = b"""
class Class1:
    pass

class Class2:
    pass
"""
    ast = await parser.parse_code(code, Language.PYTHON)
    result = await analyzer.analyze("test.py", ast, Language.PYTHON, code.decode("utf-8"))

    assert result.class_count == 2
