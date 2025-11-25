"""Tests for AST parser."""

import pytest
import tempfile
import os

from src.analysis.ast_parser import ASTParser
from src.models.analysis_models import Language


@pytest.fixture
def parser():
    """Create AST parser instance."""
    return ASTParser()


@pytest.fixture
def python_code():
    """Sample Python code for testing."""
    return b"""
def hello_world():
    '''Simple hello world function.'''
    print("Hello, World!")
    return True

class MyClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
"""


@pytest.fixture
def javascript_code():
    """Sample JavaScript code for testing."""
    return b"""
function helloWorld() {
    console.log("Hello, World!");
    return true;
}

class MyClass {
    constructor() {
        this.value = 42;
    }

    getValue() {
        return this.value;
    }
}
"""


@pytest.mark.asyncio
async def test_detect_language_python(parser):
    """Test language detection for Python files."""
    language = parser.detect_language("test.py")
    assert language == Language.PYTHON


@pytest.mark.asyncio
async def test_detect_language_javascript(parser):
    """Test language detection for JavaScript files."""
    language = parser.detect_language("test.js")
    assert language == Language.JAVASCRIPT


@pytest.mark.asyncio
async def test_parse_python_code(parser, python_code):
    """Test parsing Python code."""
    ast = await parser.parse_code(python_code, Language.PYTHON)

    assert ast is not None
    assert ast.node_type == "module"
    assert len(ast.children) > 0


@pytest.mark.asyncio
async def test_parse_javascript_code(parser, javascript_code):
    """Test parsing JavaScript code."""
    ast = await parser.parse_code(javascript_code, Language.JAVASCRIPT)

    assert ast is not None
    assert ast.node_type == "program"
    assert len(ast.children) > 0


@pytest.mark.asyncio
async def test_parse_file(parser, python_code):
    """Test parsing from file."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as f:
        f.write(python_code)
        temp_path = f.name

    try:
        ast = await parser.parse_file(temp_path)
        assert ast is not None
        assert ast.node_type == "module"
    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_traverse_ast(parser, python_code):
    """Test AST traversal."""
    ast = await parser.parse_code(python_code, Language.PYTHON)

    # Find all function definitions
    functions = parser.traverse_ast(ast, node_type="function_definition")
    assert len(functions) >= 2  # hello_world and methods


@pytest.mark.asyncio
async def test_find_nodes_by_type(parser, python_code):
    """Test finding nodes by type."""
    ast = await parser.parse_code(python_code, Language.PYTHON)

    # Find functions and classes
    nodes = parser.find_nodes_by_type(ast, ["function_definition", "class_definition"])
    assert len(nodes) >= 3  # 1 function + 1 class + class methods
