"""Tests for CodebaseAnalyzer."""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_project():
    """Create a temporary project for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample Python files
        src_dir = Path(tmpdir) / "src"
        src_dir.mkdir()

        # Main module
        main_file = src_dir / "main.py"
        main_file.write_text('''"""Main module."""

def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"


class Greeter:
    """A greeter class."""

    def __init__(self, prefix: str = "Hello"):
        self.prefix = prefix

    def greet(self, name: str) -> str:
        """Greet with custom prefix."""
        return f"{self.prefix}, {name}!"
''')

        # Utils module
        utils_file = src_dir / "utils.py"
        utils_file.write_text('''"""Utility functions."""

def format_name(first: str, last: str) -> str:
    """Format a full name."""
    return f"{first} {last}"


def validate_name(name: str) -> bool:
    """Validate a name string."""
    return bool(name and name.strip())
''')

        # Config file
        config_file = Path(tmpdir) / "pyproject.toml"
        config_file.write_text('''[project]
name = "test-project"
version = "0.1.0"
''')

        yield tmpdir


class TestCodebaseAnalyzer:
    """Tests for CodebaseAnalyzer class."""

    @pytest.mark.asyncio
    async def test_quick_analysis(self, temp_project):
        """Test quick analysis mode."""
        from src.understanding import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer(temp_project)
        understanding = await analyzer.analyze_quick()

        assert understanding is not None
        assert understanding.structure is not None
        assert understanding.structure.detected_type == "python"
        assert understanding.structure.total_files >= 2

    @pytest.mark.asyncio
    async def test_deep_analysis(self, temp_project):
        """Test deep analysis mode."""
        from src.understanding import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer(temp_project)
        understanding = await analyzer.analyze_deep()

        assert understanding is not None
        assert len(understanding.symbols) > 0
        assert len(understanding.files) >= 2

        # Check that functions were extracted
        symbol_names = [s.name for s in understanding.symbols.values()]
        assert "greet" in symbol_names
        assert "Greeter" in symbol_names

    @pytest.mark.asyncio
    async def test_symbol_extraction(self, temp_project):
        """Test symbol extraction accuracy."""
        from src.understanding import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer(temp_project)
        understanding = await analyzer.analyze_deep()

        # Find the greet function
        greet_symbols = [
            s for s in understanding.symbols.values()
            if s.name == "greet" and s.kind.value == "function"
        ]

        assert len(greet_symbols) >= 1

        greet = greet_symbols[0]
        assert greet.file_path.endswith("main.py")
        # Signature should be extracted
        assert greet.signature is not None
        assert "greet" in greet.signature

    @pytest.mark.asyncio
    async def test_statistics(self, temp_project):
        """Test statistics generation."""
        from src.understanding import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer(temp_project)
        await analyzer.analyze_deep()

        stats = analyzer.get_statistics()

        assert stats["analyzed"] is True
        assert stats["total_files"] >= 2
        assert stats["total_symbols"] > 0
        assert stats["detected_type"] == "python"


class TestFileScanner:
    """Tests for FileScanner."""

    def test_scan_python_files(self, temp_project):
        """Test scanning Python files."""
        from src.understanding import FileScanner

        scanner = FileScanner(temp_project)
        files = list(scanner.scan())

        assert len(files) >= 2

        # Check file info
        python_files = [f for f in files if f.language == "python"]
        assert len(python_files) >= 2

        for f in python_files:
            assert f.line_count > 0
            assert f.content_hash != ""

    def test_skip_patterns(self, temp_project):
        """Test that skip patterns work."""
        from src.understanding import FileScanner

        # Create a file that should be skipped
        pycache_dir = Path(temp_project) / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "test.pyc").write_bytes(b"fake pyc")

        scanner = FileScanner(temp_project)
        files = list(scanner.scan())

        # Should not include .pyc files
        pyc_files = [f for f in files if f.path.endswith(".pyc")]
        assert len(pyc_files) == 0

    def test_statistics(self, temp_project):
        """Test scanner statistics."""
        from src.understanding import FileScanner

        scanner = FileScanner(temp_project)
        stats = scanner.get_statistics()

        assert stats["total_files"] >= 2
        assert stats["total_lines"] > 0
        assert "python" in stats["by_language"]
