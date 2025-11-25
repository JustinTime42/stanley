"""Tests for framework detection."""

import pytest
import tempfile
import os
import json

from src.testing.framework_detector import FrameworkDetector
from src.models.testing_models import TestFramework
from src.models.analysis_models import Language


class TestFrameworkDetector:
    """Test framework detection functionality."""

    @pytest.fixture
    def detector(self):
        """Create framework detector instance."""
        return FrameworkDetector()

    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_detect_pytest_from_requirements(self, detector, temp_project):
        """Test pytest detection from requirements.txt."""
        # Create requirements.txt with pytest
        requirements_path = os.path.join(temp_project, "requirements.txt")
        with open(requirements_path, "w") as f:
            f.write("pytest==7.4.0\n")
            f.write("pytest-cov==4.1.0\n")

        # Detect framework
        framework = await detector.detect_framework(temp_project)

        # Verify
        assert framework == TestFramework.PYTEST

    @pytest.mark.asyncio
    async def test_detect_jest_from_package_json(self, detector, temp_project):
        """Test Jest detection from package.json."""
        # Create package.json with jest
        package_json = {
            "name": "test-project",
            "devDependencies": {"jest": "^29.0.0", "@types/jest": "^29.0.0"},
        }

        package_path = os.path.join(temp_project, "package.json")
        with open(package_path, "w") as f:
            json.dump(package_json, f)

        # Detect framework
        framework = await detector.detect_framework(temp_project)

        # Verify
        assert framework == TestFramework.JEST

    @pytest.mark.asyncio
    async def test_detect_from_test_files(self, detector, temp_project):
        """Test detection from test file patterns."""
        # Create Python test file
        test_dir = os.path.join(temp_project, "tests")
        os.makedirs(test_dir)

        test_file = os.path.join(test_dir, "test_example.py")
        with open(test_file, "w") as f:
            f.write("def test_example():\n    pass\n")

        # Detect framework
        framework = await detector.detect_framework(temp_project)

        # Should detect pytest from test_ pattern
        assert framework == TestFramework.PYTEST

    @pytest.mark.asyncio
    async def test_default_framework_for_language(self, detector, temp_project):
        """Test default framework selection by language."""
        # Create Python file but no test indicators
        src_dir = os.path.join(temp_project, "src")
        os.makedirs(src_dir)

        py_file = os.path.join(src_dir, "example.py")
        with open(py_file, "w") as f:
            f.write("def example():\n    pass\n")

        # Detect framework with language hint
        framework = await detector.detect_framework(temp_project, Language.PYTHON)

        # Should use default for Python
        assert framework == TestFramework.PYTEST

    @pytest.mark.asyncio
    async def test_get_pytest_config(self, detector, temp_project):
        """Test getting pytest configuration."""
        config = await detector.get_framework_config(
            TestFramework.PYTEST, temp_project
        )

        assert config["framework"] == "pytest"
        assert "test_dir" in config
        assert "test_pattern" in config
        assert config["fixture_support"] is True

    @pytest.mark.asyncio
    async def test_get_jest_config(self, detector, temp_project):
        """Test getting Jest configuration."""
        config = await detector.get_framework_config(
            TestFramework.JEST, temp_project
        )

        assert config["framework"] == "jest"
        assert "test_dir" in config
        assert "mock_support" in config
