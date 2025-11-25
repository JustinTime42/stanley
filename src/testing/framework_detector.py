"""Framework detection for test generation."""

import os
import json
import logging
import glob
from pathlib import Path
from typing import Optional, Dict, Any

from ..models.testing_models import TestFramework
from ..models.analysis_models import Language

logger = logging.getLogger(__name__)


class FrameworkDetector:
    """
    Detect testing framework from project structure.

    PATTERN: Priority-based detection using multiple signals
    CRITICAL: Check package files, config files, and test file patterns
    GOTCHA: Some projects use multiple frameworks
    """

    # Framework configuration file patterns
    FRAMEWORK_CONFIG_FILES = {
        TestFramework.PYTEST: ["pytest.ini", "setup.cfg", "pyproject.toml"],
        TestFramework.JEST: ["jest.config.js", "jest.config.ts", "jest.config.json"],
        TestFramework.MOCHA: [".mocharc.json", ".mocharc.js", "mocha.opts"],
        TestFramework.JUNIT: ["pom.xml", "build.gradle"],
        TestFramework.GO_TEST: ["go.mod"],
    }

    # Test file patterns by framework
    TEST_FILE_PATTERNS = {
        TestFramework.PYTEST: ["test_*.py", "*_test.py"],
        TestFramework.JEST: ["*.test.js", "*.spec.js", "*.test.ts", "*.spec.ts"],
        TestFramework.MOCHA: ["*.test.js", "*.spec.js"],
        TestFramework.JUNIT: ["*Test.java", "*Tests.java"],
        TestFramework.GO_TEST: ["*_test.go"],
    }

    # Default framework by language
    DEFAULT_FRAMEWORKS = {
        Language.PYTHON: TestFramework.PYTEST,
        Language.JAVASCRIPT: TestFramework.JEST,
        Language.TYPESCRIPT: TestFramework.JEST,
        Language.JAVA: TestFramework.JUNIT,
        Language.GO: TestFramework.GO_TEST,
    }

    def __init__(self):
        """Initialize framework detector."""
        self.logger = logger

    async def detect_framework(
        self, project_path: str, language: Optional[Language] = None
    ) -> TestFramework:
        """
        Detect testing framework from project files.

        PATTERN: Priority-based detection
        1. Check dependency files (package.json, requirements.txt)
        2. Check config files
        3. Check test file patterns
        4. Use default for language

        Args:
            project_path: Path to project directory
            language: Optional language hint

        Returns:
            Detected TestFramework
        """
        project_path = os.path.abspath(project_path)

        # Priority 1: Check package dependency files
        framework = await self._check_dependency_files(project_path)
        if framework:
            self.logger.info(f"Detected framework from dependencies: {framework.value}")
            return framework

        # Priority 2: Check config files
        framework = await self._check_config_files(project_path)
        if framework:
            self.logger.info(f"Detected framework from config: {framework.value}")
            return framework

        # Priority 3: Check test file patterns
        framework = await self._check_test_files(project_path)
        if framework:
            self.logger.info(f"Detected framework from test files: {framework.value}")
            return framework

        # Priority 4: Use default for language
        if language and language in self.DEFAULT_FRAMEWORKS:
            framework = self.DEFAULT_FRAMEWORKS[language]
            self.logger.info(
                f"Using default framework for {language.value}: {framework.value}"
            )
            return framework

        # Fallback: Try to detect language from project
        detected_language = await self._detect_primary_language(project_path)
        if detected_language and detected_language in self.DEFAULT_FRAMEWORKS:
            framework = self.DEFAULT_FRAMEWORKS[detected_language]
            self.logger.info(
                f"Using default framework for detected language {detected_language.value}: {framework.value}"
            )
            return framework

        # Ultimate fallback
        self.logger.warning("Could not detect framework, defaulting to pytest")
        return TestFramework.PYTEST

    async def _check_dependency_files(
        self, project_path: str
    ) -> Optional[TestFramework]:
        """
        Check package/dependency files for framework indicators.

        Args:
            project_path: Project directory path

        Returns:
            Detected framework or None
        """
        # Check package.json for JavaScript/TypeScript
        package_json_path = os.path.join(project_path, "package.json")
        if os.path.exists(package_json_path):
            try:
                with open(package_json_path, "r", encoding="utf-8") as f:
                    package_json = json.load(f)

                # Check devDependencies and dependencies
                dev_deps = package_json.get("devDependencies", {})
                deps = package_json.get("dependencies", {})
                all_deps = {**dev_deps, **deps}

                if "jest" in all_deps:
                    return TestFramework.JEST
                elif "mocha" in all_deps:
                    return TestFramework.MOCHA
                elif "jasmine" in all_deps:
                    return TestFramework.JASMINE

            except Exception as e:
                self.logger.warning(f"Error reading package.json: {e}")

        # Check requirements.txt for Python
        requirements_path = os.path.join(project_path, "requirements.txt")
        if os.path.exists(requirements_path):
            try:
                with open(requirements_path, "r", encoding="utf-8") as f:
                    requirements = f.read().lower()

                if "pytest" in requirements:
                    return TestFramework.PYTEST

            except Exception as e:
                self.logger.warning(f"Error reading requirements.txt: {e}")

        # Check Pipfile for Python
        pipfile_path = os.path.join(project_path, "Pipfile")
        if os.path.exists(pipfile_path):
            try:
                with open(pipfile_path, "r", encoding="utf-8") as f:
                    pipfile = f.read().lower()

                if "pytest" in pipfile:
                    return TestFramework.PYTEST

            except Exception as e:
                self.logger.warning(f"Error reading Pipfile: {e}")

        # Check pyproject.toml for Python
        pyproject_path = os.path.join(project_path, "pyproject.toml")
        if os.path.exists(pyproject_path):
            try:
                with open(pyproject_path, "r", encoding="utf-8") as f:
                    pyproject = f.read().lower()

                if "pytest" in pyproject:
                    return TestFramework.PYTEST

            except Exception as e:
                self.logger.warning(f"Error reading pyproject.toml: {e}")

        # Check pom.xml for Java
        pom_xml_path = os.path.join(project_path, "pom.xml")
        if os.path.exists(pom_xml_path):
            return TestFramework.JUNIT

        # Check go.mod for Go
        go_mod_path = os.path.join(project_path, "go.mod")
        if os.path.exists(go_mod_path):
            return TestFramework.GO_TEST

        return None

    async def _check_config_files(self, project_path: str) -> Optional[TestFramework]:
        """
        Check for framework-specific config files.

        Args:
            project_path: Project directory path

        Returns:
            Detected framework or None
        """
        for framework, config_files in self.FRAMEWORK_CONFIG_FILES.items():
            for config_file in config_files:
                config_path = os.path.join(project_path, config_file)
                if os.path.exists(config_path):
                    return framework

        return None

    async def _check_test_files(self, project_path: str) -> Optional[TestFramework]:
        """
        Check test file patterns to detect framework.

        Args:
            project_path: Project directory path

        Returns:
            Detected framework or None
        """
        # Look for test files matching framework patterns
        framework_scores: Dict[TestFramework, int] = {}

        for framework, patterns in self.TEST_FILE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                # Search for test files
                test_files = glob.glob(
                    os.path.join(project_path, "**", pattern), recursive=True
                )
                score += len(test_files)

            if score > 0:
                framework_scores[framework] = score

        # Return framework with highest score
        if framework_scores:
            best_framework = max(framework_scores.items(), key=lambda x: x[1])[0]
            return best_framework

        return None

    async def _detect_primary_language(self, project_path: str) -> Optional[Language]:
        """
        Detect primary programming language of project.

        Args:
            project_path: Project directory path

        Returns:
            Detected language or None
        """
        language_extensions = {
            ".py": Language.PYTHON,
            ".js": Language.JAVASCRIPT,
            ".jsx": Language.JAVASCRIPT,
            ".ts": Language.TYPESCRIPT,
            ".tsx": Language.TYPESCRIPT,
            ".java": Language.JAVA,
            ".go": Language.GO,
        }

        # Count files by extension
        extension_counts: Dict[Language, int] = {}

        for root, _, files in os.walk(project_path):
            # Skip common directories
            if any(
                skip in root
                for skip in [
                    "node_modules",
                    ".git",
                    "__pycache__",
                    "venv",
                    "env",
                    "dist",
                    "build",
                ]
            ):
                continue

            for file in files:
                ext = Path(file).suffix.lower()
                if ext in language_extensions:
                    lang = language_extensions[ext]
                    extension_counts[lang] = extension_counts.get(lang, 0) + 1

        # Return most common language
        if extension_counts:
            primary_language = max(extension_counts.items(), key=lambda x: x[1])[0]
            return primary_language

        return None

    async def get_framework_config(
        self, framework: TestFramework, project_path: str
    ) -> Dict[str, Any]:
        """
        Get framework-specific configuration.

        Args:
            framework: Testing framework
            project_path: Project directory

        Returns:
            Framework configuration dictionary
        """
        config = {
            "framework": framework.value,
            "project_path": project_path,
        }

        # Add framework-specific config
        if framework == TestFramework.PYTEST:
            config.update(await self._get_pytest_config(project_path))
        elif framework == TestFramework.JEST:
            config.update(await self._get_jest_config(project_path))
        elif framework == TestFramework.JUNIT:
            config.update(await self._get_junit_config(project_path))

        return config

    async def _get_pytest_config(self, project_path: str) -> Dict[str, Any]:
        """Get pytest configuration."""
        return {
            "test_dir": "tests",
            "test_pattern": "test_*.py",
            "fixture_support": True,
            "parametrize_support": True,
        }

    async def _get_jest_config(self, project_path: str) -> Dict[str, Any]:
        """Get Jest configuration."""
        return {
            "test_dir": "__tests__",
            "test_pattern": "*.test.js",
            "mock_support": True,
            "snapshot_support": True,
        }

    async def _get_junit_config(self, project_path: str) -> Dict[str, Any]:
        """Get JUnit configuration."""
        return {
            "test_dir": "src/test/java",
            "test_pattern": "*Test.java",
            "annotation_based": True,
        }
