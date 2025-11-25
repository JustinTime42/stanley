"""Project structure extractor."""

import logging
from pathlib import Path
from typing import List, Optional

from .base import BaseExtractor
from ..models import FileInfo, ProjectStructure

logger = logging.getLogger(__name__)


class StructureExtractor(BaseExtractor[ProjectStructure]):
    """
    Extract project structure information.

    PATTERN: Detect project type, frameworks, entry points
    CRITICAL: Fast analysis for quick mode
    """

    # Config files that indicate project type
    CONFIG_FILES = {
        "python": [
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "Pipfile",
            "poetry.lock",
            "tox.ini",
        ],
        "javascript": [
            "package.json",
            "webpack.config.js",
            "rollup.config.js",
            "vite.config.js",
            "tsconfig.json",
        ],
        "typescript": ["tsconfig.json", "package.json"],
        "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
        "go": ["go.mod", "go.sum"],
        "rust": ["Cargo.toml", "Cargo.lock"],
        "ruby": ["Gemfile", "Rakefile"],
        "php": ["composer.json"],
    }

    # Framework detection patterns
    FRAMEWORK_INDICATORS = {
        # Python
        "fastapi": ["from fastapi", "import fastapi", "FastAPI("],
        "django": ["from django", "import django", "DJANGO_SETTINGS"],
        "flask": ["from flask", "import flask", "Flask("],
        "pytest": ["import pytest", "from pytest", "@pytest."],
        "unittest": ["import unittest", "from unittest"],
        # JavaScript/TypeScript
        "react": ["from 'react'", 'from "react"', "import React"],
        "vue": ["from 'vue'", 'from "vue"', ".vue"],
        "angular": ["@angular/core", "@angular/common"],
        "express": ["from 'express'", 'from "express"', "express()"],
        "next": ["from 'next'", 'from "next"', "next.config"],
        "jest": ["from 'jest'", 'from "jest"', "jest.config"],
        "mocha": ["from 'mocha'", 'from "mocha"', "describe("],
    }

    # Common source directories
    SOURCE_DIRS = ["src", "lib", "app", "source", "pkg", "internal", "cmd"]

    # Common test directories
    TEST_DIRS = ["tests", "test", "__tests__", "spec", "specs"]

    # Common entry point files
    ENTRY_POINTS = {
        "python": ["main.py", "__main__.py", "app.py", "cli.py", "run.py", "manage.py"],
        "javascript": ["index.js", "main.js", "app.js", "server.js"],
        "typescript": ["index.ts", "main.ts", "app.ts", "server.ts"],
    }

    async def extract(
        self,
        root_path: Path,
        files: List[FileInfo],
    ) -> ProjectStructure:
        """
        Extract project structure from files.

        Args:
            root_path: Project root directory
            files: List of scanned files

        Returns:
            ProjectStructure with detected info
        """
        # Count files by language
        files_by_language = {}
        total_lines = 0
        for f in files:
            lang = f.language
            files_by_language[lang] = files_by_language.get(lang, 0) + 1
            total_lines += f.line_count

        # Detect primary language
        primary_language = self._detect_primary_language(files_by_language)

        # Detect framework
        framework = await self._detect_framework(root_path, files, primary_language)

        # Find directories
        source_dirs = self._find_directories(root_path, self.SOURCE_DIRS)
        test_dirs = self._find_directories(root_path, self.TEST_DIRS)

        # Find config files
        config_files = self._find_config_files(root_path)

        # Find entry points
        entry_points = self._find_entry_points(root_path, files, primary_language)

        # Detect package manager
        package_manager = self._detect_package_manager(root_path)

        # Detect test framework
        test_framework = self._detect_test_framework(root_path, files, primary_language)

        return ProjectStructure(
            root_path=str(root_path),
            project_name=root_path.name,
            detected_type=primary_language,
            detected_framework=framework,
            source_directories=source_dirs,
            test_directories=test_dirs,
            config_files=config_files,
            entry_points=entry_points,
            total_files=len(files),
            total_lines=total_lines,
            files_by_language=files_by_language,
            package_manager=package_manager,
            test_framework=test_framework,
        )

    def _detect_primary_language(self, files_by_language: dict) -> str:
        """Detect primary language based on file counts."""
        # Filter out non-code languages
        code_languages = {
            k: v
            for k, v in files_by_language.items()
            if k not in ("json", "yaml", "toml", "xml", "markdown", "text", "unknown")
        }

        if not code_languages:
            return "unknown"

        # Return language with most files
        return max(code_languages, key=code_languages.get)

    async def _detect_framework(
        self,
        root_path: Path,
        files: List[FileInfo],
        primary_language: str,
    ) -> Optional[str]:
        """Detect framework from code patterns."""
        # Sample some files to check for framework indicators
        sample_files = [f for f in files if f.language == primary_language][:20]

        framework_scores = {}

        for file_info in sample_files:
            content = self._read_file_content(file_info.path)
            if not content:
                continue

            for framework, patterns in self.FRAMEWORK_INDICATORS.items():
                for pattern in patterns:
                    if pattern in content:
                        framework_scores[framework] = framework_scores.get(framework, 0) + 1

        if framework_scores:
            return max(framework_scores, key=framework_scores.get)

        return None

    def _find_directories(self, root_path: Path, dir_names: List[str]) -> List[str]:
        """Find directories matching names."""
        found = []
        for name in dir_names:
            dir_path = root_path / name
            if dir_path.is_dir():
                found.append(str(dir_path.relative_to(root_path)))
        return found

    def _find_config_files(self, root_path: Path) -> List[str]:
        """Find configuration files in root."""
        config_files = []

        # Flatten all config files
        all_configs = set()
        for configs in self.CONFIG_FILES.values():
            all_configs.update(configs)

        for config in all_configs:
            config_path = root_path / config
            if config_path.exists():
                config_files.append(config)

        return sorted(config_files)

    def _find_entry_points(
        self,
        root_path: Path,
        files: List[FileInfo],
        primary_language: str,
    ) -> List[str]:
        """Find potential entry point files."""
        entry_points = []

        # Get entry point names for this language
        entry_names = self.ENTRY_POINTS.get(primary_language, [])

        for file_info in files:
            file_name = Path(file_info.path).name
            if file_name in entry_names:
                entry_points.append(file_info.relative_path)

        # Also check for shebang lines indicating executables
        for file_info in files[:50]:  # Sample first 50 files
            content = self._read_file_content(file_info.path)
            if content and content.startswith("#!"):
                if file_info.relative_path not in entry_points:
                    entry_points.append(file_info.relative_path)

        return entry_points

    def _detect_package_manager(self, root_path: Path) -> Optional[str]:
        """Detect package manager from lock files."""
        if (root_path / "poetry.lock").exists():
            return "poetry"
        if (root_path / "Pipfile.lock").exists():
            return "pipenv"
        if (root_path / "requirements.txt").exists():
            return "pip"
        if (root_path / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (root_path / "yarn.lock").exists():
            return "yarn"
        if (root_path / "package-lock.json").exists():
            return "npm"
        if (root_path / "Cargo.lock").exists():
            return "cargo"
        if (root_path / "go.sum").exists():
            return "go"
        if (root_path / "Gemfile.lock").exists():
            return "bundler"
        return None

    def _detect_test_framework(
        self,
        root_path: Path,
        files: List[FileInfo],
        primary_language: str,
    ) -> Optional[str]:
        """Detect test framework."""
        # Check for pytest
        if (root_path / "pytest.ini").exists() or (root_path / "conftest.py").exists():
            return "pytest"

        # Check pyproject.toml for pytest
        pyproject = root_path / "pyproject.toml"
        if pyproject.exists():
            content = self._read_file_content(pyproject)
            if content and "[tool.pytest" in content:
                return "pytest"

        # Check package.json for test frameworks
        package_json = root_path / "package.json"
        if package_json.exists():
            content = self._read_file_content(package_json)
            if content:
                if '"jest"' in content:
                    return "jest"
                if '"mocha"' in content:
                    return "mocha"
                if '"vitest"' in content:
                    return "vitest"

        # Check for test files
        test_files = [f for f in files if "test" in f.relative_path.lower()]
        if test_files:
            sample = self._read_file_content(test_files[0].path)
            if sample:
                if "import pytest" in sample or "from pytest" in sample:
                    return "pytest"
                if "import unittest" in sample:
                    return "unittest"
                if "describe(" in sample and "it(" in sample:
                    return "jest/mocha"

        return None
