"""Main test generation orchestrator."""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime

from ..models.testing_models import (
    TestSuite,
    TestGenerationRequest,
    TestFramework,
)
from ..models.analysis_models import ASTNode, Language
from .framework_detector import FrameworkDetector
from .frameworks.pytest_generator import PytestGenerator
from .frameworks.jest_generator import JestGenerator
from .coverage_analyzer import CoverageAnalyzer
from .test_enhancer import TestEnhancer

logger = logging.getLogger(__name__)


class TestGenerator:
    """
    Main test generation orchestrator.

    PATTERN: Orchestrate test generation pipeline
    CRITICAL: Validate generated tests before saving
    GOTCHA: Different frameworks have different patterns
    """

    def __init__(
        self,
        ast_parser: Optional[Any] = None,
        llm_service: Optional[Any] = None,
    ):
        """
        Initialize test generator.

        Args:
            ast_parser: Optional AST parser for code analysis
            llm_service: Optional LLM service for test generation
        """
        self.logger = logger
        self.ast_parser = ast_parser
        self.llm_service = llm_service

        # Initialize components
        self.framework_detector = FrameworkDetector()
        self.coverage_analyzer = CoverageAnalyzer()
        self.test_enhancer = TestEnhancer(self.coverage_analyzer)

        # Register framework generators
        self.framework_generators: Dict[TestFramework, Any] = {
            TestFramework.PYTEST: PytestGenerator(),
            TestFramework.JEST: JestGenerator(),
            # Add more frameworks as implemented
        }

        self.logger.info(
            f"TestGenerator initialized with {len(self.framework_generators)} frameworks"
        )

    async def generate_test_suite(
        self, target_file: str, request: Optional[TestGenerationRequest] = None
    ) -> TestSuite:
        """
        Generate complete test suite for target file.

        PATTERN: Analysis → Generation → Validation → Enhancement

        Args:
            target_file: File to generate tests for
            request: Optional test generation parameters

        Returns:
            TestSuite with generated tests
        """
        start_time = datetime.now()

        # Create default request if not provided
        if request is None:
            request = TestGenerationRequest(target_files=[target_file])

        self.logger.info(f"Generating tests for {target_file}")

        # Step 1: Detect language and framework
        project_path = os.path.dirname(target_file) or "."
        language = self._detect_language(target_file)
        framework = request.framework or await self.framework_detector.detect_framework(
            project_path, language
        )

        self.logger.info(f"Detected framework: {framework.value}")

        # Step 2: Parse code (if AST parser available)
        ast = None
        if self.ast_parser:
            try:
                ast = await self.ast_parser.parse_file(target_file, language)
            except Exception as e:
                self.logger.warning(f"AST parsing failed: {e}")

        # Step 3: Get appropriate generator
        generator = self.framework_generators.get(framework)
        if not generator:
            raise ValueError(
                f"No generator available for framework: {framework.value}"
            )

        # Step 4: Generate test suite
        test_suite = await generator.generate_test_suite(
            target_file=target_file, request=request, ast=ast
        )

        # Step 5: Validate generated tests
        if not await generator.validate_test_syntax(test_suite):
            self.logger.error("Generated tests have syntax errors")
            raise ValueError("Test generation produced invalid syntax")

        self.logger.info(
            f"Generated {len(test_suite.test_cases)} tests "
            f"and {len(test_suite.property_tests)} property tests"
        )

        # Step 6: Enhance coverage if needed
        if (
            request.coverage_target > 0
            and test_suite.total_coverage < request.coverage_target
        ):
            self.logger.info(
                f"Enhancing coverage from {test_suite.total_coverage:.2%} "
                f"to {request.coverage_target:.2%}"
            )

            # Create coverage report (simplified)
            coverage_report = {
                "total_coverage": test_suite.total_coverage,
                "files": {target_file: {"uncovered_lines": []}},
            }

            # Enhance
            test_suite = await self.test_enhancer.enhance_coverage(
                test_suite, coverage_report, request.coverage_target
            )

        # Step 7: Optimize test suite
        test_suite = await self.test_enhancer.optimize_test_suite(test_suite)

        # Calculate total generation time
        generation_time = (datetime.now() - start_time).total_seconds() * 1000
        test_suite.generation_time_ms = int(generation_time)

        self.logger.info(
            f"Test generation completed in {generation_time:.0f}ms "
            f"with {test_suite.total_coverage:.2%} coverage"
        )

        return test_suite

    async def generate_tests_for_project(
        self, project_path: str, request: Optional[TestGenerationRequest] = None
    ) -> Dict[str, TestSuite]:
        """
        Generate tests for entire project.

        Args:
            project_path: Project directory
            request: Optional test generation parameters

        Returns:
            Dictionary mapping file paths to test suites
        """
        self.logger.info(f"Generating tests for project: {project_path}")

        # Find all source files
        source_files = await self._find_source_files(project_path)

        self.logger.info(f"Found {len(source_files)} source files")

        # Generate tests for each file
        test_suites = {}

        for source_file in source_files:
            try:
                # Create request for this file
                file_request = request or TestGenerationRequest(
                    target_files=[source_file]
                )

                # Generate tests
                test_suite = await self.generate_test_suite(
                    source_file, file_request
                )

                test_suites[source_file] = test_suite

                self.logger.info(
                    f"Generated tests for {source_file}: "
                    f"{len(test_suite.test_cases)} tests"
                )

            except Exception as e:
                self.logger.error(f"Failed to generate tests for {source_file}: {e}")

        return test_suites

    async def save_test_suite(
        self, test_suite: TestSuite, output_path: Optional[str] = None
    ) -> str:
        """
        Save test suite to file.

        Args:
            test_suite: Test suite to save
            output_path: Optional output path (default: use test_file from suite)

        Returns:
            Path to saved test file
        """
        # Determine output path
        if output_path is None:
            if test_suite.test_cases:
                output_path = test_suite.test_cases[0].test_file
            else:
                raise ValueError("No output path specified and no test cases in suite")

        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Build test file content
        content = await self._build_test_file_content(test_suite)

        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        self.logger.info(f"Saved test suite to {output_path}")

        return output_path

    async def _build_test_file_content(self, test_suite: TestSuite) -> str:
        """
        Build complete test file content from suite.

        Args:
            test_suite: Test suite

        Returns:
            Test file content
        """
        lines = []

        # Add header comment
        lines.append(f"# Generated tests for {test_suite.target_module}")
        lines.append(f"# Framework: {test_suite.framework.value}")
        lines.append(f"# Generated at: {test_suite.generated_at.isoformat()}")
        lines.append("")

        # Add sys.path setup to ensure imports work
        # This adds the project root and src directory to path
        lines.append("import sys")
        lines.append("import os")
        lines.append("")
        lines.append("# Add project root and src to path for imports")
        lines.append("_test_dir = os.path.dirname(os.path.abspath(__file__))")
        lines.append("_project_root = os.path.dirname(_test_dir)")
        lines.append("_src_dir = os.path.join(_project_root, 'src')")
        lines.append("if _src_dir not in sys.path:")
        lines.append("    sys.path.insert(0, _src_dir)")
        lines.append("if _project_root not in sys.path:")
        lines.append("    sys.path.insert(0, _project_root)")
        lines.append("")

        # Add imports (framework-specific)
        if test_suite.framework == TestFramework.PYTEST:
            lines.append("import pytest")
            lines.append("")

        # Add each test case
        for test_case in test_suite.test_cases:
            lines.append(test_case.test_body)
            lines.append("")

        # Add property tests
        for prop_test in test_suite.property_tests:
            lines.append(f"# Property test: {prop_test.id}")
            lines.append("")

        return "\n".join(lines)

    def _detect_language(self, file_path: str) -> Language:
        """
        Detect programming language from file extension.

        Args:
            file_path: File path

        Returns:
            Detected language
        """
        ext = os.path.splitext(file_path)[1].lower()

        extension_map = {
            ".py": Language.PYTHON,
            ".js": Language.JAVASCRIPT,
            ".jsx": Language.JAVASCRIPT,
            ".ts": Language.TYPESCRIPT,
            ".tsx": Language.TYPESCRIPT,
            ".java": Language.JAVA,
            ".go": Language.GO,
        }

        return extension_map.get(ext, Language.UNKNOWN)

    async def _find_source_files(
        self, project_path: str, max_files: int = 100
    ) -> list[str]:
        """
        Find source files in project.

        Args:
            project_path: Project directory
            max_files: Maximum files to find

        Returns:
            List of source file paths
        """
        source_files = []

        # Common source extensions
        source_extensions = [".py", ".js", ".ts", ".java", ".go"]

        # Skip directories
        skip_dirs = [
            "node_modules",
            ".git",
            "__pycache__",
            "venv",
            "env",
            "dist",
            "build",
            "tests",
            "test",
            "__tests__",
        ]

        for root, dirs, files in os.walk(project_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in source_extensions:
                    source_files.append(os.path.join(root, file))

                    if len(source_files) >= max_files:
                        return source_files

        return source_files
