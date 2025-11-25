"""Base test generator class."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from ..models.testing_models import (
    TestSuite,
    TestCase,
    TestGenerationRequest,
    TestFramework,
    PropertyTest,
)
from ..models.analysis_models import CodeEntity, ASTNode

logger = logging.getLogger(__name__)


class BaseTestGenerator(ABC):
    """
    Abstract base class for framework-specific test generators.

    All test generators must:
    - Implement async generate_test_suite() method
    - Support unit test generation
    - Validate generated tests for syntactic correctness
    - Follow framework-specific conventions
    """

    def __init__(self, framework: TestFramework):
        """
        Initialize base test generator.

        Args:
            framework: Testing framework this generator supports
        """
        self.framework = framework
        self.logger = logging.getLogger(f"{__name__}.{framework.value}")

    @abstractmethod
    async def generate_test_suite(
        self, target_file: str, request: TestGenerationRequest, ast: Optional[ASTNode] = None
    ) -> TestSuite:
        """
        Generate complete test suite for target file.

        CRITICAL: Must be async and return TestSuite
        CRITICAL: Validate test syntax before returning

        Args:
            target_file: File to generate tests for
            request: Test generation parameters
            ast: Optional pre-parsed AST

        Returns:
            TestSuite with generated tests
        """
        pass

    @abstractmethod
    async def generate_unit_test(
        self, function: CodeEntity, include_mocks: bool = True
    ) -> TestCase:
        """
        Generate unit test for a single function.

        Args:
            function: Function to test
            include_mocks: Whether to generate mocks for dependencies

        Returns:
            TestCase for the function
        """
        pass

    @abstractmethod
    async def generate_property_test(
        self, function: CodeEntity
    ) -> Optional[PropertyTest]:
        """
        Generate property-based test if applicable.

        Args:
            function: Function to test

        Returns:
            PropertyTest if function is pure, None otherwise
        """
        pass

    @abstractmethod
    async def validate_test_syntax(self, test_suite: TestSuite) -> bool:
        """
        Validate that generated tests are syntactically correct.

        CRITICAL: Must validate before saving tests
        GOTCHA: Use framework's parser to validate

        Args:
            test_suite: Test suite to validate

        Returns:
            True if all tests are valid
        """
        pass

    async def generate_fixture(
        self, name: str, setup_code: str, scope: str = "function"
    ) -> str:
        """
        Generate test fixture code.

        Args:
            name: Fixture name
            setup_code: Fixture setup code
            scope: Fixture scope (function, module, session)

        Returns:
            Fixture code string
        """
        # Default implementation, can be overridden
        self.logger.info(f"Generating fixture: {name}")
        return setup_code

    async def generate_teardown(self, cleanup_code: str) -> str:
        """
        Generate teardown code.

        Args:
            cleanup_code: Cleanup code

        Returns:
            Teardown code string
        """
        # Default implementation, can be overridden
        return cleanup_code

    def _is_pure_function(self, function: CodeEntity) -> bool:
        """
        Check if function is pure (no side effects).

        CRITICAL: Only generate property tests for pure functions
        PATTERN: Check for file I/O, network calls, mutations

        Args:
            function: Function to check

        Returns:
            True if function appears pure
        """
        # Simple heuristic - can be enhanced
        impure_keywords = [
            "open(",
            "write(",
            "print(",
            "socket",
            "http",
            "requests",
            "fetch(",
            "axios",
            "global ",
            "nonlocal ",
        ]

        # Check function signature for indicators
        if function.signature:
            signature_lower = function.signature.lower()
            for keyword in impure_keywords:
                if keyword in signature_lower:
                    return False

        # Check metadata
        metadata = function.metadata or {}
        has_io = metadata.get("has_io", False)
        has_network = metadata.get("has_network", False)
        has_side_effects = metadata.get("has_side_effects", False)

        if has_io or has_network or has_side_effects:
            return False

        return True

    def _extract_testable_units(self, ast: ASTNode) -> List[CodeEntity]:
        """
        Extract testable functions/methods from AST.

        Args:
            ast: Parsed AST

        Returns:
            List of testable code entities
        """
        # This is a simplified extraction
        # In practice, would use AST traversal to find functions
        testable_units = []

        # Traverse AST to find function definitions
        # This is a placeholder - actual implementation would be more complex
        self.logger.debug("Extracting testable units from AST")

        return testable_units

    async def _estimate_coverage(
        self, test_suite: TestSuite, ast: ASTNode
    ) -> float:
        """
        Estimate test coverage from test suite.

        Args:
            test_suite: Generated test suite
            ast: Source code AST

        Returns:
            Estimated coverage percentage (0-1)
        """
        # Simplified coverage estimation
        # Count lines covered by all test cases
        all_lines_covered = set()

        for test_case in test_suite.test_cases:
            all_lines_covered.update(test_case.lines_covered)

        # Estimate total lines from AST
        # This is simplified - actual implementation would count executable lines
        total_lines = 100  # Placeholder

        if total_lines == 0:
            return 0.0

        coverage = len(all_lines_covered) / total_lines
        return min(coverage, 1.0)

    def _create_test_id(self, target_file: str, function_name: str) -> str:
        """
        Create unique test ID.

        Args:
            target_file: Target file path
            function_name: Function being tested

        Returns:
            Unique test ID
        """
        import hashlib
        import time

        unique_string = f"{target_file}:{function_name}:{time.time()}"
        hash_digest = hashlib.md5(unique_string.encode()).hexdigest()[:8]
        return f"test_{function_name}_{hash_digest}"
