"""Base classes for test healing components."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ...models.healing_models import (
    TestFailure,
    FailureAnalysis,
    TestRepair,
)

logger = logging.getLogger(__name__)


class BaseHealer(ABC):
    """
    Abstract base class for test healing components.

    PATTERN: Abstract interface for healing operations
    CRITICAL: All healers must implement async methods
    GOTCHA: Healing operations should be isolated and testable
    """

    def __init__(self):
        """Initialize base healer."""
        self.logger = logger

    @abstractmethod
    async def analyze_failure(
        self, failure: TestFailure, context: Optional[Dict[str, Any]] = None
    ) -> FailureAnalysis:
        """
        Analyze a test failure to identify root cause.

        Args:
            failure: Test failure to analyze
            context: Optional additional context (code changes, etc.)

        Returns:
            Analysis of the failure root cause
        """
        pass

    @abstractmethod
    async def repair_test(
        self, analysis: FailureAnalysis, max_attempts: int = 3
    ) -> Optional[TestRepair]:
        """
        Attempt to repair a failing test.

        Args:
            analysis: Failure analysis
            max_attempts: Maximum repair attempts

        Returns:
            Test repair if successful, None otherwise
        """
        pass


class BaseAnalyzer(ABC):
    """
    Abstract base for failure analyzers.

    PATTERN: Strategy pattern for different failure types
    CRITICAL: Each analyzer handles specific failure patterns
    """

    def __init__(self):
        """Initialize analyzer."""
        self.logger = logger

    @abstractmethod
    async def can_analyze(self, failure: TestFailure) -> bool:
        """
        Check if this analyzer can handle the failure type.

        Args:
            failure: Test failure

        Returns:
            True if analyzer can handle this failure
        """
        pass

    @abstractmethod
    async def analyze(
        self, failure: TestFailure, error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform detailed analysis of the failure.

        Args:
            failure: Test failure
            error_info: Parsed error information

        Returns:
            Analysis results with root cause and confidence
        """
        pass


class BaseRepairStrategy(ABC):
    """
    Abstract base for repair strategies.

    PATTERN: Strategy pattern for different repair approaches
    CRITICAL: Strategies must validate repairs before applying
    """

    def __init__(self):
        """Initialize repair strategy."""
        self.logger = logger

    @abstractmethod
    async def can_repair(self, analysis: FailureAnalysis) -> bool:
        """
        Check if this strategy can repair the failure.

        Args:
            analysis: Failure analysis

        Returns:
            True if strategy can handle this failure
        """
        pass

    @abstractmethod
    async def repair(self, test_code: str, analysis: FailureAnalysis) -> Optional[str]:
        """
        Apply repair to test code.

        Args:
            test_code: Original test code
            analysis: Failure analysis

        Returns:
            Repaired test code if successful, None otherwise
        """
        pass

    def _validate_syntax(self, code: str, language: str = "python") -> bool:
        """
        Validate syntax of repaired code.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            True if syntax is valid
        """
        if language == "python":
            try:
                compile(code, "<string>", "exec")
                return True
            except SyntaxError:
                return False
        return True  # Assume valid for other languages
