"""Assertion failure analyzer."""

import re
import logging
from typing import Dict, Any, Optional

from ..base import BaseAnalyzer
from ....models.healing_models import TestFailure, FailureType

logger = logging.getLogger(__name__)


class AssertionAnalyzer(BaseAnalyzer):
    """
    Analyze assertion failures in tests.

    PATTERN: Extract expected vs actual values
    CRITICAL: Identify value changes that broke assertions
    """

    async def can_analyze(self, failure: TestFailure) -> bool:
        """Check if this is an assertion failure."""
        return failure.failure_type == FailureType.ASSERTION_FAILED

    async def analyze(
        self, failure: TestFailure, error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze assertion failure to identify what changed.

        Args:
            failure: Test failure
            error_info: Parsed error information

        Returns:
            Analysis with expected vs actual values
        """
        error_msg = error_info.get("message", "")

        # Extract expected and actual values
        expected, actual = self._extract_values(error_msg)

        # Determine root cause
        if expected and actual:
            root_cause = f"Assertion failed: expected {expected}, got {actual}"
            confidence = 0.9
            evidence = [
                f"Expected value: {expected}",
                f"Actual value: {actual}",
                "Value mismatch detected",
            ]
        else:
            root_cause = "Assertion failed with value mismatch"
            confidence = 0.7
            evidence = ["Assertion failure detected"]

        # Check for common patterns
        if "none" in str(actual).lower() and "none" not in str(expected).lower():
            root_cause = "Function returned None unexpectedly"
            evidence.append("Unexpected None return value")
            confidence = 0.85
        elif isinstance(expected, str) and isinstance(actual, str):
            if self._is_similar(expected, actual):
                root_cause = "String format or content changed slightly"
                evidence.append("Similar but not identical strings")
                confidence = 0.8

        return {
            "description": root_cause,
            "confidence": confidence,
            "evidence": evidence,
            "expected_value": expected,
            "actual_value": actual,
            "error_type": "assertion",
        }

    def _extract_values(self, error_msg: str) -> tuple[Optional[Any], Optional[Any]]:
        """
        Extract expected and actual values from error message.

        Args:
            error_msg: Error message

        Returns:
            Tuple of (expected, actual)
        """
        # Try different assertion message formats
        patterns = [
            r"assert (.+) == (.+)",
            r"expected: (.+), got: (.+)",
            r"(.+) != (.+)",
            r"AssertionError: (.+) vs (.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()

        return None, None

    def _is_similar(self, s1: str, s2: str) -> bool:
        """Check if two strings are similar (>80% match)."""
        if not s1 or not s2:
            return False

        # Simple similarity check
        longer = max(len(s1), len(s2))
        if longer == 0:
            return True

        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        similarity = matches / longer

        return similarity > 0.8
