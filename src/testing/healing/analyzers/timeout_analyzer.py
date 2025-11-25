"""Timeout and performance analyzer."""

import logging
from typing import Dict, Any

from ..base import BaseAnalyzer
from ....models.healing_models import TestFailure, FailureType

logger = logging.getLogger(__name__)


class TimeoutAnalyzer(BaseAnalyzer):
    """
    Analyze test timeouts and performance issues.

    PATTERN: Identify performance regressions and timing issues
    CRITICAL: Distinguish between deadlocks and slow operations
    """

    async def can_analyze(self, failure: TestFailure) -> bool:
        """Check if this is a timeout."""
        return failure.failure_type == FailureType.TIMEOUT

    async def analyze(
        self, failure: TestFailure, error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze timeout to identify performance issues.

        Args:
            failure: Test failure
            error_info: Parsed error information

        Returns:
            Analysis with performance insights
        """
        error_msg = error_info.get("message", "")
        execution_time = failure.execution_time_ms or 0

        root_cause = "Test execution timeout"
        confidence = 0.8
        evidence = []

        # Analyze timeout patterns
        if "deadlock" in error_msg.lower():
            root_cause = "Potential deadlock in test execution"
            confidence = 0.9
            evidence.append("Deadlock indicators detected")
        elif "async" in error_msg.lower() or "await" in error_msg.lower():
            root_cause = (
                "Async operation not completing (missing await or event loop issue)"
            )
            confidence = 0.85
            evidence.append("Async/await timeout detected")
        elif execution_time > 30000:  # > 30 seconds
            root_cause = "Test execution too slow (performance regression)"
            confidence = 0.8
            evidence.append(f"Execution time: {execution_time}ms")
        else:
            root_cause = "Test exceeded timeout threshold"
            evidence.append("Timeout occurred")

        # Check for common timeout causes
        if "network" in error_msg.lower() or "http" in error_msg.lower():
            root_cause = (
                "Network operation timeout (missing mock or slow external service)"
            )
            confidence = 0.85
            evidence.append("Network-related timeout")

        return {
            "description": root_cause,
            "confidence": confidence,
            "evidence": evidence,
            "execution_time_ms": execution_time,
            "error_type": "timeout",
        }
