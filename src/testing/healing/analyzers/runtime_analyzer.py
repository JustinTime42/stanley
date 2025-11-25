"""Runtime error analyzer for test failures."""

import re
import logging
from typing import Dict, Any

from ..base import BaseAnalyzer
from ....models.healing_models import TestFailure, FailureType

logger = logging.getLogger(__name__)


class RuntimeAnalyzer(BaseAnalyzer):
    """
    Analyze runtime errors (AttributeError, TypeError, etc.).

    PATTERN: Parse stack traces to identify code issues
    CRITICAL: Identify attribute/method changes in target code
    """

    async def can_analyze(self, failure: TestFailure) -> bool:
        """Check if this is a runtime error."""
        return failure.failure_type in [
            FailureType.ATTRIBUTE_ERROR,
            FailureType.TYPE_ERROR,
            FailureType.KEY_ERROR,
            FailureType.RUNTIME_ERROR,
        ]

    async def analyze(
        self, failure: TestFailure, error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze runtime error to identify root cause.

        Args:
            failure: Test failure
            error_info: Parsed error information

        Returns:
            Analysis with root cause and suggested fixes
        """
        error_msg = error_info.get("message", "")

        root_cause = "Unknown runtime error"
        confidence = 0.6
        evidence = []

        # Analyze AttributeError
        if failure.failure_type == FailureType.ATTRIBUTE_ERROR:
            attr_match = re.search(
                r"'(\w+)' object has no attribute '(\w+)'", error_msg
            )
            if attr_match:
                obj_type, attr_name = attr_match.groups()
                root_cause = f"Attribute '{attr_name}' not found on {obj_type} object"
                confidence = 0.9
                evidence.append(f"Missing attribute: {attr_name}")
                evidence.append(f"Object type: {obj_type}")

        # Analyze TypeError
        elif failure.failure_type == FailureType.TYPE_ERROR:
            # Check for argument mismatch
            if "takes" in error_msg and "positional argument" in error_msg:
                root_cause = "Function signature changed (argument count mismatch)"
                confidence = 0.85
                evidence.append("Argument count mismatch detected")
            elif "unexpected keyword argument" in error_msg:
                arg_match = re.search(r"unexpected keyword argument '(\w+)'", error_msg)
                if arg_match:
                    arg_name = arg_match.group(1)
                    root_cause = (
                        f"Parameter '{arg_name}' removed from function signature"
                    )
                    confidence = 0.9
                    evidence.append(f"Removed parameter: {arg_name}")
            else:
                root_cause = "Type error in function call or operation"
                confidence = 0.7
                evidence.append("Type mismatch detected")

        # Analyze KeyError
        elif failure.failure_type == FailureType.KEY_ERROR:
            key_match = re.search(r"KeyError: ['\"](.+?)['\"]", error_msg)
            if key_match:
                missing_key = key_match.group(1)
                root_cause = f"Dictionary key '{missing_key}' not found"
                confidence = 0.85
                evidence.append(f"Missing key: {missing_key}")

        # Generic runtime error
        else:
            root_cause = f"Runtime error: {error_msg[:100]}"
            evidence.append("Runtime exception occurred")

        return {
            "description": root_cause,
            "confidence": confidence,
            "evidence": evidence,
            "error_type": "runtime",
        }
