"""Syntax error analyzer for test failures."""

import re
import logging
from typing import Dict, Any

from ..base import BaseAnalyzer
from ....models.healing_models import TestFailure, FailureType

logger = logging.getLogger(__name__)


class SyntaxErrorAnalyzer(BaseAnalyzer):
    """
    Analyze syntax errors in test code.

    PATTERN: Parse error messages to identify syntax issues
    CRITICAL: Extract line numbers and error details
    """

    async def can_analyze(self, failure: TestFailure) -> bool:
        """Check if this is a syntax error."""
        return failure.failure_type == FailureType.SYNTAX_ERROR

    async def analyze(
        self, failure: TestFailure, error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze syntax error to identify root cause.

        Args:
            failure: Test failure
            error_info: Parsed error information

        Returns:
            Analysis with root cause and confidence
        """
        error_msg = error_info.get("message", "")

        # Parse syntax error details
        line_match = re.search(r"line (\d+)", error_msg, re.IGNORECASE)
        line_number = int(line_match.group(1)) if line_match else failure.line_number

        # Identify specific syntax issue
        root_cause = "Unknown syntax error"
        confidence = 0.5
        evidence = []

        if "unexpected indent" in error_msg.lower():
            root_cause = "Unexpected indentation in test code"
            confidence = 0.9
            evidence.append("Indentation error detected")
        elif "unexpected eof" in error_msg.lower():
            root_cause = "Incomplete code block (missing closing bracket, quote, etc.)"
            confidence = 0.85
            evidence.append("EOF error indicates incomplete syntax")
        elif "invalid syntax" in error_msg.lower():
            if ":" in error_msg:
                root_cause = "Missing or incorrect colon in control statement"
                confidence = 0.8
            else:
                root_cause = "Invalid Python syntax"
                confidence = 0.7
            evidence.append("Invalid syntax detected")
        elif "expected" in error_msg.lower():
            root_cause = f"Syntax error: {error_msg}"
            confidence = 0.75
            evidence.append("Parser expected different syntax")

        return {
            "description": root_cause,
            "confidence": confidence,
            "evidence": evidence,
            "line_number": line_number,
            "error_type": "syntax",
        }
