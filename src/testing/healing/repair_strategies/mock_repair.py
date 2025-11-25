"""Repair mock configuration errors.

This module handles test failures caused by mock interface changes,
mock specification mismatches, and mock setup issues.
"""

import re
import logging
from typing import Optional, Dict, List, Any

from ..base import BaseRepairStrategy
from ....models.healing_models import (
    FailureAnalysis,
    RepairStrategy,
    FailureType,
)

logger = logging.getLogger(__name__)


class MockRepair(BaseRepairStrategy):
    """Fix mock configuration when interface changes.

    PATTERN: Update mock setup when mocked interface changes
    CRITICAL: Preserve mock behavior expectations
    GOTCHA: Handle unittest.mock, pytest-mock, and other mock libraries
    """

    async def can_repair(self, analysis: FailureAnalysis) -> bool:
        """Check if this strategy can repair the failure.

        Args:
            analysis: Failure analysis

        Returns:
            True if this strategy can handle mock-related failures
        """
        # Check if failure is related to mocks
        failure_type = analysis.failure.failure_type
        suggested_strategies = analysis.suggested_strategies

        # Can repair if:
        # 1. Mock error explicitly identified
        # 2. AttributeError with mock context
        # 3. Explicitly suggested as UPDATE_MOCK strategy
        if failure_type == FailureType.MOCK_ERROR:
            return True

        if RepairStrategy.UPDATE_MOCK in suggested_strategies:
            return True

        # Check error message and root cause for mock-related patterns
        text = (analysis.failure.error_message + " " + analysis.root_cause).lower()

        if any(
            keyword in text
            for keyword in [
                "mock",
                "patch",
                "mocker",
                "magicmock",
                "mock_open",
                "return_value",
                "side_effect",
                "assert_called",
            ]
        ):
            return True

        return False

    async def repair(self, test_code: str, analysis: FailureAnalysis) -> Optional[str]:
        """Apply mock repair to test code.

        Args:
            test_code: Original test code
            analysis: Failure analysis with mock details

        Returns:
            Repaired test code if successful, None otherwise
        """
        try:
            root_cause = analysis.root_cause
            error_msg = analysis.failure.error_message

            # Identify mock issue type
            mock_issue = self._identify_mock_issue(root_cause, error_msg)

            if mock_issue["type"] == "spec_mismatch":
                repaired = self._repair_spec_mismatch(test_code, mock_issue)
            elif mock_issue["type"] == "missing_attribute":
                repaired = self._repair_missing_attribute(test_code, mock_issue)
            elif mock_issue["type"] == "call_signature":
                repaired = self._repair_call_signature(test_code, mock_issue)
            else:
                # Generic repair - add FIXME comments
                repaired = self._add_mock_comments(test_code, root_cause)

            # Validate syntax
            if not self._validate_syntax(repaired):
                logger.warning("Repaired code has syntax errors")
                return None

            return repaired

        except Exception as e:
            logger.error(f"Mock repair failed: {e}")
            return None

    def _identify_mock_issue(self, root_cause: str, error_msg: str) -> Dict[str, Any]:
        """Identify the type of mock issue.

        Args:
            root_cause: Root cause description
            error_msg: Error message

        Returns:
            Dictionary describing the mock issue
        """
        text = root_cause + " " + error_msg

        # Issue 1: Spec mismatch (Mock called with spec that doesn't match)
        if "spec" in text.lower():
            return {"type": "spec_mismatch", "details": text}

        # Issue 2: Missing attribute on mock
        match = re.search(r"[Mm]ock.*?has no attribute ['\"](.+?)['\"]", text)
        if match:
            return {
                "type": "missing_attribute",
                "attribute": match.group(1),
                "details": text,
            }

        # Issue 3: Call signature issues (assert_called_with, etc.)
        if any(
            keyword in text
            for keyword in ["assert_called", "call_args", "call_count", "called_with"]
        ):
            return {"type": "call_signature", "details": text}

        # Issue 4: return_value or side_effect issues
        if "return_value" in text or "side_effect" in text:
            return {"type": "mock_behavior", "details": text}

        return {"type": "unknown", "details": text}

    def _repair_spec_mismatch(self, code: str, mock_issue: Dict) -> str:
        """Repair mock spec mismatches.

        Args:
            code: Original code
            mock_issue: Mock issue details

        Returns:
            Repaired code
        """
        lines = code.split("\n")
        repaired_lines = []

        for line in lines:
            # Find lines with Mock() or patch() that have spec
            if ("Mock(" in line or "patch(" in line) and "spec" in line:
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                repaired_lines.append(
                    f"{indent_str}# FIXME: Mock spec may need updating"
                )
                repaired_lines.append(
                    f"{indent_str}# The mocked interface may have changed"
                )

            repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _repair_missing_attribute(self, code: str, mock_issue: Dict) -> str:
        """Repair mock with missing attribute.

        Args:
            code: Original code
            mock_issue: Mock issue details with attribute name

        Returns:
            Repaired code
        """
        attribute = mock_issue.get("attribute", "unknown")
        lines = code.split("\n")
        repaired_lines = []

        for line in lines:
            # Find lines that access the missing attribute on a mock
            if attribute in line and any(
                mock_keyword in line
                for mock_keyword in ["mock", "Mock", "patch", "mocker"]
            ):
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                repaired_lines.append(
                    f"{indent_str}# FIXME: Mock attribute '{attribute}' not found"
                )
                repaired_lines.append(
                    f"{indent_str}# The mocked object's interface may have changed"
                )

            repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _repair_call_signature(self, code: str, mock_issue: Dict) -> str:
        """Repair mock call signature assertions.

        Args:
            code: Original code
            mock_issue: Mock issue details

        Returns:
            Repaired code
        """
        lines = code.split("\n")
        repaired_lines = []

        for line in lines:
            # Find assert_called assertions
            if any(
                keyword in line
                for keyword in [
                    "assert_called",
                    "assert_called_once",
                    "assert_called_with",
                    "assert_called_once_with",
                    "assert_any_call",
                    "assert_has_calls",
                ]
            ):
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                repaired_lines.append(
                    f"{indent_str}# FIXME: Mock call assertion may need updating"
                )
                repaired_lines.append(
                    f"{indent_str}# The function signature may have changed"
                )

            repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _add_mock_comments(self, code: str, root_cause: str) -> str:
        """Add generic mock-related comments.

        Args:
            code: Original code
            root_cause: Root cause description

        Returns:
            Code with comments added
        """
        lines = code.split("\n")
        repaired_lines = []

        mock_section_started = False

        for line in lines:
            # Detect mock-related lines
            is_mock_line = any(
                keyword in line.lower()
                for keyword in [
                    "mock",
                    "patch",
                    "mocker",
                    "magicmock",
                    "return_value",
                    "side_effect",
                ]
            )

            if is_mock_line and not mock_section_started:
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                repaired_lines.append(
                    f"{indent_str}# FIXME: Mock configuration may need updating"
                )
                repaired_lines.append(f"{indent_str}# Issue: {root_cause}")
                mock_section_started = True

            repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _find_mock_blocks(self, code: str) -> List[tuple]:
        """Find blocks of code that deal with mocks.

        Args:
            code: Source code

        Returns:
            List of (start_line, end_line) tuples for mock blocks
        """
        lines = code.split("\n")
        mock_blocks = []
        current_block_start = None

        for i, line in enumerate(lines):
            is_mock_line = any(
                keyword in line.lower()
                for keyword in ["mock", "patch", "mocker", "@mock", "@patch"]
            )

            if is_mock_line and current_block_start is None:
                current_block_start = i
            elif not is_mock_line and current_block_start is not None:
                # Check if we've left the mock block
                if line.strip() and not line.strip().startswith("#"):
                    mock_blocks.append((current_block_start, i - 1))
                    current_block_start = None

        # Close any open block
        if current_block_start is not None:
            mock_blocks.append((current_block_start, len(lines) - 1))

        return mock_blocks
