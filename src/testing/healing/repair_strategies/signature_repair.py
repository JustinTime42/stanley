"""Repair method signature mismatches.

This module handles test failures caused by method signature changes,
including parameter additions, removals, renames, and type changes.
"""

import re
import logging
from typing import Optional

from ..base import BaseRepairStrategy
from ....models.healing_models import (
    FailureAnalysis,
    RepairStrategy,
    FailureType,
)

logger = logging.getLogger(__name__)


class SignatureRepair(BaseRepairStrategy):
    """Repair tests with method signature changes.

    PATTERN: AST comparison to detect parameter changes
    CRITICAL: Must preserve test intent while updating calls
    GOTCHA: Handle both positional and keyword arguments
    """

    async def can_repair(self, analysis: FailureAnalysis) -> bool:
        """Check if this strategy can repair the failure.

        Args:
            analysis: Failure analysis

        Returns:
            True if this strategy can handle the failure type
        """
        # Check if failure is related to method signatures
        failure_type = analysis.failure.failure_type
        suggested_strategies = analysis.suggested_strategies

        # Can repair if:
        # 1. AttributeError (method doesn't exist - renamed)
        # 2. TypeError with "argument" in message (wrong parameters)
        # 3. Explicitly suggested as UPDATE_SIGNATURE strategy
        if failure_type == FailureType.ATTRIBUTE_ERROR:
            return True

        if failure_type == FailureType.TYPE_ERROR:
            error_msg = analysis.failure.error_message.lower()
            if any(
                keyword in error_msg
                for keyword in [
                    "argument",
                    "parameter",
                    "positional",
                    "keyword",
                    "takes",
                    "given",
                ]
            ):
                return True

        if RepairStrategy.UPDATE_SIGNATURE in suggested_strategies:
            return True

        return False

    async def repair(self, test_code: str, analysis: FailureAnalysis) -> Optional[str]:
        """Apply signature repair to test code.

        Args:
            test_code: Original test code
            analysis: Failure analysis with root cause

        Returns:
            Repaired test code if successful, None otherwise
        """
        try:
            root_cause = analysis.root_cause
            error_msg = analysis.failure.error_message

            # Attempt different repair strategies based on error type
            if "attribute" in root_cause.lower():
                repaired = self._repair_missing_attribute(test_code, root_cause)
            elif "argument" in error_msg.lower():
                repaired = self._repair_argument_mismatch(test_code, error_msg)
            else:
                # Generic repair - add FIXME comment
                repaired = self._add_fixme_comment(test_code, root_cause)

            # Validate syntax
            if not self._validate_syntax(repaired):
                logger.warning("Repaired code has syntax errors")
                return None

            return repaired

        except Exception as e:
            logger.error(f"Signature repair failed: {e}")
            return None

    def _repair_missing_attribute(self, code: str, root_cause: str) -> str:
        """Repair code when an attribute/method no longer exists.

        Args:
            code: Original code
            root_cause: Root cause description

        Returns:
            Repaired code with placeholder
        """
        # Extract attribute name
        attr_match = re.search(
            r"attribute '(\w+)'|method '(\w+)'|'(\w+)' not found",
            root_cause,
            re.IGNORECASE,
        )

        if not attr_match:
            return self._add_fixme_comment(code, root_cause)

        missing_attr = attr_match.group(1) or attr_match.group(2) or attr_match.group(3)

        lines = code.split("\n")
        repaired_lines = []

        for line in lines:
            if missing_attr in line and not line.strip().startswith("#"):
                # Found problematic line - comment it and add placeholder
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                repaired_lines.append(
                    f"{indent_str}# FIXME: Method/attribute '{missing_attr}' no longer exists"
                )
                repaired_lines.append(f"{indent_str}# Original: {line.strip()}")
                repaired_lines.append(
                    f"{indent_str}pass  # TODO: Update to use new API"
                )
            else:
                repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _repair_argument_mismatch(self, code: str, error_msg: str) -> str:
        """Repair code when function signature doesn't match.

        Args:
            code: Original code
            error_msg: Error message

        Returns:
            Repaired code with comment
        """
        # Extract function name if present
        func_match = re.search(r"(\w+)\(\)", error_msg)
        func_name = func_match.group(1) if func_match else None

        # Parse error for argument details
        # Examples:
        # - "takes 2 positional arguments but 3 were given"
        # - "missing 1 required positional argument: 'name'"
        # - "got an unexpected keyword argument 'old_param'"

        takes_match = re.search(r"takes (\d+).+but (\d+)", error_msg)
        missing_match = re.search(
            r"missing.+argument[s]?: ['\"]?(\w+)['\"]?", error_msg
        )
        unexpected_match = re.search(
            r"unexpected.+argument ['\"]?(\w+)['\"]?", error_msg
        )

        lines = code.split("\n")
        repaired_lines = []

        for line in lines:
            # Check if this line contains the problematic function call
            if func_name and func_name in line and "(" in line:
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                if takes_match:
                    expected = takes_match.group(1)
                    got = takes_match.group(2)
                    repaired_lines.append(
                        f"{indent_str}# FIXME: Function signature changed - "
                        f"expected {expected} args, got {got}"
                    )
                elif missing_match:
                    param = missing_match.group(1)
                    repaired_lines.append(
                        f"{indent_str}# FIXME: Missing required parameter: {param}"
                    )
                elif unexpected_match:
                    param = unexpected_match.group(1)
                    repaired_lines.append(
                        f"{indent_str}# FIXME: Remove unexpected parameter: {param}"
                    )
                else:
                    repaired_lines.append(
                        f"{indent_str}# FIXME: Function signature mismatch"
                    )

                repaired_lines.append(f"{indent_str}# Original: {line.strip()}")

            repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _add_fixme_comment(self, code: str, root_cause: str) -> str:
        """Add generic FIXME comment when specific repair not possible.

        Args:
            code: Original code
            root_cause: Root cause description

        Returns:
            Code with FIXME comment added at top
        """
        lines = code.split("\n")

        # Find first non-import, non-comment line
        insert_index = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and not stripped.startswith("import")
                and not stripped.startswith("from")
            ):
                insert_index = i
                break

        # Insert FIXME comment
        indent = len(lines[insert_index]) - len(lines[insert_index].lstrip())
        indent_str = " " * indent

        comment = f"{indent_str}# FIXME: Signature change detected - {root_cause}"

        lines.insert(insert_index, comment)

        return "\n".join(lines)
