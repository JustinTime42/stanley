"""Repair assertion failures.

This module handles test failures caused by assertion mismatches,
including updated expected values, changed return types, and assertion format changes.
"""

import re
import logging
from typing import Optional, Tuple

from ..base import BaseRepairStrategy
from ....models.healing_models import (
    FailureAnalysis,
    RepairStrategy,
    FailureType,
)

logger = logging.getLogger(__name__)


class AssertionRepair(BaseRepairStrategy):
    """Update assertions when expected values change.

    PATTERN: Extract expected/actual from error, update assertion
    CRITICAL: Only update if value change is reasonable
    GOTCHA: Handle different assertion styles (pytest, unittest, jest)
    """

    async def can_repair(self, analysis: FailureAnalysis) -> bool:
        """Check if this strategy can repair the failure.

        Args:
            analysis: Failure analysis

        Returns:
            True if this strategy can handle assertion failures
        """
        # Check if failure is an assertion error
        failure_type = analysis.failure.failure_type
        suggested_strategies = analysis.suggested_strategies

        # Can repair if:
        # 1. Assertion failed
        # 2. Explicitly suggested as UPDATE_ASSERTION strategy
        if failure_type == FailureType.ASSERTION_FAILED:
            return True

        if RepairStrategy.UPDATE_ASSERTION in suggested_strategies:
            return True

        # Check error message for assertion patterns
        error_msg = analysis.failure.error_message.lower()
        if any(
            keyword in error_msg
            for keyword in [
                "assertion",
                "assert",
                "expected",
                "actual",
                "assertEqual",
                "toBe",
                "toEqual",
            ]
        ):
            return True

        return False

    async def repair(self, test_code: str, analysis: FailureAnalysis) -> Optional[str]:
        """Apply assertion repair to test code.

        Args:
            test_code: Original test code
            analysis: Failure analysis with expected/actual values

        Returns:
            Repaired test code if successful, None otherwise
        """
        try:
            root_cause = analysis.root_cause
            error_msg = analysis.failure.error_message

            # Extract expected and actual values
            expected, actual = self._extract_values(root_cause, error_msg)

            if not expected or not actual:
                logger.warning("Could not extract expected/actual values")
                return self._add_comment(test_code, root_cause)

            # Attempt to repair assertions
            repaired = self._update_assertions(test_code, expected, actual)

            # Validate syntax
            if not self._validate_syntax(repaired):
                logger.warning("Repaired code has syntax errors")
                return None

            return repaired

        except Exception as e:
            logger.error(f"Assertion repair failed: {e}")
            return None

    def _extract_values(
        self, root_cause: str, error_msg: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract expected and actual values from error message.

        Args:
            root_cause: Root cause description
            error_msg: Error message

        Returns:
            Tuple of (expected, actual) values
        """
        # Try multiple patterns to extract values
        patterns = [
            # "expected X, got Y"
            r"expected\s+(.+?),\s+(?:but\s+)?got\s+(.+?)(?:\s|$|\.)",
            # "AssertionError: X != Y"
            r"AssertionError:\s*(.+?)\s*!=\s*(.+?)(?:\s|$|\.)",
            # "assert X == Y" where X is actual, Y is expected
            r"assert\s+(.+?)\s*==\s*(.+?)(?:\s|$|\.)",
            # pytest format: "X == Y" where left is actual
            r"^\s*(.+?)\s*==\s*(.+?)$",
        ]

        text = root_cause + " " + error_msg

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                expected = match.group(1).strip()
                actual = match.group(2).strip()
                return expected, actual

        return None, None

    def _update_assertions(self, code: str, old_value: str, new_value: str) -> str:
        """Update assertions with new expected value.

        Args:
            code: Original code
            old_value: Old expected value
            new_value: New expected value

        Returns:
            Repaired code
        """
        lines = code.split("\n")
        repaired_lines = []

        for line in lines:
            original_line = line
            modified = False

            # Skip if line is already a comment
            if line.strip().startswith("#"):
                repaired_lines.append(line)
                continue

            # Pattern 1: pytest style - assert x == value
            if "assert" in line and "==" in line:
                # Try to replace the expected value
                pattern = rf"(assert\s+.+?\s*==\s*){re.escape(old_value)}"
                if re.search(pattern, line):
                    line = re.sub(pattern, rf"\g<1>{new_value}", line)
                    modified = True

            # Pattern 2: unittest style - self.assertEqual(x, value)
            if "assertEqual" in line:
                pattern = rf"(assertEqual\([^,]+,\s*){re.escape(old_value)}(\s*\))"
                if re.search(pattern, line):
                    line = re.sub(pattern, rf"\g<1>{new_value}\g<2>", line)
                    modified = True

            # Pattern 3: assertIs, assertTrue with value
            if any(
                method in line for method in ["assertIs", "assertTrue", "assertFalse"]
            ):
                pattern = rf"({'|'.join(['assertIs', 'assertTrue', 'assertFalse'])})\([^,]+,\s*{re.escape(old_value)}"
                if re.search(pattern, line):
                    line = re.sub(
                        rf"({'assertIs|assertTrue|assertFalse'})\(([^,]+),\s*{re.escape(old_value)}",
                        rf"\g<1>(\g<2>, {new_value}",
                        line,
                    )
                    modified = True

            # Pattern 4: Jest/JavaScript style - expect(x).toBe(value)
            if "toBe" in line or "toEqual" in line:
                pattern = rf"(toBe|toEqual)\({re.escape(old_value)}\)"
                if re.search(pattern, line):
                    line = re.sub(pattern, rf"\g<1>({new_value})", line)
                    modified = True

            # Add comment if modification was made
            if modified:
                indent = len(original_line) - len(original_line.lstrip())
                indent_str = " " * indent
                repaired_lines.append(
                    f"{indent_str}# AUTO-REPAIRED: Updated assertion from {old_value} to {new_value}"
                )

            repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _add_comment(self, code: str, root_cause: str) -> str:
        """Add comment when automatic repair not possible.

        Args:
            code: Original code
            root_cause: Root cause description

        Returns:
            Code with comment added
        """
        lines = code.split("\n")
        repaired_lines = []

        # Find assertion lines and add comments
        for line in lines:
            if "assert" in line.lower() or any(
                method in line
                for method in [
                    "assertEqual",
                    "assertNotEqual",
                    "assertTrue",
                    "assertFalse",
                    "assertIs",
                    "assertIsNot",
                    "toBe",
                    "toEqual",
                    "toMatch",
                ]
            ):
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent
                repaired_lines.append(
                    f"{indent_str}# FIXME: Assertion may need updating - {root_cause}"
                )

            repaired_lines.append(line)

        return "\n".join(repaired_lines)
