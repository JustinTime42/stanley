"""Repair import errors.

This module handles test failures caused by import path changes,
module renames, moved classes, and missing dependencies.
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


class ImportRepair(BaseRepairStrategy):
    """Fix import path changes.

    PATTERN: Update import statements when modules move
    CRITICAL: Validate new import path exists
    GOTCHA: Handle both 'import x' and 'from x import y' formats
    """

    async def can_repair(self, analysis: FailureAnalysis) -> bool:
        """Check if this strategy can repair the failure.

        Args:
            analysis: Failure analysis

        Returns:
            True if this strategy can handle import errors
        """
        # Check if failure is related to imports
        failure_type = analysis.failure.failure_type
        suggested_strategies = analysis.suggested_strategies

        # Can repair if:
        # 1. Import error
        # 2. AttributeError with module context
        # 3. ModuleNotFoundError
        # 4. Explicitly suggested as UPDATE_IMPORT strategy
        if failure_type == FailureType.IMPORT_ERROR:
            return True

        if RepairStrategy.UPDATE_IMPORT in suggested_strategies:
            return True

        # Check error message for import-related patterns
        error_msg = analysis.failure.error_message.lower()
        if any(
            keyword in error_msg
            for keyword in [
                "importerror",
                "modulenotfounderror",
                "no module named",
                "cannot import",
                "import error",
            ]
        ):
            return True

        return False

    async def repair(self, test_code: str, analysis: FailureAnalysis) -> Optional[str]:
        """Apply import repair to test code.

        Args:
            test_code: Original test code
            analysis: Failure analysis with import details

        Returns:
            Repaired test code if successful, None otherwise
        """
        try:
            root_cause = analysis.root_cause
            error_msg = analysis.failure.error_message

            # Extract module information
            module_info = self._extract_module_info(root_cause, error_msg)

            if not module_info:
                logger.warning("Could not extract module information")
                return self._add_comment(test_code, root_cause)

            # Attempt repair based on error type
            if module_info["type"] == "missing_module":
                repaired = self._repair_missing_module(test_code, module_info["module"])
            elif module_info["type"] == "missing_attribute":
                repaired = self._repair_missing_attribute(
                    test_code, module_info["module"], module_info.get("attribute")
                )
            else:
                repaired = self._add_comment(test_code, root_cause)

            # Validate syntax
            if not self._validate_syntax(repaired):
                logger.warning("Repaired code has syntax errors")
                return None

            return repaired

        except Exception as e:
            logger.error(f"Import repair failed: {e}")
            return None

    def _extract_module_info(self, root_cause: str, error_msg: str) -> Optional[dict]:
        """Extract module and import information from error.

        Args:
            root_cause: Root cause description
            error_msg: Error message

        Returns:
            Dictionary with module information
        """
        text = root_cause + " " + error_msg

        # Pattern 1: No module named 'X'
        match = re.search(r"[Nn]o module named ['\"](.+?)['\"]", text)
        if match:
            return {"type": "missing_module", "module": match.group(1)}

        # Pattern 2: cannot import name 'X' from 'Y'
        match = re.search(
            r"cannot import name ['\"](.+?)['\"] from ['\"](.+?)['\"]", text
        )
        if match:
            return {
                "type": "missing_attribute",
                "attribute": match.group(1),
                "module": match.group(2),
            }

        # Pattern 3: module 'X' has no attribute 'Y'
        match = re.search(
            r"module ['\"](.+?)['\"] has no attribute ['\"](.+?)['\"]", text
        )
        if match:
            return {
                "type": "missing_attribute",
                "module": match.group(1),
                "attribute": match.group(2),
            }

        # Pattern 4: ModuleNotFoundError: X
        match = re.search(r"ModuleNotFoundError:?\s*(.+?)(?:\s|$)", text)
        if match:
            return {"type": "missing_module", "module": match.group(1).strip()}

        return None

    def _repair_missing_module(self, code: str, module_name: str) -> str:
        """Repair code when a module is not found.

        Args:
            code: Original code
            module_name: Name of missing module

        Returns:
            Repaired code
        """
        lines = code.split("\n")
        repaired_lines = []

        for line in lines:
            # Check if this line imports the missing module
            if self._line_imports_module(line, module_name):
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                # Add comment before the problematic import
                repaired_lines.append(
                    f"{indent_str}# FIXME: Module '{module_name}' not found"
                )
                repaired_lines.append(f"{indent_str}# Original import: {line.strip()}")
                repaired_lines.append(
                    f"{indent_str}# Check if module was renamed or moved"
                )

                # Comment out the original import
                repaired_lines.append(f"{indent_str}# {line.strip()}")
            else:
                repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _repair_missing_attribute(
        self, code: str, module_name: str, attribute_name: Optional[str]
    ) -> str:
        """Repair code when an attribute cannot be imported.

        Args:
            code: Original code
            module_name: Module name
            attribute_name: Attribute that's missing

        Returns:
            Repaired code
        """
        lines = code.split("\n")
        repaired_lines = []

        for line in lines:
            # Check if this line imports the attribute
            is_problematic = (
                module_name in line
                and (attribute_name is None or attribute_name in line)
                and ("import" in line or "from" in line)
            )

            if is_problematic:
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                # Add comment
                if attribute_name:
                    repaired_lines.append(
                        f"{indent_str}# FIXME: '{attribute_name}' not found in module '{module_name}'"
                    )
                    repaired_lines.append(
                        f"{indent_str}# The attribute may have been renamed, moved, or removed"
                    )
                else:
                    repaired_lines.append(
                        f"{indent_str}# FIXME: Import from '{module_name}' failed"
                    )

                repaired_lines.append(f"{indent_str}# Original: {line.strip()}")

                # Comment out the import
                repaired_lines.append(f"{indent_str}# {line.strip()}")
            else:
                repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _line_imports_module(self, line: str, module_name: str) -> bool:
        """Check if a line imports a specific module.

        Args:
            line: Line of code
            module_name: Module name to check

        Returns:
            True if line imports the module
        """
        stripped = line.strip()

        # Check various import patterns
        patterns = [
            rf"^import\s+{re.escape(module_name)}(?:\s|$|,)",
            rf"^from\s+{re.escape(module_name)}\s+import",
            rf"^import\s+\w+\.{re.escape(module_name)}(?:\s|$|,)",
        ]

        for pattern in patterns:
            if re.search(pattern, stripped):
                return True

        return False

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

        # Find first import and add comment there
        first_import_found = False

        for line in lines:
            if not first_import_found and ("import" in line or "from" in line):
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent
                repaired_lines.append(
                    f"{indent_str}# FIXME: Import error detected - {root_cause}"
                )
                first_import_found = True

            repaired_lines.append(line)

        return "\n".join(repaired_lines)
