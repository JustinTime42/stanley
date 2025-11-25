"""Add async/await when needed.

This module handles test failures caused by missing async/await keywords,
coroutine not awaited errors, and async context issues.
"""

import re
import logging
from typing import Optional, List

from ..base import BaseRepairStrategy
from ....models.healing_models import (
    FailureAnalysis,
    RepairStrategy,
    FailureType,
)

logger = logging.getLogger(__name__)


class AsyncRepair(BaseRepairStrategy):
    """Add async/await to tests when needed.

    PATTERN: Detect coroutine errors and add await
    CRITICAL: Must handle async context properly
    GOTCHA: Need to convert test function to async if not already
    """

    async def can_repair(self, analysis: FailureAnalysis) -> bool:
        """Check if this strategy can repair the failure.

        Args:
            analysis: Failure analysis

        Returns:
            True if this strategy can handle async-related failures
        """
        # Check if failure is related to async/await
        suggested_strategies = analysis.suggested_strategies

        # Explicitly suggested
        if RepairStrategy.ADD_ASYNC in suggested_strategies:
            return True

        # Check error message and root cause for async patterns
        text = (analysis.failure.error_message + " " + analysis.root_cause).lower()

        async_keywords = [
            "coroutine",
            "was never awaited",
            "object cannot be used in 'await'",
            "object is not awaitable",
            "asyncio",
            "async",
            "await",
            "event loop",
            "runtime warning: coroutine",
        ]

        if any(keyword in text for keyword in async_keywords):
            return True

        # Check for RuntimeWarning about coroutines
        if analysis.failure.failure_type == FailureType.RUNTIME_ERROR:
            if "coroutine" in text:
                return True

        return False

    async def repair(self, test_code: str, analysis: FailureAnalysis) -> Optional[str]:
        """Apply async/await repair to test code.

        Args:
            test_code: Original test code
            analysis: Failure analysis with async details

        Returns:
            Repaired test code if successful, None otherwise
        """
        try:
            root_cause = analysis.root_cause
            error_msg = analysis.failure.error_message

            # Determine what needs to be fixed
            needs_async_def = self._needs_async_function(test_code, error_msg)
            missing_awaits = self._find_missing_awaits(test_code, error_msg)

            # Apply repairs
            repaired = test_code

            if needs_async_def:
                repaired = self._add_async_to_function(repaired)

            if missing_awaits:
                repaired = self._add_awaits(repaired, missing_awaits)

            # If no specific repairs identified, add general comments
            if not needs_async_def and not missing_awaits:
                repaired = self._add_async_comments(repaired, root_cause)

            # Validate syntax
            if not self._validate_syntax(repaired):
                logger.warning("Repaired code has syntax errors")
                return None

            return repaired

        except Exception as e:
            logger.error(f"Async repair failed: {e}")
            return None

    def _needs_async_function(self, code: str, error_msg: str) -> bool:
        """Check if test function needs to be made async.

        Args:
            code: Test code
            error_msg: Error message

        Returns:
            True if function needs async keyword
        """
        # Check if error is about coroutine not being awaited
        if "was never awaited" in error_msg.lower():
            return True

        # Check if code has awaits but function isn't async
        has_await = "await " in code
        has_async_def = re.search(r"async\s+def\s+test_", code) is not None

        if has_await and not has_async_def:
            return True

        return False

    def _find_missing_awaits(self, code: str, error_msg: str) -> List[str]:
        """Find function calls that need await.

        Args:
            code: Test code
            error_msg: Error message

        Returns:
            List of patterns that need await
        """
        missing_awaits = []

        # Parse error message for specific coroutine reference
        coroutine_match = re.search(
            r"coroutine ['\"]?(\w+)['\"]? was never awaited", error_msg
        )

        if coroutine_match:
            missing_awaits.append(coroutine_match.group(1))

        # Look for common async patterns in code that might need await
        lines = code.split("\n")
        for line in lines:
            # Skip if already awaited
            if "await" in line:
                continue

            # Common async patterns
            # 1. Variable assignment from async-looking functions
            if "= " in line and "(" in line:
                # Look for patterns like: result = async_function()
                match = re.search(r"=\s*(\w+)\(", line)
                if match:
                    func_name = match.group(1)
                    # Heuristic: functions with these prefixes are often async
                    if any(
                        prefix in func_name.lower()
                        for prefix in [
                            "async",
                            "get_",
                            "fetch_",
                            "load_",
                            "create_",
                            "update_",
                            "delete_",
                            "execute_",
                            "run_",
                        ]
                    ):
                        if func_name not in missing_awaits:
                            missing_awaits.append(func_name)

        return missing_awaits

    def _add_async_to_function(self, code: str) -> str:
        """Add async keyword to test function definition.

        Args:
            code: Original code

        Returns:
            Code with async def
        """
        lines = code.split("\n")
        repaired_lines = []

        for line in lines:
            # Find test function definition without async
            if re.match(r"\s*def\s+test_\w+", line):
                # Add async keyword
                line = re.sub(r"(\s*)def(\s+test_\w+)", r"\1async def\2", line)

                # Add comment
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent
                repaired_lines.append(
                    f"{indent_str}# AUTO-REPAIRED: Made function async"
                )

            repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _add_awaits(self, code: str, patterns: List[str]) -> str:
        """Add await keywords to specified patterns.

        Args:
            code: Original code
            patterns: Function names or patterns that need await

        Returns:
            Code with await added
        """
        lines = code.split("\n")
        repaired_lines = []

        for line in lines:
            original_line = line
            modified = False

            # Skip if already has await
            if "await" in line:
                repaired_lines.append(line)
                continue

            # Check if line contains any of the patterns
            for pattern in patterns:
                if pattern in line and "(" in line:
                    # Try to add await before the function call
                    # Pattern: result = function() -> result = await function()
                    if "= " in line:
                        line = re.sub(
                            rf"(=\s*)({re.escape(pattern)}\()", r"\1await \2", line
                        )
                        modified = True
                    # Pattern: function() standalone -> await function()
                    elif line.strip().startswith(pattern):
                        indent = len(line) - len(line.lstrip())
                        stripped = line.strip()
                        line = " " * indent + f"await {stripped}"
                        modified = True

            if modified:
                indent = len(original_line) - len(original_line.lstrip())
                indent_str = " " * indent
                repaired_lines.append(f"{indent_str}# AUTO-REPAIRED: Added await")

            repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _add_async_comments(self, code: str, root_cause: str) -> str:
        """Add comments about async issues when specific repair unclear.

        Args:
            code: Original code
            root_cause: Root cause description

        Returns:
            Code with comments
        """
        lines = code.split("\n")
        repaired_lines = []

        # Find test function and add comment
        for line in lines:
            if re.match(r"\s*(?:async\s+)?def\s+test_\w+", line):
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent

                repaired_lines.append(
                    f"{indent_str}# FIXME: Async/await issue detected"
                )
                repaired_lines.append(f"{indent_str}# {root_cause}")
                repaired_lines.append(
                    f"{indent_str}# Check if function should be 'async def' and calls need 'await'"
                )

            repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _is_async_function(self, code: str) -> bool:
        """Check if test function is already async.

        Args:
            code: Test code

        Returns:
            True if function uses async def
        """
        return bool(re.search(r"async\s+def\s+test_", code))

    def _extract_function_calls(self, line: str) -> List[str]:
        """Extract function calls from a line of code.

        Args:
            line: Line of code

        Returns:
            List of function names called on this line
        """
        # Find patterns like: function_name(...)
        pattern = r"(\w+)\s*\("
        matches = re.findall(pattern, line)

        # Filter out common non-function keywords
        keywords = {
            "if",
            "elif",
            "while",
            "for",
            "with",
            "def",
            "class",
            "return",
            "assert",
            "raise",
            "import",
            "from",
            "print",
        }

        return [match for match in matches if match not in keywords]
