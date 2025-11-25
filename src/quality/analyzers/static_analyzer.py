"""Static code analysis integrating multiple linters.

PATTERN: Aggregate results from Ruff, Mypy, Prospector
CRITICAL: Tools run in subprocess, need async coordination
GOTCHA: Tool availability varies by environment
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StaticAnalyzer:
    """
    Static code analyzer integrating multiple linting tools.

    PATTERN: Aggregate results from multiple static analysis tools
    CRITICAL: Tools execute in subprocess, coordinate asynchronously
    GOTCHA: Not all tools may be installed, graceful degradation required
    """

    def __init__(
        self,
        enabled_tools: Optional[List[str]] = None,
        strict_mode: bool = False
    ):
        """
        Initialize static analyzer.

        Args:
            enabled_tools: List of tools to enable (ruff, mypy, prospector)
            strict_mode: Enable strict checking
        """
        self.logger = logger
        self.enabled_tools = enabled_tools or ["ruff", "mypy", "prospector"]
        self.strict_mode = strict_mode

    async def analyze_code_quality(
        self,
        source_path: str,
        include_external: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze code quality using multiple static analysis tools.

        PATTERN: Parallel execution of all enabled tools
        CRITICAL: Aggregate and normalize results across tools

        Args:
            source_path: Path to code to analyze
            include_external: Include external dependencies in analysis

        Returns:
            Aggregated quality analysis results
        """
        start_time = datetime.now()

        # Run all enabled tools in parallel
        tasks = []

        if "ruff" in self.enabled_tools:
            tasks.append(self._run_ruff(source_path))

        if "mypy" in self.enabled_tools:
            tasks.append(self._run_mypy(source_path))

        if "prospector" in self.enabled_tools:
            tasks.append(self._run_prospector(source_path))

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        tool_results = {}
        all_issues = []

        for i, tool_name in enumerate(self.enabled_tools):
            if i < len(results):
                result = results[i]
                if isinstance(result, Exception):
                    self.logger.error(f"{tool_name} analysis failed: {result}")
                    tool_results[tool_name] = {
                        "error": str(result),
                        "available": False,
                    }
                else:
                    tool_results[tool_name] = result
                    all_issues.extend(result.get("issues", []))

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(tool_results, all_issues)

        # Categorize issues
        issue_categories = self._categorize_issues(all_issues)

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "timestamp": datetime.now().isoformat(),
            "source_path": source_path,
            "quality_score": quality_score,
            "tool_results": tool_results,
            "total_issues": len(all_issues),
            "issues_by_severity": issue_categories["by_severity"],
            "issues_by_type": issue_categories["by_type"],
            "all_issues": all_issues,
            "execution_time_seconds": round(execution_time, 2),
        }

    async def _run_ruff(self, source_path: str) -> Dict[str, Any]:
        """
        Run Ruff linter.

        PATTERN: Execute ruff check with JSON output
        CRITICAL: Parse JSON output for structured results

        Args:
            source_path: Path to analyze

        Returns:
            Ruff analysis results
        """
        try:
            cmd = [
                "ruff",
                "check",
                source_path,
                "--output-format=json",
            ]

            if not self.strict_mode:
                cmd.append("--ignore=E501")  # Line too long

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60
            )

            # Parse JSON output
            if stdout:
                ruff_results = json.loads(stdout.decode())
            else:
                ruff_results = []

            # Convert to standard format
            issues = []
            for issue in ruff_results:
                issues.append({
                    "tool": "ruff",
                    "file": issue.get("filename", ""),
                    "line": issue.get("location", {}).get("row", 0),
                    "column": issue.get("location", {}).get("column", 0),
                    "severity": self._map_ruff_severity(issue.get("code", "")),
                    "type": issue.get("code", ""),
                    "message": issue.get("message", ""),
                    "fixable": issue.get("fix", {}).get("applicability") == "automatic",
                })

            return {
                "available": True,
                "success": process.returncode == 0,
                "issues_count": len(issues),
                "issues": issues,
                "fixable_count": sum(1 for i in issues if i.get("fixable")),
            }

        except asyncio.TimeoutError:
            self.logger.error("Ruff analysis timeout")
            raise
        except FileNotFoundError:
            self.logger.warning("Ruff not installed")
            return {
                "available": False,
                "error": "ruff not installed",
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Ruff output: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Ruff execution failed: {e}")
            raise

    async def _run_mypy(self, source_path: str) -> Dict[str, Any]:
        """
        Run Mypy type checker.

        PATTERN: Execute mypy with JSON output
        CRITICAL: Parse structured type errors

        Args:
            source_path: Path to analyze

        Returns:
            Mypy analysis results
        """
        try:
            cmd = [
                "mypy",
                source_path,
                "--show-column-numbers",
                "--show-error-codes",
                "--no-error-summary",
            ]

            if self.strict_mode:
                cmd.append("--strict")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120
            )

            output = stdout.decode()

            # Parse mypy output (format: file:line:col: error: message [code])
            issues = []
            for line in output.splitlines():
                parsed = self._parse_mypy_line(line)
                if parsed:
                    issues.append(parsed)

            # Categorize by error type
            error_types = {}
            for issue in issues:
                error_type = issue.get("type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

            return {
                "available": True,
                "success": process.returncode == 0,
                "issues_count": len(issues),
                "issues": issues,
                "error_types": error_types,
            }

        except asyncio.TimeoutError:
            self.logger.error("Mypy analysis timeout")
            raise
        except FileNotFoundError:
            self.logger.warning("Mypy not installed")
            return {
                "available": False,
                "error": "mypy not installed",
            }
        except Exception as e:
            self.logger.error(f"Mypy execution failed: {e}")
            raise

    async def _run_prospector(self, source_path: str) -> Dict[str, Any]:
        """
        Run Prospector meta-linter.

        PATTERN: Execute prospector with JSON output
        CRITICAL: Aggregates multiple tools (pylint, pep8, etc.)

        Args:
            source_path: Path to analyze

        Returns:
            Prospector analysis results
        """
        try:
            cmd = [
                "prospector",
                source_path,
                "--output-format=json",
            ]

            if self.strict_mode:
                cmd.append("--strictness=veryhigh")
            else:
                cmd.append("--strictness=medium")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=180
            )

            # Parse JSON output
            if stdout:
                prospector_results = json.loads(stdout.decode())
            else:
                prospector_results = {"messages": []}

            # Convert to standard format
            issues = []
            for msg in prospector_results.get("messages", []):
                issues.append({
                    "tool": msg.get("source", "prospector"),
                    "file": msg.get("location", {}).get("path", ""),
                    "line": msg.get("location", {}).get("line", 0),
                    "column": msg.get("location", {}).get("character", 0),
                    "severity": msg.get("severity", "medium"),
                    "type": msg.get("code", ""),
                    "message": msg.get("message", ""),
                })

            # Get summary
            summary = prospector_results.get("summary", {})

            return {
                "available": True,
                "success": True,
                "issues_count": len(issues),
                "issues": issues,
                "summary": {
                    "started": summary.get("started", 0),
                    "completed": summary.get("completed", 0),
                    "time_taken": summary.get("time_taken", 0),
                    "tools_run": summary.get("tools_run", []),
                },
            }

        except asyncio.TimeoutError:
            self.logger.error("Prospector analysis timeout")
            raise
        except FileNotFoundError:
            self.logger.warning("Prospector not installed")
            return {
                "available": False,
                "error": "prospector not installed",
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Prospector output: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Prospector execution failed: {e}")
            raise

    def _parse_mypy_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single line of mypy output.

        PATTERN: Regex parsing of mypy error format
        Format: file.py:line:col: severity: message [code]

        Args:
            line: Line from mypy output

        Returns:
            Parsed issue or None
        """
        import re

        # Pattern: file:line:col: level: message [code]
        pattern = r"^(.+?):(\d+):(\d+):\s+(\w+):\s+(.+?)(?:\s+\[(.+?)\])?$"
        match = re.match(pattern, line)

        if match:
            file_path, line_num, col_num, severity, message, code = match.groups()
            return {
                "tool": "mypy",
                "file": file_path,
                "line": int(line_num),
                "column": int(col_num),
                "severity": severity.lower(),
                "type": code or "type-error",
                "message": message,
            }

        return None

    def _map_ruff_severity(self, code: str) -> str:
        """
        Map Ruff error codes to severity levels.

        Args:
            code: Ruff error code (e.g., E501, F401)

        Returns:
            Severity level
        """
        if not code:
            return "medium"

        # Error codes by severity
        if code.startswith("E9") or code.startswith("F"):
            return "high"  # Syntax errors, undefined names
        elif code.startswith("E") or code.startswith("W"):
            return "medium"  # Style and warning
        elif code.startswith("C") or code.startswith("N"):
            return "low"  # Convention and naming

        return "medium"

    def _calculate_quality_score(
        self,
        tool_results: Dict[str, Any],
        all_issues: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall code quality score.

        PATTERN: Weighted scoring based on issue severity and density
        CRITICAL: Score from 0-100, higher is better

        Args:
            tool_results: Results from all tools
            all_issues: All detected issues

        Returns:
            Quality score (0-100)
        """
        # Start with perfect score
        score = 100.0

        # Count available tools
        available_tools = sum(
            1 for result in tool_results.values()
            if result.get("available", False)
        )

        if available_tools == 0:
            return 0.0

        # Deduct points based on issues
        severity_weights = {
            "critical": 5.0,
            "high": 3.0,
            "error": 3.0,
            "medium": 1.0,
            "warning": 1.0,
            "low": 0.5,
            "info": 0.1,
        }

        for issue in all_issues:
            severity = issue.get("severity", "medium")
            weight = severity_weights.get(severity, 1.0)
            score -= weight

        # Don't go below 0
        score = max(0.0, score)

        # Normalize to 0-100 range
        # Assume 50 issues = score of 0
        max_deduction = 50 * severity_weights["medium"]
        normalized_score = (score / 100) * 100

        return round(normalized_score, 2)

    def _categorize_issues(
        self,
        issues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Categorize issues by severity and type.

        Args:
            issues: List of all issues

        Returns:
            Categorized issue counts
        """
        by_severity = {}
        by_type = {}

        for issue in issues:
            # By severity
            severity = issue.get("severity", "unknown")
            by_severity[severity] = by_severity.get(severity, 0) + 1

            # By type
            issue_type = issue.get("type", "unknown")
            by_type[issue_type] = by_type.get(issue_type, 0) + 1

        return {
            "by_severity": by_severity,
            "by_type": by_type,
        }

    async def get_fixable_issues(
        self,
        source_path: str
    ) -> List[Dict[str, Any]]:
        """
        Get issues that can be automatically fixed.

        Args:
            source_path: Path to analyze

        Returns:
            List of fixable issues
        """
        results = await self.analyze_code_quality(source_path)

        fixable = []
        for issue in results.get("all_issues", []):
            if issue.get("fixable", False):
                fixable.append(issue)

        return fixable

    async def apply_fixes(
        self,
        source_path: str,
        issues: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Apply automatic fixes for detected issues.

        PATTERN: Run tools in fix mode
        CRITICAL: Only apply safe automated fixes

        Args:
            source_path: Path to fix
            issues: Specific issues to fix (None = all fixable)

        Returns:
            Fix results
        """
        try:
            # Run ruff with --fix
            cmd = ["ruff", "check", source_path, "--fix"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60
            )

            return {
                "success": process.returncode == 0,
                "output": stdout.decode(),
                "errors": stderr.decode(),
            }

        except Exception as e:
            self.logger.error(f"Failed to apply fixes: {e}")
            return {
                "success": False,
                "error": str(e),
            }
