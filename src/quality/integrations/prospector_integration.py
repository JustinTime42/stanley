"""Prospector meta-linter integration.

This module provides integration with Prospector, which aggregates multiple
Python linting tools into a single comprehensive analysis.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...models.quality_models import SeverityLevel

logger = logging.getLogger(__name__)


class ProspectorIntegration:
    """Integration with Prospector meta-linter.

    Prospector aggregates multiple linters:
    - pylint: General purpose linter
    - pyflakes: Logical errors
    - pep8/pycodestyle: Style checking
    - mccabe: Complexity analysis
    - dodgy: Secret detection
    - pydocstyle: Docstring checking
    - pyroma: Package quality
    """

    # Severity mapping from Prospector
    SEVERITY_MAP = {
        "high": SeverityLevel.HIGH,
        "medium": SeverityLevel.MEDIUM,
        "low": SeverityLevel.LOW,
    }

    # Tool weights for aggregation
    TOOL_WEIGHTS = {
        "pylint": 1.0,
        "pyflakes": 0.8,
        "pep8": 0.5,
        "pycodestyle": 0.5,
        "mccabe": 0.7,
        "dodgy": 0.9,
        "pydocstyle": 0.4,
        "pyroma": 0.6,
    }

    def __init__(
        self,
        strictness: str = "medium",
        with_tools: Optional[List[str]] = None,
        without_tools: Optional[List[str]] = None,
        max_line_length: int = 100
    ):
        """Initialize Prospector integration.

        Args:
            strictness: Analysis strictness (verylow, low, medium, high, veryhigh)
            with_tools: Specific tools to enable
            without_tools: Tools to disable
            max_line_length: Maximum line length
        """
        self.strictness = strictness
        self.with_tools = with_tools or []
        self.without_tools = without_tools or []
        self.max_line_length = max_line_length

    async def analyze_code_quality(
        self,
        target_path: str,
        include_external: bool = False
    ) -> Dict[str, Any]:
        """Run Prospector analysis.

        PATTERN: Meta-linter aggregation
        Combines multiple linters for comprehensive analysis

        Args:
            target_path: Path to analyze
            include_external: Include external dependencies

        Returns:
            Dictionary containing:
                - issues: List of all issues found
                - by_tool: Issues grouped by tool
                - by_severity: Issues grouped by severity
                - quality_score: Overall quality score (0-100)
                - summary: Summary statistics
                - tool_results: Individual tool results
        """
        logger.info(f"Starting Prospector analysis for {target_path}")

        try:
            # Build Prospector command
            cmd = self._build_prospector_command(target_path, include_external)

            # Run Prospector
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Parse JSON output
            try:
                raw_results = json.loads(stdout.decode())
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Prospector output")
                return self._empty_results(reason="Failed to parse output")

            # Process results
            results = self._process_results(raw_results)

            logger.info(
                f"Prospector analysis complete: "
                f"{results['summary']['total_issues']} issues, "
                f"quality score {results['quality_score']:.1f}"
            )

            return results

        except FileNotFoundError:
            logger.warning("Prospector not installed, skipping analysis")
            return self._empty_results(reason="Prospector not installed")

        except Exception as e:
            logger.error(f"Prospector analysis failed: {e}")
            return self._empty_results(reason=str(e))

    def _build_prospector_command(
        self,
        target_path: str,
        include_external: bool
    ) -> List[str]:
        """Build Prospector command.

        Args:
            target_path: Path to analyze
            include_external: Include external dependencies

        Returns:
            Command as list of strings
        """
        cmd = [
            "prospector",
            target_path,
            "--output-format", "json",
            "--strictness", self.strictness,
            "--max-line-length", str(self.max_line_length),
        ]

        # Add specific tools
        if self.with_tools:
            for tool in self.with_tools:
                cmd.extend(["--with-tool", tool])

        # Disable tools
        if self.without_tools:
            for tool in self.without_tools:
                cmd.extend(["--without-tool", tool])

        # External dependencies
        if not include_external:
            cmd.append("--no-external-config")

        # Additional options for better results
        cmd.extend([
            "--die-on-tool-error",  # Fail if a tool errors
            "--messages-only",  # Only show messages
        ])

        return cmd

    def _process_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process Prospector results.

        Args:
            raw_results: Raw results from Prospector

        Returns:
            Processed results
        """
        messages = raw_results.get("messages", [])

        # Group by tool
        by_tool: Dict[str, List[Dict[str, Any]]] = {}
        for message in messages:
            tool = message.get("source", "unknown")
            if tool not in by_tool:
                by_tool[tool] = []
            by_tool[tool].append(message)

        # Group by severity
        by_severity: Dict[str, List[Dict[str, Any]]] = {}
        for message in messages:
            severity = message.get("severity", "low")
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(message)

        # Calculate quality score
        quality_score = self._calculate_quality_score(messages, by_tool)

        # Generate summary
        summary = self._generate_summary(messages, by_tool, by_severity)

        # Extract tool-specific results
        tool_results = self._extract_tool_results(by_tool)

        return {
            "issues": messages,
            "by_tool": {tool: len(issues) for tool, issues in by_tool.items()},
            "by_severity": {sev: len(issues) for sev, issues in by_severity.items()},
            "quality_score": quality_score,
            "summary": summary,
            "tool_results": tool_results,
            "raw_results": raw_results
        }

    def _calculate_quality_score(
        self,
        messages: List[Dict[str, Any]],
        by_tool: Dict[str, List[Dict[str, Any]]]
    ) -> float:
        """Calculate overall quality score.

        PATTERN: Weighted aggregation from multiple linters

        Args:
            messages: All messages
            by_tool: Messages grouped by tool

        Returns:
            Quality score (0-100)
        """
        if not messages:
            return 100.0

        # Calculate weighted deductions
        total_deductions = 0.0

        for tool, tool_messages in by_tool.items():
            weight = self.TOOL_WEIGHTS.get(tool, 0.5)

            for message in tool_messages:
                severity = message.get("severity", "low")

                # Deduction based on severity
                if severity == "high":
                    deduction = 5.0
                elif severity == "medium":
                    deduction = 2.0
                else:  # low
                    deduction = 0.5

                # Apply tool weight
                total_deductions += deduction * weight

        # Calculate score (max deduction capped at 100)
        score = max(0.0, 100.0 - min(100.0, total_deductions))

        return round(score, 2)

    def _generate_summary(
        self,
        messages: List[Dict[str, Any]],
        by_tool: Dict[str, List[Dict[str, Any]]],
        by_severity: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Generate analysis summary.

        Args:
            messages: All messages
            by_tool: Messages by tool
            by_severity: Messages by severity

        Returns:
            Summary statistics
        """
        # Count unique files with issues
        files_with_issues = set(
            msg.get("location", {}).get("path", "")
            for msg in messages
        )

        # Find most common issue types
        issue_codes = [msg.get("code", "") for msg in messages]
        code_counts = {}
        for code in issue_codes:
            if code:
                code_counts[code] = code_counts.get(code, 0) + 1

        top_issues = sorted(
            code_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "total_issues": len(messages),
            "files_with_issues": len(files_with_issues),
            "tools_used": len(by_tool),
            "severity_breakdown": {
                "high": len(by_severity.get("high", [])),
                "medium": len(by_severity.get("medium", [])),
                "low": len(by_severity.get("low", [])),
            },
            "top_issues": [
                {"code": code, "count": count}
                for code, count in top_issues
            ]
        }

    def _extract_tool_results(
        self,
        by_tool: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract individual tool results.

        Args:
            by_tool: Messages grouped by tool

        Returns:
            Tool-specific results
        """
        tool_results = {}

        for tool, messages in by_tool.items():
            # Calculate tool-specific metrics
            severity_counts = {
                "high": 0,
                "medium": 0,
                "low": 0,
            }

            for message in messages:
                severity = message.get("severity", "low")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Tool score
            deductions = (
                severity_counts["high"] * 5.0 +
                severity_counts["medium"] * 2.0 +
                severity_counts["low"] * 0.5
            )
            tool_score = max(0.0, 100.0 - deductions)

            tool_results[tool] = {
                "issue_count": len(messages),
                "severity_counts": severity_counts,
                "score": tool_score,
                "sample_issues": messages[:3]  # First 3 issues as examples
            }

        return tool_results

    def _empty_results(self, reason: str = "No analysis run") -> Dict[str, Any]:
        """Return empty results structure.

        Args:
            reason: Reason for empty results

        Returns:
            Empty results dictionary
        """
        return {
            "issues": [],
            "by_tool": {},
            "by_severity": {},
            "quality_score": 100.0,
            "summary": {
                "total_issues": 0,
                "files_with_issues": 0,
                "tools_used": 0,
                "severity_breakdown": {
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                },
                "top_issues": []
            },
            "tool_results": {},
            "reason": reason
        }

    async def check_installation(self) -> bool:
        """Check if Prospector is installed.

        Returns:
            True if Prospector is available
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "prospector", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            return process.returncode == 0
        except FileNotFoundError:
            return False

    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tools.

        Returns:
            List of tool names
        """
        # Default Prospector tools
        default_tools = [
            "pylint",
            "pyflakes",
            "pycodestyle",
            "mccabe",
            "dodgy",
            "pydocstyle",
        ]

        # Apply with_tools and without_tools
        enabled = set(default_tools)

        if self.with_tools:
            enabled.update(self.with_tools)

        if self.without_tools:
            enabled -= set(self.without_tools)

        return list(enabled)

    def configure_strictness(self, strictness: str) -> None:
        """Configure analysis strictness.

        Args:
            strictness: Strictness level (verylow, low, medium, high, veryhigh)
        """
        valid_levels = ["verylow", "low", "medium", "high", "veryhigh"]
        if strictness not in valid_levels:
            raise ValueError(
                f"Invalid strictness level: {strictness}. "
                f"Must be one of {valid_levels}"
            )
        self.strictness = strictness

    def enable_tool(self, tool: str) -> None:
        """Enable a specific tool.

        Args:
            tool: Tool name to enable
        """
        if tool not in self.with_tools:
            self.with_tools.append(tool)

        # Remove from disabled if present
        if tool in self.without_tools:
            self.without_tools.remove(tool)

    def disable_tool(self, tool: str) -> None:
        """Disable a specific tool.

        Args:
            tool: Tool name to disable
        """
        if tool not in self.without_tools:
            self.without_tools.append(tool)

        # Remove from enabled if present
        if tool in self.with_tools:
            self.with_tools.remove(tool)
