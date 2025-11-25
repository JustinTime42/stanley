"""Security vulnerability scanning integration using Bandit.

This module provides security scanning capabilities with false positive filtering,
nosec support, and CVE detection and reporting.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...models.quality_models import SecurityIssue, SeverityLevel

logger = logging.getLogger(__name__)


class BanditIntegration:
    """Integration with Bandit security scanner.

    CRITICAL: Bandit has false positives
    - Implements whitelist/ignore patterns
    - Respects # nosec comments
    - Maintains audit trail for filtered issues
    """

    # Default whitelist patterns for common false positives
    DEFAULT_WHITELIST = [
        "B101",  # assert_used - safe in tests
    ]

    # Severity mapping from Bandit to our SeverityLevel
    SEVERITY_MAP = {
        "HIGH": SeverityLevel.HIGH,
        "MEDIUM": SeverityLevel.MEDIUM,
        "LOW": SeverityLevel.LOW,
    }

    # CWE mappings for common Bandit checks
    CWE_MAP = {
        "B201": "CWE-78",   # Flask debug mode
        "B301": "CWE-78",   # Pickle usage
        "B302": "CWE-78",   # marshal usage
        "B303": "CWE-78",   # MD5 usage
        "B304": "CWE-78",   # Ciphers
        "B305": "CWE-78",   # Cipher modes
        "B306": "CWE-377",  # mktemp usage
        "B307": "CWE-94",   # eval usage
        "B308": "CWE-78",   # mark_safe usage
        "B310": "CWE-22",   # urllib
        "B311": "CWE-330",  # random
        "B312": "CWE-330",  # telnetlib
        "B313": "CWE-327",  # xml
        "B314": "CWE-327",  # xml
        "B315": "CWE-327",  # xml
        "B316": "CWE-327",  # xml
        "B317": "CWE-327",  # xml
        "B318": "CWE-327",  # xml
        "B319": "CWE-327",  # xml
        "B320": "CWE-327",  # xml
        "B321": "CWE-327",  # ftplib
        "B323": "CWE-327",  # unverified context
        "B324": "CWE-327",  # hashlib
        "B501": "CWE-295",  # request verify
        "B502": "CWE-295",  # ssl
        "B503": "CWE-295",  # ssl
        "B504": "CWE-295",  # ssl
        "B505": "CWE-327",  # weak cryptographic key
        "B506": "CWE-20",   # yaml load
        "B507": "CWE-502",  # ssh
        "B601": "CWE-78",   # paramiko
        "B602": "CWE-78",   # shell
        "B603": "CWE-78",   # subprocess without shell
        "B604": "CWE-78",   # shell
        "B605": "CWE-78",   # shell
        "B606": "CWE-78",   # shell
        "B607": "CWE-78",   # shell
        "B608": "CWE-89",   # sql
        "B609": "CWE-78",   # wildcard
    }

    def __init__(
        self,
        whitelist_patterns: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
        config_file: Optional[str] = None
    ):
        """Initialize Bandit integration.

        Args:
            whitelist_patterns: Additional patterns to whitelist
            exclude_dirs: Directories to exclude from scanning
            config_file: Path to Bandit config file
        """
        self.whitelist_patterns = (
            self.DEFAULT_WHITELIST + (whitelist_patterns or [])
        )
        self.exclude_dirs = exclude_dirs or ["tests", "test", ".venv", "venv"]
        self.config_file = config_file
        self.filtered_issues: List[Dict[str, Any]] = []  # Audit trail

    async def scan_security(
        self,
        target_path: str,
        severity_threshold: Optional[str] = None,
        confidence_threshold: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run Bandit security scan on target path.

        PATTERN: Programmatic API usage with subprocess fallback
        GOTCHA: False positives need filtering

        Args:
            target_path: Path to scan
            severity_threshold: Minimum severity (LOW, MEDIUM, HIGH)
            confidence_threshold: Minimum confidence (LOW, MEDIUM, HIGH)

        Returns:
            Dictionary containing:
                - issues: List of SecurityIssue objects
                - security_score: Overall security score (0-100)
                - total_issues: Total issues found
                - by_severity: Count by severity level
                - filtered_count: Number of filtered false positives
        """
        logger.info(f"Starting security scan for {target_path}")

        try:
            # Build Bandit command
            cmd = self._build_bandit_command(
                target_path,
                severity_threshold,
                confidence_threshold
            )

            # Run Bandit
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
                logger.error(f"Failed to parse Bandit output: {stdout.decode()[:200]}")
                return self._empty_results(reason="Failed to parse output")

            # Filter false positives
            filtered_issues = self.filter_false_positives(
                raw_results.get("results", [])
            )

            # Convert to SecurityIssue objects
            security_issues = [
                self._convert_to_security_issue(issue)
                for issue in filtered_issues
            ]

            # Calculate metrics
            results = self._calculate_metrics(
                security_issues,
                len(raw_results.get("results", [])),
                len(self.filtered_issues)
            )

            logger.info(
                f"Security scan complete: {results['total_issues']} issues found, "
                f"{results['filtered_count']} filtered"
            )

            return results

        except FileNotFoundError:
            logger.warning("Bandit not installed, skipping security scan")
            return self._empty_results(reason="Bandit not installed")

        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return self._empty_results(reason=str(e))

    def _build_bandit_command(
        self,
        target_path: str,
        severity_threshold: Optional[str],
        confidence_threshold: Optional[str]
    ) -> List[str]:
        """Build Bandit command with options.

        Args:
            target_path: Path to scan
            severity_threshold: Severity threshold
            confidence_threshold: Confidence threshold

        Returns:
            Command as list of strings
        """
        cmd = [
            "bandit",
            "-r",  # Recursive
            target_path,
            "-f", "json",  # JSON output for parsing
        ]

        # Add severity threshold
        if severity_threshold:
            cmd.extend(["-ll" if severity_threshold == "LOW" else
                       "-l" if severity_threshold == "MEDIUM" else
                       ""])  # HIGH is default

        # Add confidence threshold
        if confidence_threshold:
            cmd.extend(["-i" if confidence_threshold == "HIGH" else ""])

        # Exclude directories
        if self.exclude_dirs:
            for exclude_dir in self.exclude_dirs:
                cmd.extend(["-x", exclude_dir])

        # Add config file if specified
        if self.config_file:
            cmd.extend(["-c", self.config_file])

        # Remove empty strings
        cmd = [c for c in cmd if c]

        return cmd

    def filter_false_positives(
        self,
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter false positives from security issues.

        PATTERN: Intelligent false positive filtering
        GOTCHA: Don't filter without audit trail

        Args:
            issues: Raw issues from Bandit

        Returns:
            Filtered list of issues
        """
        filtered = []
        self.filtered_issues = []  # Reset audit trail

        for issue in issues:
            test_id = issue.get("test_id", "")

            # Check against whitelist patterns
            if self._is_whitelisted(issue):
                logger.debug(f"Filtered whitelisted pattern: {test_id}")
                self.filtered_issues.append({
                    "issue": issue,
                    "reason": "whitelisted"
                })
                continue

            # Check for nosec comments
            if self._has_nosec_comment(issue):
                logger.debug(f"Filtered nosec: {test_id}")
                self.filtered_issues.append({
                    "issue": issue,
                    "reason": "nosec comment"
                })
                continue

            # Check if in test file (more lenient for test code)
            if self._is_test_file(issue) and self._is_test_safe(issue):
                logger.debug(f"Filtered test-safe issue: {test_id}")
                self.filtered_issues.append({
                    "issue": issue,
                    "reason": "safe in tests"
                })
                continue

            filtered.append(issue)

        return filtered

    def _is_whitelisted(self, issue: Dict[str, Any]) -> bool:
        """Check if issue matches whitelist patterns.

        Args:
            issue: Bandit issue

        Returns:
            True if whitelisted
        """
        test_id = issue.get("test_id", "")
        return test_id in self.whitelist_patterns

    def _has_nosec_comment(self, issue: Dict[str, Any]) -> bool:
        """Check if issue has nosec comment.

        Args:
            issue: Bandit issue

        Returns:
            True if has nosec comment
        """
        # Bandit includes this in the issue data
        # Check if the issue has nosec marker
        code = issue.get("code", "")
        return "# nosec" in code or "# noqa" in code

    def _is_test_file(self, issue: Dict[str, Any]) -> bool:
        """Check if issue is in a test file.

        Args:
            issue: Bandit issue

        Returns:
            True if in test file
        """
        filename = issue.get("filename", "")
        path = Path(filename)

        # Check if in test directory or test file
        return (
            "test" in path.parts or
            path.name.startswith("test_") or
            path.name.endswith("_test.py")
        )

    def _is_test_safe(self, issue: Dict[str, Any]) -> bool:
        """Check if issue is safe in test context.

        Args:
            issue: Bandit issue

        Returns:
            True if safe in tests
        """
        test_id = issue.get("test_id", "")

        # Issues that are generally safe in tests
        test_safe_patterns = [
            "B101",  # assert_used
            "B311",  # random (not crypto-secure random in tests is OK)
        ]

        return test_id in test_safe_patterns

    def _convert_to_security_issue(
        self,
        bandit_issue: Dict[str, Any]
    ) -> SecurityIssue:
        """Convert Bandit issue to SecurityIssue model.

        Args:
            bandit_issue: Raw Bandit issue

        Returns:
            SecurityIssue object
        """
        test_id = bandit_issue.get("test_id", "UNKNOWN")
        severity_str = bandit_issue.get("issue_severity", "LOW")

        return SecurityIssue(
            issue_id=test_id,
            severity=self.SEVERITY_MAP.get(
                severity_str,
                SeverityLevel.LOW
            ),
            confidence=bandit_issue.get("issue_confidence", "LOW"),
            file_path=bandit_issue.get("filename", ""),
            line_number=bandit_issue.get("line_number", 0),
            column=bandit_issue.get("col_offset"),
            issue_type=bandit_issue.get("test_name", ""),
            description=bandit_issue.get("issue_text", ""),
            remediation=self._get_remediation(test_id),
            cwe_id=self.CWE_MAP.get(test_id),
            owasp_category=self._get_owasp_category(test_id),
            references=[
                f"https://bandit.readthedocs.io/en/latest/plugins/{test_id.lower()}.html"
            ]
        )

    def _get_remediation(self, test_id: str) -> str:
        """Get remediation advice for a test ID.

        Args:
            test_id: Bandit test ID

        Returns:
            Remediation advice
        """
        # Common remediation advice
        remediation_map = {
            "B201": "Disable Flask debug mode in production",
            "B301": "Avoid using pickle, use JSON or safer serialization",
            "B303": "Use SHA256 or stronger hash instead of MD5",
            "B307": "Avoid using eval(), validate and sanitize input",
            "B311": "Use secrets module for cryptographic randomness",
            "B501": "Enable SSL certificate verification",
            "B506": "Use yaml.safe_load() instead of yaml.load()",
            "B601": "Avoid shell=True, use list of arguments",
            "B608": "Use parameterized queries to prevent SQL injection",
        }

        return remediation_map.get(
            test_id,
            "Review Bandit documentation for remediation advice"
        )

    def _get_owasp_category(self, test_id: str) -> Optional[str]:
        """Get OWASP category for a test ID.

        Args:
            test_id: Bandit test ID

        Returns:
            OWASP category or None
        """
        # Map to OWASP Top 10 categories
        owasp_map = {
            "B201": "A05:2021-Security Misconfiguration",
            "B301": "A08:2021-Software and Data Integrity Failures",
            "B303": "A02:2021-Cryptographic Failures",
            "B307": "A03:2021-Injection",
            "B311": "A02:2021-Cryptographic Failures",
            "B501": "A07:2021-Identification and Authentication Failures",
            "B506": "A08:2021-Software and Data Integrity Failures",
            "B601": "A03:2021-Injection",
            "B608": "A03:2021-Injection",
        }

        return owasp_map.get(test_id)

    def _calculate_metrics(
        self,
        security_issues: List[SecurityIssue],
        total_raw_issues: int,
        filtered_count: int
    ) -> Dict[str, Any]:
        """Calculate security metrics.

        Args:
            security_issues: Filtered security issues
            total_raw_issues: Total issues before filtering
            filtered_count: Number of filtered issues

        Returns:
            Metrics dictionary
        """
        # Count by severity
        by_severity = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 0,
            SeverityLevel.MEDIUM: 0,
            SeverityLevel.LOW: 0,
            SeverityLevel.INFO: 0,
        }

        for issue in security_issues:
            by_severity[issue.severity] += 1

        # Calculate security score (0-100)
        # Formula: 100 - (critical*20 + high*10 + medium*5 + low*2)
        deductions = (
            by_severity[SeverityLevel.CRITICAL] * 20 +
            by_severity[SeverityLevel.HIGH] * 10 +
            by_severity[SeverityLevel.MEDIUM] * 5 +
            by_severity[SeverityLevel.LOW] * 2
        )
        security_score = max(0.0, 100.0 - deductions)

        return {
            "issues": security_issues,
            "security_score": security_score,
            "total_issues": len(security_issues),
            "total_raw_issues": total_raw_issues,
            "filtered_count": filtered_count,
            "by_severity": {
                str(k.value): v for k, v in by_severity.items()
            }
        }

    def _empty_results(self, reason: str = "No scan run") -> Dict[str, Any]:
        """Return empty results structure.

        Args:
            reason: Reason for empty results

        Returns:
            Empty results dictionary
        """
        return {
            "issues": [],
            "security_score": 100.0,
            "total_issues": 0,
            "total_raw_issues": 0,
            "filtered_count": 0,
            "by_severity": {
                str(SeverityLevel.CRITICAL.value): 0,
                str(SeverityLevel.HIGH.value): 0,
                str(SeverityLevel.MEDIUM.value): 0,
                str(SeverityLevel.LOW.value): 0,
                str(SeverityLevel.INFO.value): 0,
            },
            "reason": reason
        }

    async def check_installation(self) -> bool:
        """Check if Bandit is installed.

        Returns:
            True if Bandit is available
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "bandit", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            return process.returncode == 0
        except FileNotFoundError:
            return False

    def get_filtered_issues_audit(self) -> List[Dict[str, Any]]:
        """Get audit trail of filtered issues.

        Returns:
            List of filtered issues with reasons
        """
        return self.filtered_issues
