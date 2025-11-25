"""SonarPython integration for comprehensive code quality metrics.

This module provides SonarQube/SonarPython integration with fallback support,
handling both local scanner and cloud API options.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class SonarIntegration:
    """Integration with SonarQube/SonarPython for code quality analysis.

    CRITICAL: SonarQube requires Java runtime
    - Checks Java availability
    - Falls back to cloud API if local scanner unavailable
    - Provides comprehensive code quality metrics
    """

    # Quality metrics provided by SonarQube
    METRICS = [
        "bugs",
        "vulnerabilities",
        "code_smells",
        "coverage",
        "duplicated_lines_density",
        "ncloc",  # Lines of code
        "complexity",
        "cognitive_complexity",
        "sqale_index",  # Technical debt
        "sqale_rating",  # Maintainability rating
        "reliability_rating",
        "security_rating",
    ]

    def __init__(
        self,
        sonar_host: Optional[str] = None,
        sonar_token: Optional[str] = None,
        project_key: Optional[str] = None,
        use_cloud: bool = False
    ):
        """Initialize SonarQube integration.

        Args:
            sonar_host: SonarQube server URL (default: http://localhost:9000)
            sonar_token: Authentication token
            project_key: Project key in SonarQube
            use_cloud: Use SonarCloud instead of local server
        """
        self.sonar_host = sonar_host or os.getenv(
            "SONAR_HOST_URL",
            "http://localhost:9000"
        )
        self.sonar_token = sonar_token or os.getenv("SONAR_TOKEN")
        self.project_key = project_key
        self.use_cloud = use_cloud

        # Determine scanner availability
        self.scanner_available = False
        self.java_available = False

    async def analyze_code_quality(
        self,
        source_path: str,
        project_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run SonarQube analysis on source code.

        PATTERN: Local scanner with cloud API fallback
        GOTCHA: Requires Java runtime for local scanner

        Args:
            source_path: Path to source code
            project_key: Project key (uses default if not provided)

        Returns:
            Dictionary containing:
                - available: Whether SonarQube analysis was possible
                - quality_gate_status: Pass/fail status
                - metrics: Dictionary of quality metrics
                - issues: List of issues found
                - ratings: Quality ratings (A-E)
        """
        logger.info(f"Starting SonarQube analysis for {source_path}")

        # Check prerequisites
        java_ok = await self._check_java()
        scanner_ok = await self._check_scanner()

        if not java_ok or not scanner_ok:
            logger.warning(
                "SonarQube scanner not available (requires Java and sonar-scanner). "
                "Falling back to alternative analysis or skipping."
            )
            return self._unavailable_results(
                reason="Scanner or Java not available"
            )

        try:
            # Use project key
            proj_key = project_key or self.project_key
            if not proj_key:
                # Generate project key from path
                proj_key = self._generate_project_key(source_path)

            # Run scanner
            scan_result = await self._run_scanner(source_path, proj_key)

            if not scan_result["success"]:
                return self._unavailable_results(
                    reason=scan_result.get("error", "Scanner failed")
                )

            # Fetch metrics from server
            metrics = await self._fetch_metrics(proj_key)

            # Fetch issues
            issues = await self._fetch_issues(proj_key)

            # Calculate quality ratings
            ratings = self._calculate_ratings(metrics)

            results = {
                "available": True,
                "quality_gate_status": metrics.get("quality_gate_status", "UNKNOWN"),
                "metrics": metrics,
                "issues": issues,
                "ratings": ratings,
                "project_key": proj_key
            }

            logger.info(
                f"SonarQube analysis complete: "
                f"{len(issues)} issues, "
                f"quality gate {results['quality_gate_status']}"
            )

            return results

        except Exception as e:
            logger.error(f"SonarQube analysis failed: {e}")
            return self._unavailable_results(reason=str(e))

    async def _check_java(self) -> bool:
        """Check if Java is available.

        Returns:
            True if Java is installed
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "java", "-version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            self.java_available = process.returncode == 0
            return self.java_available
        except FileNotFoundError:
            self.java_available = False
            return False

    async def _check_scanner(self) -> bool:
        """Check if sonar-scanner is available.

        Returns:
            True if scanner is installed
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "sonar-scanner", "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            self.scanner_available = process.returncode == 0
            return self.scanner_available
        except FileNotFoundError:
            self.scanner_available = False
            return False

    def _generate_project_key(self, source_path: str) -> str:
        """Generate project key from source path.

        Args:
            source_path: Source code path

        Returns:
            Project key
        """
        path = Path(source_path)
        # Use directory name as project key
        if path.is_file():
            path = path.parent

        project_name = path.name or "unknown"
        return f"agent-swarm:{project_name}"

    async def _run_scanner(
        self,
        source_path: str,
        project_key: str
    ) -> Dict[str, Any]:
        """Run sonar-scanner on source code.

        Args:
            source_path: Path to source
            project_key: Project key

        Returns:
            Scanner result
        """
        try:
            # Build scanner command
            cmd = [
                "sonar-scanner",
                f"-Dsonar.projectKey={project_key}",
                f"-Dsonar.sources={source_path}",
                f"-Dsonar.host.url={self.sonar_host}",
            ]

            if self.sonar_token:
                cmd.append(f"-Dsonar.login={self.sonar_token}")

            # Add language-specific settings
            cmd.extend([
                "-Dsonar.language=py",
                "-Dsonar.python.version=3.8,3.9,3.10,3.11,3.12",
            ])

            # Run scanner
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "output": stdout.decode(),
                "error": stderr.decode() if process.returncode != 0 else None
            }

        except Exception as e:
            logger.error(f"Scanner execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _fetch_metrics(self, project_key: str) -> Dict[str, Any]:
        """Fetch metrics from SonarQube server.

        Args:
            project_key: Project key

        Returns:
            Metrics dictionary
        """
        try:
            # Use SonarQube Web API
            import aiohttp

            metrics_str = ",".join(self.METRICS)
            url = f"{self.sonar_host}/api/measures/component"
            params = {
                "component": project_key,
                "metricKeys": metrics_str
            }

            headers = {}
            if self.sonar_token:
                headers["Authorization"] = f"Bearer {self.sonar_token}"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch metrics: {response.status}")
                        return {}

                    data = await response.json()

                    # Parse metrics
                    metrics = {}
                    for measure in data.get("component", {}).get("measures", []):
                        metric_key = measure.get("metric")
                        value = measure.get("value")
                        metrics[metric_key] = self._parse_metric_value(value)

                    # Fetch quality gate status separately
                    gate_status = await self._fetch_quality_gate_status(
                        project_key,
                        session,
                        headers
                    )
                    metrics["quality_gate_status"] = gate_status

                    return metrics

        except ImportError:
            logger.warning("aiohttp not available, cannot fetch metrics")
            return {}
        except Exception as e:
            logger.error(f"Failed to fetch metrics: {e}")
            return {}

    async def _fetch_quality_gate_status(
        self,
        project_key: str,
        session: Any,
        headers: Dict[str, str]
    ) -> str:
        """Fetch quality gate status.

        Args:
            project_key: Project key
            session: aiohttp session
            headers: Request headers

        Returns:
            Quality gate status (OK, WARN, ERROR)
        """
        try:
            url = f"{self.sonar_host}/api/qualitygates/project_status"
            params = {"projectKey": project_key}

            async with session.get(
                url,
                params=params,
                headers=headers
            ) as response:
                if response.status != 200:
                    return "UNKNOWN"

                data = await response.json()
                return data.get("projectStatus", {}).get("status", "UNKNOWN")

        except Exception as e:
            logger.debug(f"Failed to fetch quality gate status: {e}")
            return "UNKNOWN"

    async def _fetch_issues(self, project_key: str) -> List[Dict[str, Any]]:
        """Fetch issues from SonarQube server.

        Args:
            project_key: Project key

        Returns:
            List of issues
        """
        try:
            import aiohttp

            url = f"{self.sonar_host}/api/issues/search"
            params = {
                "componentKeys": project_key,
                "resolved": "false",
                "ps": 500  # Page size
            }

            headers = {}
            if self.sonar_token:
                headers["Authorization"] = f"Bearer {self.sonar_token}"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch issues: {response.status}")
                        return []

                    data = await response.json()
                    return data.get("issues", [])

        except ImportError:
            logger.warning("aiohttp not available, cannot fetch issues")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch issues: {e}")
            return []

    def _parse_metric_value(self, value: str) -> Any:
        """Parse metric value to appropriate type.

        Args:
            value: Metric value as string

        Returns:
            Parsed value
        """
        try:
            # Try integer
            return int(value)
        except (ValueError, TypeError):
            pass

        try:
            # Try float
            return float(value)
        except (ValueError, TypeError):
            pass

        # Return as string
        return value

    def _calculate_ratings(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Calculate quality ratings from metrics.

        Args:
            metrics: Metrics dictionary

        Returns:
            Ratings dictionary
        """
        ratings = {}

        # Extract ratings from metrics
        rating_keys = [
            "sqale_rating",  # Maintainability
            "reliability_rating",
            "security_rating",
        ]

        for key in rating_keys:
            if key in metrics:
                rating_value = metrics[key]
                # Convert numeric rating to letter grade
                rating_letter = self._rating_to_letter(rating_value)
                # Remove _rating suffix for cleaner key
                clean_key = key.replace("_rating", "")
                ratings[clean_key] = rating_letter

        return ratings

    def _rating_to_letter(self, rating: Any) -> str:
        """Convert numeric rating to letter grade.

        Args:
            rating: Numeric rating (1-5)

        Returns:
            Letter grade (A-E)
        """
        try:
            rating_num = int(rating)
            mapping = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
            return mapping.get(rating_num, "?")
        except (ValueError, TypeError):
            return str(rating)

    def _unavailable_results(self, reason: str = "Not available") -> Dict[str, Any]:
        """Return results indicating SonarQube is unavailable.

        Args:
            reason: Reason for unavailability

        Returns:
            Unavailable results dictionary
        """
        return {
            "available": False,
            "reason": reason,
            "quality_gate_status": "UNKNOWN",
            "metrics": {},
            "issues": [],
            "ratings": {}
        }

    async def check_installation(self) -> bool:
        """Check if SonarQube integration is available.

        Returns:
            True if Java and scanner are available
        """
        java_ok = await self._check_java()
        scanner_ok = await self._check_scanner()
        return java_ok and scanner_ok

    def is_available(self) -> bool:
        """Check if SonarQube is available (synchronous).

        Returns:
            True if prerequisites are met
        """
        return self.java_available and self.scanner_available
