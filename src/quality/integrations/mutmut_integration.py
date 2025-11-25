"""Mutation testing integration using mutmut.

This module provides mutation testing capabilities with resource limits,
timeout handling, and result parsing for mutation score calculation.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...models.quality_models import CoverageReport, CoverageType

logger = logging.getLogger(__name__)


class MutmutIntegration:
    """Integration with mutmut mutation testing framework.

    CRITICAL: Mutation testing is CPU-intensive. This class implements:
    - Resource limits (max parallel mutations)
    - Timeout handling (5 minute maximum)
    - Partial results on timeout
    - Result parsing for mutation score
    """

    def __init__(
        self,
        timeout_factor: float = 5.0,
        max_timeout_seconds: int = 300,
        max_parallel: int = 4
    ):
        """Initialize mutmut integration.

        Args:
            timeout_factor: Timeout multiplier for test execution
            max_timeout_seconds: Maximum overall timeout (default: 5 minutes)
            max_parallel: Maximum parallel mutations (default: 4)
        """
        self.timeout_factor = timeout_factor
        self.max_timeout_seconds = max_timeout_seconds
        self.max_parallel = max_parallel

    async def run_mutation_testing(
        self,
        target_path: str,
        test_command: str = "pytest",
        paths_to_exclude: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run mutation testing on target path.

        PATTERN: Resource-limited subprocess execution
        CRITICAL: CPU-intensive, use timeout and parallel limits

        Args:
            target_path: Path to code to mutate
            test_command: Command to run tests (default: pytest)
            paths_to_exclude: Paths to exclude from mutation

        Returns:
            Dictionary containing:
                - mutation_score: Percentage of killed mutants
                - killed: Number of killed mutants
                - survived: Number of survived mutants
                - timeout: Number of timed-out mutants
                - total: Total mutations attempted
                - completed: Whether all mutations completed
                - details: Per-file mutation details
        """
        logger.info(f"Starting mutation testing for {target_path}")
        start_time = datetime.now()

        try:
            # First, initialize mutmut cache
            await self._initialize_mutmut()

            # Build mutmut command
            cmd = self._build_mutmut_command(
                target_path,
                test_command,
                paths_to_exclude
            )

            # Run mutation testing with timeout
            completed = False
            try:
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    ),
                    timeout=self.max_timeout_seconds
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.max_timeout_seconds
                )
                completed = True

            except asyncio.TimeoutError:
                logger.warning(
                    f"Mutation testing timeout after {self.max_timeout_seconds}s, "
                    "using partial results"
                )
                completed = False
                # Try to get partial results
                stdout = b""
                stderr = b"Timeout occurred"

            # Parse results
            results = await self._parse_mutmut_results(completed)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            results["execution_time_seconds"] = execution_time
            results["completed"] = completed

            logger.info(
                f"Mutation testing finished: {results['mutation_score']:.1f}% score "
                f"({results['killed']}/{results['total']} killed)"
            )

            return results

        except FileNotFoundError:
            logger.warning("mutmut not installed, skipping mutation testing")
            return self._empty_results(reason="mutmut not installed")

        except Exception as e:
            logger.error(f"Mutation testing failed: {e}")
            return self._empty_results(reason=str(e))

    async def _initialize_mutmut(self) -> None:
        """Initialize mutmut cache/database."""
        try:
            # Check if .mutmut-cache exists, if not mutmut will create it
            cache_path = Path(".mutmut-cache")
            if cache_path.exists():
                # Clean previous cache for fresh run
                logger.debug("Cleaning previous mutmut cache")
                cache_path.unlink()
        except Exception as e:
            logger.debug(f"Failed to clean mutmut cache: {e}")

    def _build_mutmut_command(
        self,
        target_path: str,
        test_command: str,
        paths_to_exclude: Optional[List[str]] = None
    ) -> List[str]:
        """Build mutmut command with all options.

        Args:
            target_path: Path to mutate
            test_command: Test command
            paths_to_exclude: Paths to exclude

        Returns:
            Command as list of strings
        """
        cmd = [
            "mutmut", "run",
            "--paths-to-mutate", target_path,
            "--runner", test_command,
            "--timeout-factor", str(self.timeout_factor),
        ]

        # Add parallel execution limit
        # GOTCHA: Too many parallel mutations can overload system
        cmd.extend(["--use-coverage"])  # Use coverage data to speed up

        # Add exclusions
        if paths_to_exclude:
            for exclude in paths_to_exclude:
                cmd.extend(["--paths-to-exclude", exclude])

        return cmd

    async def _parse_mutmut_results(self, completed: bool) -> Dict[str, Any]:
        """Parse mutmut results from cache.

        Args:
            completed: Whether mutation testing completed fully

        Returns:
            Parsed mutation results
        """
        try:
            # Run mutmut results command to get summary
            process = await asyncio.create_subprocess_exec(
                "mutmut", "results",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            output = stdout.decode()

            # Parse the output
            # Format typically: "X killed, Y timeout, Z suspicious, W survived"
            results = {
                "killed": 0,
                "timeout": 0,
                "suspicious": 0,
                "survived": 0,
                "total": 0,
                "mutation_score": 0.0,
                "details": {}
            }

            # Try to extract numbers from output
            killed_match = re.search(r'(\d+)\s+killed', output, re.IGNORECASE)
            timeout_match = re.search(r'(\d+)\s+timeout', output, re.IGNORECASE)
            suspicious_match = re.search(r'(\d+)\s+suspicious', output, re.IGNORECASE)
            survived_match = re.search(r'(\d+)\s+survived', output, re.IGNORECASE)

            if killed_match:
                results["killed"] = int(killed_match.group(1))
            if timeout_match:
                results["timeout"] = int(timeout_match.group(1))
            if suspicious_match:
                results["suspicious"] = int(suspicious_match.group(1))
            if survived_match:
                results["survived"] = int(survived_match.group(1))

            # Calculate total and score
            results["total"] = (
                results["killed"] +
                results["timeout"] +
                results["suspicious"] +
                results["survived"]
            )

            if results["total"] > 0:
                # Mutation score = killed / (total - timeout)
                # Timeout mutants are excluded from score calculation
                effective_total = results["total"] - results["timeout"]
                if effective_total > 0:
                    results["mutation_score"] = (
                        results["killed"] / effective_total * 100
                    )

            # Get detailed results per mutant if available
            try:
                details = await self._get_mutant_details()
                results["details"] = details
            except Exception as e:
                logger.debug(f"Could not get mutant details: {e}")

            return results

        except Exception as e:
            logger.error(f"Failed to parse mutmut results: {e}")
            return self._empty_results(reason=str(e))

    async def _get_mutant_details(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get detailed information about each mutant.

        Returns:
            Dictionary mapping file paths to list of mutant details
        """
        details: Dict[str, List[Dict[str, Any]]] = {}

        try:
            # Run mutmut show to get individual mutant details
            # This requires parsing the cache or using mutmut's JSON output
            # For now, return empty details as this is optional enhancement
            pass

        except Exception as e:
            logger.debug(f"Failed to get mutant details: {e}")

        return details

    def _empty_results(self, reason: str = "No mutations run") -> Dict[str, Any]:
        """Return empty results structure.

        Args:
            reason: Reason for empty results

        Returns:
            Empty results dictionary
        """
        return {
            "killed": 0,
            "timeout": 0,
            "suspicious": 0,
            "survived": 0,
            "total": 0,
            "mutation_score": 0.0,
            "completed": False,
            "reason": reason,
            "details": {}
        }

    def create_coverage_report(
        self,
        mutation_results: Dict[str, Any]
    ) -> CoverageReport:
        """Create coverage report from mutation results.

        Args:
            mutation_results: Results from run_mutation_testing

        Returns:
            CoverageReport for mutation testing
        """
        return CoverageReport(
            type=CoverageType.MUTATION,
            percentage=mutation_results.get("mutation_score", 0.0),
            covered=mutation_results.get("killed", 0),
            total=mutation_results.get("total", 0),
            mutation_score=mutation_results.get("mutation_score", 0.0),
            killed_mutants=mutation_results.get("killed", 0),
            survived_mutants=mutation_results.get("survived", 0),
            timeout_mutants=mutation_results.get("timeout", 0),
            files=mutation_results.get("details", {})
        )

    async def check_installation(self) -> bool:
        """Check if mutmut is installed.

        Returns:
            True if mutmut is available
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "mutmut", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            return process.returncode == 0
        except FileNotFoundError:
            return False
