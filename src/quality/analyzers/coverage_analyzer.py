"""Enhanced coverage analyzer with multi-level support.

PATTERN: Extends existing coverage_analyzer.py with branch and mutation coverage
CRITICAL: Uses Coverage.py with --branch flag for branch coverage
GOTCHA: Branch coverage data structure differs from line coverage
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedCoverageAnalyzer:
    """
    Enhanced coverage analyzer supporting line, branch, and mutation coverage.

    PATTERN: Multi-level coverage analysis with parallel execution
    CRITICAL: Branch coverage requires Coverage.py --branch flag
    GOTCHA: Mutation coverage is CPU-intensive, use timeouts
    """

    def __init__(self, mutation_enabled: bool = False):
        """
        Initialize enhanced coverage analyzer.

        Args:
            mutation_enabled: Enable mutation testing (resource intensive)
        """
        self.logger = logger
        self.mutation_enabled = mutation_enabled

    async def analyze_coverage(
        self,
        test_results: Dict[str, Any],
        source_files: List[str],
        coverage_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze multiple coverage types in parallel.

        PATTERN: Parallel analysis of different coverage types
        CRITICAL: Each coverage type may require different data sources

        Args:
            test_results: Test execution results with coverage data
            source_files: Source files being tested
            coverage_types: Types to analyze (line, branch, mutation)

        Returns:
            Dictionary with coverage reports by type
        """
        if coverage_types is None:
            coverage_types = ["line", "branch"]
            if self.mutation_enabled:
                coverage_types.append("mutation")

        tasks = []

        if "line" in coverage_types:
            tasks.append(self.analyze_line_coverage(test_results, source_files))

        if "branch" in coverage_types:
            tasks.append(self.analyze_branch_coverage(test_results, source_files))

        if "mutation" in coverage_types and self.mutation_enabled:
            tasks.append(self.analyze_mutation_coverage(source_files))

        # Run analyses in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        coverage_report = {
            "timestamp": datetime.now().isoformat(),
            "source_files": source_files,
            "coverage_types": {},
        }

        for i, coverage_type in enumerate(coverage_types):
            if i < len(results):
                result = results[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Error analyzing {coverage_type} coverage: {result}")
                    coverage_report["coverage_types"][coverage_type] = {
                        "error": str(result),
                        "percentage": 0.0
                    }
                else:
                    coverage_report["coverage_types"][coverage_type] = result

        # Calculate overall percentage
        coverage_report["total_percentage"] = self._calculate_total_coverage(
            coverage_report["coverage_types"]
        )

        return coverage_report

    async def analyze_line_coverage(
        self,
        test_results: Dict[str, Any],
        source_files: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze line coverage from test results.

        PATTERN: Parse .coverage data file or coverage JSON
        CRITICAL: Requires coverage run during test execution

        Args:
            test_results: Test execution results
            source_files: Source files to analyze

        Returns:
            Line coverage report
        """
        try:
            # Check for coverage data in multiple formats
            coverage_data = None

            # Try to load from .coverage file
            coverage_file = Path(".coverage")
            if coverage_file.exists():
                coverage_data = await self._load_coverage_file(coverage_file)
            # Try to load from test_results
            elif "coverage" in test_results:
                coverage_data = test_results["coverage"]

            if not coverage_data:
                self.logger.warning("No coverage data found")
                return self._empty_coverage_report("line")

            # Analyze each source file
            files_coverage = {}
            total_lines = 0
            covered_lines = 0

            for file_path in source_files:
                file_data = await self._analyze_file_line_coverage(
                    file_path,
                    coverage_data
                )
                files_coverage[file_path] = file_data
                total_lines += file_data["total_lines"]
                covered_lines += file_data["covered_lines"]

            percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

            return {
                "type": "line",
                "percentage": round(percentage, 2),
                "covered": covered_lines,
                "total": total_lines,
                "files": files_coverage,
            }

        except Exception as e:
            self.logger.error(f"Line coverage analysis failed: {e}")
            raise

    async def analyze_branch_coverage(
        self,
        test_results: Dict[str, Any],
        source_files: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze branch coverage from test results.

        PATTERN: Requires coverage run --branch
        CRITICAL: Branch coverage tracks decision path coverage
        GOTCHA: Branch data structure differs from line coverage

        Args:
            test_results: Test execution results
            source_files: Source files to analyze

        Returns:
            Branch coverage report
        """
        try:
            # Load branch coverage data
            coverage_file = Path(".coverage")
            if not coverage_file.exists():
                self.logger.warning("No .coverage file found for branch analysis")
                return self._empty_coverage_report("branch")

            coverage_data = await self._load_coverage_file(coverage_file)

            # Extract branch information
            files_coverage = {}
            total_branches = 0
            covered_branches = 0

            for file_path in source_files:
                file_data = await self._analyze_file_branch_coverage(
                    file_path,
                    coverage_data
                )
                files_coverage[file_path] = file_data
                total_branches += file_data["total_branches"]
                covered_branches += file_data["covered_branches"]

            percentage = (covered_branches / total_branches * 100) if total_branches > 0 else 0.0

            return {
                "type": "branch",
                "percentage": round(percentage, 2),
                "covered": covered_branches,
                "total": total_branches,
                "files": files_coverage,
                "branch_details": self._extract_branch_details(coverage_data, source_files),
            }

        except Exception as e:
            self.logger.error(f"Branch coverage analysis failed: {e}")
            raise

    async def analyze_mutation_coverage(
        self,
        source_files: List[str],
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Analyze mutation testing coverage.

        PATTERN: Resource-limited mutation testing
        CRITICAL: CPU-intensive, requires timeout and parallel limits
        GOTCHA: May timeout on large codebases

        Args:
            source_files: Source files to test
            timeout: Maximum execution time in seconds

        Returns:
            Mutation coverage report
        """
        try:
            # Run mutation testing with mutmut
            result = await self._run_mutmut(source_files, timeout)

            if result.get("timeout"):
                self.logger.warning("Mutation testing timeout, using partial results")

            mutation_score = result.get("mutation_score", 0.0)

            return {
                "type": "mutation",
                "percentage": round(mutation_score, 2),
                "mutation_score": round(mutation_score, 2),
                "killed_mutants": result.get("killed", 0),
                "survived_mutants": result.get("survived", 0),
                "timeout_mutants": result.get("timeout", 0),
                "total_mutants": result.get("total", 0),
                "files": result.get("files", {}),
            }

        except Exception as e:
            self.logger.error(f"Mutation coverage analysis failed: {e}")
            raise

    async def _load_coverage_file(self, coverage_file: Path) -> Dict[str, Any]:
        """
        Load coverage data from .coverage file.

        PATTERN: Use coverage library to load data
        CRITICAL: Handle both SQLite and JSON formats

        Args:
            coverage_file: Path to .coverage file

        Returns:
            Parsed coverage data
        """
        try:
            import coverage

            # Create coverage instance
            cov = coverage.Coverage(data_file=str(coverage_file))
            cov.load()

            # Get data
            data = cov.get_data()

            # Convert to dictionary format
            coverage_dict = {}

            for file_path in data.measured_files():
                coverage_dict[file_path] = {
                    "lines": list(data.lines(file_path) or []),
                    "arcs": list(data.arcs(file_path) or []),
                }

            return coverage_dict

        except ImportError:
            self.logger.error("coverage library not installed")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load coverage file: {e}")
            raise

    async def _analyze_file_line_coverage(
        self,
        file_path: str,
        coverage_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze line coverage for a single file.

        Args:
            file_path: Path to source file
            coverage_data: Coverage data dictionary

        Returns:
            File line coverage data
        """
        file_path_abs = str(Path(file_path).resolve())

        # Find matching entry in coverage data
        file_coverage = None
        for cov_path, cov_data in coverage_data.items():
            if Path(cov_path).resolve() == Path(file_path_abs):
                file_coverage = cov_data
                break

        if not file_coverage:
            return {
                "total_lines": 0,
                "covered_lines": 0,
                "uncovered_lines": [],
                "percentage": 0.0,
            }

        # Get covered lines
        covered_lines = set(file_coverage.get("lines", []))

        # Count executable lines in file
        executable_lines = await self._count_executable_lines(file_path)
        uncovered_lines = executable_lines - covered_lines

        total = len(executable_lines)
        covered = len(covered_lines)
        percentage = (covered / total * 100) if total > 0 else 0.0

        return {
            "total_lines": total,
            "covered_lines": covered,
            "uncovered_lines": sorted(list(uncovered_lines)),
            "percentage": round(percentage, 2),
        }

    async def _analyze_file_branch_coverage(
        self,
        file_path: str,
        coverage_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze branch coverage for a single file.

        Args:
            file_path: Path to source file
            coverage_data: Coverage data with branch information

        Returns:
            File branch coverage data
        """
        file_path_abs = str(Path(file_path).resolve())

        # Find matching entry in coverage data
        file_coverage = None
        for cov_path, cov_data in coverage_data.items():
            if Path(cov_path).resolve() == Path(file_path_abs):
                file_coverage = cov_data
                break

        if not file_coverage:
            return {
                "total_branches": 0,
                "covered_branches": 0,
                "uncovered_branches": [],
                "percentage": 0.0,
            }

        # Get branch arcs (source_line, dest_line) tuples
        arcs = file_coverage.get("arcs", [])

        # Identify decision points and branches
        branches = self._extract_branches_from_arcs(arcs)

        total_branches = len(branches["all"])
        covered_branches = len(branches["covered"])

        percentage = (covered_branches / total_branches * 100) if total_branches > 0 else 0.0

        return {
            "total_branches": total_branches,
            "covered_branches": covered_branches,
            "uncovered_branches": branches["uncovered"],
            "percentage": round(percentage, 2),
            "branch_points": branches["points"],
        }

    async def _run_mutmut(
        self,
        source_files: List[str],
        timeout: int
    ) -> Dict[str, Any]:
        """
        Run mutmut mutation testing.

        PATTERN: Resource-limited subprocess execution
        CRITICAL: Use timeout and parallel limits

        Args:
            source_files: Files to mutate
            timeout: Maximum execution time

        Returns:
            Mutation testing results
        """
        try:
            # Build mutmut command
            paths = ",".join(source_files)
            cmd = [
                "mutmut",
                "run",
                "--paths-to-mutate", paths,
                "--runner", "pytest -x",
                "--timeout-factor", "5.0",
            ]

            # Run with timeout
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=timeout
            )

            stdout, stderr = await process.communicate()

            # Parse results
            return await self._parse_mutmut_results(stdout.decode())

        except asyncio.TimeoutError:
            self.logger.warning(f"Mutation testing timeout after {timeout}s")
            return {
                "timeout": True,
                "mutation_score": 0.0,
                "killed": 0,
                "survived": 0,
                "total": 0,
            }
        except FileNotFoundError:
            self.logger.error("mutmut not installed")
            return {
                "error": "mutmut not installed",
                "mutation_score": 0.0,
            }
        except Exception as e:
            self.logger.error(f"Mutation testing failed: {e}")
            raise

    async def _parse_mutmut_results(self, output: str) -> Dict[str, Any]:
        """
        Parse mutmut output to extract results.

        Args:
            output: mutmut stdout

        Returns:
            Parsed mutation results
        """
        # Parse mutmut output
        # Format: "X killed, Y survived, Z timeout, W total"
        result = {
            "killed": 0,
            "survived": 0,
            "timeout": 0,
            "total": 0,
            "mutation_score": 0.0,
        }

        # Extract numbers from output
        import re

        killed_match = re.search(r"(\d+)\s+killed", output, re.IGNORECASE)
        survived_match = re.search(r"(\d+)\s+survived", output, re.IGNORECASE)
        timeout_match = re.search(r"(\d+)\s+timeout", output, re.IGNORECASE)

        if killed_match:
            result["killed"] = int(killed_match.group(1))
        if survived_match:
            result["survived"] = int(survived_match.group(1))
        if timeout_match:
            result["timeout"] = int(timeout_match.group(1))

        result["total"] = result["killed"] + result["survived"] + result["timeout"]

        if result["total"] > 0:
            result["mutation_score"] = (result["killed"] / result["total"]) * 100

        return result

    async def _count_executable_lines(self, file_path: str) -> Set[int]:
        """
        Count executable lines in a Python file.

        PATTERN: Use AST to identify executable statements
        CRITICAL: Exclude comments, docstrings, blank lines

        Args:
            file_path: Path to source file

        Returns:
            Set of executable line numbers
        """
        try:
            import ast

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content)

            # Collect line numbers of all statements
            executable_lines = set()

            for node in ast.walk(tree):
                if hasattr(node, "lineno"):
                    # Skip docstrings
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                        if isinstance(node.value.value, str):
                            continue
                    executable_lines.add(node.lineno)

            return executable_lines

        except Exception as e:
            self.logger.error(f"Failed to count executable lines in {file_path}: {e}")
            return set()

    def _extract_branches_from_arcs(self, arcs: List[tuple]) -> Dict[str, Any]:
        """
        Extract branch information from coverage arcs.

        PATTERN: Group arcs by decision points
        CRITICAL: Each decision point should have 2+ outgoing arcs

        Args:
            arcs: List of (from_line, to_line) tuples

        Returns:
            Branch analysis data
        """
        # Group arcs by source line (decision point)
        decision_points = {}

        for from_line, to_line in arcs:
            if from_line not in decision_points:
                decision_points[from_line] = []
            decision_points[from_line].append(to_line)

        # Identify branches (decision points with multiple destinations)
        branches = {
            "all": [],
            "covered": [],
            "uncovered": [],
            "points": {},
        }

        for from_line, to_lines in decision_points.items():
            if len(to_lines) > 1:
                # This is a branch point
                for to_line in to_lines:
                    branch_id = f"{from_line}->{to_line}"
                    branches["all"].append(branch_id)
                    branches["covered"].append(branch_id)  # All in arcs are covered

                branches["points"][from_line] = {
                    "destinations": to_lines,
                    "branch_count": len(to_lines),
                }

        return branches

    def _extract_branch_details(
        self,
        coverage_data: Dict[str, Any],
        source_files: List[str]
    ) -> Dict[str, Any]:
        """
        Extract detailed branch coverage information.

        Args:
            coverage_data: Coverage data dictionary
            source_files: Source files to analyze

        Returns:
            Detailed branch information
        """
        branch_details = {}

        for file_path in source_files:
            file_path_abs = str(Path(file_path).resolve())

            for cov_path, cov_data in coverage_data.items():
                if Path(cov_path).resolve() == Path(file_path_abs):
                    arcs = cov_data.get("arcs", [])
                    branch_details[file_path] = self._extract_branches_from_arcs(arcs)
                    break

        return branch_details

    def _calculate_total_coverage(self, coverage_types: Dict[str, Any]) -> float:
        """
        Calculate weighted average of all coverage types.

        Args:
            coverage_types: Dictionary of coverage results by type

        Returns:
            Overall coverage percentage
        """
        if not coverage_types:
            return 0.0

        # Weighted average: line=40%, branch=40%, mutation=20%
        weights = {
            "line": 0.4,
            "branch": 0.4,
            "mutation": 0.2,
        }

        total = 0.0
        weight_sum = 0.0

        for cov_type, cov_data in coverage_types.items():
            if cov_type in weights and "percentage" in cov_data:
                total += cov_data["percentage"] * weights[cov_type]
                weight_sum += weights[cov_type]

        if weight_sum > 0:
            return round(total / weight_sum, 2)
        return 0.0

    def _empty_coverage_report(self, coverage_type: str) -> Dict[str, Any]:
        """
        Create empty coverage report for a type.

        Args:
            coverage_type: Type of coverage

        Returns:
            Empty report structure
        """
        return {
            "type": coverage_type,
            "percentage": 0.0,
            "covered": 0,
            "total": 0,
            "files": {},
        }
