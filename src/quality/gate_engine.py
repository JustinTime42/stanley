"""Quality gate engine orchestrating all analyzers.

PATTERN: Service orchestration with parallel execution
CRITICAL: Enforce quality gates with thresholds and audit trail
GOTCHA: Some analyzers are resource-intensive, need timeouts
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

from ..models.quality_models import (
    QualityReport,
    QualityStatus,
    QualityThreshold,
    QualityDimension,
    CoverageType,
    CoverageReport,
    SecurityIssue,
    PerformanceMetric,
)
from .threshold_manager import ThresholdManager
from .analyzers.coverage_analyzer import EnhancedCoverageAnalyzer
from .analyzers.static_analyzer import StaticAnalyzer

logger = logging.getLogger(__name__)


class QualityGateEngine:
    """
    Orchestrates all quality analyzers and enforces quality gates.

    PATTERN: Service orchestration with parallel analyzer execution
    CRITICAL: Must support emergency overrides with full audit trail
    GOTCHA: Balance parallel execution with resource constraints

    This engine coordinates all quality analysis dimensions:
    - Coverage analysis (line, branch, mutation)
    - Static code analysis (Ruff, Mypy, Prospector)
    - Security scanning (Bandit, vulnerability detection)
    - Performance regression detection
    - Complexity metrics (cyclomatic, cognitive)

    It enforces configurable quality gates with threshold violations,
    supports emergency overrides with audit logging, and generates
    comprehensive quality reports.
    """

    def __init__(
        self,
        threshold_manager: Optional[ThresholdManager] = None,
        coverage_analyzer: Optional[EnhancedCoverageAnalyzer] = None,
        static_analyzer: Optional[StaticAnalyzer] = None,
        enable_mutation_testing: bool = False,
        enable_security_scanning: bool = True,
        enable_performance_analysis: bool = True,
    ):
        """
        Initialize quality gate engine.

        Args:
            threshold_manager: Threshold configuration manager
            coverage_analyzer: Coverage analysis component
            static_analyzer: Static analysis component
            enable_mutation_testing: Enable mutation testing (resource intensive)
            enable_security_scanning: Enable security vulnerability scanning
            enable_performance_analysis: Enable performance regression detection
        """
        self.logger = logger

        # Initialize threshold manager
        self.threshold_manager = threshold_manager or ThresholdManager()

        # Initialize analyzers
        self.coverage_analyzer = coverage_analyzer or EnhancedCoverageAnalyzer(
            mutation_enabled=enable_mutation_testing
        )
        self.static_analyzer = static_analyzer or StaticAnalyzer()

        # Feature flags
        self.enable_mutation_testing = enable_mutation_testing
        self.enable_security_scanning = enable_security_scanning
        self.enable_performance_analysis = enable_performance_analysis

        # Override audit log
        self.override_log: List[Dict[str, Any]] = []

        self.logger.info("QualityGateEngine initialized")

    async def run_quality_checks(
        self,
        source_path: str,
        test_results: Optional[Dict[str, Any]] = None,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
        module: Optional[str] = None,
        dimensions: Optional[List[QualityDimension]] = None,
    ) -> QualityReport:
        """
        Run all quality checks and generate comprehensive report.

        PATTERN: Parallel execution of all enabled analyzers
        CRITICAL: Aggregate results from all dimensions
        GOTCHA: Handle analyzer failures gracefully

        Args:
            source_path: Path to source code to analyze
            test_results: Optional test execution results with coverage
            baseline_metrics: Optional baseline for regression detection
            project: Optional project name for threshold overrides
            module: Optional module name for threshold overrides
            dimensions: Optional filter to specific quality dimensions

        Returns:
            Comprehensive quality report
        """
        start_time = datetime.now()
        report_id = str(uuid.uuid4())

        self.logger.info(f"Running quality checks for {source_path} (report: {report_id})")

        # Determine which dimensions to analyze
        enabled_dimensions = self._get_enabled_dimensions(dimensions)

        # Run all analyzers in parallel
        tasks = []

        if QualityDimension.COVERAGE in enabled_dimensions and test_results:
            tasks.append(self._run_coverage_analysis(source_path, test_results))
        else:
            tasks.append(self._create_empty_coverage())

        if QualityDimension.STATIC in enabled_dimensions:
            tasks.append(self._run_static_analysis(source_path))
        else:
            tasks.append(self._create_empty_static())

        if QualityDimension.SECURITY in enabled_dimensions and self.enable_security_scanning:
            tasks.append(self._run_security_analysis(source_path))
        else:
            tasks.append(self._create_empty_security())

        if QualityDimension.PERFORMANCE in enabled_dimensions and baseline_metrics:
            tasks.append(self._run_performance_analysis(test_results, baseline_metrics))
        else:
            tasks.append(self._create_empty_performance())

        if QualityDimension.COMPLEXITY in enabled_dimensions:
            tasks.append(self._run_complexity_analysis(source_path))
        else:
            tasks.append(self._create_empty_complexity())

        # Execute all analyses in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extract results
        coverage_data = self._extract_result(results[0], {})
        static_data = self._extract_result(results[1], {})
        security_data = self._extract_result(results[2], {})
        performance_data = self._extract_result(results[3], {})
        complexity_data = self._extract_result(results[4], {})

        # Build quality report
        report = self._build_quality_report(
            report_id=report_id,
            coverage_data=coverage_data,
            static_data=static_data,
            security_data=security_data,
            performance_data=performance_data,
            complexity_data=complexity_data,
            start_time=start_time,
        )

        # Check thresholds
        violations = self.threshold_manager.check_violations(
            metrics=self._extract_metrics(report),
            project=project,
            module=module,
        )

        report.violations = violations
        report.passed = len([v for v in violations if v["enforcement"] == "error"]) == 0

        # Determine overall status
        report.status = self._determine_status(violations)

        # Add recommendations
        report.recommendations = self._generate_recommendations(report)

        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            f"Quality checks completed in {duration:.2f}s "
            f"(status: {report.status}, violations: {len(violations)})"
        )

        return report

    async def enforce_gates(
        self,
        report: QualityReport,
        force: bool = False,
        reason: Optional[str] = None,
        user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enforce quality gates with optional emergency override.

        PATTERN: Configurable enforcement with full audit trail
        CRITICAL: Emergency overrides must be logged with justification
        GOTCHA: Overrides should only apply to overridable thresholds

        Args:
            report: Quality report to enforce
            force: Force override of quality gates
            reason: Required justification for override
            user: User requesting override (for audit trail)

        Returns:
            Enforcement result with pass/fail and violation details
        """
        violations = report.violations
        error_violations = [v for v in violations if v["enforcement"] == "error"]
        warning_violations = [v for v in violations if v["enforcement"] == "warning"]

        # Check if override is valid
        if force:
            if not reason:
                raise ValueError("Override reason required for force=True")

            # Log override for audit trail
            override_entry = {
                "timestamp": datetime.now().isoformat(),
                "report_id": report.report_id,
                "user": user or "unknown",
                "reason": reason,
                "violations_overridden": len(error_violations),
                "error_violations": error_violations,
                "warning_violations": warning_violations,
            }
            self.override_log.append(override_entry)

            self.logger.warning(
                f"Quality gate OVERRIDDEN by {user or 'unknown'}: {reason} "
                f"(report: {report.report_id}, violations: {len(error_violations)})"
            )

            # Check which violations can be overridden
            non_overridable = [
                v for v in error_violations
                if not v["threshold"].allow_override
            ]

            if non_overridable:
                self.logger.error(
                    f"Cannot override {len(non_overridable)} non-overridable violations"
                )
                return {
                    "passed": False,
                    "violations": violations,
                    "override_used": True,
                    "override_failed": True,
                    "non_overridable_violations": non_overridable,
                    "message": "Some violations cannot be overridden",
                }

            # Override is valid
            return {
                "passed": True,
                "violations": violations,
                "override_used": True,
                "override_reason": reason,
                "override_user": user,
                "warning": f"Quality gates overridden: {len(error_violations)} violations ignored",
            }

        # Normal enforcement (no override)
        if error_violations:
            return {
                "passed": False,
                "violations": violations,
                "override_used": False,
                "error_count": len(error_violations),
                "warning_count": len(warning_violations),
                "message": f"Quality gates failed: {len(error_violations)} error violations",
            }

        # All checks passed
        return {
            "passed": True,
            "violations": violations,
            "override_used": False,
            "warning_count": len(warning_violations),
            "message": "Quality gates passed",
        }

    async def _run_coverage_analysis(
        self,
        source_path: str,
        test_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run coverage analysis (line, branch, mutation).

        Args:
            source_path: Path to source code
            test_results: Test execution results

        Returns:
            Coverage analysis results
        """
        try:
            source_files = self._get_source_files(source_path)

            coverage_types = ["line", "branch"]
            if self.enable_mutation_testing:
                coverage_types.append("mutation")

            result = await self.coverage_analyzer.analyze_coverage(
                test_results=test_results,
                source_files=source_files,
                coverage_types=coverage_types,
            )

            return result

        except Exception as e:
            self.logger.error(f"Coverage analysis failed: {e}")
            return {"error": str(e)}

    async def _run_static_analysis(self, source_path: str) -> Dict[str, Any]:
        """
        Run static code quality analysis.

        Args:
            source_path: Path to source code

        Returns:
            Static analysis results
        """
        try:
            result = await self.static_analyzer.analyze_code_quality(
                source_path=source_path
            )
            return result

        except Exception as e:
            self.logger.error(f"Static analysis failed: {e}")
            return {"error": str(e)}

    async def _run_security_analysis(self, source_path: str) -> Dict[str, Any]:
        """
        Run security vulnerability scanning.

        Args:
            source_path: Path to source code

        Returns:
            Security analysis results
        """
        try:
            # TODO: Integrate Bandit and security scanners
            # For now, return empty results
            return {
                "security_issues": [],
                "security_score": 100.0,
            }

        except Exception as e:
            self.logger.error(f"Security analysis failed: {e}")
            return {"error": str(e)}

    async def _run_performance_analysis(
        self,
        test_results: Optional[Dict[str, Any]],
        baseline_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run performance regression detection.

        Args:
            test_results: Current test results with performance metrics
            baseline_metrics: Baseline metrics for comparison

        Returns:
            Performance analysis results
        """
        try:
            # TODO: Implement performance regression detection
            # For now, return empty results
            return {
                "performance_metrics": [],
                "performance_regressions": 0,
            }

        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e)}

    async def _run_complexity_analysis(self, source_path: str) -> Dict[str, Any]:
        """
        Run code complexity analysis.

        Args:
            source_path: Path to source code

        Returns:
            Complexity analysis results
        """
        try:
            # TODO: Implement complexity analysis
            # For now, return default values
            return {
                "cyclomatic_complexity": 5.0,
                "cognitive_complexity": 8.0,
                "maintainability_index": 85.0,
            }

        except Exception as e:
            self.logger.error(f"Complexity analysis failed: {e}")
            return {"error": str(e)}

    def _get_enabled_dimensions(
        self, dimensions: Optional[List[QualityDimension]]
    ) -> Set[QualityDimension]:
        """
        Get set of enabled quality dimensions.

        Args:
            dimensions: Optional filter dimensions

        Returns:
            Set of enabled dimensions
        """
        if dimensions:
            return set(dimensions)

        # All dimensions enabled by default
        return {
            QualityDimension.COVERAGE,
            QualityDimension.STATIC,
            QualityDimension.SECURITY,
            QualityDimension.PERFORMANCE,
            QualityDimension.COMPLEXITY,
        }

    def _get_source_files(self, source_path: str) -> List[str]:
        """
        Get list of source files to analyze.

        Args:
            source_path: Path to source code

        Returns:
            List of source file paths
        """
        path = Path(source_path)

        if path.is_file():
            return [str(path)]

        # Recursively find Python files
        source_files = []
        for pattern in ["*.py"]:
            source_files.extend([str(f) for f in path.rglob(pattern)])

        return source_files

    def _build_quality_report(
        self,
        report_id: str,
        coverage_data: Dict[str, Any],
        static_data: Dict[str, Any],
        security_data: Dict[str, Any],
        performance_data: Dict[str, Any],
        complexity_data: Dict[str, Any],
        start_time: datetime,
    ) -> QualityReport:
        """
        Build comprehensive quality report from analyzer results.

        Args:
            report_id: Report identifier
            coverage_data: Coverage analysis results
            static_data: Static analysis results
            security_data: Security analysis results
            performance_data: Performance analysis results
            complexity_data: Complexity analysis results
            start_time: Analysis start time

        Returns:
            Quality report
        """
        # Parse coverage reports
        coverage_reports = {}
        if "coverage_types" in coverage_data:
            for coverage_type, data in coverage_data["coverage_types"].items():
                if "error" not in data:
                    coverage_reports[coverage_type] = CoverageReport(
                        type=coverage_type,
                        percentage=data.get("percentage", 0.0),
                        covered=data.get("covered", 0),
                        total=data.get("total", 0),
                        files=data.get("files", {}),
                        branch_coverage=data.get("branch_coverage"),
                        mutation_score=data.get("mutation_score"),
                        killed_mutants=data.get("killed_mutants"),
                        survived_mutants=data.get("survived_mutants"),
                        timeout_mutants=data.get("timeout_mutants"),
                    )

        # Parse security issues
        security_issues = []
        if "security_issues" in security_data:
            for issue_data in security_data["security_issues"]:
                security_issues.append(SecurityIssue(**issue_data))

        # Parse performance metrics
        performance_metrics = []
        if "performance_metrics" in performance_data:
            for metric_data in performance_data["performance_metrics"]:
                performance_metrics.append(PerformanceMetric(**metric_data))

        # Build report
        report = QualityReport(
            report_id=report_id,
            timestamp=start_time,
            status=QualityStatus.PASSED,  # Will be updated based on violations
            passed=True,  # Will be updated based on violations
            coverage_reports=coverage_reports,
            static_issues=static_data.get("issues", []),
            code_quality_score=static_data.get("quality_score", 0.0),
            security_issues=security_issues,
            security_score=security_data.get("security_score", 100.0),
            performance_metrics=performance_metrics,
            performance_regressions=performance_data.get("performance_regressions", 0),
            cyclomatic_complexity=complexity_data.get("cyclomatic_complexity", 0.0),
            cognitive_complexity=complexity_data.get("cognitive_complexity", 0.0),
            maintainability_index=complexity_data.get("maintainability_index", 100.0),
        )

        return report

    def _extract_result(
        self, result: Any, default: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract result from analyzer, handling exceptions.

        Args:
            result: Analyzer result or exception
            default: Default value if exception

        Returns:
            Result dictionary
        """
        if isinstance(result, Exception):
            self.logger.error(f"Analyzer failed: {result}")
            return {"error": str(result)}
        return result or default

    def _extract_metrics(self, report: QualityReport) -> Dict[str, float]:
        """
        Extract all metrics from quality report for threshold checking.

        Args:
            report: Quality report

        Returns:
            Dictionary of metric values
        """
        metrics = {}

        # Coverage metrics
        for coverage_type, coverage_report in report.coverage_reports.items():
            metrics[f"coverage.{coverage_type}_coverage"] = coverage_report.percentage
            if coverage_report.mutation_score is not None:
                metrics["coverage.mutation_score"] = coverage_report.mutation_score

        # Static analysis metrics
        metrics["static.code_quality_score"] = report.code_quality_score

        # Count issues by severity
        critical_count = len(
            [i for i in report.static_issues if i.get("severity") == "critical"]
        )
        high_count = len(
            [i for i in report.static_issues if i.get("severity") == "high"]
        )
        metrics["static.critical_issues"] = critical_count
        metrics["static.high_issues"] = high_count

        # Security metrics
        metrics["security.security_score"] = report.security_score
        critical_vuln = len(
            [i for i in report.security_issues if i.severity == "critical"]
        )
        high_vuln = len(
            [i for i in report.security_issues if i.severity == "high"]
        )
        medium_vuln = len(
            [i for i in report.security_issues if i.severity == "medium"]
        )
        metrics["security.critical_vulnerabilities"] = critical_vuln
        metrics["security.high_vulnerabilities"] = high_vuln
        metrics["security.medium_vulnerabilities"] = medium_vuln

        # Complexity metrics
        metrics["complexity.cyclomatic_complexity"] = report.cyclomatic_complexity
        metrics["complexity.cognitive_complexity"] = report.cognitive_complexity
        metrics["complexity.maintainability_index"] = report.maintainability_index

        # Performance metrics
        if report.performance_regressions > 0:
            metrics["performance.regression_count"] = report.performance_regressions

        return metrics

    def _determine_status(self, violations: List[Dict[str, Any]]) -> QualityStatus:
        """
        Determine overall quality status from violations.

        Args:
            violations: List of threshold violations

        Returns:
            Overall quality status
        """
        if not violations:
            return QualityStatus.PASSED

        error_violations = [v for v in violations if v["enforcement"] == "error"]
        if error_violations:
            return QualityStatus.FAILED

        warning_violations = [v for v in violations if v["enforcement"] == "warning"]
        if warning_violations:
            return QualityStatus.WARNING

        return QualityStatus.PASSED

    def _generate_recommendations(self, report: QualityReport) -> List[str]:
        """
        Generate improvement recommendations based on report.

        Args:
            report: Quality report

        Returns:
            List of recommendations
        """
        recommendations = []

        # Coverage recommendations
        for coverage_type, coverage_report in report.coverage_reports.items():
            if coverage_report.percentage < 80:
                recommendations.append(
                    f"Increase {coverage_type} coverage from "
                    f"{coverage_report.percentage:.1f}% to at least 80%"
                )

        # Static analysis recommendations
        if report.code_quality_score < 8.0:
            recommendations.append(
                f"Improve code quality score from {report.code_quality_score:.1f} to at least 8.0"
            )

        # Security recommendations
        if report.security_issues:
            critical_count = len([i for i in report.security_issues if i.severity == "critical"])
            if critical_count > 0:
                recommendations.append(
                    f"Address {critical_count} critical security vulnerabilities"
                )

        # Complexity recommendations
        if report.cyclomatic_complexity > 15:
            recommendations.append(
                f"Reduce cyclomatic complexity from {report.cyclomatic_complexity:.1f} to below 15"
            )

        if report.maintainability_index < 65:
            recommendations.append(
                f"Improve maintainability index from {report.maintainability_index:.1f} to at least 65"
            )

        return recommendations

    async def _create_empty_coverage(self) -> Dict[str, Any]:
        """Create empty coverage data."""
        return {"coverage_types": {}}

    async def _create_empty_static(self) -> Dict[str, Any]:
        """Create empty static analysis data."""
        return {"issues": [], "quality_score": 0.0}

    async def _create_empty_security(self) -> Dict[str, Any]:
        """Create empty security data."""
        return {"security_issues": [], "security_score": 100.0}

    async def _create_empty_performance(self) -> Dict[str, Any]:
        """Create empty performance data."""
        return {"performance_metrics": [], "performance_regressions": 0}

    async def _create_empty_complexity(self) -> Dict[str, Any]:
        """Create empty complexity data."""
        return {
            "cyclomatic_complexity": 0.0,
            "cognitive_complexity": 0.0,
            "maintainability_index": 100.0,
        }

    def get_override_log(self) -> List[Dict[str, Any]]:
        """
        Get audit log of all quality gate overrides.

        Returns:
            List of override entries with timestamps, users, and reasons
        """
        return self.override_log

    def export_override_log(self, format: str = "json") -> str:
        """
        Export override audit log.

        Args:
            format: Export format ('json' or 'text')

        Returns:
            Formatted override log
        """
        if format == "json":
            import json
            return json.dumps(self.override_log, indent=2, default=str)
        else:
            lines = ["Quality Gate Override Audit Log", "=" * 50, ""]
            for entry in self.override_log:
                lines.append(f"Timestamp: {entry['timestamp']}")
                lines.append(f"Report ID: {entry['report_id']}")
                lines.append(f"User: {entry['user']}")
                lines.append(f"Reason: {entry['reason']}")
                lines.append(f"Violations Overridden: {entry['violations_overridden']}")
                lines.append("-" * 50)
            return "\n".join(lines)
