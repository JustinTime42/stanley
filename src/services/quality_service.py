"""High-level quality gate service.

This service provides a facade for quality analysis, gate enforcement, and report
generation. It integrates with the testing service, analytics, and other components
following the established service architecture patterns.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from ..models.quality_models import (
    QualityReport,
    QualityThreshold,
    QualityDimension,
    QualityStatus,
)
from ..quality.gate_engine import QualityGateEngine
from ..quality.threshold_manager import ThresholdManager
from ..quality.report_generator import ReportGenerator, ReportFormat

logger = logging.getLogger(__name__)


class QualityService:
    """
    High-level quality gate service.

    PATTERN: Service facade for quality analysis and reporting
    CRITICAL: Coordinates gate engine, threshold management, and report generation
    GOTCHA: Must handle missing dependencies gracefully

    This service provides the main entry point for quality analysis in the agent swarm.
    It orchestrates quality checks, enforces quality gates, generates reports,
    and integrates with other services like testing and analytics.
    """

    def __init__(
        self,
        gate_engine: Optional[QualityGateEngine] = None,
        threshold_manager: Optional[ThresholdManager] = None,
        report_generator: Optional[ReportGenerator] = None,
        analytics_service: Optional[Any] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize quality service.

        Args:
            gate_engine: Quality gate engine instance
            threshold_manager: Threshold configuration manager
            report_generator: Report generator instance
            analytics_service: Analytics service for tracking
            config_path: Path to quality configuration file
        """
        self.logger = logger

        # Initialize components
        self.threshold_manager = threshold_manager or ThresholdManager(
            config_path=config_path
        )
        self.gate_engine = gate_engine or QualityGateEngine(
            threshold_manager=self.threshold_manager
        )
        self.report_generator = report_generator or ReportGenerator()
        self.analytics_service = analytics_service

        # Configuration
        self.config_path = config_path
        self.default_output_dir = Path("reports/quality")
        self.default_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("QualityService initialized")

    async def check_quality(
        self,
        source_path: str,
        test_results: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
        module: Optional[str] = None,
        dimensions: Optional[List[QualityDimension]] = None,
        generate_report: bool = True,
        report_format: str = "html",
    ) -> Dict[str, Any]:
        """
        Run comprehensive quality checks on source code.

        Args:
            source_path: Path to source code to analyze
            test_results: Optional test execution results
            project: Project identifier for threshold lookup
            module: Module identifier for specific thresholds
            dimensions: Specific quality dimensions to check
            generate_report: Whether to generate report
            report_format: Format for report (html, json, markdown, all)

        Returns:
            Dictionary containing quality report and metadata
        """
        self.logger.info(f"Running quality checks on: {source_path}")
        start_time = datetime.now()

        try:
            # Load thresholds for project/module
            thresholds = await self.threshold_manager.load_thresholds_for_context(
                project=project, module=module
            )

            # Run quality analysis
            quality_report = await self.gate_engine.run_quality_checks(
                source_path=source_path,
                test_results=test_results,
                project=project,
                module=module,
                dimensions=dimensions,
            )

            # Track metrics if analytics available
            if self.analytics_service:
                await self._record_quality_metrics(quality_report, project, module)

            # Generate reports if requested
            report_paths = {}
            if generate_report:
                report_paths = await self._generate_reports(
                    quality_report, report_format
                )

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            result = {
                "success": True,
                "report": quality_report,
                "passed": quality_report.passed,
                "status": quality_report.status.value,
                "violations": len(quality_report.violations),
                "report_paths": report_paths,
                "execution_time_seconds": execution_time,
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(
                f"Quality check completed: {quality_report.status.value} "
                f"({execution_time:.2f}s)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Quality check failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "passed": False,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }

    async def enforce_gates(
        self,
        report: QualityReport,
        force: bool = False,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enforce quality gates on a report.

        Args:
            report: Quality report to check
            force: Force deployment despite violations
            reason: Reason for override (required if force=True)

        Returns:
            Dictionary with enforcement results
        """
        self.logger.info(f"Enforcing quality gates for report: {report.report_id}")

        if force and not reason:
            self.logger.error("Force override requires a reason")
            return {
                "success": False,
                "error": "Force override requires a reason",
                "passed": False,
            }

        try:
            # Get thresholds
            thresholds = self.threshold_manager.get_thresholds()

            # Enforce gates
            enforcement_result = await self.gate_engine.enforce_gates(
                report=report, thresholds=thresholds, force=force, reason=reason
            )

            # Log override if used
            if force and enforcement_result.get("override_used"):
                self.logger.warning(
                    f"Quality gate override used: {reason} "
                    f"(Report: {report.report_id})"
                )

            return {
                "success": True,
                "passed": enforcement_result["passed"],
                "violations": enforcement_result.get("violations", []),
                "override_used": enforcement_result.get("override_used", False),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Gate enforcement failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "passed": False}

    async def generate_report(
        self,
        report: QualityReport,
        format: str = "html",
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate quality report in specified format.

        Args:
            report: Quality report to generate
            format: Output format (html, json, markdown, all)
            output_path: Optional custom output path
            **kwargs: Additional format-specific options

        Returns:
            Report content or path to saved file
        """
        self.logger.info(f"Generating {format} report for {report.report_id}")

        try:
            # Generate report
            result = self.report_generator.generate(
                report=report, format=format, output_path=output_path, **kwargs
            )

            self.logger.info(f"Report generated: {format}")
            return result

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}", exc_info=True)
            raise

    async def generate_pr_comment(self, report: QualityReport) -> str:
        """
        Generate concise PR comment for quality report.

        Args:
            report: Quality report

        Returns:
            Markdown string for PR comment
        """
        self.logger.info(f"Generating PR comment for {report.report_id}")
        return self.report_generator.generate_pr_comment(report)

    async def generate_ci_output(self, report: QualityReport) -> str:
        """
        Generate CI/CD-friendly output.

        Args:
            report: Quality report

        Returns:
            JSON string for CI systems
        """
        self.logger.info(f"Generating CI output for {report.report_id}")
        return self.report_generator.generate_ci_output(report)

    async def get_quality_trend(
        self,
        project: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get quality trend analysis for a project.

        Args:
            project: Project identifier
            days: Number of days to analyze

        Returns:
            Trend analysis data
        """
        self.logger.info(f"Retrieving quality trend for {project} ({days} days)")

        if not self.analytics_service:
            self.logger.warning("Analytics service not available")
            return {"error": "Analytics service not available"}

        try:
            # Get trend from analytics
            trend = await self.analytics_service.get_quality_trend(
                project_id=project, days=days
            )

            return trend

        except Exception as e:
            self.logger.error(f"Failed to get quality trend: {e}", exc_info=True)
            return {"error": str(e)}

    async def check_with_testing_integration(
        self,
        source_files: List[str],
        test_files: Optional[List[str]] = None,
        run_tests: bool = True,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run quality checks integrated with test execution.

        Args:
            source_files: Source files to analyze
            test_files: Test files to run (auto-discovered if None)
            run_tests: Whether to run tests first
            project: Project identifier

        Returns:
            Combined testing and quality results
        """
        self.logger.info(
            f"Running integrated quality and testing checks for {len(source_files)} files"
        )

        try:
            test_results = None

            # Run tests if requested
            if run_tests:
                # Import here to avoid circular dependency
                from .testing_service import TestingOrchestrator

                testing_service = TestingOrchestrator()

                if test_files:
                    test_results = await testing_service.run_tests(test_files)
                else:
                    self.logger.info("No test files provided, skipping test execution")

            # Run quality checks on all source files
            quality_results = []
            for source_file in source_files:
                result = await self.check_quality(
                    source_path=source_file,
                    test_results=test_results,
                    project=project,
                    generate_report=False,  # Generate combined report later
                )
                quality_results.append(result)

            # Determine overall status
            all_passed = all(r.get("passed", False) for r in quality_results)

            # Generate combined report if all results available
            combined_report_path = None
            if quality_results and all(r.get("report") for r in quality_results):
                # Use the first report as base (could merge in future)
                first_report = quality_results[0]["report"]
                combined_report_path = self.report_generator.generate_and_save(
                    first_report, format="html", filename="combined_quality_report"
                )

            return {
                "success": True,
                "passed": all_passed,
                "test_results": test_results,
                "quality_results": quality_results,
                "combined_report": combined_report_path,
                "summary": {
                    "total_files": len(source_files),
                    "passed_files": sum(1 for r in quality_results if r.get("passed")),
                    "failed_files": sum(
                        1 for r in quality_results if not r.get("passed")
                    ),
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Integrated check failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "passed": False}

    async def configure_thresholds(
        self,
        project: Optional[str] = None,
        module: Optional[str] = None,
        thresholds: Optional[List[QualityThreshold]] = None,
        config_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Configure quality thresholds.

        Args:
            project: Project identifier
            module: Module identifier
            thresholds: List of threshold configurations
            config_file: Path to configuration file

        Returns:
            Configuration result
        """
        self.logger.info(f"Configuring thresholds for project={project}, module={module}")

        try:
            if config_file:
                # Load from file
                loaded = await self.threshold_manager.load_from_file(config_file)
                return {"success": True, "loaded": loaded, "source": "file"}

            elif thresholds:
                # Set programmatically
                for threshold in thresholds:
                    await self.threshold_manager.set_threshold(
                        threshold, project=project, module=module
                    )

                return {
                    "success": True,
                    "configured": len(thresholds),
                    "source": "programmatic",
                }

            else:
                return {"success": False, "error": "No thresholds or config file provided"}

        except Exception as e:
            self.logger.error(f"Threshold configuration failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_service_stats(self) -> Dict[str, Any]:
        """
        Get quality service statistics.

        Returns:
            Service statistics
        """
        # Get thresholds count
        try:
            thresholds = self.threshold_manager.get_thresholds()
            threshold_count = len(thresholds) if thresholds else 0
        except Exception:
            threshold_count = 0

        stats = {
            "service": "QualityService",
            "gate_engine": {
                "mutation_testing_enabled": self.gate_engine.enable_mutation_testing,
                "security_scanning_enabled": self.gate_engine.enable_security_scanning,
                "performance_analysis_enabled": self.gate_engine.enable_performance_analysis,
            },
            "threshold_manager": {
                "total_thresholds": threshold_count,
            },
            "report_generator": {
                "output_dir": str(self.report_generator.output_dir),
                "available_formats": self.report_generator.get_available_formats(),
            },
            "analytics_enabled": self.analytics_service is not None,
        }

        return stats

    async def _generate_reports(
        self, report: QualityReport, format: str
    ) -> Dict[str, str]:
        """Generate reports in requested format(s)."""
        report_paths = {}

        try:
            if format == "all":
                # Generate all formats
                for fmt in ["html", "json", "markdown"]:
                    path = self.report_generator.generate_and_save(
                        report, format=fmt, filename=f"quality_report_{report.report_id}"
                    )
                    report_paths[fmt] = path
            else:
                # Generate single format
                path = self.report_generator.generate_and_save(
                    report, format=format, filename=f"quality_report_{report.report_id}"
                )
                report_paths[format] = path

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}", exc_info=True)

        return report_paths

    async def _record_quality_metrics(
        self, report: QualityReport, project: Optional[str], module: Optional[str]
    ) -> None:
        """Record quality metrics to analytics service."""
        if not self.analytics_service:
            return

        try:
            # Extract key metrics
            metrics = {
                "project": project or "unknown",
                "module": module or "unknown",
                "timestamp": report.timestamp,
                "status": report.status.value,
                "passed": report.passed,
                "coverage_average": (
                    sum(cr.percentage for cr in report.coverage_reports.values())
                    / len(report.coverage_reports)
                    if report.coverage_reports
                    else 0.0
                ),
                "security_score": report.security_score,
                "code_quality_score": report.code_quality_score,
                "maintainability_index": report.maintainability_index,
                "violations": len(report.violations),
                "security_issues": len(report.security_issues),
                "performance_regressions": report.performance_regressions,
            }

            # Record to analytics
            await self.analytics_service.record_quality_metrics(metrics)

        except Exception as e:
            self.logger.warning(f"Failed to record quality metrics: {e}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Cleanup old reports
        try:
            deleted = self.report_generator.cleanup_old_reports(keep_count=50)
            self.logger.info(f"Cleaned up {deleted} old reports")
        except Exception as e:
            self.logger.warning(f"Report cleanup failed: {e}")

        self.logger.info("QualityService cleaned up")

    async def close(self) -> None:
        """Alias for cleanup()."""
        await self.cleanup()
