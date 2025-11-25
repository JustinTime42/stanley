"""JSON report generator for quality analysis.

This module generates structured JSON reports for API consumption and integration
with external systems. Provides complete quality data in machine-readable format.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from ...models.quality_models import (
    QualityReport,
    CoverageType,
    SecurityIssue,
    PerformanceMetric,
)

logger = logging.getLogger(__name__)


class JsonReporter:
    """
    Generate JSON reports for API consumption.

    PATTERN: Structured JSON output for programmatic access
    CRITICAL: Ensures complete data serialization
    GOTCHA: Handles datetime and enum serialization
    """

    def __init__(self, pretty: bool = True):
        """
        Initialize JSON reporter.

        Args:
            pretty: Whether to pretty-print JSON output
        """
        self.pretty = pretty
        self.logger = logger
        self.logger.info("JsonReporter initialized")

    def generate_report(self, report: QualityReport) -> str:
        """
        Generate JSON report from quality analysis.

        Args:
            report: Quality report to serialize

        Returns:
            JSON string
        """
        self.logger.info(f"Generating JSON report: {report.report_id}")

        # Convert to dictionary
        report_dict = self._report_to_dict(report)

        # Serialize to JSON
        if self.pretty:
            json_str = json.dumps(report_dict, indent=2, default=str)
        else:
            json_str = json.dumps(report_dict, default=str)

        self.logger.info(f"JSON report generated ({len(json_str)} bytes)")
        return json_str

    def _report_to_dict(self, report: QualityReport) -> Dict[str, Any]:
        """
        Convert quality report to dictionary.

        Args:
            report: Quality report

        Returns:
            Dictionary representation
        """
        return {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "status": {
                "overall": report.status.value,
                "passed": report.passed,
                "trend": report.trend,
            },
            "coverage": self._serialize_coverage(report),
            "security": self._serialize_security(report),
            "static_analysis": {
                "code_quality_score": report.code_quality_score,
                "total_issues": len(report.static_issues),
                "issues": report.static_issues,
            },
            "performance": self._serialize_performance(report),
            "complexity": {
                "cyclomatic_complexity": report.cyclomatic_complexity,
                "cognitive_complexity": report.cognitive_complexity,
                "maintainability_index": report.maintainability_index,
            },
            "violations": report.violations,
            "recommendations": report.recommendations,
            "metadata": {
                "previous_score": report.previous_score,
                "generation_timestamp": datetime.now().isoformat(),
            },
        }

    def _serialize_coverage(self, report: QualityReport) -> Dict[str, Any]:
        """Serialize coverage data."""
        coverage_data = {}

        for cov_type, cov_report in report.coverage_reports.items():
            coverage_data[cov_type.value] = {
                "percentage": cov_report.percentage,
                "covered": cov_report.covered,
                "total": cov_report.total,
                "files": cov_report.files,
            }

            # Add type-specific data
            if cov_type == CoverageType.BRANCH and cov_report.branch_coverage:
                coverage_data[cov_type.value]["branch_details"] = (
                    cov_report.branch_coverage
                )

            if cov_type == CoverageType.MUTATION:
                coverage_data[cov_type.value]["mutation"] = {
                    "score": cov_report.mutation_score,
                    "killed": cov_report.killed_mutants,
                    "survived": cov_report.survived_mutants,
                    "timeout": cov_report.timeout_mutants,
                }

        # Calculate summary
        if coverage_data:
            avg_coverage = sum(
                data["percentage"] for data in coverage_data.values()
            ) / len(coverage_data)
            coverage_data["summary"] = {
                "average_coverage": round(avg_coverage, 2),
                "types_analyzed": len(coverage_data),
            }

        return coverage_data

    def _serialize_security(self, report: QualityReport) -> Dict[str, Any]:
        """Serialize security data."""
        # Group issues by severity
        issues_by_severity = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": [],
        }

        for issue in report.security_issues:
            issue_dict = {
                "issue_id": issue.issue_id,
                "type": issue.issue_type,
                "description": issue.description,
                "severity": issue.severity.value,
                "confidence": issue.confidence,
                "location": {
                    "file": issue.file_path,
                    "line": issue.line_number,
                    "column": issue.column,
                },
                "remediation": issue.remediation,
                "references": {
                    "cwe_id": issue.cwe_id,
                    "owasp_category": issue.owasp_category,
                    "links": issue.references,
                },
            }
            issues_by_severity[issue.severity.value].append(issue_dict)

        return {
            "security_score": report.security_score,
            "total_issues": len(report.security_issues),
            "issues_by_severity": {
                severity: {
                    "count": len(issues),
                    "issues": issues,
                }
                for severity, issues in issues_by_severity.items()
                if issues
            },
            "summary": {
                "critical": len(issues_by_severity["critical"]),
                "high": len(issues_by_severity["high"]),
                "medium": len(issues_by_severity["medium"]),
                "low": len(issues_by_severity["low"]),
                "info": len(issues_by_severity["info"]),
            },
        }

    def _serialize_performance(self, report: QualityReport) -> Dict[str, Any]:
        """Serialize performance data."""
        metrics_data = []
        regressions = []

        for metric in report.performance_metrics:
            metric_dict = {
                "name": metric.metric_name,
                "test": metric.test_name,
                "current_value": metric.current_value,
                "baseline_value": metric.baseline_value,
                "regression_detected": metric.regression_detected,
                "regression_percentage": metric.regression_percentage,
                "statistical_data": {
                    "confidence_interval": metric.confidence_interval,
                    "p_value": metric.p_value,
                },
                "environment": metric.environment,
            }
            metrics_data.append(metric_dict)

            if metric.regression_detected:
                regressions.append(metric_dict)

        return {
            "total_regressions": report.performance_regressions,
            "metrics_analyzed": len(metrics_data),
            "regressions": regressions,
            "all_metrics": metrics_data,
        }

    def generate_metrics_only(self, report: QualityReport) -> str:
        """
        Generate simplified JSON with only metrics (no detailed issues).

        Args:
            report: Quality report

        Returns:
            JSON string with summary metrics only
        """
        metrics = {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "passed": report.passed,
            "status": report.status.value,
            "metrics": {
                "coverage_average": (
                    sum(cr.percentage for cr in report.coverage_reports.values())
                    / len(report.coverage_reports)
                    if report.coverage_reports
                    else 0.0
                ),
                "security_score": report.security_score,
                "code_quality_score": report.code_quality_score,
                "maintainability_index": report.maintainability_index,
                "performance_regressions": report.performance_regressions,
            },
            "counts": {
                "violations": len(report.violations),
                "security_issues": len(report.security_issues),
                "static_issues": len(report.static_issues),
                "performance_metrics": len(report.performance_metrics),
            },
        }

        if self.pretty:
            return json.dumps(metrics, indent=2, default=str)
        else:
            return json.dumps(metrics, default=str)

    def generate_api_response(
        self, report: QualityReport, include_details: bool = True
    ) -> Dict[str, Any]:
        """
        Generate API-friendly response format.

        Args:
            report: Quality report
            include_details: Whether to include detailed issue lists

        Returns:
            Dictionary suitable for API response
        """
        response = {
            "success": True,
            "data": {
                "report_id": report.report_id,
                "timestamp": report.timestamp.isoformat(),
                "quality_gate": {
                    "passed": report.passed,
                    "status": report.status.value,
                    "violations_count": len(report.violations),
                },
                "scores": {
                    "overall": self._calculate_overall_score(report),
                    "coverage": (
                        sum(cr.percentage for cr in report.coverage_reports.values())
                        / len(report.coverage_reports)
                        if report.coverage_reports
                        else 0.0
                    ),
                    "security": report.security_score,
                    "code_quality": report.code_quality_score,
                    "maintainability": report.maintainability_index,
                },
                "summary": {
                    "coverage_types": len(report.coverage_reports),
                    "security_issues": len(report.security_issues),
                    "static_issues": len(report.static_issues),
                    "performance_regressions": report.performance_regressions,
                    "recommendations": len(report.recommendations),
                },
            },
        }

        if include_details:
            response["data"]["details"] = self._report_to_dict(report)

        return response

    def _calculate_overall_score(self, report: QualityReport) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            "coverage": 0.25,
            "security": 0.30,
            "code_quality": 0.25,
            "maintainability": 0.20,
        }

        # Calculate coverage average
        coverage_score = (
            sum(cr.percentage for cr in report.coverage_reports.values())
            / len(report.coverage_reports)
            if report.coverage_reports
            else 0.0
        )

        overall = (
            weights["coverage"] * coverage_score
            + weights["security"] * report.security_score
            + weights["code_quality"] * report.code_quality_score
            + weights["maintainability"] * report.maintainability_index
        )

        return round(overall, 2)

    def save_report(
        self,
        report: QualityReport,
        output_path: str,
        metrics_only: bool = False,
    ) -> str:
        """
        Generate and save JSON report to file.

        Args:
            report: Quality report to serialize
            output_path: Path to save JSON file
            metrics_only: Whether to save only metrics summary

        Returns:
            Path to saved file
        """
        if metrics_only:
            json_str = self.generate_metrics_only(report)
        else:
            json_str = self.generate_report(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)

        self.logger.info(f"JSON report saved to: {output_path}")
        return output_path

    def generate_ci_output(self, report: QualityReport) -> str:
        """
        Generate CI/CD-friendly JSON output.

        Args:
            report: Quality report

        Returns:
            JSON string formatted for CI systems
        """
        ci_data = {
            "status": "passed" if report.passed else "failed",
            "exit_code": 0 if report.passed else 1,
            "summary": f"Quality gate {'PASSED' if report.passed else 'FAILED'}",
            "metrics": {
                "coverage": (
                    sum(cr.percentage for cr in report.coverage_reports.values())
                    / len(report.coverage_reports)
                    if report.coverage_reports
                    else 0.0
                ),
                "security_score": report.security_score,
                "violations": len(report.violations),
            },
            "details_url": f"reports/{report.report_id}.html",
            "timestamp": report.timestamp.isoformat(),
        }

        return json.dumps(ci_data, indent=2)

    def export_for_integration(
        self, report: QualityReport, system: str = "generic"
    ) -> str:
        """
        Export report in format suitable for external system integration.

        Args:
            report: Quality report
            system: Target system (generic, sonarqube, codecov, etc.)

        Returns:
            JSON string in system-specific format
        """
        if system == "sonarqube":
            return self._export_sonarqube_format(report)
        elif system == "codecov":
            return self._export_codecov_format(report)
        else:
            return self.generate_report(report)

    def _export_sonarqube_format(self, report: QualityReport) -> str:
        """Export in SonarQube-compatible format."""
        # Simplified SonarQube format
        sonar_data = {
            "projectKey": report.report_id,
            "measures": [
                {
                    "metric": "coverage",
                    "value": (
                        sum(cr.percentage for cr in report.coverage_reports.values())
                        / len(report.coverage_reports)
                        if report.coverage_reports
                        else 0.0
                    ),
                },
                {
                    "metric": "security_rating",
                    "value": self._convert_to_rating(report.security_score),
                },
                {
                    "metric": "sqale_rating",
                    "value": self._convert_to_rating(report.maintainability_index),
                },
            ],
            "issues": [
                {
                    "severity": issue.severity.value.upper(),
                    "component": issue.file_path,
                    "line": issue.line_number,
                    "message": issue.description,
                }
                for issue in report.security_issues[:100]  # Limit for performance
            ],
        }

        return json.dumps(sonar_data, indent=2)

    def _export_codecov_format(self, report: QualityReport) -> str:
        """Export in Codecov-compatible format."""
        codecov_data = {
            "coverage": {
                "totals": {
                    "lines": (
                        report.coverage_reports[CoverageType.LINE].percentage
                        if CoverageType.LINE in report.coverage_reports
                        else 0.0
                    ),
                    "branches": (
                        report.coverage_reports[CoverageType.BRANCH].percentage
                        if CoverageType.BRANCH in report.coverage_reports
                        else 0.0
                    ),
                },
                "files": {},
            }
        }

        # Add per-file coverage
        for cov_type, cov_report in report.coverage_reports.items():
            for file_path, file_data in cov_report.files.items():
                if file_path not in codecov_data["coverage"]["files"]:
                    codecov_data["coverage"]["files"][file_path] = {}
                codecov_data["coverage"]["files"][file_path][cov_type.value] = file_data

        return json.dumps(codecov_data, indent=2)

    def _convert_to_rating(self, score: float) -> str:
        """Convert score to A-E rating."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "E"
