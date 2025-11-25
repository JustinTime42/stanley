"""HTML report generator for quality analysis.

This module generates visual HTML reports with dashboards for quality analysis results.
Uses template-based approach for clean, maintainable report generation.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ...models.quality_models import (
    QualityReport,
    QualityStatus,
    CoverageType,
    SeverityLevel,
)

logger = logging.getLogger(__name__)


class HtmlReporter:
    """
    Generate HTML quality reports with dashboard.

    PATTERN: Template-based HTML generation
    CRITICAL: Creates visual dashboards for human review
    GOTCHA: Handles missing data gracefully
    """

    def __init__(self):
        """Initialize HTML reporter."""
        self.logger = logger
        self.logger.info("HtmlReporter initialized")

    def generate_report(self, report: QualityReport) -> str:
        """
        Generate HTML report from quality analysis.

        Args:
            report: Quality report to render

        Returns:
            HTML string
        """
        self.logger.info(f"Generating HTML report: {report.report_id}")

        html = self._generate_html_structure(report)

        self.logger.info(f"HTML report generated ({len(html)} bytes)")
        return html

    def _generate_html_structure(self, report: QualityReport) -> str:
        """Generate complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Report - {report.report_id}</title>
    {self._generate_styles()}
</head>
<body>
    <div class="container">
        {self._generate_header(report)}
        {self._generate_summary_dashboard(report)}
        {self._generate_coverage_section(report)}
        {self._generate_security_section(report)}
        {self._generate_static_analysis_section(report)}
        {self._generate_performance_section(report)}
        {self._generate_complexity_section(report)}
        {self._generate_violations_section(report)}
        {self._generate_recommendations_section(report)}
        {self._generate_footer(report)}
    </div>
</body>
</html>"""

    def _generate_styles(self) -> str:
        """Generate CSS styles."""
        return """<style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header .meta {
            opacity: 0.9;
            font-size: 0.9em;
        }

        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }

        .metric-card h3 {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .metric-value.passed {
            color: #10b981;
        }

        .metric-value.warning {
            color: #f59e0b;
        }

        .metric-value.failed {
            color: #ef4444;
        }

        .section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .section h2 {
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }

        .progress-bar {
            background: #e5e7eb;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
            position: relative;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.3s;
        }

        .progress-fill.low {
            background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
        }

        .progress-fill.medium {
            background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
        }

        .issue-list {
            list-style: none;
        }

        .issue-item {
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            background: #f9fafb;
            border-radius: 5px;
        }

        .issue-item.critical {
            border-left-color: #ef4444;
            background: #fef2f2;
        }

        .issue-item.high {
            border-left-color: #f59e0b;
            background: #fffbeb;
        }

        .issue-item.medium {
            border-left-color: #3b82f6;
            background: #eff6ff;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }

        .badge.critical {
            background: #ef4444;
            color: white;
        }

        .badge.high {
            background: #f59e0b;
            color: white;
        }

        .badge.medium {
            background: #3b82f6;
            color: white;
        }

        .badge.low {
            background: #10b981;
            color: white;
        }

        .badge.info {
            background: #6b7280;
            color: white;
        }

        .recommendations {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }

        .recommendations li {
            margin: 8px 0;
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: #6b7280;
            font-size: 0.9em;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }

        th {
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }

        tr:hover {
            background: #f9fafb;
        }

        .status-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2em;
        }

        .status-badge.passed {
            background: #d1fae5;
            color: #065f46;
        }

        .status-badge.warning {
            background: #fef3c7;
            color: #92400e;
        }

        .status-badge.failed {
            background: #fee2e2;
            color: #991b1b;
        }
    </style>"""

    def _generate_header(self, report: QualityReport) -> str:
        """Generate report header."""
        status_class = report.status.value
        status_symbol = "✓" if report.passed else "✗"

        return f"""<div class="header">
        <h1>Quality Analysis Report</h1>
        <div class="meta">
            <p><strong>Report ID:</strong> {report.report_id}</p>
            <p><strong>Generated:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Status:</strong> <span class="status-badge {status_class}">{status_symbol} {report.status.value.upper()}</span></p>
            <p><strong>Trend:</strong> {report.trend}</p>
        </div>
    </div>"""

    def _generate_summary_dashboard(self, report: QualityReport) -> str:
        """Generate summary dashboard with key metrics."""
        # Calculate overall coverage
        coverage_avg = 0.0
        if report.coverage_reports:
            coverage_avg = sum(
                cr.percentage for cr in report.coverage_reports.values()
            ) / len(report.coverage_reports)

        # Count issues by severity
        critical_issues = sum(
            1 for issue in report.security_issues
            if issue.severity == SeverityLevel.CRITICAL
        )
        high_issues = sum(
            1 for issue in report.security_issues
            if issue.severity == SeverityLevel.HIGH
        )

        return f"""<div class="dashboard">
        <div class="metric-card">
            <h3>Overall Status</h3>
            <div class="metric-value {report.status.value}">
                {"PASS" if report.passed else "FAIL"}
            </div>
            <p>{len(report.violations)} violations</p>
        </div>

        <div class="metric-card">
            <h3>Test Coverage</h3>
            <div class="metric-value {'passed' if coverage_avg >= 80 else 'warning' if coverage_avg >= 60 else 'failed'}">
                {coverage_avg:.1f}%
            </div>
            <p>Average coverage</p>
        </div>

        <div class="metric-card">
            <h3>Security Score</h3>
            <div class="metric-value {'passed' if report.security_score >= 90 else 'warning' if report.security_score >= 70 else 'failed'}">
                {report.security_score:.1f}
            </div>
            <p>{len(report.security_issues)} issues found</p>
        </div>

        <div class="metric-card">
            <h3>Code Quality</h3>
            <div class="metric-value {'passed' if report.code_quality_score >= 80 else 'warning' if report.code_quality_score >= 60 else 'failed'}">
                {report.code_quality_score:.1f}
            </div>
            <p>{len(report.static_issues)} static issues</p>
        </div>

        <div class="metric-card">
            <h3>Performance</h3>
            <div class="metric-value {'passed' if report.performance_regressions == 0 else 'failed'}">
                {report.performance_regressions}
            </div>
            <p>Regressions detected</p>
        </div>

        <div class="metric-card">
            <h3>Maintainability</h3>
            <div class="metric-value {'passed' if report.maintainability_index >= 80 else 'warning' if report.maintainability_index >= 60 else 'failed'}">
                {report.maintainability_index:.1f}
            </div>
            <p>Maintainability index</p>
        </div>
    </div>"""

    def _generate_coverage_section(self, report: QualityReport) -> str:
        """Generate coverage analysis section."""
        if not report.coverage_reports:
            return ""

        coverage_html = '<div class="section"><h2>Test Coverage Analysis</h2>'

        for cov_type, cov_report in report.coverage_reports.items():
            percentage = cov_report.percentage
            bar_class = (
                "low" if percentage < 60 else "medium" if percentage < 80 else ""
            )

            coverage_html += f"""
            <div style="margin-bottom: 25px;">
                <h3>{cov_type.value.title()} Coverage</h3>
                <div class="progress-bar">
                    <div class="progress-fill {bar_class}" style="width: {percentage}%">
                        {percentage:.1f}%
                    </div>
                </div>
                <p><strong>{cov_report.covered}</strong> of <strong>{cov_report.total}</strong> {cov_type.value} covered</p>
            """

            # Add mutation-specific details
            if cov_type == CoverageType.MUTATION and cov_report.mutation_score:
                coverage_html += f"""
                <p style="margin-top: 10px;">
                    <strong>Mutation Score:</strong> {cov_report.mutation_score:.1f}% |
                    Killed: {cov_report.killed_mutants or 0} |
                    Survived: {cov_report.survived_mutants or 0} |
                    Timeout: {cov_report.timeout_mutants or 0}
                </p>
                """

            coverage_html += "</div>"

        coverage_html += "</div>"
        return coverage_html

    def _generate_security_section(self, report: QualityReport) -> str:
        """Generate security analysis section."""
        if not report.security_issues:
            return '<div class="section"><h2>Security Analysis</h2><p>No security issues detected. ✓</p></div>'

        issues_html = '<div class="section"><h2>Security Analysis</h2><ul class="issue-list">'

        # Sort by severity
        sorted_issues = sorted(
            report.security_issues,
            key=lambda x: ["critical", "high", "medium", "low", "info"].index(
                x.severity.value
            ),
        )

        for issue in sorted_issues:
            issues_html += f"""
            <li class="issue-item {issue.severity.value}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong>{issue.issue_type}</strong>
                    <span class="badge {issue.severity.value}">{issue.severity.value}</span>
                </div>
                <p>{issue.description}</p>
                <p style="margin-top: 10px; color: #6b7280;">
                    <strong>File:</strong> {issue.file_path}:{issue.line_number} |
                    <strong>Confidence:</strong> {issue.confidence}
                </p>
                <p style="margin-top: 5px; color: #059669;"><strong>Fix:</strong> {issue.remediation}</p>
            </li>
            """

        issues_html += "</ul></div>"
        return issues_html

    def _generate_static_analysis_section(self, report: QualityReport) -> str:
        """Generate static analysis section."""
        if not report.static_issues:
            return '<div class="section"><h2>Static Analysis</h2><p>No static analysis issues detected. ✓</p></div>'

        html = f"""<div class="section">
        <h2>Static Analysis</h2>
        <p><strong>Code Quality Score:</strong> {report.code_quality_score:.1f}/100</p>
        <p><strong>Total Issues:</strong> {len(report.static_issues)}</p>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Message</th>
                    <th>Location</th>
                </tr>
            </thead>
            <tbody>
        """

        for issue in report.static_issues[:20]:  # Limit to first 20
            category = issue.get("category", "Unknown")
            message = issue.get("message", "No message")
            location = issue.get("location", "Unknown")

            html += f"""
                <tr>
                    <td>{category}</td>
                    <td>{message}</td>
                    <td>{location}</td>
                </tr>
            """

        if len(report.static_issues) > 20:
            html += f"""
                <tr>
                    <td colspan="3" style="text-align: center; color: #6b7280;">
                        ... and {len(report.static_issues) - 20} more issues
                    </td>
                </tr>
            """

        html += "</tbody></table></div>"
        return html

    def _generate_performance_section(self, report: QualityReport) -> str:
        """Generate performance analysis section."""
        if not report.performance_metrics:
            return ""

        html = '<div class="section"><h2>Performance Analysis</h2>'

        if report.performance_regressions > 0:
            html += f'<p style="color: #ef4444; font-weight: bold;">⚠ {report.performance_regressions} performance regression(s) detected!</p>'

        html += "<table><thead><tr><th>Metric</th><th>Current</th><th>Baseline</th><th>Change</th></tr></thead><tbody>"

        for metric in report.performance_metrics:
            if metric.regression_detected:
                change_color = "color: #ef4444;"
                change_symbol = "↑"
            else:
                change_color = "color: #10b981;"
                change_symbol = "↓"

            html += f"""
                <tr>
                    <td>{metric.metric_name}</td>
                    <td>{metric.current_value:.2f}</td>
                    <td>{metric.baseline_value or 'N/A'}</td>
                    <td style="{change_color}">
                        {change_symbol} {metric.regression_percentage or 0:.1f}%
                    </td>
                </tr>
            """

        html += "</tbody></table></div>"
        return html

    def _generate_complexity_section(self, report: QualityReport) -> str:
        """Generate code complexity section."""
        return f"""<div class="section">
        <h2>Code Complexity Metrics</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div>
                <h4>Cyclomatic Complexity</h4>
                <p style="font-size: 2em; font-weight: bold; color: {'#10b981' if report.cyclomatic_complexity <= 10 else '#f59e0b' if report.cyclomatic_complexity <= 15 else '#ef4444'};">
                    {report.cyclomatic_complexity:.1f}
                </p>
            </div>
            <div>
                <h4>Cognitive Complexity</h4>
                <p style="font-size: 2em; font-weight: bold; color: {'#10b981' if report.cognitive_complexity <= 15 else '#f59e0b' if report.cognitive_complexity <= 25 else '#ef4444'};">
                    {report.cognitive_complexity:.1f}
                </p>
            </div>
            <div>
                <h4>Maintainability Index</h4>
                <p style="font-size: 2em; font-weight: bold; color: {'#10b981' if report.maintainability_index >= 80 else '#f59e0b' if report.maintainability_index >= 60 else '#ef4444'};">
                    {report.maintainability_index:.1f}
                </p>
            </div>
        </div>
    </div>"""

    def _generate_violations_section(self, report: QualityReport) -> str:
        """Generate quality gate violations section."""
        if not report.violations:
            return ""

        html = '<div class="section"><h2>Quality Gate Violations</h2><ul class="issue-list">'

        for violation in report.violations:
            threshold = violation.get("threshold", {})
            actual = violation.get("actual", "Unknown")
            severity = violation.get("severity", "error")

            html += f"""
            <li class="issue-item {severity}">
                <strong>{threshold.get('dimension', 'Unknown')} - {threshold.get('metric', 'Unknown')}</strong>
                <p>Expected: {threshold.get('min_value', 'N/A')} | Actual: {actual}</p>
                <span class="badge {severity}">{severity}</span>
            </li>
            """

        html += "</ul></div>"
        return html

    def _generate_recommendations_section(self, report: QualityReport) -> str:
        """Generate recommendations section."""
        if not report.recommendations:
            return ""

        html = """<div class="section">
        <h2>Recommendations</h2>
        <div class="recommendations">
            <ul>"""

        for rec in report.recommendations:
            html += f"<li>{rec}</li>"

        html += "</ul></div></div>"
        return html

    def _generate_footer(self, report: QualityReport) -> str:
        """Generate report footer."""
        return f"""<div class="footer">
        <p>Generated by Agent Swarm Quality Gate System</p>
        <p>Report ID: {report.report_id} | {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>"""

    def save_report(self, report: QualityReport, output_path: str) -> str:
        """
        Generate and save HTML report to file.

        Args:
            report: Quality report to render
            output_path: Path to save HTML file

        Returns:
            Path to saved file
        """
        html = self.generate_report(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        self.logger.info(f"HTML report saved to: {output_path}")
        return output_path
