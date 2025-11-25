"""Markdown report generator for quality analysis.

This module generates GitHub-flavored markdown reports suitable for PR comments,
documentation, and developer-friendly summaries.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from ...models.quality_models import (
    QualityReport,
    QualityStatus,
    CoverageType,
    SeverityLevel,
)

logger = logging.getLogger(__name__)


class MarkdownReporter:
    """
    Generate Markdown reports for PR comments and documentation.

    PATTERN: GitHub-flavored markdown with tables and badges
    CRITICAL: Formats for readability in PR comments
    GOTCHA: Handles special characters in markdown
    """

    def __init__(self, include_badges: bool = True):
        """
        Initialize Markdown reporter.

        Args:
            include_badges: Whether to include visual badges
        """
        self.include_badges = include_badges
        self.logger = logger
        self.logger.info("MarkdownReporter initialized")

    def generate_report(self, report: QualityReport) -> str:
        """
        Generate full Markdown report from quality analysis.

        Args:
            report: Quality report to render

        Returns:
            Markdown string
        """
        self.logger.info(f"Generating Markdown report: {report.report_id}")

        sections = [
            self._generate_header(report),
            self._generate_summary(report),
            self._generate_coverage_section(report),
            self._generate_security_section(report),
            self._generate_static_analysis_section(report),
            self._generate_performance_section(report),
            self._generate_complexity_section(report),
            self._generate_violations_section(report),
            self._generate_recommendations_section(report),
            self._generate_footer(report),
        ]

        markdown = "\n\n".join(filter(None, sections))

        self.logger.info(f"Markdown report generated ({len(markdown)} chars)")
        return markdown

    def _generate_header(self, report: QualityReport) -> str:
        """Generate report header with status badge."""
        status_emoji = "✅" if report.passed else "❌"
        status_badge = self._create_status_badge(report)

        header = f"# Quality Analysis Report {status_emoji}\n\n"

        if self.include_badges:
            header += f"{status_badge}\n\n"

        header += f"""**Report ID:** `{report.report_id}`
**Generated:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Status:** {report.status.value.upper()}
**Trend:** {self._trend_emoji(report.trend)} {report.trend}
"""
        return header

    def _generate_summary(self, report: QualityReport) -> str:
        """Generate executive summary with key metrics."""
        # Calculate overall coverage
        coverage_avg = 0.0
        if report.coverage_reports:
            coverage_avg = sum(
                cr.percentage for cr in report.coverage_reports.values()
            ) / len(report.coverage_reports)

        summary = "## Summary\n\n"
        summary += "| Metric | Score | Status |\n"
        summary += "|--------|-------|--------|\n"

        # Coverage
        summary += f"| Test Coverage | {coverage_avg:.1f}% | {self._status_emoji(coverage_avg, 80, 60)} |\n"

        # Security
        summary += f"| Security Score | {report.security_score:.1f} | {self._status_emoji(report.security_score, 90, 70)} |\n"

        # Code Quality
        summary += f"| Code Quality | {report.code_quality_score:.1f} | {self._status_emoji(report.code_quality_score, 80, 60)} |\n"

        # Maintainability
        summary += f"| Maintainability | {report.maintainability_index:.1f} | {self._status_emoji(report.maintainability_index, 80, 60)} |\n"

        # Performance
        perf_status = "✅" if report.performance_regressions == 0 else "❌"
        summary += f"| Performance | {report.performance_regressions} regressions | {perf_status} |\n"

        return summary

    def _generate_coverage_section(self, report: QualityReport) -> str:
        """Generate coverage analysis section."""
        if not report.coverage_reports:
            return ""

        section = "## Test Coverage\n\n"

        for cov_type, cov_report in report.coverage_reports.items():
            percentage = cov_report.percentage
            bar = self._create_progress_bar(percentage)
            status = self._status_emoji(percentage, 80, 60)

            section += f"### {cov_type.value.title()} Coverage {status}\n\n"
            section += f"{bar} **{percentage:.1f}%**\n\n"
            section += f"- **Covered:** {cov_report.covered}/{cov_report.total}\n"

            # Add mutation-specific details
            if cov_type == CoverageType.MUTATION and cov_report.mutation_score:
                section += f"- **Mutation Score:** {cov_report.mutation_score:.1f}%\n"
                section += f"- **Killed Mutants:** {cov_report.killed_mutants or 0}\n"
                section += f"- **Survived Mutants:** {cov_report.survived_mutants or 0}\n"
                section += f"- **Timeout Mutants:** {cov_report.timeout_mutants or 0}\n"

            section += "\n"

        return section

    def _generate_security_section(self, report: QualityReport) -> str:
        """Generate security analysis section."""
        if not report.security_issues:
            return "## Security Analysis\n\n✅ No security issues detected!\n"

        # Group by severity
        by_severity = {
            SeverityLevel.CRITICAL: [],
            SeverityLevel.HIGH: [],
            SeverityLevel.MEDIUM: [],
            SeverityLevel.LOW: [],
            SeverityLevel.INFO: [],
        }

        for issue in report.security_issues:
            by_severity[issue.severity].append(issue)

        section = f"## Security Analysis\n\n**Security Score:** {report.security_score:.1f}/100\n\n"

        # Summary table
        section += "| Severity | Count |\n"
        section += "|----------|-------|\n"
        for severity in SeverityLevel:
            count = len(by_severity[severity])
            if count > 0:
                emoji = self._severity_emoji(severity)
                section += f"| {emoji} {severity.value.upper()} | {count} |\n"

        section += "\n"

        # Detailed issues (critical and high only)
        critical_and_high = (
            by_severity[SeverityLevel.CRITICAL] + by_severity[SeverityLevel.HIGH]
        )

        if critical_and_high:
            section += "### Critical & High Severity Issues\n\n"

            for issue in critical_and_high[:10]:  # Limit to first 10
                emoji = self._severity_emoji(issue.severity)
                section += f"#### {emoji} {issue.issue_type}\n\n"
                section += f"**Severity:** {issue.severity.value.upper()} ({issue.confidence} confidence)\n\n"
                section += f"{issue.description}\n\n"
                section += f"**Location:** `{issue.file_path}:{issue.line_number}`\n\n"
                section += f"**Fix:** {issue.remediation}\n\n"

                if issue.cwe_id:
                    section += f"**CWE:** [{issue.cwe_id}](https://cwe.mitre.org/data/definitions/{issue.cwe_id.replace('CWE-', '')}.html)\n\n"

            if len(critical_and_high) > 10:
                section += f"<details>\n<summary>... and {len(critical_and_high) - 10} more issues</summary>\n\n"
                section += "_View full report for complete details_\n</details>\n\n"

        return section

    def _generate_static_analysis_section(self, report: QualityReport) -> str:
        """Generate static analysis section."""
        if not report.static_issues:
            return "## Static Analysis\n\n✅ No static analysis issues detected!\n"

        section = f"## Static Analysis\n\n**Code Quality Score:** {report.code_quality_score:.1f}/100\n\n"
        section += f"**Total Issues:** {len(report.static_issues)}\n\n"

        # Show first 10 issues in a table
        if report.static_issues:
            section += "| Category | Message | Location |\n"
            section += "|----------|---------|----------|\n"

            for issue in report.static_issues[:10]:
                category = issue.get("category", "Unknown")
                message = self._escape_markdown(issue.get("message", "No message"))
                location = issue.get("location", "Unknown")
                section += f"| {category} | {message} | `{location}` |\n"

            if len(report.static_issues) > 10:
                section += f"\n_... and {len(report.static_issues) - 10} more issues_\n"

        return section + "\n"

    def _generate_performance_section(self, report: QualityReport) -> str:
        """Generate performance analysis section."""
        if not report.performance_metrics:
            return ""

        section = "## Performance Analysis\n\n"

        if report.performance_regressions > 0:
            section += f"⚠️ **{report.performance_regressions} performance regression(s) detected!**\n\n"

        section += "| Metric | Current | Baseline | Change | Status |\n"
        section += "|--------|---------|----------|--------|--------|\n"

        for metric in report.performance_metrics:
            change_emoji = "📈" if metric.regression_detected else "📉"
            status_emoji = "❌" if metric.regression_detected else "✅"

            baseline_str = (
                f"{metric.baseline_value:.2f}" if metric.baseline_value else "N/A"
            )
            change_str = (
                f"{metric.regression_percentage:+.1f}%"
                if metric.regression_percentage
                else "N/A"
            )

            section += f"| {metric.metric_name} | {metric.current_value:.2f} | {baseline_str} | {change_emoji} {change_str} | {status_emoji} |\n"

        return section + "\n"

    def _generate_complexity_section(self, report: QualityReport) -> str:
        """Generate code complexity section."""
        section = "## Code Complexity\n\n"
        section += "| Metric | Value | Status |\n"
        section += "|--------|-------|--------|\n"

        # Cyclomatic
        cyclo_status = self._status_emoji(
            20 - report.cyclomatic_complexity, 10, 5
        )  # Inverse scoring
        section += f"| Cyclomatic Complexity | {report.cyclomatic_complexity:.1f} | {cyclo_status} |\n"

        # Cognitive
        cogn_status = self._status_emoji(
            30 - report.cognitive_complexity, 15, 10
        )  # Inverse scoring
        section += f"| Cognitive Complexity | {report.cognitive_complexity:.1f} | {cogn_status} |\n"

        # Maintainability
        maint_status = self._status_emoji(report.maintainability_index, 80, 60)
        section += f"| Maintainability Index | {report.maintainability_index:.1f} | {maint_status} |\n"

        return section + "\n"

    def _generate_violations_section(self, report: QualityReport) -> str:
        """Generate quality gate violations section."""
        if not report.violations:
            return "## Quality Gate Status\n\n✅ All quality gates passed!\n"

        section = f"## Quality Gate Violations\n\n❌ **{len(report.violations)} violation(s) detected**\n\n"

        for violation in report.violations:
            threshold = violation.get("threshold", {})
            actual = violation.get("actual", "Unknown")
            severity = violation.get("severity", "error")

            emoji = "🔴" if severity == "error" else "🟡"

            section += f"- {emoji} **{threshold.get('dimension', 'Unknown')} - {threshold.get('metric', 'Unknown')}**\n"
            section += f"  - Expected: {threshold.get('min_value', 'N/A')}\n"
            section += f"  - Actual: {actual}\n"
            section += f"  - Severity: {severity}\n\n"

        return section

    def _generate_recommendations_section(self, report: QualityReport) -> str:
        """Generate recommendations section."""
        if not report.recommendations:
            return ""

        section = "## Recommendations\n\n"

        for i, rec in enumerate(report.recommendations, 1):
            section += f"{i}. {rec}\n"

        return section + "\n"

    def _generate_footer(self, report: QualityReport) -> str:
        """Generate report footer."""
        return f"""---

<sub>Generated by Agent Swarm Quality Gate System | Report ID: `{report.report_id}` | {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</sub>"""

    def generate_pr_comment(self, report: QualityReport) -> str:
        """
        Generate concise PR comment summary.

        Args:
            report: Quality report

        Returns:
            Markdown string for PR comment
        """
        self.logger.info(f"Generating PR comment: {report.report_id}")

        status_emoji = "✅" if report.passed else "❌"

        # Calculate overall coverage
        coverage_avg = 0.0
        if report.coverage_reports:
            coverage_avg = sum(
                cr.percentage for cr in report.coverage_reports.values()
            ) / len(report.coverage_reports)

        comment = f"## {status_emoji} Quality Gate Report\n\n"

        # Quick stats
        comment += "| Metric | Value |\n"
        comment += "|--------|-------|\n"
        comment += f"| Status | **{report.status.value.upper()}** |\n"
        comment += f"| Coverage | {coverage_avg:.1f}% |\n"
        comment += f"| Security Score | {report.security_score:.1f}/100 |\n"
        comment += f"| Code Quality | {report.code_quality_score:.1f}/100 |\n"
        comment += f"| Violations | {len(report.violations)} |\n"

        # Highlight critical issues
        critical_security = sum(
            1
            for issue in report.security_issues
            if issue.severity == SeverityLevel.CRITICAL
        )

        if critical_security > 0 or report.performance_regressions > 0:
            comment += "\n### ⚠️ Action Required\n\n"

            if critical_security > 0:
                comment += f"- 🔴 **{critical_security} critical security issue(s)** must be resolved\n"

            if report.performance_regressions > 0:
                comment += f"- 📉 **{report.performance_regressions} performance regression(s)** detected\n"

        # Quick recommendations
        if report.recommendations:
            comment += "\n### 📋 Top Recommendations\n\n"
            for rec in report.recommendations[:3]:
                comment += f"- {rec}\n"

        comment += f"\n<sub>View [full report]({report.report_id}.html) for details</sub>"

        return comment

    def generate_summary_table(self, reports: List[QualityReport]) -> str:
        """
        Generate comparison table for multiple reports.

        Args:
            reports: List of quality reports to compare

        Returns:
            Markdown table comparing reports
        """
        if not reports:
            return ""

        table = "## Quality Reports Comparison\n\n"
        table += "| Report ID | Date | Status | Coverage | Security | Quality |\n"
        table += "|-----------|------|--------|----------|----------|--------|\n"

        for report in reports:
            coverage_avg = (
                sum(cr.percentage for cr in report.coverage_reports.values())
                / len(report.coverage_reports)
                if report.coverage_reports
                else 0.0
            )

            status_emoji = "✅" if report.passed else "❌"
            date_str = report.timestamp.strftime("%Y-%m-%d")

            table += f"| `{report.report_id}` | {date_str} | {status_emoji} | {coverage_avg:.1f}% | {report.security_score:.1f} | {report.code_quality_score:.1f} |\n"

        return table + "\n"

    def _create_status_badge(self, report: QualityReport) -> str:
        """Create status badge."""
        if report.passed:
            return "![Status](https://img.shields.io/badge/quality-passing-brightgreen)"
        else:
            return "![Status](https://img.shields.io/badge/quality-failing-red)"

    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create ASCII progress bar."""
        filled = int(width * percentage / 100)
        empty = width - filled
        return f"`[{'█' * filled}{'░' * empty}]`"

    def _status_emoji(
        self, value: float, good_threshold: float, warn_threshold: float
    ) -> str:
        """Get status emoji based on value and thresholds."""
        if value >= good_threshold:
            return "✅"
        elif value >= warn_threshold:
            return "⚠️"
        else:
            return "❌"

    def _trend_emoji(self, trend: str) -> str:
        """Get emoji for trend."""
        if trend == "improving":
            return "📈"
        elif trend == "degrading":
            return "📉"
        else:
            return "➡️"

    def _severity_emoji(self, severity: SeverityLevel) -> str:
        """Get emoji for severity level."""
        mapping = {
            SeverityLevel.CRITICAL: "🔴",
            SeverityLevel.HIGH: "🟠",
            SeverityLevel.MEDIUM: "🟡",
            SeverityLevel.LOW: "🟢",
            SeverityLevel.INFO: "🔵",
        }
        return mapping.get(severity, "⚪")

    def _escape_markdown(self, text: str) -> str:
        """Escape special markdown characters."""
        # Escape pipe character for tables
        return text.replace("|", "\\|").replace("\n", " ")

    def save_report(self, report: QualityReport, output_path: str) -> str:
        """
        Generate and save Markdown report to file.

        Args:
            report: Quality report to render
            output_path: Path to save markdown file

        Returns:
            Path to saved file
        """
        markdown = self.generate_report(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        self.logger.info(f"Markdown report saved to: {output_path}")
        return output_path
