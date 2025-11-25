"""Accessibility compliance reporter for WCAG audits.

This module provides the AccessibilityReporter class for generating WCAG
compliance reports from accessibility audit results, grouping issues by
impact level and rule, with fix suggestions and exportable formats.
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from datetime import datetime
import json

from src.models.browser_models import AccessibilityIssue

logger = logging.getLogger(__name__)


class AccessibilityReporter:
    """Generate WCAG compliance reports with issues grouped by impact and rule.

    This class creates comprehensive accessibility reports in HTML or JSON format,
    showing violations organized by severity, WCAG criteria, and providing
    actionable fix suggestions for developers.

    PATTERN: Group and aggregate issues for better actionability and reporting.
    """

    def __init__(self, output_dir: str = "accessibility-reports"):
        """Initialize the accessibility reporter.

        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Accessibility reporter initialized: {self.output_dir}")

    def generate_report(
        self,
        issues: List[AccessibilityIssue],
        url: str,
        wcag_level: Literal["A", "AA", "AAA"] = "AA",
        format: Literal["html", "json"] = "html",
        report_name: Optional[str] = None,
    ) -> str:
        """Generate an accessibility compliance report.

        Args:
            issues: List of accessibility issues found
            url: URL that was tested
            wcag_level: WCAG level tested against
            format: Output format (html or json)
            report_name: Custom report name (auto-generated if None)

        Returns:
            Path to the generated report file

        Example:
            reporter = AccessibilityReporter()
            issues = await tester.run_audit(page, wcag_level="AA")
            report_path = reporter.generate_report(
                issues,
                url="https://example.com",
                wcag_level="AA",
                format="html"
            )
        """
        logger.info(f"Generating accessibility report for {url} ({len(issues)} issues)")

        try:
            # Generate report name if not provided
            if not report_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_url = url.replace("://", "_").replace("/", "_")[:50]
                report_name = f"a11y_{safe_url}_{timestamp}"

            # Generate report based on format
            if format == "json":
                report_path = self._generate_json_report(
                    issues, url, wcag_level, report_name
                )
            else:
                report_path = self._generate_html_report(
                    issues, url, wcag_level, report_name
                )

            logger.info(f"Accessibility report generated: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to generate accessibility report: {e}")
            raise

    def _generate_json_report(
        self,
        issues: List[AccessibilityIssue],
        url: str,
        wcag_level: Literal["A", "AA", "AAA"],
        report_name: str,
    ) -> Path:
        """Generate a JSON accessibility report.

        Args:
            issues: List of accessibility issues
            url: Tested URL
            wcag_level: WCAG level
            report_name: Report name

        Returns:
            Path to JSON report
        """
        # Group issues
        by_impact = self.group_by_impact(issues)
        by_rule = self.group_by_rule(issues)

        # Build report data
        report_data = {
            "metadata": {
                "url": url,
                "wcag_level": wcag_level,
                "timestamp": datetime.now().isoformat(),
                "total_issues": len(issues),
            },
            "summary": {
                "by_impact": {
                    impact: len(issues_list)
                    for impact, issues_list in by_impact.items()
                },
                "by_rule": {
                    rule_id: len(issues_list)
                    for rule_id, issues_list in by_rule.items()
                },
            },
            "issues": [issue.model_dump() for issue in issues],
            "grouped_by_impact": {
                impact: [issue.model_dump() for issue in issues_list]
                for impact, issues_list in by_impact.items()
            },
            "grouped_by_rule": {
                rule_id: [issue.model_dump() for issue in issues_list]
                for rule_id, issues_list in by_rule.items()
            },
        }

        # Save JSON report
        report_path = self.output_dir / f"{report_name}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return report_path

    def _generate_html_report(
        self,
        issues: List[AccessibilityIssue],
        url: str,
        wcag_level: Literal["A", "AA", "AAA"],
        report_name: str,
    ) -> Path:
        """Generate an HTML accessibility report.

        Args:
            issues: List of accessibility issues
            url: Tested URL
            wcag_level: WCAG level
            report_name: Report name

        Returns:
            Path to HTML report
        """
        # Group issues
        by_impact = self.group_by_impact(issues)
        by_rule = self.group_by_rule(issues)

        # Build HTML
        html_content = self._build_html_structure(
            issues, by_impact, by_rule, url, wcag_level
        )

        # Save HTML report
        report_path = self.output_dir / f"{report_name}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_path

    def _build_html_structure(
        self,
        issues: List[AccessibilityIssue],
        by_impact: Dict[str, List[AccessibilityIssue]],
        by_rule: Dict[str, List[AccessibilityIssue]],
        url: str,
        wcag_level: Literal["A", "AA", "AAA"],
    ) -> str:
        """Build complete HTML structure for accessibility report.

        Args:
            issues: All issues
            by_impact: Issues grouped by impact
            by_rule: Issues grouped by rule
            url: Tested URL
            wcag_level: WCAG level

        Returns:
            Complete HTML content
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessibility Report - {url}</title>
    {self._get_styles()}
</head>
<body>
    <div class="container">
        <header>
            <h1>Accessibility Compliance Report</h1>
            <p class="url">{url}</p>
            <p class="meta">WCAG Level: {wcag_level} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>

        {self._build_summary_section(issues, by_impact)}
        {self._build_impact_section(by_impact)}
        {self._build_rules_section(by_rule)}

        <footer>
            <p>Accessibility Report - Generated by Agent Swarm</p>
        </footer>
    </div>

    {self._get_scripts()}
</body>
</html>"""
        return html

    def _get_styles(self) -> str:
        """Get inline CSS styles for the report.

        Returns:
            HTML style tag with CSS
        """
        return """<style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        h1 {
            color: #2c3e50;
            font-size: 2em;
            margin-bottom: 10px;
        }

        h2 {
            color: #34495e;
            font-size: 1.5em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }

        h3 {
            color: #2c3e50;
            font-size: 1.2em;
            margin: 15px 0 10px;
            cursor: pointer;
            user-select: none;
        }

        h3:hover {
            color: #3498db;
        }

        .url {
            color: #3498db;
            font-size: 1.1em;
            margin: 10px 0;
            word-break: break-all;
        }

        .meta {
            color: #7f8c8d;
            font-size: 0.9em;
        }

        .section {
            background: #fff;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .summary-card {
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }

        .summary-card.total {
            background: #ecf0f1;
            border-left: 4px solid #95a5a6;
        }

        .summary-card.critical {
            background: #fadbd8;
            border-left: 4px solid #c0392b;
        }

        .summary-card.serious {
            background: #ffe6e6;
            border-left: 4px solid #e74c3c;
        }

        .summary-card.moderate {
            background: #fef5e7;
            border-left: 4px solid #f39c12;
        }

        .summary-card.minor {
            background: #f0f0f0;
            border-left: 4px solid #95a5a6;
        }

        .summary-card .value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .summary-card .label {
            font-size: 0.9em;
            text-transform: uppercase;
            color: #7f8c8d;
        }

        .impact-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
            text-transform: uppercase;
            margin-right: 8px;
        }

        .impact-badge.critical {
            background: #c0392b;
            color: white;
        }

        .impact-badge.serious {
            background: #e74c3c;
            color: white;
        }

        .impact-badge.moderate {
            background: #f39c12;
            color: white;
        }

        .impact-badge.minor {
            background: #95a5a6;
            color: white;
        }

        .issue-item {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
            background: #fafafa;
        }

        .issue-item.critical {
            border-left: 4px solid #c0392b;
        }

        .issue-item.serious {
            border-left: 4px solid #e74c3c;
        }

        .issue-item.moderate {
            border-left: 4px solid #f39c12;
        }

        .issue-item.minor {
            border-left: 4px solid #95a5a6;
        }

        .issue-header {
            font-weight: bold;
            font-size: 1.05em;
            margin-bottom: 10px;
        }

        .issue-description {
            color: #555;
            margin-bottom: 10px;
        }

        .issue-details {
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            font-size: 0.9em;
        }

        .issue-details p {
            margin: 5px 0;
        }

        .fix-suggestion {
            background: #e8f4f8;
            border-left: 3px solid #3498db;
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
        }

        .fix-suggestion strong {
            color: #2980b9;
        }

        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #c7254e;
        }

        .wcag-tag {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.75em;
            margin: 2px 2px 2px 0;
        }

        .collapsible-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .collapsible-content.active {
            max-height: 10000px;
        }

        .rule-group {
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }

        .rule-header {
            background: #f8f9fa;
            padding: 15px;
            cursor: pointer;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .rule-header:hover {
            background: #e9ecef;
        }

        .rule-count {
            background: #e74c3c;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }

        .rule-body {
            padding: 15px;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
        }

        .no-issues {
            text-align: center;
            padding: 40px;
            color: #27ae60;
            font-size: 1.2em;
        }

        .no-issues::before {
            content: "✓";
            display: block;
            font-size: 3em;
            margin-bottom: 10px;
        }
    </style>"""

    def _build_summary_section(
        self,
        issues: List[AccessibilityIssue],
        by_impact: Dict[str, List[AccessibilityIssue]],
    ) -> str:
        """Build the summary section.

        Args:
            issues: All issues
            by_impact: Issues grouped by impact

        Returns:
            HTML for summary section
        """
        if not issues:
            return '<div class="section"><div class="no-issues">No accessibility issues found!</div></div>'

        critical_count = len(by_impact.get("critical", []))
        serious_count = len(by_impact.get("serious", []))
        moderate_count = len(by_impact.get("moderate", []))
        minor_count = len(by_impact.get("minor", []))

        return f"""
        <div class="section">
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-card total">
                    <div class="value">{len(issues)}</div>
                    <div class="label">Total Issues</div>
                </div>
                <div class="summary-card critical">
                    <div class="value">{critical_count}</div>
                    <div class="label">Critical</div>
                </div>
                <div class="summary-card serious">
                    <div class="value">{serious_count}</div>
                    <div class="label">Serious</div>
                </div>
                <div class="summary-card moderate">
                    <div class="value">{moderate_count}</div>
                    <div class="label">Moderate</div>
                </div>
                <div class="summary-card minor">
                    <div class="value">{minor_count}</div>
                    <div class="label">Minor</div>
                </div>
            </div>
        </div>"""

    def _build_impact_section(
        self, by_impact: Dict[str, List[AccessibilityIssue]]
    ) -> str:
        """Build the section grouped by impact level.

        Args:
            by_impact: Issues grouped by impact

        Returns:
            HTML for impact section
        """
        if not by_impact:
            return ""

        section_html = '<div class="section"><h2>Issues by Impact Level</h2>'

        # Order by severity
        impact_order = ["critical", "serious", "moderate", "minor"]

        for impact in impact_order:
            impact_issues = by_impact.get(impact, [])
            if not impact_issues:
                continue

            section_html += f"""
            <h3 class="collapsible">
                <span class="impact-badge {impact}">{impact}</span>
                {len(impact_issues)} issue{"s" if len(impact_issues) != 1 else ""}
            </h3>
            <div class="collapsible-content">"""

            for issue in impact_issues:
                section_html += self._render_issue(issue)

            section_html += "</div>"

        section_html += "</div>"
        return section_html

    def _build_rules_section(self, by_rule: Dict[str, List[AccessibilityIssue]]) -> str:
        """Build the section grouped by rule.

        Args:
            by_rule: Issues grouped by rule

        Returns:
            HTML for rules section
        """
        if not by_rule:
            return ""

        section_html = '<div class="section"><h2>Issues by WCAG Rule</h2>'

        # Sort rules by number of issues (descending)
        sorted_rules = sorted(by_rule.items(), key=lambda x: len(x[1]), reverse=True)

        for rule_id, rule_issues in sorted_rules:
            # Get description from first issue
            description = rule_issues[0].description if rule_issues else ""

            section_html += f"""
            <div class="rule-group">
                <div class="rule-header" onclick="toggleRule(this)">
                    <div>
                        <strong>{rule_id}</strong>
                        <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;">{description}</p>
                    </div>
                    <span class="rule-count">{len(rule_issues)}</span>
                </div>
                <div class="rule-body" style="display: none;">"""

            for issue in rule_issues:
                section_html += self._render_issue(issue)

            section_html += """
                </div>
            </div>"""

        section_html += "</div>"
        return section_html

    def _render_issue(self, issue: AccessibilityIssue) -> str:
        """Render a single issue.

        Args:
            issue: Accessibility issue

        Returns:
            HTML for the issue
        """
        wcag_tags = "".join(
            f'<span class="wcag-tag">{criteria}</span>'
            for criteria in issue.wcag_criteria
        )

        return f"""
        <div class="issue-item {issue.impact}">
            <div class="issue-header">
                <span class="impact-badge {issue.impact}">{issue.impact}</span>
                {issue.rule_id}
            </div>
            <div class="issue-description">
                {issue.description}
            </div>
            <div class="issue-details">
                <p><strong>Element:</strong> <code>{self._escape_html(issue.selector)}</code></p>
                <p><strong>HTML:</strong> <code>{self._escape_html(issue.html[:100])}...</code></p>
                <p><strong>WCAG Criteria:</strong> {wcag_tags if wcag_tags else "None"}</p>
            </div>
            <div class="fix-suggestion">
                <strong>How to Fix:</strong><br>
                {issue.fix_suggestion or issue.help_text}
            </div>
        </div>"""

    def _get_scripts(self) -> str:
        """Get inline JavaScript for interactive features.

        Returns:
            HTML script tag with JavaScript
        """
        return """<script>
        // Toggle collapsible sections
        document.addEventListener('DOMContentLoaded', function() {
            const collapsibles = document.querySelectorAll('.collapsible');
            collapsibles.forEach(function(element) {
                element.addEventListener('click', function() {
                    const content = this.nextElementSibling;
                    content.classList.toggle('active');
                });
            });
        });

        // Toggle rule groups
        function toggleRule(header) {
            const body = header.nextElementSibling;
            if (body.style.display === 'none') {
                body.style.display = 'block';
            } else {
                body.style.display = 'none';
            }
        }
    </script>"""

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Text to escape

        Returns:
            HTML-escaped text
        """
        if not text:
            return ""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    def group_by_impact(
        self, issues: List[AccessibilityIssue]
    ) -> Dict[str, List[AccessibilityIssue]]:
        """Group issues by impact level.

        Args:
            issues: List of accessibility issues

        Returns:
            Dictionary mapping impact level to list of issues
        """
        grouped: Dict[str, List[AccessibilityIssue]] = {
            "critical": [],
            "serious": [],
            "moderate": [],
            "minor": [],
        }

        for issue in issues:
            if issue.impact in grouped:
                grouped[issue.impact].append(issue)

        return grouped

    def group_by_rule(
        self, issues: List[AccessibilityIssue]
    ) -> Dict[str, List[AccessibilityIssue]]:
        """Group issues by rule ID.

        Args:
            issues: List of accessibility issues

        Returns:
            Dictionary mapping rule ID to list of issues
        """
        grouped: Dict[str, List[AccessibilityIssue]] = {}

        for issue in issues:
            if issue.rule_id not in grouped:
                grouped[issue.rule_id] = []
            grouped[issue.rule_id].append(issue)

        return grouped

    def filter_by_wcag_level(
        self,
        issues: List[AccessibilityIssue],
        wcag_level: Literal["A", "AA", "AAA"],
    ) -> List[AccessibilityIssue]:
        """Filter issues to only those relevant to a specific WCAG level.

        Args:
            issues: List of accessibility issues
            wcag_level: WCAG level to filter by

        Returns:
            Filtered list of issues
        """
        level_order = {"A": 0, "AA": 1, "AAA": 2}
        target_level = level_order.get(wcag_level, 1)

        filtered = [
            issue
            for issue in issues
            if level_order.get(issue.wcag_level, 0) <= target_level
        ]

        logger.debug(
            f"Filtered {len(issues)} issues to {len(filtered)} for WCAG level {wcag_level}"
        )

        return filtered

    def get_statistics(self, issues: List[AccessibilityIssue]) -> Dict[str, Any]:
        """Get statistics about accessibility issues.

        Args:
            issues: List of accessibility issues

        Returns:
            Dictionary with statistics

        Example:
            stats = reporter.get_statistics(issues)
            print(f"Critical issues: {stats['by_impact']['critical']}")
        """
        by_impact = self.group_by_impact(issues)
        by_rule = self.group_by_rule(issues)

        return {
            "total": len(issues),
            "by_impact": {
                impact: len(issues_list) for impact, issues_list in by_impact.items()
            },
            "by_rule": {
                rule_id: len(issues_list) for rule_id, issues_list in by_rule.items()
            },
            "unique_rules": len(by_rule),
            "most_common_rule": (
                max(by_rule.items(), key=lambda x: len(x[1]))[0] if by_rule else None
            ),
        }
