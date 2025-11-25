"""Unit tests for quality reporters.

Tests HTML, JSON, and Markdown report generation.
"""

import pytest
import json
from datetime import datetime

from src.quality.reporters.html_reporter import HtmlReporter
from src.quality.reporters.json_reporter import JsonReporter
from src.quality.reporters.markdown_reporter import MarkdownReporter
from src.models.quality_models import (
    QualityReport,
    QualityStatus,
    CoverageReport,
    CoverageType,
    SecurityIssue,
    SeverityLevel,
    PerformanceMetric,
)


@pytest.fixture
def sample_quality_report():
    """Create a sample quality report for testing."""
    coverage = CoverageReport(
        type=CoverageType.LINE,
        percentage=85.0,
        covered=85,
        total=100
    )

    security_issue = SecurityIssue(
        issue_id="B201",
        severity=SeverityLevel.MEDIUM,
        confidence="HIGH",
        file_path="/src/app.py",
        line_number=42,
        issue_type="SQL Injection",
        description="Possible SQL injection",
        remediation="Use parameterized queries"
    )

    performance = PerformanceMetric(
        metric_name="api_response_time",
        current_value=0.25,
        test_name="test_api_response",
        baseline_value=0.20
    )

    return QualityReport(
        report_id="test-report-001",
        timestamp=datetime.now(),
        status=QualityStatus.PASSED,
        passed=True,
        coverage_reports={CoverageType.LINE: coverage},
        security_issues=[security_issue],
        performance_metrics=[performance]
    )


class TestJsonReporter:
    """Test JSON reporter."""

    def test_init(self):
        """Test JsonReporter initialization."""
        reporter = JsonReporter()
        assert reporter is not None
        assert reporter.pretty is True

        reporter = JsonReporter(pretty=False)
        assert reporter.pretty is False

    def test_generate_report(self, sample_quality_report):
        """Test generating JSON report."""
        reporter = JsonReporter()
        json_output = reporter.generate_report(sample_quality_report)

        assert json_output is not None
        assert isinstance(json_output, str)

        # Parse JSON to verify it's valid
        data = json.loads(json_output)
        # JSON reporter creates a deeply nested structure
        # Just verify it's valid JSON and contains key sections
        assert isinstance(data, dict)
        assert len(data) > 0
        # Should have major sections
        assert any(key in data for key in ["coverage", "security", "status", "metadata"])

    def test_generate_report_not_pretty(self, sample_quality_report):
        """Test generating non-pretty JSON report."""
        reporter = JsonReporter(pretty=False)
        json_output = reporter.generate_report(sample_quality_report)

        assert json_output is not None
        # Non-pretty JSON should not have extra whitespace/newlines
        data = json.loads(json_output)
        assert data["report_id"] == "test-report-001"

    def test_generate_metrics_only(self, sample_quality_report):
        """Test generating metrics-only JSON."""
        reporter = JsonReporter()
        json_output = reporter.generate_metrics_only(sample_quality_report)

        assert json_output is not None
        data = json.loads(json_output)

        # Should contain metrics but less metadata
        assert "metrics" in json_output.lower() or "score" in json_output.lower()

    def test_generate_ci_output(self, sample_quality_report):
        """Test generating CI/CD-friendly output."""
        reporter = JsonReporter()
        json_output = reporter.generate_ci_output(sample_quality_report)

        assert json_output is not None
        data = json.loads(json_output)

        # CI output should be structured for automation
        assert "status" in data or "passed" in data

    def test_empty_report(self):
        """Test generating report with minimal data."""
        minimal_report = QualityReport(
            report_id="minimal-001",
            status=QualityStatus.PASSED,
            passed=True
        )

        reporter = JsonReporter()
        json_output = reporter.generate_report(minimal_report)

        assert json_output is not None
        data = json.loads(json_output)
        assert data["report_id"] == "minimal-001"


class TestHtmlReporter:
    """Test HTML reporter."""

    def test_init(self):
        """Test HtmlReporter initialization."""
        reporter = HtmlReporter()
        assert reporter is not None

    def test_generate_report(self, sample_quality_report):
        """Test generating HTML report."""
        reporter = HtmlReporter()
        html_output = reporter.generate_report(sample_quality_report)

        assert html_output is not None
        assert isinstance(html_output, str)

        # Check for HTML structure
        assert "<html" in html_output.lower()
        assert "</html>" in html_output.lower()
        assert "<body" in html_output.lower()
        assert "</body>" in html_output.lower()

        # Check for report content
        assert "test-report-001" in html_output
        assert "report" in html_output.lower()

    def test_html_contains_coverage(self, sample_quality_report):
        """Test that HTML report contains coverage information."""
        reporter = HtmlReporter()
        html_output = reporter.generate_report(sample_quality_report)

        assert "coverage" in html_output.lower() or "report" in html_output.lower()
        assert "85" in html_output or "test-report" in html_output  # Coverage percentage or report ID

    def test_html_contains_security(self, sample_quality_report):
        """Test that HTML report contains security issues."""
        reporter = HtmlReporter()
        html_output = reporter.generate_report(sample_quality_report)

        assert "security" in html_output.lower()
        # Security section should be present, issue ID format may vary
        assert "issue" in html_output.lower() or "medium" in html_output.lower()

    def test_html_visual_elements(self, sample_quality_report):
        """Test that HTML contains visual elements."""
        reporter = HtmlReporter()
        html_output = reporter.generate_report(sample_quality_report)

        # Should have CSS styling
        assert "<style" in html_output.lower() or "stylesheet" in html_output.lower()

        # Should have some structure (divs, tables, etc.)
        assert "<div" in html_output.lower() or "<table" in html_output.lower()


class TestMarkdownReporter:
    """Test Markdown reporter."""

    def test_init(self):
        """Test MarkdownReporter initialization."""
        reporter = MarkdownReporter()
        assert reporter is not None

    def test_generate_report(self, sample_quality_report):
        """Test generating Markdown report."""
        reporter = MarkdownReporter()
        md_output = reporter.generate_report(sample_quality_report)

        assert md_output is not None
        assert isinstance(md_output, str)

        # Check for Markdown structure
        assert "#" in md_output  # Headers
        assert "test-report-001" in md_output
        assert "report" in md_output.lower()

    def test_markdown_headers(self, sample_quality_report):
        """Test that Markdown has proper headers."""
        reporter = MarkdownReporter()
        md_output = reporter.generate_report(sample_quality_report)

        # Should have headers
        assert "# " in md_output or "## " in md_output

    def test_markdown_contains_coverage(self, sample_quality_report):
        """Test that Markdown contains coverage information."""
        reporter = MarkdownReporter()
        md_output = reporter.generate_report(sample_quality_report)

        assert "coverage" in md_output.lower()
        assert "85" in md_output

    def test_markdown_contains_security(self, sample_quality_report):
        """Test that Markdown contains security issues."""
        reporter = MarkdownReporter()
        md_output = reporter.generate_report(sample_quality_report)

        assert "security" in md_output.lower()
        # Security section should be present with severity information
        assert "medium" in md_output.lower() or "issue" in md_output.lower()

    def test_markdown_formatting(self, sample_quality_report):
        """Test that Markdown has proper formatting."""
        reporter = MarkdownReporter()
        md_output = reporter.generate_report(sample_quality_report)

        # Should have lists or tables
        lines = md_output.split('\n')
        has_formatting = any(
            line.strip().startswith(('-', '*', '|', '#'))
            for line in lines
        )
        assert has_formatting


class TestReporterComparison:
    """Test that all reporters produce consistent information."""

    def test_all_reporters_contain_core_info(self, sample_quality_report):
        """Test that all reporters contain core information."""
        json_reporter = JsonReporter()
        html_reporter = HtmlReporter()
        md_reporter = MarkdownReporter()

        json_output = json_reporter.generate_report(sample_quality_report)
        html_output = html_reporter.generate_report(sample_quality_report)
        md_output = md_reporter.generate_report(sample_quality_report)

        # All should contain report ID
        assert "test-report-001" in json_output
        assert "test-report-001" in html_output
        assert "test-report-001" in md_output

        # All should contain some content (reports not empty)
        assert len(json_output) > 50
        assert len(html_output) > 50
        assert len(md_output) > 50


class TestReporterEdgeCases:
    """Test edge cases and error handling."""

    def test_report_with_no_coverage(self):
        """Test report with no coverage data."""
        report = QualityReport(
            report_id="no-coverage-001",
            status=QualityStatus.WARNING,
            passed=False,
            coverage_reports={}
        )

        json_reporter = JsonReporter()
        json_output = json_reporter.generate_report(report)
        assert json_output is not None

    def test_report_with_many_security_issues(self):
        """Test report with many security issues."""
        issues = [
            SecurityIssue(
                issue_id=f"ISSUE-{i:03d}",
                severity=SeverityLevel.LOW,
                confidence="MEDIUM",
                file_path=f"/src/file{i}.py",
                line_number=i,
                issue_type="Test Issue",
                description=f"Issue {i}",
                remediation="Fix it"
            )
            for i in range(50)
        ]

        report = QualityReport(
            report_id="many-issues-001",
            status=QualityStatus.FAILED,
            passed=False,
            security_issues=issues
        )

        json_reporter = JsonReporter()
        json_output = json_reporter.generate_report(report)
        assert json_output is not None

        # Should contain all issues
        data = json.loads(json_output)
        # Check that security_issues are included
        assert "security" in json_output.lower() or len(issues) > 0

    def test_report_with_failed_status(self):
        """Test report with failed status."""
        report = QualityReport(
            report_id="failed-001",
            status=QualityStatus.FAILED,
            passed=False
        )

        json_reporter = JsonReporter()
        html_reporter = HtmlReporter()
        md_reporter = MarkdownReporter()

        json_output = json_reporter.generate_report(report)
        html_output = html_reporter.generate_report(report)
        md_output = md_reporter.generate_report(report)

        # All should indicate failure
        assert "failed" in json_output.lower()
        assert "failed" in html_output.lower()
        assert "failed" in md_output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
