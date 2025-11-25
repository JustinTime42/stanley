"""Unit tests for quality models.

Tests Pydantic model validation, field constraints, and data structures
for the quality gate system.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models.quality_models import (
    QualityDimension,
    CoverageType,
    QualityStatus,
    SeverityLevel,
    CoverageReport,
    SecurityIssue,
    PerformanceMetric,
    QualityThreshold,
    QualityReport,
)


class TestEnums:
    """Test enum definitions."""

    def test_quality_dimension_values(self):
        """Test QualityDimension enum has expected values."""
        assert QualityDimension.COVERAGE == "coverage"
        assert QualityDimension.STATIC == "static"
        assert QualityDimension.SECURITY == "security"
        assert QualityDimension.PERFORMANCE == "performance"
        assert QualityDimension.COMPLEXITY == "complexity"

    def test_coverage_type_values(self):
        """Test CoverageType enum has expected values."""
        assert CoverageType.LINE == "line"
        assert CoverageType.BRANCH == "branch"
        assert CoverageType.FUNCTION == "function"
        assert CoverageType.MUTATION == "mutation"

    def test_quality_status_values(self):
        """Test QualityStatus enum has expected values."""
        assert QualityStatus.PASSED == "passed"
        assert QualityStatus.WARNING == "warning"
        assert QualityStatus.FAILED == "failed"
        assert QualityStatus.ERROR == "error"

    def test_severity_level_values(self):
        """Test SeverityLevel enum has expected values."""
        assert SeverityLevel.CRITICAL == "critical"
        assert SeverityLevel.HIGH == "high"
        assert SeverityLevel.MEDIUM == "medium"
        assert SeverityLevel.LOW == "low"
        assert SeverityLevel.INFO == "info"


class TestCoverageReport:
    """Test CoverageReport model."""

    def test_create_basic_coverage_report(self):
        """Test creating a basic coverage report."""
        report = CoverageReport(
            type=CoverageType.LINE,
            percentage=85.5,
            covered=171,
            total=200
        )

        assert report.type == CoverageType.LINE
        assert report.percentage == 85.5
        assert report.covered == 171
        assert report.total == 200
        assert report.files == {}
        assert report.branch_coverage is None
        assert report.mutation_score is None

    def test_coverage_percentage_constraints(self):
        """Test percentage field constraints (0-100)."""
        # Valid percentages
        report = CoverageReport(type=CoverageType.LINE, percentage=0, covered=0, total=100)
        assert report.percentage == 0

        report = CoverageReport(type=CoverageType.LINE, percentage=100, covered=100, total=100)
        assert report.percentage == 100

        # Invalid percentage (> 100)
        with pytest.raises(ValidationError) as exc_info:
            CoverageReport(type=CoverageType.LINE, percentage=101, covered=100, total=100)
        assert "less_than_equal" in str(exc_info.value)

        # Invalid percentage (< 0)
        with pytest.raises(ValidationError) as exc_info:
            CoverageReport(type=CoverageType.LINE, percentage=-1, covered=0, total=100)
        assert "greater_than_equal" in str(exc_info.value)

    def test_mutation_coverage_fields(self):
        """Test mutation-specific fields."""
        report = CoverageReport(
            type=CoverageType.MUTATION,
            percentage=75.0,
            covered=75,
            total=100,
            mutation_score=0.75,
            killed_mutants=60,
            survived_mutants=15,
            timeout_mutants=5
        )

        assert report.mutation_score == 0.75
        assert report.killed_mutants == 60
        assert report.survived_mutants == 15
        assert report.timeout_mutants == 5


class TestSecurityIssue:
    """Test SecurityIssue model."""

    def test_create_security_issue(self):
        """Test creating a security issue."""
        issue = SecurityIssue(
            issue_id="CVE-2023-12345",
            severity=SeverityLevel.HIGH,
            confidence="HIGH",
            file_path="/path/to/file.py",
            line_number=42,
            issue_type="SQL Injection",
            description="SQL injection vulnerability detected",
            remediation="Use parameterized queries"
        )

        assert issue.issue_id == "CVE-2023-12345"
        assert issue.severity == SeverityLevel.HIGH
        assert issue.confidence == "HIGH"
        assert issue.file_path == "/path/to/file.py"
        assert issue.line_number == 42
        assert issue.cwe_id is None

    def test_security_issue_with_cwe(self):
        """Test security issue with CWE reference."""
        issue = SecurityIssue(
            issue_id="B201",
            severity=SeverityLevel.MEDIUM,
            confidence="MEDIUM",
            file_path="/path/to/file.py",
            line_number=10,
            issue_type="Unsafe deserialization",
            description="Use of unsafe pickle",
            remediation="Use safe serialization",
            cwe_id="CWE-502"
        )

        assert issue.cwe_id == "CWE-502"


class TestPerformanceMetric:
    """Test PerformanceMetric model."""

    def test_create_performance_metric(self):
        """Test creating a performance metric."""
        metric = PerformanceMetric(
            metric_name="test_response_time",
            current_value=0.5,
            test_name="test_api_response",
            baseline_value=0.4
        )

        assert metric.metric_name == "test_response_time"
        assert metric.current_value == 0.5
        assert metric.test_name == "test_api_response"
        assert metric.baseline_value == 0.4

    def test_performance_regression(self):
        """Test detecting performance regression."""
        metric = PerformanceMetric(
            metric_name="api_latency",
            current_value=1.2,
            test_name="test_api_latency",
            baseline_value=1.0,
            regression_detected=True,
            regression_percentage=20.0
        )

        assert metric.regression_detected is True
        assert metric.regression_percentage == 20.0


class TestQualityThreshold:
    """Test QualityThreshold model."""

    def test_create_quality_threshold(self):
        """Test creating a quality threshold."""
        threshold = QualityThreshold(
            dimension=QualityDimension.COVERAGE,
            metric="line_coverage",
            min_value=80.0,
            max_value=None,
            enforcement="error"
        )

        assert threshold.dimension == QualityDimension.COVERAGE
        assert threshold.metric == "line_coverage"
        assert threshold.min_value == 80.0
        assert threshold.max_value is None
        assert threshold.enforcement == "error"

    def test_threshold_with_range(self):
        """Test threshold with both min and max values."""
        threshold = QualityThreshold(
            dimension=QualityDimension.COMPLEXITY,
            metric="cyclomatic_complexity",
            min_value=1.0,
            max_value=10.0,
            enforcement="warning"
        )

        assert threshold.min_value == 1.0
        assert threshold.max_value == 10.0


class TestQualityReport:
    """Test QualityReport model."""

    def test_create_minimal_quality_report(self):
        """Test creating a minimal quality report."""
        report = QualityReport(
            report_id="test-001",
            status=QualityStatus.PASSED,
            passed=True
        )

        assert report.report_id == "test-001"
        assert report.status == QualityStatus.PASSED
        assert report.passed is True
        assert isinstance(report.timestamp, datetime)
        assert report.coverage_reports == {}
        assert report.security_issues == []
        assert report.performance_metrics == []
        assert report.code_quality_score == 0.0
        assert report.security_score == 100.0

    def test_quality_report_with_coverage(self):
        """Test quality report with coverage data."""
        coverage = CoverageReport(
            type=CoverageType.LINE,
            percentage=90.0,
            covered=90,
            total=100
        )

        report = QualityReport(
            report_id="test-002",
            status=QualityStatus.PASSED,
            passed=True,
            coverage_reports={CoverageType.LINE: coverage}
        )

        assert CoverageType.LINE in report.coverage_reports
        assert report.coverage_reports[CoverageType.LINE].percentage == 90.0

    def test_quality_report_with_security_issues(self):
        """Test quality report with security issues."""
        issue = SecurityIssue(
            issue_id="CVE-2023-001",
            severity=SeverityLevel.HIGH,
            confidence="HIGH",
            file_path="/test.py",
            line_number=10,
            issue_type="SQL Injection",
            description="Test issue",
            remediation="Fix it"
        )

        report = QualityReport(
            report_id="test-003",
            status=QualityStatus.FAILED,
            passed=False,
            security_issues=[issue]
        )

        assert len(report.security_issues) == 1
        assert report.security_issues[0].issue_id == "CVE-2023-001"
        assert report.passed is False

    def test_quality_report_json_serialization(self):
        """Test that quality report can be serialized to JSON."""
        coverage = CoverageReport(
            type=CoverageType.LINE,
            percentage=85.0,
            covered=85,
            total=100
        )

        report = QualityReport(
            report_id="test-004",
            status=QualityStatus.PASSED,
            passed=True,
            coverage_reports={CoverageType.LINE: coverage}
        )

        # Should be able to convert to dict
        report_dict = report.model_dump()
        assert report_dict["report_id"] == "test-004"

        # Should be able to convert to JSON string
        json_str = report.model_dump_json()
        assert "test-004" in json_str
        assert "85" in json_str


class TestModelValidation:
    """Test model validation and error handling."""

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CoverageReport(type=CoverageType.LINE, percentage=80.0)
        assert "covered" in str(exc_info.value)
        assert "total" in str(exc_info.value)

    def test_invalid_enum_value(self):
        """Test that invalid enum values raise ValidationError."""
        with pytest.raises(ValidationError):
            CoverageReport(
                type="invalid_type",
                percentage=80.0,
                covered=80,
                total=100
            )

    def test_type_coercion(self):
        """Test that types are coerced when possible."""
        # Integer percentage should be converted to float
        report = CoverageReport(
            type=CoverageType.LINE,
            percentage=85,  # int instead of float
            covered=85,
            total=100
        )
        assert isinstance(report.percentage, float)
        assert report.percentage == 85.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
