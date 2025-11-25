"""Data models for quality gate system.

This module contains all Pydantic models for quality analysis, coverage reports,
security scanning, performance metrics, and quality gate enforcement.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field


class QualityDimension(str, Enum):
    """Quality analysis dimensions."""

    COVERAGE = "coverage"
    STATIC = "static"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLEXITY = "complexity"
    DOCUMENTATION = "documentation"


class CoverageType(str, Enum):
    """Types of coverage analysis."""

    LINE = "line"
    BRANCH = "branch"
    FUNCTION = "function"
    MUTATION = "mutation"


class QualityStatus(str, Enum):
    """Quality gate status."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    ERROR = "error"


class SeverityLevel(str, Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class CoverageReport(BaseModel):
    """Multi-level coverage report."""

    type: CoverageType = Field(description="Type of coverage")
    percentage: float = Field(ge=0, le=100, description="Coverage percentage")
    covered: int = Field(description="Number of covered items")
    total: int = Field(description="Total number of items")

    # Detailed breakdown
    files: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-file coverage data"
    )

    # Branch-specific
    branch_coverage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Branch coverage details"
    )

    # Mutation-specific
    mutation_score: Optional[float] = Field(
        default=None,
        description="Mutation testing score"
    )
    killed_mutants: Optional[int] = Field(default=None)
    survived_mutants: Optional[int] = Field(default=None)
    timeout_mutants: Optional[int] = Field(default=None)


class SecurityIssue(BaseModel):
    """Security vulnerability finding."""

    issue_id: str = Field(description="Issue identifier (e.g., CVE)")
    severity: SeverityLevel = Field(description="Issue severity")
    confidence: str = Field(description="Detection confidence")

    # Location
    file_path: str = Field(description="File containing issue")
    line_number: int = Field(description="Line number")
    column: Optional[int] = Field(default=None)

    # Details
    issue_type: str = Field(description="Type of vulnerability")
    description: str = Field(description="Issue description")
    remediation: str = Field(description="Suggested fix")

    # References
    cwe_id: Optional[str] = Field(default=None, description="CWE identifier")
    owasp_category: Optional[str] = Field(default=None)
    references: List[str] = Field(default_factory=list)


class PerformanceMetric(BaseModel):
    """Performance measurement."""

    metric_name: str = Field(description="Metric identifier")
    current_value: float = Field(description="Current measurement")
    baseline_value: Optional[float] = Field(default=None)

    # Regression detection
    regression_detected: bool = Field(default=False)
    regression_percentage: Optional[float] = Field(default=None)

    # Statistical significance
    confidence_interval: Optional[Tuple[float, float]] = Field(default=None)
    p_value: Optional[float] = Field(default=None)

    # Context
    test_name: str = Field(description="Associated test")
    environment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment details"
    )


class QualityThreshold(BaseModel):
    """Configurable quality threshold."""

    dimension: QualityDimension = Field(description="Quality dimension")
    metric: str = Field(description="Specific metric")

    # Thresholds
    min_value: Optional[float] = Field(default=None, description="Minimum acceptable")
    max_value: Optional[float] = Field(default=None, description="Maximum acceptable")
    target_value: Optional[float] = Field(default=None, description="Target value")

    # Gate configuration
    enforcement: str = Field(
        default="error",
        description="Action on violation: error, warning, info"
    )

    # Overrides
    allow_override: bool = Field(default=False)
    override_reason: Optional[str] = Field(default=None)


class QualityReport(BaseModel):
    """Comprehensive quality analysis report."""

    report_id: str = Field(description="Report identifier")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Overall status
    status: QualityStatus = Field(description="Overall quality gate status")
    passed: bool = Field(description="Whether quality gates passed")

    # Coverage analysis
    coverage_reports: Dict[CoverageType, CoverageReport] = Field(
        default_factory=dict,
        description="Coverage analysis by type"
    )

    # Static analysis
    static_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Static analysis findings"
    )
    code_quality_score: float = Field(
        default=0.0,
        description="Overall code quality score"
    )

    # Security analysis
    security_issues: List[SecurityIssue] = Field(
        default_factory=list,
        description="Security vulnerabilities"
    )
    security_score: float = Field(default=100.0)

    # Performance analysis
    performance_metrics: List[PerformanceMetric] = Field(
        default_factory=list,
        description="Performance measurements"
    )
    performance_regressions: int = Field(default=0)

    # Complexity metrics
    cyclomatic_complexity: float = Field(default=0.0)
    cognitive_complexity: float = Field(default=0.0)
    maintainability_index: float = Field(default=100.0)

    # Threshold violations
    violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Threshold violations"
    )

    # Historical comparison
    trend: str = Field(
        default="stable",
        description="Quality trend: improving, stable, degrading"
    )
    previous_score: Optional[float] = Field(default=None)

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )
