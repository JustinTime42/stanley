"""Data models for self-healing test system.

This module contains all Pydantic models for test failure analysis, repair,
flaky detection, optimization, and historical tracking.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import statistics


class FailureType(str, Enum):
    """Types of test failures."""

    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    ATTRIBUTE_ERROR = "attribute_error"
    ASSERTION_FAILED = "assertion_failed"
    TIMEOUT = "timeout"
    MOCK_ERROR = "mock_error"
    TYPE_ERROR = "type_error"
    KEY_ERROR = "key_error"
    RUNTIME_ERROR = "runtime_error"
    FLAKY = "flaky"


class RepairStrategy(str, Enum):
    """Test repair strategies."""

    UPDATE_SIGNATURE = "update_signature"  # Method signature changed
    UPDATE_ASSERTION = "update_assertion"  # Expected value changed
    UPDATE_IMPORT = "update_import"  # Module/import path changed
    UPDATE_MOCK = "update_mock"  # Mock configuration changed
    ADD_ASYNC = "add_async"  # Add async/await
    UPDATE_SELECTOR = "update_selector"  # UI selector changed
    REGENERATE = "regenerate"  # Complete regeneration needed
    ADD_WAIT = "add_wait"  # Add wait/delay for timing
    UPDATE_SETUP = "update_setup"  # Fix test setup/teardown


class TestFailure(BaseModel):
    """Represents a test failure."""

    test_id: str = Field(description="Unique test identifier")
    test_name: str = Field(description="Test name")
    test_file: str = Field(description="Test file path")
    failure_type: FailureType = Field(description="Type of failure")

    # Failure details
    error_message: str = Field(description="Error message from test runner")
    stack_trace: Optional[str] = Field(default=None, description="Full stack trace")
    line_number: Optional[int] = Field(
        default=None, description="Line where failure occurred"
    )

    # Context
    target_file: str = Field(description="File being tested")
    target_function: Optional[str] = Field(
        default=None, description="Function being tested"
    )
    test_framework: str = Field(description="Testing framework")

    # Timing
    execution_time_ms: Optional[int] = Field(
        default=None, description="Test execution time"
    )
    timestamp: datetime = Field(default_factory=datetime.now)

    # Previous attempts
    retry_count: int = Field(default=0, description="Number of retry attempts")
    previous_repairs: List[str] = Field(
        default_factory=list, description="Previous repair attempts"
    )


class FailureAnalysis(BaseModel):
    """Analysis of test failure root cause."""

    failure: TestFailure = Field(description="Original failure")
    root_cause: str = Field(description="Identified root cause")
    confidence: float = Field(ge=0, le=1, description="Confidence in analysis")

    # Detailed analysis
    code_changes: Optional[Dict[str, Any]] = Field(
        default=None, description="Related code changes"
    )
    dependency_changes: List[str] = Field(
        default_factory=list, description="Changed dependencies"
    )
    suggested_strategies: List[RepairStrategy] = Field(
        default_factory=list, description="Repair strategies"
    )

    # Supporting evidence
    evidence: List[str] = Field(default_factory=list, description="Evidence for root cause")
    similar_failures: List[str] = Field(
        default_factory=list, description="Similar historical failures"
    )


class TestRepair(BaseModel):
    """Represents a test repair action."""

    repair_id: str = Field(description="Unique repair ID")
    failure_analysis: FailureAnalysis = Field(description="Failure analysis")
    strategy: RepairStrategy = Field(description="Repair strategy used")

    # Repair details
    original_code: str = Field(description="Original test code")
    repaired_code: str = Field(description="Repaired test code")
    changes: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of changes made"
    )

    # Validation
    syntax_valid: bool = Field(default=False, description="Syntax validation passed")
    test_passes: bool = Field(default=False, description="Repaired test passes")
    coverage_maintained: bool = Field(
        default=False, description="Coverage maintained"
    )

    # Metadata
    repair_time_ms: int = Field(description="Time taken to repair")
    confidence: float = Field(ge=0, le=1, description="Confidence in repair")
    created_at: datetime = Field(default_factory=datetime.now)


class FlakyTestResult(BaseModel):
    """Result of flaky test detection."""

    test_id: str = Field(description="Test identifier")
    test_name: str = Field(description="Test name")
    flakiness_score: float = Field(
        ge=0, le=1, description="Probability of being flaky"
    )

    # Execution history
    total_runs: int = Field(description="Total number of runs")
    pass_count: int = Field(description="Number of passing runs")
    fail_count: int = Field(description="Number of failing runs")
    pass_rate: float = Field(ge=0, le=1, description="Pass rate")

    # Statistical analysis
    execution_times: List[float] = Field(
        default_factory=list, description="Execution times"
    )
    mean_time_ms: float = Field(description="Mean execution time")
    std_dev_time_ms: float = Field(description="Standard deviation of execution time")

    # Failure patterns
    failure_types: Dict[FailureType, int] = Field(
        default_factory=dict, description="Failure type counts"
    )
    failure_messages: List[str] = Field(
        default_factory=list, description="Unique failure messages"
    )

    # Recommendations
    is_flaky: bool = Field(description="Determined to be flaky")
    recommended_action: str = Field(description="Recommended action")
    root_causes: List[str] = Field(
        default_factory=list, description="Potential flakiness causes"
    )


class TestOptimization(BaseModel):
    """Test optimization suggestion."""

    optimization_id: str = Field(description="Unique optimization ID")
    test_id: str = Field(description="Test to optimize")
    optimization_type: str = Field(description="Type of optimization")

    # Performance impact
    current_time_ms: float = Field(description="Current execution time")
    estimated_time_ms: float = Field(description="Estimated time after optimization")
    time_saving_ms: float = Field(description="Expected time saving")
    time_saving_percent: float = Field(description="Percentage improvement")

    # Optimization details
    description: str = Field(description="Optimization description")
    implementation: str = Field(description="How to implement optimization")
    code_changes: Optional[str] = Field(
        default=None, description="Suggested code changes"
    )

    # Impact
    risk_level: str = Field(description="low|medium|high risk")
    affects_coverage: bool = Field(
        description="Whether optimization affects coverage"
    )
    priority: int = Field(description="Priority (1=highest)")


class TestPerformanceHistory(BaseModel):
    """Historical test performance data."""

    test_id: str = Field(description="Test identifier")

    # Execution history
    executions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Execution records"
    )

    # Aggregated metrics
    total_executions: int = Field(default=0, description="Total number of executions")
    total_failures: int = Field(default=0, description="Total failures")
    failure_rate: float = Field(default=0.0, description="Overall failure rate")

    # Performance trends
    avg_execution_time_ms: float = Field(description="Average execution time")
    execution_time_trend: str = Field(
        description="increasing|stable|decreasing"
    )

    # Failure analysis
    common_failure_types: Dict[FailureType, int] = Field(default_factory=dict)
    repair_history: List[TestRepair] = Field(default_factory=list)

    # Predictions
    predicted_failure_probability: float = Field(
        default=0.0, description="Failure probability"
    )
    predicted_maintenance_needed: bool = Field(default=False)

    # Metadata
    first_seen: datetime = Field(description="First test execution")
    last_updated: datetime = Field(default_factory=datetime.now)


class HealingRequest(BaseModel):
    """Request to heal failing tests."""

    test_failures: List[TestFailure] = Field(description="Failed tests to heal")
    auto_repair: bool = Field(default=True, description="Automatically apply repairs")
    detect_flaky: bool = Field(default=True, description="Detect flaky tests")
    optimize: bool = Field(default=False, description="Include optimization suggestions")
    max_repair_attempts: int = Field(
        default=3, description="Maximum repair attempts per test"
    )
    confidence_threshold: float = Field(
        default=0.7, description="Minimum confidence for auto-repair"
    )


class HealingResult(BaseModel):
    """Result of test healing process."""

    total_failures: int = Field(description="Total failures processed")
    successful_repairs: int = Field(description="Successfully repaired tests")
    failed_repairs: int = Field(description="Failed repair attempts")
    flaky_tests_detected: int = Field(description="Number of flaky tests detected")

    # Detailed results
    repairs: List[TestRepair] = Field(
        default_factory=list, description="All repair attempts"
    )
    flaky_tests: List[FlakyTestResult] = Field(
        default_factory=list, description="Flaky test results"
    )
    optimizations: List[TestOptimization] = Field(
        default_factory=list, description="Optimization suggestions"
    )

    # Summary statistics
    repair_success_rate: float = Field(description="Percentage of successful repairs")
    total_time_saved_ms: float = Field(
        description="Total time saved through optimizations"
    )

    # Metadata
    healing_time_ms: int = Field(description="Total healing process time")
    timestamp: datetime = Field(default_factory=datetime.now)
