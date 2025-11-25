"""Data models for test generation."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Set, Literal
from enum import Enum
from datetime import datetime


class TestFramework(str, Enum):
    """Supported testing frameworks."""

    PYTEST = "pytest"
    JEST = "jest"
    JUNIT = "junit"
    GO_TEST = "go_test"
    MOCHA = "mocha"
    JASMINE = "jasmine"
    RSPEC = "rspec"
    XUNIT = "xunit"


class TestType(str, Enum):
    """Types of tests to generate."""

    UNIT = "unit"
    INTEGRATION = "integration"
    PROPERTY = "property"
    REGRESSION = "regression"
    SMOKE = "smoke"
    EDGE_CASE = "edge_case"
    PERFORMANCE = "performance"


class TestDataStrategy(str, Enum):
    """Test data generation strategies."""

    BOUNDARY = "boundary"  # Boundary value analysis
    EQUIVALENCE = "equivalence"  # Equivalence partitioning
    RANDOM = "random"  # Random generation
    PROPERTY = "property"  # Property-based strategies
    COMBINATORIAL = "combinatorial"  # All-pairs testing
    MUTATION = "mutation"  # Mutation-based


class MockType(str, Enum):
    """Types of test doubles."""

    MOCK = "mock"  # Full mock with expectations
    STUB = "stub"  # Simple return values
    SPY = "spy"  # Records calls
    FAKE = "fake"  # Working implementation
    DUMMY = "dummy"  # Placeholder object


class TestCase(BaseModel):
    """Represents a single test case."""

    id: str = Field(description="Unique test ID")
    name: str = Field(description="Test case name")
    description: str = Field(description="What the test validates")
    type: TestType = Field(description="Type of test")

    # Test structure
    target_function: str = Field(description="Function/method being tested")
    target_file: str = Field(description="File containing target")
    test_file: str = Field(description="Generated test file path")

    # Test components
    setup: Optional[str] = Field(default=None, description="Setup code")
    teardown: Optional[str] = Field(default=None, description="Teardown code")
    test_body: str = Field(description="Main test code")
    assertions: List[str] = Field(default_factory=list, description="Test assertions")

    # Test data
    inputs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Test input data"
    )
    expected_outputs: List[Any] = Field(
        default_factory=list, description="Expected results"
    )
    edge_cases: List[Dict[str, Any]] = Field(
        default_factory=list, description="Edge case inputs"
    )

    # Mocking
    mocks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Mock configurations"
    )
    stubs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Stub configurations"
    )

    # Coverage
    lines_covered: Set[int] = Field(
        default_factory=set, description="Lines covered by test"
    )
    branches_covered: Set[str] = Field(
        default_factory=set, description="Branches covered"
    )
    coverage_percentage: float = Field(default=0.0, description="Coverage achieved")

    # Metadata
    framework: TestFramework = Field(description="Testing framework used")
    language: str = Field(description="Programming language")
    generated_at: datetime = Field(default_factory=datetime.now)
    execution_time_ms: Optional[int] = Field(
        default=None, description="Test execution time"
    )
    passed: Optional[bool] = Field(default=None, description="Test pass status")


class PropertyTest(BaseModel):
    """Property-based test specification."""

    id: str = Field(description="Property test ID")
    function_name: str = Field(description="Function to test")
    properties: List[str] = Field(description="Properties to verify")

    # Strategies
    input_strategies: Dict[str, str] = Field(
        description="Input generation strategies per parameter"
    )
    invariants: List[str] = Field(description="Invariants to check")

    # Examples
    example_inputs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Example test inputs"
    )

    # Framework-specific
    framework_config: Dict[str, Any] = Field(
        default_factory=dict, description="Framework-specific configuration"
    )


class TestSuite(BaseModel):
    """Collection of related test cases."""

    id: str = Field(description="Test suite ID")
    name: str = Field(description="Suite name")
    target_module: str = Field(description="Module being tested")
    framework: TestFramework = Field(description="Testing framework")

    # Test cases
    test_cases: List[TestCase] = Field(
        default_factory=list, description="Individual tests"
    )
    property_tests: List[PropertyTest] = Field(
        default_factory=list, description="Property tests"
    )

    # Setup/Teardown
    suite_setup: Optional[str] = Field(default=None, description="Suite-level setup")
    suite_teardown: Optional[str] = Field(
        default=None, description="Suite-level teardown"
    )

    # Fixtures/Helpers
    fixtures: Dict[str, str] = Field(default_factory=dict, description="Test fixtures")
    helpers: Dict[str, str] = Field(
        default_factory=dict, description="Helper functions"
    )

    # Coverage
    total_coverage: float = Field(default=0.0, description="Overall coverage")
    line_coverage: float = Field(default=0.0, description="Line coverage")
    branch_coverage: float = Field(default=0.0, description="Branch coverage")

    # Quality metrics
    mutation_score: Optional[float] = Field(
        default=None, description="Mutation testing score"
    )
    test_quality_score: float = Field(default=0.0, description="Overall test quality")

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    generation_time_ms: int = Field(default=0, description="Generation duration")


class TestGenerationRequest(BaseModel):
    """Request to generate tests."""

    target_files: List[str] = Field(description="Files to generate tests for")
    test_types: List[TestType] = Field(
        default_factory=lambda: [TestType.UNIT],
        description="Types of tests to generate",
    )
    framework: Optional[TestFramework] = Field(
        default=None, description="Force specific framework"
    )
    coverage_target: float = Field(
        default=0.8, ge=0, le=1, description="Target coverage"
    )
    include_property_tests: bool = Field(
        default=True, description="Generate property tests"
    )
    include_edge_cases: bool = Field(default=True, description="Include edge case testing")
    include_mocks: bool = Field(default=True, description="Generate mocks for dependencies")
    max_test_cases: int = Field(default=50, description="Maximum tests per file")


class TestDataSpec(BaseModel):
    """Specification for test data generation."""

    parameter_name: str = Field(description="Parameter to generate data for")
    data_type: str = Field(description="Data type of parameter")
    constraints: Dict[str, Any] = Field(
        default_factory=dict, description="Value constraints"
    )

    # Strategies
    strategy: TestDataStrategy = Field(default=TestDataStrategy.BOUNDARY)

    # Generated values
    values: List[Any] = Field(default_factory=list, description="Generated test values")
    edge_cases: List[Any] = Field(default_factory=list, description="Edge case values")
    invalid_values: List[Any] = Field(
        default_factory=list, description="Invalid test values"
    )


class CoverageGap(BaseModel):
    """Represents uncovered code that needs tests."""

    file_path: str = Field(description="File with coverage gap")
    start_line: int = Field(description="Start of uncovered region")
    end_line: int = Field(description="End of uncovered region")
    type: Literal["line", "branch", "function"] = Field(description="Type of gap")
    complexity: int = Field(default=1, description="Complexity of uncovered code")
    suggested_test_type: TestType = Field(description="Recommended test type")
    reason: str = Field(description="Why this gap exists")


class MockSpecification(BaseModel):
    """Specification for mock/stub generation."""

    target: str = Field(description="What to mock (module/class/function)")
    mock_type: MockType = Field(description="Type of test double")

    # Mock behavior
    return_value: Optional[Any] = Field(default=None, description="Return value")
    side_effect: Optional[str] = Field(default=None, description="Side effect code")
    call_assertions: List[str] = Field(
        default_factory=list, description="Call expectations"
    )

    # Framework-specific
    framework_syntax: str = Field(description="Framework-specific mock syntax")


class TestQualityMetrics(BaseModel):
    """Metrics for test quality assessment."""

    coverage: float = Field(description="Code coverage percentage")
    mutation_score: float = Field(default=0.0, description="Mutation testing score")
    assertion_density: float = Field(description="Assertions per test")
    test_maintainability: float = Field(description="Maintainability score")
    duplication: float = Field(description="Test code duplication percentage")
    edge_case_coverage: float = Field(description="Edge case coverage")
    mock_usage: float = Field(description="Appropriate mock usage score")
