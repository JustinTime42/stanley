# PRP-09: Multi-Framework Test Generation

## Goal

**Feature Goal**: Implement a comprehensive multi-framework test generation system that automatically creates unit tests, integration tests, and property-based tests for multiple testing frameworks (Pytest, Jest, JUnit, Go test, etc.) with intelligent test data generation and coverage optimization.

**Deliverable**: Test generation service with framework detection, code analysis-based test creation, test scaffolding, property-based testing support, intelligent test data generation, and coverage-guided test enhancement integrated with the Tester agent and existing code analysis infrastructure.

**Success Definition**:

- Support 5+ testing frameworks (Pytest, Jest, JUnit, Go test, Mocha)
- Generate unit tests achieving 80%+ code coverage automatically
- Create property-based tests for 90%+ of pure functions
- Test generation completes in <5 seconds for 100-line modules
- Generated tests have 95%+ syntactic correctness
- Test data generation covers edge cases with 85%+ effectiveness
- Integration test scaffolding reduces manual setup by 70%
- Mutation testing shows 75%+ test quality score

## Why

- Current Tester agent has placeholder implementation with no actual test generation
- No automatic test creation from code analysis
- Missing framework detection for polyglot codebases
- No property-based testing generation capabilities
- Test data generation is manual and incomplete
- No integration between code analysis (AST) and test generation
- Critical for achieving 80%+ test coverage goal autonomously
- Essential for validating generated code quality

## What

Implement a sophisticated multi-framework test generation system that analyzes code structure using AST, automatically detects the appropriate testing framework, generates comprehensive test suites including unit tests, integration tests, and property-based tests, with intelligent test data generation covering edge cases and boundary conditions.

### Success Criteria

- [ ] Framework detection accurate for 5+ testing frameworks
- [ ] Unit tests generated with 80%+ line coverage
- [ ] Property-based tests for applicable functions
- [ ] Integration test scaffolding with mocks/stubs
- [ ] Edge case test data generation
- [ ] Tests pass on first run 95%+ of the time
- [ ] Coverage-guided test enhancement operational

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete test generation patterns, framework-specific templates, property-based testing strategies, test data generation algorithms, and integration with existing systems.

### Documentation & References

```yaml
- url: https://hypothesis.readthedocs.io/en/latest/
  why: Hypothesis library for property-based testing in Python
  critical: Shows strategies for generating test data and properties
  section: Strategies and data generation

- url: https://github.com/dubzzz/fast-check
  why: Property-based testing for JavaScript/TypeScript
  critical: Fast-check library for Jest property tests

- url: https://docs.pytest.org/en/stable/how-to/fixtures.html
  why: Pytest fixture patterns for test setup
  critical: Shows fixture scopes, parameterization, and dependency injection

- url: https://jestjs.io/docs/mock-functions
  why: Jest mocking patterns for unit and integration tests
  critical: Mock creation, spy functions, module mocking

- file: src/agents/tester.py
  why: Existing Tester agent that will use test generation
  pattern: BaseAgent inheritance, execute method placeholder
  gotcha: Currently has _create_tests placeholder method

- file: src/analysis/ast_parser.py
  why: AST parser for understanding code structure
  pattern: ASTParser for extracting functions, classes, dependencies
  gotcha: Use for identifying testable units and their contracts

- file: src/tools/implementations/test_tools.py
  why: Test execution tools that run generated tests
  pattern: PytestTool, JestTool implementations
  gotcha: Need to integrate test generation with execution

- file: src/services/llm_service.py
  why: LLM service for generating test implementations
  pattern: LLMOrchestrator for code generation
  gotcha: Use appropriate model for test code generation

- file: src/analysis/dependency_analyzer.py
  why: Dependency analysis for identifying what to mock
  pattern: DependencyAnalyzer for import/call relationships
  gotcha: Critical for integration test setup

- url: https://github.com/boxed/mutmut
  why: Mutation testing for Python to validate test quality
  critical: Shows how to evaluate test effectiveness

- url: https://stryker-mutator.io/
  why: Mutation testing framework for JavaScript
  critical: Test quality validation through mutation analysis
```

### Current Codebase Tree

```bash
agent-swarm/
├── src/
│   ├── agents/
│   │   ├── tester.py               # Has _create_tests placeholder
│   │   ├── validator.py            # Uses test results
│   │   └── base.py                 # BaseAgent class
│   ├── analysis/
│   │   ├── ast_parser.py           # Code structure analysis
│   │   ├── dependency_analyzer.py  # Dependency graphs
│   │   ├── pattern_detector.py     # Code pattern detection
│   │   └── complexity_analyzer.py  # Complexity metrics
│   ├── tools/
│   │   └── implementations/
│   │       ├── test_tools.py       # Test execution tools
│   │       └── validation_tools.py # Code validation
│   ├── services/
│   │   ├── llm_service.py          # LLM orchestration
│   │   ├── analysis_service.py     # Code analysis service
│   │   └── tool_service.py         # Tool orchestration
│   └── models/
│       ├── analysis_models.py      # Code analysis models
│       └── tool_models.py          # Tool execution models
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── testing/                            # NEW: Test generation subsystem
│   │   ├── __init__.py                     # Export main interfaces
│   │   ├── base.py                         # BaseTestGenerator abstract class
│   │   ├── framework_detector.py           # Testing framework detection
│   │   ├── test_generator.py               # Main test generation orchestrator
│   │   ├── property_generator.py           # Property-based test generation
│   │   ├── data_generator.py               # Test data generation engine
│   │   ├── mock_generator.py               # Mock/stub generation
│   │   ├── coverage_analyzer.py            # Coverage analysis and gap detection
│   │   ├── test_enhancer.py                # Coverage-guided test enhancement
│   │   ├── frameworks/                     # Framework-specific generators
│   │   │   ├── __init__.py
│   │   │   ├── pytest_generator.py         # Pytest test generation
│   │   │   ├── jest_generator.py           # Jest test generation
│   │   │   ├── junit_generator.py          # JUnit test generation
│   │   │   ├── go_test_generator.py        # Go test generation
│   │   │   └── mocha_generator.py          # Mocha test generation
│   │   ├── templates/                      # Test templates
│   │   │   ├── __init__.py
│   │   │   ├── pytest_templates.py         # Pytest test templates
│   │   │   ├── jest_templates.py           # Jest test templates
│   │   │   └── junit_templates.py          # JUnit test templates
│   │   └── strategies/                     # Test generation strategies
│   │       ├── __init__.py
│   │       ├── boundary_testing.py         # Boundary value analysis
│   │       ├── equivalence_partitioning.py # Equivalence class testing
│   │       ├── path_coverage.py            # Path coverage strategies
│   │       └── mutation_guided.py          # Mutation-guided generation
│   ├── models/
│   │   └── testing_models.py              # NEW: Testing-related models
│   ├── services/
│   │   └── testing_service.py             # NEW: High-level testing service
│   ├── agents/
│   │   └── tester.py                      # MODIFY: Integrate test generation
│   └── tests/
│       └── testing/                        # NEW: Test generation tests
│           ├── test_framework_detection.py
│           ├── test_generation.py
│           ├── test_property_generation.py
│           └── test_data_generation.py
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: Generated tests must be syntactically valid for target framework
# Validate with framework's parser before saving

# CRITICAL: Test isolation is essential - each test must be independent
# Reset state, clear mocks between tests

# CRITICAL: Property-based testing requires pure functions
# Detect side effects and skip property generation for impure functions

# CRITICAL: Mock generation must handle circular dependencies
# Detect and break circular imports in test setup

# CRITICAL: Test data must respect type constraints
# Use type hints and docstrings to guide data generation

# CRITICAL: Coverage metrics vary by language/framework
# Normalize coverage reporting across different tools

# CRITICAL: Integration tests need environment setup
# Generate docker-compose or setup scripts when needed

# CRITICAL: Async/await testing requires special handling
# Different patterns for pytest-asyncio vs jest async tests

# CRITICAL: Test naming must follow framework conventions
# test_ prefix for pytest, .test. or .spec. for jest
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/models/testing_models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Set, Tuple, Literal
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
    BOUNDARY = "boundary"              # Boundary value analysis
    EQUIVALENCE = "equivalence"        # Equivalence partitioning
    RANDOM = "random"                  # Random generation
    PROPERTY = "property"              # Property-based strategies
    COMBINATORIAL = "combinatorial"    # All-pairs testing
    MUTATION = "mutation"              # Mutation-based

class MockType(str, Enum):
    """Types of test doubles."""
    MOCK = "mock"          # Full mock with expectations
    STUB = "stub"          # Simple return values
    SPY = "spy"            # Records calls
    FAKE = "fake"          # Working implementation
    DUMMY = "dummy"        # Placeholder object

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
    inputs: List[Dict[str, Any]] = Field(default_factory=list, description="Test input data")
    expected_outputs: List[Any] = Field(default_factory=list, description="Expected results")
    edge_cases: List[Dict[str, Any]] = Field(default_factory=list, description="Edge case inputs")

    # Mocking
    mocks: List[Dict[str, Any]] = Field(default_factory=list, description="Mock configurations")
    stubs: List[Dict[str, Any]] = Field(default_factory=list, description="Stub configurations")

    # Coverage
    lines_covered: Set[int] = Field(default_factory=set, description="Lines covered by test")
    branches_covered: Set[str] = Field(default_factory=set, description="Branches covered")
    coverage_percentage: float = Field(default=0.0, description="Coverage achieved")

    # Metadata
    framework: TestFramework = Field(description="Testing framework used")
    language: str = Field(description="Programming language")
    generated_at: datetime = Field(default_factory=datetime.now)
    execution_time_ms: Optional[int] = Field(default=None, description="Test execution time")
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
        default_factory=list,
        description="Example test inputs"
    )

    # Framework-specific
    framework_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Framework-specific configuration"
    )

class TestSuite(BaseModel):
    """Collection of related test cases."""
    id: str = Field(description="Test suite ID")
    name: str = Field(description="Suite name")
    target_module: str = Field(description="Module being tested")
    framework: TestFramework = Field(description="Testing framework")

    # Test cases
    test_cases: List[TestCase] = Field(default_factory=list, description="Individual tests")
    property_tests: List[PropertyTest] = Field(default_factory=list, description="Property tests")

    # Setup/Teardown
    suite_setup: Optional[str] = Field(default=None, description="Suite-level setup")
    suite_teardown: Optional[str] = Field(default=None, description="Suite-level teardown")

    # Fixtures/Helpers
    fixtures: Dict[str, str] = Field(default_factory=dict, description="Test fixtures")
    helpers: Dict[str, str] = Field(default_factory=dict, description="Helper functions")

    # Coverage
    total_coverage: float = Field(default=0.0, description="Overall coverage")
    line_coverage: float = Field(default=0.0, description="Line coverage")
    branch_coverage: float = Field(default=0.0, description="Branch coverage")

    # Quality metrics
    mutation_score: Optional[float] = Field(default=None, description="Mutation testing score")
    test_quality_score: float = Field(default=0.0, description="Overall test quality")

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    generation_time_ms: int = Field(default=0, description="Generation duration")

class TestGenerationRequest(BaseModel):
    """Request to generate tests."""
    target_files: List[str] = Field(description="Files to generate tests for")
    test_types: List[TestType] = Field(
        default_factory=lambda: [TestType.UNIT],
        description="Types of tests to generate"
    )
    framework: Optional[TestFramework] = Field(
        default=None,
        description="Force specific framework"
    )
    coverage_target: float = Field(default=0.8, ge=0, le=1, description="Target coverage")
    include_property_tests: bool = Field(default=True, description="Generate property tests")
    include_edge_cases: bool = Field(default=True, description="Include edge case testing")
    include_mocks: bool = Field(default=True, description="Generate mocks for dependencies")
    max_test_cases: int = Field(default=50, description="Maximum tests per file")

class TestDataSpec(BaseModel):
    """Specification for test data generation."""
    parameter_name: str = Field(description="Parameter to generate data for")
    data_type: str = Field(description="Data type of parameter")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Value constraints")

    # Strategies
    strategy: TestDataStrategy = Field(default=TestDataStrategy.BOUNDARY)

    # Generated values
    values: List[Any] = Field(default_factory=list, description="Generated test values")
    edge_cases: List[Any] = Field(default_factory=list, description="Edge case values")
    invalid_values: List[Any] = Field(default_factory=list, description="Invalid test values")

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
    call_assertions: List[str] = Field(default_factory=list, description="Call expectations")

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
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/testing/base.py
  - IMPLEMENT: BaseTestGenerator abstract class
  - FOLLOW pattern: Abstract base class with async methods
  - NAMING: BaseTestGenerator, generate_tests, analyze_code methods
  - PLACEMENT: Testing subsystem base module

Task 2: CREATE src/models/testing_models.py
  - IMPLEMENT: TestCase, TestSuite, PropertyTest models
  - FOLLOW pattern: Pydantic models with validation
  - NAMING: Models as specified in data models section
  - PLACEMENT: Models directory

Task 3: CREATE src/testing/framework_detector.py
  - IMPLEMENT: FrameworkDetector for auto-detecting test framework
  - FOLLOW pattern: File pattern matching, config file parsing
  - NAMING: FrameworkDetector, detect_framework, get_framework_config methods
  - DEPENDENCIES: File analysis, package.json/requirements.txt parsing
  - PLACEMENT: Testing subsystem

Task 4: CREATE src/testing/templates/pytest_templates.py
  - IMPLEMENT: Pytest test templates and patterns
  - FOLLOW pattern: Template strings with placeholders
  - NAMING: PYTEST_UNIT_TEMPLATE, PYTEST_FIXTURE_TEMPLATE constants
  - DEPENDENCIES: Python syntax knowledge
  - PLACEMENT: Templates module

Task 5: CREATE src/testing/templates/jest_templates.py
  - IMPLEMENT: Jest test templates and patterns
  - FOLLOW pattern: JavaScript/TypeScript test templates
  - NAMING: JEST_UNIT_TEMPLATE, JEST_MOCK_TEMPLATE constants
  - DEPENDENCIES: JavaScript syntax
  - PLACEMENT: Templates module

Task 6: CREATE src/testing/data_generator.py
  - IMPLEMENT: TestDataGenerator for intelligent test data
  - FOLLOW pattern: Type-based generation, constraint satisfaction
  - NAMING: TestDataGenerator, generate_boundary_values, generate_edge_cases methods
  - DEPENDENCIES: Type analysis, constraint solving
  - PLACEMENT: Testing subsystem

Task 7: CREATE src/testing/mock_generator.py
  - IMPLEMENT: MockGenerator for creating test doubles
  - FOLLOW pattern: Dependency analysis, interface extraction
  - NAMING: MockGenerator, generate_mock, generate_stub methods
  - DEPENDENCIES: AST parser, dependency analyzer
  - PLACEMENT: Testing subsystem

Task 8: CREATE src/testing/property_generator.py
  - IMPLEMENT: PropertyTestGenerator for property-based tests
  - FOLLOW pattern: Function purity analysis, property inference
  - NAMING: PropertyTestGenerator, generate_properties, create_strategies methods
  - DEPENDENCIES: AST parser, type analysis
  - PLACEMENT: Testing subsystem

Task 9: CREATE src/testing/frameworks/pytest_generator.py
  - IMPLEMENT: PytestGenerator for Python test generation
  - FOLLOW pattern: BaseTestGenerator inheritance
  - NAMING: PytestGenerator, generate_unit_test, generate_fixture methods
  - DEPENDENCIES: Pytest templates, Python AST
  - PLACEMENT: Framework-specific generators

Task 10: CREATE src/testing/frameworks/jest_generator.py
  - IMPLEMENT: JestGenerator for JavaScript test generation
  - FOLLOW pattern: BaseTestGenerator inheritance
  - NAMING: JestGenerator, generate_test_suite, generate_mock methods
  - DEPENDENCIES: Jest templates, JavaScript AST
  - PLACEMENT: Framework-specific generators

Task 11: CREATE src/testing/coverage_analyzer.py
  - IMPLEMENT: CoverageAnalyzer for identifying test gaps
  - FOLLOW pattern: Coverage report parsing, gap identification
  - NAMING: CoverageAnalyzer, analyze_coverage, identify_gaps methods
  - DEPENDENCIES: Coverage.py, jest-coverage parsing
  - PLACEMENT: Testing subsystem

Task 12: CREATE src/testing/test_enhancer.py
  - IMPLEMENT: TestEnhancer for coverage-guided enhancement
  - FOLLOW pattern: Iterative improvement based on coverage
  - NAMING: TestEnhancer, enhance_coverage, generate_missing_tests methods
  - DEPENDENCIES: Coverage analyzer, test generator
  - PLACEMENT: Testing subsystem

Task 13: CREATE src/testing/test_generator.py
  - IMPLEMENT: TestGenerator main orchestrator
  - FOLLOW pattern: Pipeline orchestration
  - NAMING: TestGenerator, generate_test_suite, validate_tests methods
  - DEPENDENCIES: All testing components
  - PLACEMENT: Testing subsystem root

Task 14: CREATE src/services/testing_service.py
  - IMPLEMENT: TestingOrchestrator high-level service
  - FOLLOW pattern: Service facade pattern
  - NAMING: TestingOrchestrator, generate_tests, run_tests methods
  - DEPENDENCIES: Test generator, test tools
  - PLACEMENT: Services layer

Task 15: MODIFY src/agents/tester.py
  - INTEGRATE: Replace _create_tests placeholder with test generation
  - FIND pattern: _create_tests method
  - REPLACE: Use TestingOrchestrator for test generation
  - PRESERVE: BaseAgent inheritance, state management

Task 16: CREATE src/testing/strategies/boundary_testing.py
  - IMPLEMENT: BoundaryTestStrategy for boundary value analysis
  - FOLLOW pattern: Strategy pattern for test generation
  - NAMING: BoundaryTestStrategy, generate_boundary_tests methods
  - DEPENDENCIES: Data generator, type analysis
  - PLACEMENT: Strategies module

Task 17: CREATE src/tests/testing/test_framework_detection.py
  - IMPLEMENT: Unit tests for framework detection
  - FOLLOW pattern: pytest with fixtures
  - COVERAGE: Multiple languages, edge cases
  - PLACEMENT: Testing tests directory

Task 18: CREATE src/tests/testing/test_generation.py
  - IMPLEMENT: Tests for test generation pipeline
  - FOLLOW pattern: Mock AST, verify generated tests
  - COVERAGE: All test types, frameworks
  - PLACEMENT: Testing tests directory
```

### Implementation Patterns & Key Details

```python
# Framework detection pattern
class FrameworkDetector:
    """
    PATTERN: Detect testing framework from project structure
    CRITICAL: Check multiple signals for accuracy
    """

    def detect_framework(self, project_path: str) -> TestFramework:
        """
        Detect testing framework from project files.
        PATTERN: Priority-based detection
        """
        # Check package files
        if os.path.exists(os.path.join(project_path, "package.json")):
            package_json = self._read_json("package.json")

            # Check devDependencies
            dev_deps = package_json.get("devDependencies", {})
            if "jest" in dev_deps:
                return TestFramework.JEST
            elif "mocha" in dev_deps:
                return TestFramework.MOCHA
            elif "jasmine" in dev_deps:
                return TestFramework.JASMINE

        if os.path.exists(os.path.join(project_path, "requirements.txt")):
            requirements = self._read_file("requirements.txt")
            if "pytest" in requirements:
                return TestFramework.PYTEST

        # Check for test files
        test_files = glob.glob(os.path.join(project_path, "**/*test*"), recursive=True)
        for file in test_files:
            if file.endswith("_test.go"):
                return TestFramework.GO_TEST
            elif "test_" in file and file.endswith(".py"):
                return TestFramework.PYTEST
            elif ".test." in file or ".spec." in file:
                return TestFramework.JEST

        # Default based on language
        return self._default_framework_for_language(project_path)

# Test generation pattern
class TestGenerator:
    """
    PATTERN: Orchestrate test generation pipeline
    CRITICAL: Validate generated tests before saving
    """

    def __init__(self, ast_parser, llm_service, framework_detector):
        self.ast_parser = ast_parser
        self.llm_service = llm_service
        self.framework_detector = framework_detector
        self.framework_generators = {}
        self._register_generators()

    async def generate_test_suite(
        self,
        target_file: str,
        request: TestGenerationRequest
    ) -> TestSuite:
        """
        Generate complete test suite for target file.
        PATTERN: Analysis → Generation → Validation
        """
        # Analyze code structure
        ast = await self.ast_parser.parse_file(target_file)
        functions = self._extract_testable_units(ast)

        # Detect or use specified framework
        framework = request.framework or self.framework_detector.detect_framework(
            os.path.dirname(target_file)
        )

        # Get appropriate generator
        generator = self.framework_generators[framework]

        # Generate tests for each function
        test_cases = []
        for func in functions:
            # Generate unit test
            if TestType.UNIT in request.test_types:
                unit_test = await generator.generate_unit_test(
                    func,
                    request.include_mocks
                )
                test_cases.append(unit_test)

            # Generate property test if applicable
            if request.include_property_tests and self._is_pure_function(func):
                prop_test = await self._generate_property_test(func, framework)
                test_cases.append(prop_test)

            # Generate edge case tests
            if request.include_edge_cases:
                edge_tests = await self._generate_edge_case_tests(func)
                test_cases.extend(edge_tests)

        # Create test suite
        suite = TestSuite(
            id=f"suite_{target_file}",
            name=f"Test suite for {os.path.basename(target_file)}",
            target_module=target_file,
            framework=framework,
            test_cases=test_cases
        )

        # Calculate coverage
        suite.total_coverage = await self._estimate_coverage(suite, ast)

        # Validate generated tests
        if not await self._validate_test_syntax(suite):
            raise ValueError("Generated tests have syntax errors")

        return suite

# Property-based test generation pattern
class PropertyTestGenerator:
    """
    PATTERN: Generate property-based tests
    CRITICAL: Only for pure functions without side effects
    """

    async def generate_properties(
        self,
        function: CodeEntity,
        framework: TestFramework
    ) -> PropertyTest:
        """
        Generate property-based test for function.
        PATTERN: Infer properties from function signature and docs
        """
        # Analyze function for properties
        properties = await self._infer_properties(function)

        # Generate input strategies based on types
        input_strategies = self._create_input_strategies(
            function.signature,
            framework
        )

        # Framework-specific generation
        if framework == TestFramework.PYTEST:
            return self._generate_hypothesis_test(
                function,
                properties,
                input_strategies
            )
        elif framework == TestFramework.JEST:
            return self._generate_fast_check_test(
                function,
                properties,
                input_strategies
            )

    def _infer_properties(self, function: CodeEntity) -> List[str]:
        """
        Infer properties from function analysis.
        PATTERN: Common property patterns
        """
        properties = []

        # Idempotence: f(f(x)) = f(x)
        if self._is_idempotent(function):
            properties.append("idempotent")

        # Commutativity: f(a, b) = f(b, a)
        if self._is_commutative(function):
            properties.append("commutative")

        # Invariants from docstring
        if function.docstring:
            invariants = self._extract_invariants(function.docstring)
            properties.extend(invariants)

        # Round-trip: decode(encode(x)) = x
        if self._is_encoder_decoder_pair(function):
            properties.append("round_trip")

        return properties

# Test data generation pattern
class TestDataGenerator:
    """
    PATTERN: Generate test data based on types and constraints
    CRITICAL: Cover edge cases systematically
    """

    def generate_test_data(
        self,
        parameter_spec: TestDataSpec,
        strategy: TestDataStrategy = TestDataStrategy.BOUNDARY
    ) -> List[Any]:
        """
        Generate test data for parameter.
        PATTERN: Type-aware generation with strategies
        """
        if strategy == TestDataStrategy.BOUNDARY:
            return self._generate_boundary_values(parameter_spec)
        elif strategy == TestDataStrategy.EQUIVALENCE:
            return self._generate_equivalence_partitions(parameter_spec)
        elif strategy == TestDataStrategy.PROPERTY:
            return self._generate_property_based(parameter_spec)

    def _generate_boundary_values(
        self,
        spec: TestDataSpec
    ) -> List[Any]:
        """
        Generate boundary test values.
        PATTERN: Type-specific boundaries
        """
        values = []
        data_type = spec.data_type

        if data_type == "int":
            # Integer boundaries
            min_val = spec.constraints.get("min", -sys.maxsize)
            max_val = spec.constraints.get("max", sys.maxsize)

            values.extend([
                min_val,           # Lower boundary
                min_val + 1,       # Just above lower
                max_val - 1,       # Just below upper
                max_val,           # Upper boundary
                0,                 # Zero
                -1,                # Negative
                1,                 # Positive
            ])

            # Add invalid values for negative testing
            spec.invalid_values = [min_val - 1, max_val + 1]

        elif data_type == "str":
            # String boundaries
            min_len = spec.constraints.get("min_length", 0)
            max_len = spec.constraints.get("max_length", 1000)

            values.extend([
                "",                                    # Empty
                "a" * min_len,                        # Minimum length
                "a" * max_len,                        # Maximum length
                "a" * (max_len // 2),                 # Medium length
                "Special!@#$%^&*()_+",                # Special characters
                "Unicode: Ω≈ç√∫˜µ≤≥÷",               # Unicode
                "\\n\\t\\r",                         # Escape sequences
            ])

        elif data_type == "list":
            # List boundaries
            values.extend([
                [],                                    # Empty list
                [None],                               # Single None
                [1],                                  # Single element
                list(range(1000)),                   # Large list
                [1, 2, 3, 2, 1],                     # Duplicates
            ])

        return values

# Mock generation pattern
class MockGenerator:
    """
    PATTERN: Generate appropriate test doubles
    CRITICAL: Respect dependency interfaces
    """

    async def generate_mock(
        self,
        dependency: str,
        mock_type: MockType,
        framework: TestFramework
    ) -> MockSpecification:
        """
        Generate mock for dependency.
        PATTERN: Framework-specific mock syntax
        """
        # Analyze dependency interface
        interface = await self._extract_interface(dependency)

        # Generate mock based on type
        if framework == TestFramework.PYTEST:
            return self._generate_pytest_mock(
                dependency,
                interface,
                mock_type
            )
        elif framework == TestFramework.JEST:
            return self._generate_jest_mock(
                dependency,
                interface,
                mock_type
            )

    def _generate_pytest_mock(
        self,
        dependency: str,
        interface: Dict[str, Any],
        mock_type: MockType
    ) -> MockSpecification:
        """
        Generate pytest mock.
        PATTERN: pytest-mock or unittest.mock
        """
        if mock_type == MockType.MOCK:
            syntax = f"""
@pytest.fixture
def mock_{dependency}(mocker):
    mock = mocker.patch('{dependency}')
    mock.return_value = {interface.get('default_return')}
    return mock
"""
        elif mock_type == MockType.STUB:
            syntax = f"""
@pytest.fixture
def stub_{dependency}():
    class Stub{dependency}:
        def __call__(self, *args, **kwargs):
            return {interface.get('default_return')}
    return Stub{dependency}()
"""

        return MockSpecification(
            target=dependency,
            mock_type=mock_type,
            framework_syntax=syntax,
            return_value=interface.get('default_return')
        )

# Coverage enhancement pattern
class TestEnhancer:
    """
    PATTERN: Iteratively improve test coverage
    CRITICAL: Target uncovered branches and paths
    """

    async def enhance_coverage(
        self,
        test_suite: TestSuite,
        coverage_report: Dict[str, Any],
        target_coverage: float = 0.8
    ) -> TestSuite:
        """
        Enhance test suite to improve coverage.
        PATTERN: Gap analysis → Targeted generation
        """
        current_coverage = coverage_report['total_coverage']

        while current_coverage < target_coverage:
            # Identify coverage gaps
            gaps = self._identify_coverage_gaps(coverage_report)

            if not gaps:
                break  # No more gaps to cover

            # Prioritize gaps by complexity
            prioritized_gaps = sorted(
                gaps,
                key=lambda g: g.complexity,
                reverse=True
            )

            # Generate tests for top gaps
            for gap in prioritized_gaps[:5]:  # Process top 5 gaps
                new_test = await self._generate_gap_test(gap)
                test_suite.test_cases.append(new_test)

            # Re-run coverage analysis
            coverage_report = await self._analyze_coverage(test_suite)
            current_coverage = coverage_report['total_coverage']

            # Avoid infinite loops
            if len(test_suite.test_cases) > 100:
                break

        test_suite.total_coverage = current_coverage
        return test_suite

# Template-based generation pattern
PYTEST_UNIT_TEMPLATE = """
import pytest
from unittest.mock import Mock, patch
{imports}

class Test{class_name}:
    \"\"\"Test suite for {target_function}\"\"\"

    {fixtures}

    def test_{test_name}(self{fixture_params}):
        \"\"\"Test: {test_description}\"\"\"
        # Arrange
        {setup_code}

        # Act
        {action_code}

        # Assert
        {assertions}

        {teardown_code}
"""

JEST_UNIT_TEMPLATE = """
import {{ {imports} }} from '{module_path}';

describe('{suite_name}', () => {{
    {setup_code}

    {teardown_code}

    it('{test_description}', async () => {{
        // Arrange
        {arrange_code}

        // Act
        {act_code}

        // Assert
        {assertions}
    }});
}});
"""
```

### Integration Points

```yaml
AST_PARSER:
  - integration: "Extract testable units from code"
  - pattern: "Identify functions, classes, methods"
  - usage: "Understand code structure for test generation"

DEPENDENCY_ANALYZER:
  - integration: "Identify dependencies to mock"
  - pattern: "Extract imports and calls"
  - usage: "Generate appropriate mocks and stubs"

LLM_SERVICE:
  - integration: "Generate test implementation code"
  - pattern: "Use code generation models"
  - temperature: 0.3 for deterministic test code

TOOL_SERVICE:
  - integration: "Execute generated tests"
  - pattern: "Use PytestTool, JestTool"
  - validation: "Run tests to verify correctness"

MEMORY_SERVICE:
  - integration: "Cache test templates and patterns"
  - pattern: "Store successful test patterns"
  - reuse: "Learn from previously generated tests"

CONFIG:
  - add to: .env
  - variables: |
      # Test Generation Configuration
      TEST_GENERATION_MAX_TESTS_PER_FILE=50
      TEST_GENERATION_COVERAGE_TARGET=0.8
      TEST_GENERATION_TIMEOUT_SECONDS=10

      # Framework Detection
      TEST_FRAMEWORK_AUTO_DETECT=true
      TEST_FRAMEWORK_DEFAULT=pytest

      # Property Testing
      PROPERTY_TEST_EXAMPLES=100
      PROPERTY_TEST_MAX_SHRINKS=500

      # Mock Generation
      MOCK_GENERATION_AUTO=true
      MOCK_DEFAULT_TYPE=mock

      # Coverage Analysis
      COVERAGE_MIN_LINE=0.8
      COVERAGE_MIN_BRANCH=0.7
      COVERAGE_MIN_FUNCTION=0.9

      # Test Data Generation
      TEST_DATA_STRATEGY=boundary
      TEST_DATA_MAX_VALUES=20
      TEST_DATA_INCLUDE_INVALID=true

DEPENDENCIES:
  - hypothesis: "Property-based testing for Python"
  - fast-check: "Property-based testing for JavaScript"
  - coverage: "Python coverage analysis"
  - jest: "JavaScript testing framework"
  - pytest: "Python testing framework"
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# After creating each file
ruff check src/testing/ --fix
mypy src/testing/ --strict
ruff format src/testing/

# Verify imports
python -c "from src.testing import TestGenerator; print('Testing imports OK')"
python -c "from src.services.testing_service import TestingOrchestrator; print('Service imports OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test framework detection
pytest src/tests/testing/test_framework_detection.py -v --cov=src/testing/framework_detector

# Test data generation
pytest src/tests/testing/test_data_generation.py -v --cov=src/testing/data_generator

# Test property generation
pytest src/tests/testing/test_property_generation.py -v --cov=src/testing/property_generator

# Test mock generation
pytest src/tests/testing/test_mock_generation.py -v --cov=src/testing/mock_generator

# Full testing suite
pytest src/tests/testing/ -v --cov=src/testing --cov-report=term-missing

# Expected: 90%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Test framework detection on real projects
python scripts/test_framework_detection.py \
  --projects ./test-projects/ \
  --verify-accuracy
# Expected: 95%+ detection accuracy

# Generate tests for Python module
python scripts/test_python_generation.py \
  --file src/agents/implementer.py \
  --framework pytest \
  --verify-syntax \
  --run-tests
# Expected: Valid pytest tests that pass

# Generate tests for JavaScript module
python scripts/test_javascript_generation.py \
  --file src/components/calculator.js \
  --framework jest \
  --verify-syntax \
  --run-tests
# Expected: Valid jest tests that pass

# Test property-based generation
python scripts/test_property_generation.py \
  --file src/utils/math_functions.py \
  --verify-properties \
  --run-hypothesis
# Expected: Property tests with valid strategies

# Test data generation validation
python scripts/test_data_generation_quality.py \
  --types "int,str,list,dict" \
  --verify-boundaries \
  --check-edge-cases
# Expected: Comprehensive test data including edge cases

# Mock generation validation
python scripts/test_mock_generation.py \
  --analyze-dependencies \
  --generate-mocks \
  --verify-interfaces
# Expected: Mocks respect dependency interfaces

# Coverage enhancement test
python scripts/test_coverage_enhancement.py \
  --initial-suite ./basic-tests/ \
  --target-coverage 0.8 \
  --verify-improvement
# Expected: Coverage improves to target level
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Large Codebase Test Generation
python scripts/test_large_codebase.py \
  --repository https://github.com/fastapi/fastapi \
  --sample-files 20 \
  --measure-time \
  --verify-quality
# Expected: <5s per 100-line module, 80%+ coverage

# Multi-Language Project Testing
python scripts/test_polyglot_project.py \
  --project ./full-stack-app/ \
  --languages "python,javascript,go" \
  --verify-framework-selection
# Expected: Correct framework for each language

# Property Test Quality Assessment
python scripts/test_property_quality.py \
  --pure-functions ./test-functions/ \
  --verify-properties \
  --measure-shrinking
# Expected: Properties cover main invariants

# Edge Case Effectiveness
python scripts/test_edge_case_effectiveness.py \
  --inject-bugs ./buggy-code/ \
  --generate-tests \
  --measure-bug-detection
# Expected: 85%+ edge case bugs detected

# Test Maintainability Analysis
python scripts/test_maintainability.py \
  --generated-tests ./generated/ \
  --measure-complexity \
  --check-duplication
# Expected: Low complexity, <10% duplication

# Mutation Testing Validation
python scripts/test_mutation_testing.py \
  --run-mutmut \
  --generated-tests ./tests/ \
  --verify-quality
# Expected: 75%+ mutation score

# Integration Test Generation
python scripts/test_integration_generation.py \
  --api-specs ./openapi.yaml \
  --generate-integration-tests \
  --verify-mocking
# Expected: Complete API test coverage

# Performance Test Generation
python scripts/test_performance_generation.py \
  --hot-paths ./profiling-data.json \
  --generate-benchmarks \
  --verify-metrics
# Expected: Performance tests for critical paths

# Test Execution Validation
python scripts/test_execution_validation.py \
  --generated-suites ./test-suites/ \
  --run-all-frameworks \
  --verify-results
# Expected: 95%+ generated tests pass

# Agent Integration Test
python scripts/test_tester_agent_integration.py \
  --task "Generate tests for authentication module" \
  --verify-agent-usage \
  --check-coverage
# Expected: Tester agent successfully uses test generation
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] Testing module tests achieve 90%+ coverage: `pytest src/tests/testing/ --cov=src/testing`
- [ ] No linting errors: `ruff check src/testing/`
- [ ] No type errors: `mypy src/testing/ --strict`
- [ ] All framework generators working

### Feature Validation

- [ ] 5+ testing frameworks supported
- [ ] Unit tests achieve 80%+ coverage automatically
- [ ] Property tests generated for pure functions
- [ ] Test data covers edge cases effectively
- [ ] Generated tests syntactically correct 95%+ of the time
- [ ] Mutation testing score 75%+
- [ ] Integration test scaffolding operational

### Code Quality Validation

- [ ] Follows existing agent and service patterns
- [ ] All operations async-compatible
- [ ] Proper error handling for test generation failures
- [ ] Templates validated for each framework
- [ ] Mock generation respects interfaces
- [ ] Test isolation maintained

### Documentation & Deployment

- [ ] Environment variables documented
- [ ] Test templates documented
- [ ] Framework detection rules documented
- [ ] Property strategies explained
- [ ] API endpoints for testing service
- [ ] Integration examples provided

---

## Anti-Patterns to Avoid

- ❌ Don't generate tests without validating syntax
- ❌ Don't ignore test isolation (shared state between tests)
- ❌ Don't generate property tests for impure functions
- ❌ Don't create circular dependencies in mocks
- ❌ Don't skip framework detection for polyglot projects
- ❌ Don't hardcode test data values
- ❌ Don't generate tests without assertions
- ❌ Don't exceed maximum test file size
- ❌ Don't ignore async/await test patterns
- ❌ Don't generate duplicate test cases
