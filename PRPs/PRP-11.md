# PRP-11: Self-Healing Test System

## Goal

**Feature Goal**: Implement an intelligent self-healing test system that automatically adapts tests to code changes, detects and fixes test failures, identifies flaky tests, suggests optimizations, and maintains historical test performance tracking to minimize manual test maintenance effort.

**Deliverable**: Self-healing test service with test failure analysis engine, automatic test repair mechanisms, flaky test detection, test optimization recommendations, and historical performance tracking integrated with the existing test generation infrastructure from PRP-09 and code analysis capabilities from PRP-05.

**Success Definition**:

- Automatically fix 75%+ of test failures caused by code changes
- Detect flaky tests with 90%+ accuracy
- Reduce manual test maintenance effort by 70%
- Test repair suggestions have 85%+ acceptance rate
- False positive reduction of 80%+ in test failures
- Test suite execution time optimization of 30%+
- Historical tracking enables predictive test failure detection
- Self-healing completes in <10 seconds for typical failures

## Why

- Generated tests from PRP-09 need maintenance when code evolves
- Test failures often caused by minor changes (renamed methods, parameter changes)
- Flaky tests waste developer time and reduce confidence
- No automatic adaptation when implementation changes but behavior remains
- Missing historical analysis to predict and prevent test failures
- Manual test maintenance is time-consuming and error-prone
- Critical for maintaining high test coverage as codebase evolves
- Essential for autonomous development with minimal human intervention

## What

Implement a comprehensive self-healing test system that analyzes test failures, automatically repairs tests when code changes, detects and isolates flaky tests, provides optimization suggestions, and maintains historical performance data to enable predictive maintenance and continuous test suite improvement.

### Success Criteria

- [ ] Test failure root cause analysis with 90%+ accuracy
- [ ] Automatic test repair for common failure patterns
- [ ] Flaky test detection using statistical analysis
- [ ] Test optimization suggestions reduce execution time by 30%+
- [ ] Historical tracking of test performance metrics
- [ ] Integration with existing test generation from PRP-09

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete self-healing patterns, failure analysis algorithms, flaky test detection strategies, optimization techniques, and integration with existing test and analysis systems.

### Documentation & References

```yaml
- url: https://github.com/facebook/jest/blob/main/packages/jest-circus/src/utils.ts
  why: Jest's test retry and failure analysis patterns
  critical: Shows how to analyze test failures and implement retry logic
  section: Error handling and retry mechanisms

- url: https://martinfowler.com/articles/nonDeterminism.html
  why: Comprehensive guide on dealing with non-deterministic tests
  critical: Strategies for identifying and fixing flaky tests

- url: https://engineering.salesforce.com/flaky-test-detection-d6d8c6c6e9e1
  why: Statistical approaches to flaky test detection
  critical: Algorithms for identifying non-deterministic behavior

- url: https://github.com/pytest-dev/pytest-rerunfailures
  why: Pytest plugin for rerunning failed tests
  critical: Patterns for test retry and failure analysis

- file: src/testing/test_generator.py
  why: Existing test generation system to integrate with
  pattern: TestGenerator class, generate_test_suite method
  gotcha: Must maintain compatibility with generated test format

- file: src/analysis/ast_parser.py
  why: AST analysis for understanding code changes
  pattern: ASTParser for detecting method renames, signature changes
  gotcha: Use for mapping old test to new code structure

- file: src/testing/coverage_analyzer.py
  why: Coverage analysis for test effectiveness
  pattern: CoverageAnalyzer for identifying test gaps
  gotcha: Use to ensure repairs maintain coverage

- file: src/tools/implementations/test_tools.py
  why: Test execution tools for running and analyzing tests
  pattern: PytestTool, JestTool for test execution
  gotcha: Need to capture detailed failure information

- file: src/llm/providers/openai_provider.py
  why: LLM service for generating test repairs
  pattern: LLM integration for code generation
  gotcha: Use for complex test repairs requiring understanding

- file: src/memory/project.py
  why: Project memory for storing test history
  pattern: QdrantProjectMemory for historical tracking
  gotcha: Store test performance metrics over time

- url: https://github.com/bazelbuild/bazel/tree/master/src/main/java/com/google/devtools/build/lib/analysis/test
  why: Bazel's test analysis and caching strategies
  critical: Test result caching and incremental testing patterns
```

### Current Codebase Tree

```bash
agent-swarm/
├── src/
│   ├── testing/                    # Test generation from PRP-09
│   │   ├── base.py                 # BaseTestGenerator
│   │   ├── test_generator.py       # Test generation orchestrator
│   │   ├── coverage_analyzer.py    # Coverage analysis
│   │   ├── test_enhancer.py        # Coverage enhancement
│   │   ├── data_generator.py       # Test data generation
│   │   ├── mock_generator.py       # Mock generation
│   │   ├── property_generator.py   # Property-based tests
│   │   └── frameworks/              # Framework-specific generators
│   ├── analysis/                   # Code analysis from PRP-05
│   │   ├── ast_parser.py           # AST parsing
│   │   ├── dependency_analyzer.py  # Dependency analysis
│   │   └── pattern_detector.py     # Pattern detection
│   ├── tools/
│   │   └── implementations/
│   │       └── test_tools.py       # Test execution tools
│   ├── memory/                     # Memory system from PRP-01
│   │   ├── project.py              # Project memory
│   │   └── hybrid.py               # Hybrid search
│   ├── llm/                        # LLM service from PRP-03
│   │   └── providers/              # LLM providers
│   └── models/
│       └── testing_models.py       # Testing data models
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── testing/
│   │   ├── healing/                         # NEW: Self-healing subsystem
│   │   │   ├── __init__.py                  # Export main interfaces
│   │   │   ├── base.py                      # BaseHealer abstract class
│   │   │   ├── failure_analyzer.py          # Test failure root cause analysis
│   │   │   ├── test_repairer.py             # Automatic test repair engine
│   │   │   ├── flaky_detector.py            # Flaky test detection
│   │   │   ├── test_optimizer.py            # Test optimization suggestions
│   │   │   ├── history_tracker.py           # Historical performance tracking
│   │   │   ├── repair_strategies/           # Repair strategy implementations
│   │   │   │   ├── __init__.py
│   │   │   │   ├── signature_repair.py      # Method signature change repairs
│   │   │   │   ├── assertion_repair.py      # Assertion update repairs
│   │   │   │   ├── import_repair.py         # Import/module change repairs
│   │   │   │   ├── mock_repair.py           # Mock configuration repairs
│   │   │   │   └── async_repair.py          # Async/await pattern repairs
│   │   │   └── analyzers/                   # Failure analysis strategies
│   │   │       ├── __init__.py
│   │   │       ├── syntax_analyzer.py       # Syntax error analysis
│   │   │       ├── runtime_analyzer.py      # Runtime error analysis
│   │   │       ├── assertion_analyzer.py    # Assertion failure analysis
│   │   │       └── timeout_analyzer.py      # Timeout/performance analysis
│   ├── models/
│   │   └── healing_models.py                # NEW: Self-healing related models
│   ├── services/
│   │   └── healing_service.py               # NEW: High-level healing service
│   ├── agents/
│   │   └── tester.py                        # MODIFY: Integrate self-healing
│   └── tests/
│       └── testing/
│           └── healing/                      # NEW: Self-healing tests
│               ├── test_failure_analysis.py
│               ├── test_repair.py
│               ├── test_flaky_detection.py
│               └── test_optimization.py
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: Test repairs must maintain test validity and coverage
# Validate repaired tests still test the intended behavior

# CRITICAL: Flaky test detection requires multiple test runs
# Need statistical significance (minimum 10 runs recommended)

# CRITICAL: AST changes don't always map 1:1 to test changes
# Complex refactoring may require LLM-based repair

# CRITICAL: Test execution isolation is essential for accurate analysis
# Each test run must be independent to detect flakiness

# CRITICAL: Historical data storage can grow large
# Implement data retention policies and aggregation

# CRITICAL: Different test frameworks have different failure formats
# Normalize error messages across pytest, jest, etc.

# CRITICAL: Race conditions in async tests need special handling
# Timing-based failures are common source of flakiness

# CRITICAL: Mock repairs must preserve interface contracts
# Validate mock behavior matches actual dependencies

# CRITICAL: Performance regression detection needs baselines
# Establish performance baselines for comparison
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/models/healing_models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Set, Tuple, Literal
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
    UPDATE_SIGNATURE = "update_signature"      # Method signature changed
    UPDATE_ASSERTION = "update_assertion"      # Expected value changed
    UPDATE_IMPORT = "update_import"           # Module/import path changed
    UPDATE_MOCK = "update_mock"               # Mock configuration changed
    ADD_ASYNC = "add_async"                   # Add async/await
    UPDATE_SELECTOR = "update_selector"       # UI selector changed
    REGENERATE = "regenerate"                 # Complete regeneration needed
    ADD_WAIT = "add_wait"                     # Add wait/delay for timing
    UPDATE_SETUP = "update_setup"             # Fix test setup/teardown

class TestFailure(BaseModel):
    """Represents a test failure."""
    test_id: str = Field(description="Unique test identifier")
    test_name: str = Field(description="Test name")
    test_file: str = Field(description="Test file path")
    failure_type: FailureType = Field(description="Type of failure")

    # Failure details
    error_message: str = Field(description="Error message from test runner")
    stack_trace: Optional[str] = Field(default=None, description="Full stack trace")
    line_number: Optional[int] = Field(default=None, description="Line where failure occurred")

    # Context
    target_file: str = Field(description="File being tested")
    target_function: Optional[str] = Field(default=None, description="Function being tested")
    test_framework: str = Field(description="Testing framework")

    # Timing
    execution_time_ms: Optional[int] = Field(default=None, description="Test execution time")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Previous attempts
    retry_count: int = Field(default=0, description="Number of retry attempts")
    previous_repairs: List[str] = Field(default_factory=list, description="Previous repair attempts")

class FailureAnalysis(BaseModel):
    """Analysis of test failure root cause."""
    failure: TestFailure = Field(description="Original failure")
    root_cause: str = Field(description="Identified root cause")
    confidence: float = Field(ge=0, le=1, description="Confidence in analysis")

    # Detailed analysis
    code_changes: Optional[Dict[str, Any]] = Field(default=None, description="Related code changes")
    dependency_changes: List[str] = Field(default_factory=list, description="Changed dependencies")
    suggested_strategies: List[RepairStrategy] = Field(default_factory=list, description="Repair strategies")

    # Supporting evidence
    evidence: List[str] = Field(default_factory=list, description="Evidence for root cause")
    similar_failures: List[str] = Field(default_factory=list, description="Similar historical failures")

class TestRepair(BaseModel):
    """Represents a test repair action."""
    repair_id: str = Field(description="Unique repair ID")
    failure_analysis: FailureAnalysis = Field(description="Failure analysis")
    strategy: RepairStrategy = Field(description="Repair strategy used")

    # Repair details
    original_code: str = Field(description="Original test code")
    repaired_code: str = Field(description="Repaired test code")
    changes: List[Dict[str, Any]] = Field(default_factory=list, description="List of changes made")

    # Validation
    syntax_valid: bool = Field(default=False, description="Syntax validation passed")
    test_passes: bool = Field(default=False, description="Repaired test passes")
    coverage_maintained: bool = Field(default=False, description="Coverage maintained")

    # Metadata
    repair_time_ms: int = Field(description="Time taken to repair")
    confidence: float = Field(ge=0, le=1, description="Confidence in repair")
    created_at: datetime = Field(default_factory=datetime.now)

class FlakyTestResult(BaseModel):
    """Result of flaky test detection."""
    test_id: str = Field(description="Test identifier")
    test_name: str = Field(description="Test name")
    flakiness_score: float = Field(ge=0, le=1, description="Probability of being flaky")

    # Execution history
    total_runs: int = Field(description="Total number of runs")
    pass_count: int = Field(description="Number of passing runs")
    fail_count: int = Field(description="Number of failing runs")
    pass_rate: float = Field(ge=0, le=1, description="Pass rate")

    # Statistical analysis
    execution_times: List[float] = Field(default_factory=list, description="Execution times")
    mean_time_ms: float = Field(description="Mean execution time")
    std_dev_time_ms: float = Field(description="Standard deviation of execution time")

    # Failure patterns
    failure_types: Dict[FailureType, int] = Field(default_factory=dict, description="Failure type counts")
    failure_messages: List[str] = Field(default_factory=list, description="Unique failure messages")

    # Recommendations
    is_flaky: bool = Field(description="Determined to be flaky")
    recommended_action: str = Field(description="Recommended action")
    root_causes: List[str] = Field(default_factory=list, description="Potential flakiness causes")

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
    code_changes: Optional[str] = Field(default=None, description="Suggested code changes")

    # Impact
    risk_level: str = Field(description="low|medium|high risk")
    affects_coverage: bool = Field(description="Whether optimization affects coverage")
    priority: int = Field(description="Priority (1=highest)")

class TestPerformanceHistory(BaseModel):
    """Historical test performance data."""
    test_id: str = Field(description="Test identifier")

    # Execution history
    executions: List[Dict[str, Any]] = Field(default_factory=list, description="Execution records")

    # Aggregated metrics
    total_executions: int = Field(default=0, description="Total number of executions")
    total_failures: int = Field(default=0, description="Total failures")
    failure_rate: float = Field(default=0.0, description="Overall failure rate")

    # Performance trends
    avg_execution_time_ms: float = Field(description="Average execution time")
    execution_time_trend: str = Field(description="increasing|stable|decreasing")

    # Failure analysis
    common_failure_types: Dict[FailureType, int] = Field(default_factory=dict)
    repair_history: List[TestRepair] = Field(default_factory=list)

    # Predictions
    predicted_failure_probability: float = Field(default=0.0, description="Failure probability")
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
    max_repair_attempts: int = Field(default=3, description="Maximum repair attempts per test")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for auto-repair")

class HealingResult(BaseModel):
    """Result of test healing process."""
    total_failures: int = Field(description="Total failures processed")
    successful_repairs: int = Field(description="Successfully repaired tests")
    failed_repairs: int = Field(description="Failed repair attempts")
    flaky_tests_detected: int = Field(description="Number of flaky tests detected")

    # Detailed results
    repairs: List[TestRepair] = Field(default_factory=list, description="All repair attempts")
    flaky_tests: List[FlakyTestResult] = Field(default_factory=list, description="Flaky test results")
    optimizations: List[TestOptimization] = Field(default_factory=list, description="Optimization suggestions")

    # Summary statistics
    repair_success_rate: float = Field(description="Percentage of successful repairs")
    total_time_saved_ms: float = Field(description="Total time saved through optimizations")

    # Metadata
    healing_time_ms: int = Field(description="Total healing process time")
    timestamp: datetime = Field(default_factory=datetime.now)
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/testing/healing/base.py
  - IMPLEMENT: BaseHealer abstract class defining healing interface
  - FOLLOW pattern: Abstract base class with async methods
  - NAMING: BaseHealer, analyze_failure, repair_test methods
  - PLACEMENT: Healing subsystem base module

Task 2: CREATE src/models/healing_models.py
  - IMPLEMENT: TestFailure, FailureAnalysis, TestRepair models
  - FOLLOW pattern: Pydantic models with validation
  - NAMING: Models as specified in data models section
  - PLACEMENT: Models directory

Task 3: CREATE src/testing/healing/failure_analyzer.py
  - IMPLEMENT: FailureAnalyzer for root cause analysis
  - FOLLOW pattern: Multi-strategy analysis with confidence scoring
  - NAMING: FailureAnalyzer, analyze_failure, identify_root_cause methods
  - DEPENDENCIES: AST parser, error parsing, pattern matching
  - PLACEMENT: Healing subsystem

Task 4: CREATE src/testing/healing/analyzers/syntax_analyzer.py
  - IMPLEMENT: SyntaxErrorAnalyzer for syntax error analysis
  - FOLLOW pattern: Error message parsing, line number extraction
  - NAMING: SyntaxErrorAnalyzer, analyze_syntax_error methods
  - DEPENDENCIES: Language-specific parsers
  - PLACEMENT: Analyzers module

Task 5: CREATE src/testing/healing/analyzers/assertion_analyzer.py
  - IMPLEMENT: AssertionAnalyzer for assertion failure analysis
  - FOLLOW pattern: Expected vs actual comparison, value extraction
  - NAMING: AssertionAnalyzer, analyze_assertion_failure methods
  - DEPENDENCIES: Test framework output parsing
  - PLACEMENT: Analyzers module

Task 6: CREATE src/testing/healing/repair_strategies/signature_repair.py
  - IMPLEMENT: SignatureRepair for method signature changes
  - FOLLOW pattern: AST comparison, parameter mapping
  - NAMING: SignatureRepair, repair_signature_mismatch methods
  - DEPENDENCIES: AST parser, diff analysis
  - PLACEMENT: Repair strategies module

Task 7: CREATE src/testing/healing/repair_strategies/assertion_repair.py
  - IMPLEMENT: AssertionRepair for updating assertions
  - FOLLOW pattern: Value update, assertion type changes
  - NAMING: AssertionRepair, update_assertion methods
  - DEPENDENCIES: Test execution, value extraction
  - PLACEMENT: Repair strategies module

Task 8: CREATE src/testing/healing/repair_strategies/mock_repair.py
  - IMPLEMENT: MockRepair for fixing mock configurations
  - FOLLOW pattern: Mock interface updates, call signature fixes
  - NAMING: MockRepair, repair_mock_configuration methods
  - DEPENDENCIES: Mock generator from PRP-09
  - PLACEMENT: Repair strategies module

Task 9: CREATE src/testing/healing/test_repairer.py
  - IMPLEMENT: TestRepairer orchestrating repair strategies
  - FOLLOW pattern: Strategy selection, repair application
  - NAMING: TestRepairer, repair_test, apply_repair methods
  - DEPENDENCIES: All repair strategies, test validator
  - PLACEMENT: Healing subsystem

Task 10: CREATE src/testing/healing/flaky_detector.py
  - IMPLEMENT: FlakyDetector for identifying non-deterministic tests
  - FOLLOW pattern: Statistical analysis, multiple test runs
  - NAMING: FlakyDetector, detect_flaky_tests, calculate_flakiness methods
  - DEPENDENCIES: Test runner, statistical analysis
  - PLACEMENT: Healing subsystem

Task 11: CREATE src/testing/healing/test_optimizer.py
  - IMPLEMENT: TestOptimizer for performance improvements
  - FOLLOW pattern: Performance profiling, optimization suggestions
  - NAMING: TestOptimizer, analyze_performance, suggest_optimizations methods
  - DEPENDENCIES: Performance metrics, test analysis
  - PLACEMENT: Healing subsystem

Task 12: CREATE src/testing/healing/history_tracker.py
  - IMPLEMENT: HistoryTracker for test performance tracking
  - FOLLOW pattern: Time-series data storage, trend analysis
  - NAMING: HistoryTracker, record_execution, analyze_trends methods
  - DEPENDENCIES: Memory service from PRP-01, Redis
  - PLACEMENT: Healing subsystem

Task 13: CREATE src/services/healing_service.py
  - IMPLEMENT: HealingOrchestrator high-level service
  - FOLLOW pattern: Service facade pattern
  - NAMING: HealingOrchestrator, heal_tests, analyze_test_health methods
  - DEPENDENCIES: All healing components
  - PLACEMENT: Services layer

Task 14: MODIFY src/agents/tester.py
  - INTEGRATE: Add self-healing capabilities to tester agent
  - FIND pattern: execute method, test execution logic
  - ADD: Automatic healing on test failures
  - PRESERVE: Existing test generation from PRP-09

Task 15: CREATE src/tests/testing/healing/test_failure_analysis.py
  - IMPLEMENT: Unit tests for failure analysis
  - FOLLOW pattern: pytest-asyncio with fixtures
  - COVERAGE: All failure types, root cause detection
  - PLACEMENT: Healing test directory

Task 16: CREATE src/tests/testing/healing/test_repair.py
  - IMPLEMENT: Tests for repair strategies
  - FOLLOW pattern: Mock failures, verify repairs
  - COVERAGE: All repair strategies, edge cases
  - PLACEMENT: Healing test directory

Task 17: CREATE src/tests/testing/healing/test_flaky_detection.py
  - IMPLEMENT: Tests for flaky detection
  - FOLLOW pattern: Simulated flaky behavior
  - COVERAGE: Statistical accuracy, various flakiness patterns
  - PLACEMENT: Healing test directory

Task 18: CREATE src/tests/testing/healing/test_optimization.py
  - IMPLEMENT: Tests for optimization suggestions
  - FOLLOW pattern: Performance analysis validation
  - COVERAGE: Time savings, risk assessment
  - PLACEMENT: Healing test directory
```

### Implementation Patterns & Key Details

```python
# Failure analysis pattern
class FailureAnalyzer:
    """
    PATTERN: Multi-strategy failure root cause analysis
    CRITICAL: Must handle framework-specific error formats
    """

    def __init__(self, ast_parser, dependency_analyzer):
        self.ast_parser = ast_parser
        self.dependency_analyzer = dependency_analyzer
        self.analyzers = {}
        self._register_analyzers()

    async def analyze_failure(
        self,
        failure: TestFailure,
        code_diff: Optional[Dict] = None
    ) -> FailureAnalysis:
        """
        Analyze test failure to identify root cause.
        PATTERN: Chain of responsibility for analyzers
        """
        # Parse error message and stack trace
        error_info = self._parse_error(failure.error_message, failure.stack_trace)

        # Determine failure type
        failure_type = self._classify_failure(error_info)
        failure.failure_type = failure_type

        # Get appropriate analyzer
        analyzer = self.analyzers.get(failure_type)
        if not analyzer:
            analyzer = self.analyzers['default']

        # Perform detailed analysis
        root_cause = await analyzer.analyze(failure, error_info)

        # Check code changes if available
        if code_diff:
            related_changes = self._find_related_changes(
                failure.target_function,
                code_diff
            )
            root_cause['code_changes'] = related_changes

        # Suggest repair strategies
        strategies = self._suggest_strategies(failure_type, root_cause)

        return FailureAnalysis(
            failure=failure,
            root_cause=root_cause['description'],
            confidence=root_cause['confidence'],
            code_changes=root_cause.get('code_changes'),
            suggested_strategies=strategies,
            evidence=root_cause.get('evidence', [])
        )

    def _classify_failure(self, error_info: Dict) -> FailureType:
        """
        Classify failure based on error patterns.
        PATTERN: Keyword and pattern matching
        """
        error_msg = error_info.get('message', '').lower()

        if 'syntaxerror' in error_msg:
            return FailureType.SYNTAX_ERROR
        elif 'importerror' in error_msg or 'modulenotfound' in error_msg:
            return FailureType.IMPORT_ERROR
        elif 'attributeerror' in error_msg:
            return FailureType.ATTRIBUTE_ERROR
        elif 'assertion' in error_msg or 'assert' in error_msg:
            return FailureType.ASSERTION_FAILED
        elif 'timeout' in error_msg or 'timed out' in error_msg:
            return FailureType.TIMEOUT
        elif 'mock' in error_msg:
            return FailureType.MOCK_ERROR
        else:
            return FailureType.RUNTIME_ERROR

# Test repair pattern
class TestRepairer:
    """
    PATTERN: Orchestrate test repair with validation
    CRITICAL: Must validate repairs don't break other tests
    """

    def __init__(self, repair_strategies, test_validator, llm_service):
        self.repair_strategies = repair_strategies
        self.test_validator = test_validator
        self.llm_service = llm_service

    async def repair_test(
        self,
        failure_analysis: FailureAnalysis,
        max_attempts: int = 3
    ) -> TestRepair:
        """
        Repair failing test based on analysis.
        PATTERN: Try strategies in order of confidence
        """
        original_code = await self._read_test_file(
            failure_analysis.failure.test_file
        )

        # Sort strategies by likelihood of success
        strategies = sorted(
            failure_analysis.suggested_strategies,
            key=lambda s: self._strategy_priority(s),
            reverse=True
        )

        for attempt, strategy in enumerate(strategies[:max_attempts]):
            try:
                # Get repair strategy implementation
                repairer = self.repair_strategies[strategy]

                # Generate repair
                repaired_code = await repairer.repair(
                    original_code,
                    failure_analysis
                )

                # Validate syntax
                if not self._validate_syntax(repaired_code):
                    continue

                # Write temporary test file
                temp_file = await self._write_temp_test(repaired_code)

                # Run test to validate repair
                test_result = await self.test_validator.run_test(temp_file)

                if test_result.passed:
                    # Verify coverage maintained
                    coverage_ok = await self._verify_coverage(
                        original_code,
                        repaired_code
                    )

                    return TestRepair(
                        repair_id=f"repair_{failure_analysis.failure.test_id}",
                        failure_analysis=failure_analysis,
                        strategy=strategy,
                        original_code=original_code,
                        repaired_code=repaired_code,
                        changes=self._diff_changes(original_code, repaired_code),
                        syntax_valid=True,
                        test_passes=True,
                        coverage_maintained=coverage_ok,
                        confidence=0.9
                    )
            except Exception as e:
                continue  # Try next strategy

        # If simple strategies fail, try LLM-based repair
        return await self._llm_repair(failure_analysis, original_code)

    async def _llm_repair(
        self,
        failure_analysis: FailureAnalysis,
        original_code: str
    ) -> TestRepair:
        """
        Use LLM for complex repairs.
        PATTERN: Provide context and let LLM fix
        """
        prompt = f"""
        Fix the failing test based on the following information:

        Error: {failure_analysis.failure.error_message}
        Root Cause: {failure_analysis.root_cause}

        Original test code:
        {original_code}

        Code changes that might have caused the failure:
        {failure_analysis.code_changes}

        Generate the fixed test code that:
        1. Resolves the error
        2. Maintains the test's intent
        3. Preserves code coverage
        """

        response = await self.llm_service.generate_response(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # Low temperature for deterministic fixes
        )

        repaired_code = self._extract_code(response.content)

        return TestRepair(
            repair_id=f"llm_repair_{failure_analysis.failure.test_id}",
            failure_analysis=failure_analysis,
            strategy=RepairStrategy.REGENERATE,
            original_code=original_code,
            repaired_code=repaired_code,
            changes=[{"type": "llm_generated", "description": "Complete LLM repair"}],
            confidence=0.7
        )

# Flaky test detection pattern
class FlakyDetector:
    """
    PATTERN: Statistical flaky test detection
    CRITICAL: Requires multiple test runs for accuracy
    """

    def __init__(self, test_runner, min_runs: int = 10):
        self.test_runner = test_runner
        self.min_runs = min_runs

    async def detect_flaky_tests(
        self,
        test_ids: List[str],
        runs_per_test: int = 20
    ) -> List[FlakyTestResult]:
        """
        Detect flaky tests through statistical analysis.
        PATTERN: Multiple runs with statistical analysis
        """
        results = []

        for test_id in test_ids:
            # Run test multiple times
            run_results = []
            execution_times = []
            failure_messages = set()
            failure_types = {}

            for _ in range(runs_per_test):
                # Ensure clean state between runs
                await self._clean_test_state()

                # Run test
                result = await self.test_runner.run_single_test(test_id)
                run_results.append(result.passed)
                execution_times.append(result.execution_time_ms)

                if not result.passed:
                    failure_messages.add(result.error_message)
                    failure_type = self._classify_failure(result.error_message)
                    failure_types[failure_type] = failure_types.get(failure_type, 0) + 1

            # Calculate statistics
            pass_count = sum(run_results)
            fail_count = len(run_results) - pass_count
            pass_rate = pass_count / len(run_results)

            # Execution time statistics
            mean_time = statistics.mean(execution_times)
            std_dev_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0

            # Determine flakiness
            is_flaky = self._is_flaky(
                pass_rate,
                std_dev_time / mean_time if mean_time > 0 else 0,
                len(failure_messages)
            )

            # Calculate flakiness score
            flakiness_score = self._calculate_flakiness_score(
                pass_rate,
                std_dev_time / mean_time if mean_time > 0 else 0,
                len(failure_messages)
            )

            # Identify root causes
            root_causes = self._identify_flakiness_causes(
                failure_types,
                std_dev_time / mean_time if mean_time > 0 else 0,
                failure_messages
            )

            results.append(FlakyTestResult(
                test_id=test_id,
                test_name=await self._get_test_name(test_id),
                flakiness_score=flakiness_score,
                total_runs=len(run_results),
                pass_count=pass_count,
                fail_count=fail_count,
                pass_rate=pass_rate,
                execution_times=execution_times,
                mean_time_ms=mean_time,
                std_dev_time_ms=std_dev_time,
                failure_types=failure_types,
                failure_messages=list(failure_messages),
                is_flaky=is_flaky,
                recommended_action=self._recommend_action(is_flaky, root_causes),
                root_causes=root_causes
            ))

        return results

    def _is_flaky(
        self,
        pass_rate: float,
        cv_time: float,  # Coefficient of variation for execution time
        unique_failures: int
    ) -> bool:
        """
        Determine if test is flaky based on heuristics.
        PATTERN: Multiple signals for flakiness
        """
        # Test is flaky if:
        # 1. Pass rate between 10% and 90% (sometimes passes, sometimes fails)
        # 2. High variation in execution time (CV > 0.3)
        # 3. Multiple different failure messages

        if 0.1 < pass_rate < 0.9:
            return True

        if cv_time > 0.3:  # High time variation
            return True

        if unique_failures > 2:  # Multiple failure modes
            return True

        return False

    def _calculate_flakiness_score(
        self,
        pass_rate: float,
        cv_time: float,
        unique_failures: int
    ) -> float:
        """
        Calculate flakiness probability score.
        PATTERN: Weighted combination of signals
        """
        # Score based on pass rate variance from stable (0 or 1)
        pass_rate_score = 2 * min(pass_rate, 1 - pass_rate)

        # Score based on timing variation
        time_score = min(cv_time, 1.0)

        # Score based on failure diversity
        failure_score = min(unique_failures / 5, 1.0)

        # Weighted combination
        weights = [0.5, 0.3, 0.2]  # Pass rate most important
        score = (
            weights[0] * pass_rate_score +
            weights[1] * time_score +
            weights[2] * failure_score
        )

        return min(score, 1.0)

# Test optimization pattern
class TestOptimizer:
    """
    PATTERN: Identify and suggest test performance improvements
    CRITICAL: Balance speed with test effectiveness
    """

    def __init__(self, performance_analyzer, coverage_analyzer):
        self.performance_analyzer = performance_analyzer
        self.coverage_analyzer = coverage_analyzer

    async def suggest_optimizations(
        self,
        test_suite: str,
        performance_data: Dict[str, Any]
    ) -> List[TestOptimization]:
        """
        Analyze and suggest test optimizations.
        PATTERN: Multi-dimensional optimization analysis
        """
        optimizations = []

        # Analyze slow tests
        slow_tests = self._identify_slow_tests(performance_data)
        for test_id, metrics in slow_tests.items():
            # Check for common performance issues
            if self._has_unnecessary_waits(test_id):
                optimizations.append(self._optimize_waits(test_id, metrics))

            if self._has_redundant_setup(test_id):
                optimizations.append(self._optimize_setup(test_id, metrics))

            if self._can_parallelize(test_id):
                optimizations.append(self._suggest_parallelization(test_id, metrics))

        # Analyze test redundancy
        redundant_tests = await self._find_redundant_tests(test_suite)
        for test_group in redundant_tests:
            optimizations.append(self._merge_redundant_tests(test_group))

        # Analyze mock usage
        mock_optimizations = await self._optimize_mocks(test_suite)
        optimizations.extend(mock_optimizations)

        # Sort by priority (time savings * inverse risk)
        optimizations.sort(
            key=lambda o: o.time_saving_ms * (1 / self._risk_score(o.risk_level)),
            reverse=True
        )

        return optimizations

    def _optimize_waits(
        self,
        test_id: str,
        metrics: Dict
    ) -> TestOptimization:
        """
        Optimize unnecessary waits and sleeps.
        PATTERN: Replace fixed waits with smart waits
        """
        current_time = metrics['execution_time_ms']
        estimated_time = current_time * 0.3  # Assume 70% time is in waits

        return TestOptimization(
            optimization_id=f"opt_waits_{test_id}",
            test_id=test_id,
            optimization_type="remove_waits",
            current_time_ms=current_time,
            estimated_time_ms=estimated_time,
            time_saving_ms=current_time - estimated_time,
            time_saving_percent=70,
            description="Replace fixed waits with smart waits",
            implementation="""
            Replace:
              await asyncio.sleep(5)
            With:
              await wait_for_condition(condition_func, timeout=5)
            """,
            risk_level="low",
            affects_coverage=False,
            priority=1
        )

# History tracking pattern
class HistoryTracker:
    """
    PATTERN: Track test performance over time
    CRITICAL: Efficient storage and retrieval of time-series data
    """

    def __init__(self, memory_service, redis_client):
        self.memory_service = memory_service
        self.redis = redis_client

    async def record_execution(
        self,
        test_id: str,
        execution_data: Dict[str, Any]
    ):
        """
        Record test execution for historical tracking.
        PATTERN: Time-series data with aggregation
        """
        # Store in Redis with TTL for recent data
        key = f"test:execution:{test_id}:{execution_data['timestamp']}"
        await self.redis.setex(
            key,
            86400 * 30,  # 30 days TTL
            json.dumps(execution_data)
        )

        # Update aggregated metrics
        await self._update_aggregates(test_id, execution_data)

        # Store in long-term memory if significant
        if self._is_significant_execution(execution_data):
            await self.memory_service.store_memory(
                content=f"Test execution: {test_id}",
                metadata=execution_data,
                memory_type="project"
            )

    async def analyze_trends(
        self,
        test_id: str,
        time_window: int = 30  # days
    ) -> TestPerformanceHistory:
        """
        Analyze historical trends for test.
        PATTERN: Time-series analysis with predictions
        """
        # Retrieve execution history
        executions = await self._get_executions(test_id, time_window)

        if not executions:
            return self._empty_history(test_id)

        # Calculate aggregated metrics
        total_executions = len(executions)
        failures = [e for e in executions if not e['passed']]
        failure_rate = len(failures) / total_executions

        # Analyze execution time trends
        times = [e['execution_time_ms'] for e in executions]
        time_trend = self._calculate_trend(times)

        # Analyze failure patterns
        failure_types = {}
        for failure in failures:
            f_type = failure.get('failure_type', 'unknown')
            failure_types[f_type] = failure_types.get(f_type, 0) + 1

        # Predict future failures
        failure_probability = self._predict_failure_probability(
            executions,
            failure_rate,
            time_trend
        )

        return TestPerformanceHistory(
            test_id=test_id,
            executions=executions,
            total_executions=total_executions,
            total_failures=len(failures),
            failure_rate=failure_rate,
            avg_execution_time_ms=statistics.mean(times),
            execution_time_trend=time_trend,
            common_failure_types=failure_types,
            predicted_failure_probability=failure_probability,
            predicted_maintenance_needed=failure_probability > 0.3,
            first_seen=executions[0]['timestamp'],
            last_updated=datetime.now()
        )
```

### Integration Points

```yaml
TESTING_SYSTEM:
  - integration: "Build on test generation from PRP-09"
  - pattern: "Use TestGenerator for regeneration"
  - enhancement: "Add healing capabilities to test pipeline"

AST_PARSER:
  - integration: "Detect code changes affecting tests"
  - pattern: "Compare AST before/after for signature changes"
  - usage: "Map code changes to test failures"

TEST_TOOLS:
  - integration: "Execute tests for validation"
  - pattern: "Use PytestTool, JestTool for test runs"
  - enhancement: "Capture detailed failure information"

MEMORY_SERVICE:
  - integration: "Store test performance history"
  - pattern: "Use project memory for historical data"
  - optimization: "Aggregate metrics for efficient storage"

LLM_SERVICE:
  - integration: "Generate complex test repairs"
  - pattern: "Use for repairs beyond simple patterns"
  - model_selection: "Use code generation models"

COVERAGE_ANALYZER:
  - integration: "Verify repairs maintain coverage"
  - pattern: "Compare coverage before/after repair"
  - validation: "Ensure no coverage regression"

CONFIG:
  - add to: .env
  - variables: |
      # Self-Healing Configuration
      HEALING_AUTO_REPAIR=true
      HEALING_CONFIDENCE_THRESHOLD=0.7
      HEALING_MAX_REPAIR_ATTEMPTS=3

      # Flaky Test Detection
      FLAKY_MIN_RUNS=10
      FLAKY_MAX_RUNS=50
      FLAKY_DETECTION_THRESHOLD=0.3

      # Test Optimization
      OPTIMIZATION_TARGET_REDUCTION=0.3
      OPTIMIZATION_RISK_TOLERANCE=medium

      # History Tracking
      HISTORY_RETENTION_DAYS=90
      HISTORY_AGGREGATION_INTERVAL=daily

      # Performance Thresholds
      PERFORMANCE_SLOW_TEST_MS=5000
      PERFORMANCE_REGRESSION_THRESHOLD=1.5

REDIS:
  - usage: "Time-series test execution data"
  - keys: |
      test:execution:{test_id}:{timestamp} - Individual execution
      test:aggregate:{test_id}:daily - Daily aggregates
      test:flaky:{test_id} - Flaky test detection data
      test:repair:{test_id} - Repair history

DEPENDENCIES:
  - statistics: "Statistical analysis for flaky detection"
  - difflib: "Code diff generation for repairs"
  - dateutil: "Time-series analysis"
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Check new healing module
ruff check src/testing/healing/ --fix
mypy src/testing/healing/ --strict
ruff format src/testing/healing/

# Verify imports
python -c "from src.testing.healing import TestRepairer; print('Healing imports OK')"
python -c "from src.services.healing_service import HealingOrchestrator; print('Service imports OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test failure analysis
pytest src/tests/testing/healing/test_failure_analysis.py -v --cov=src/testing/healing/failure_analyzer

# Test repair strategies
pytest src/tests/testing/healing/test_repair.py -v --cov=src/testing/healing/test_repairer

# Test flaky detection
pytest src/tests/testing/healing/test_flaky_detection.py -v --cov=src/testing/healing/flaky_detector

# Test optimization suggestions
pytest src/tests/testing/healing/test_optimization.py -v --cov=src/testing/healing/test_optimizer

# Full healing test suite
pytest src/tests/testing/healing/ -v --cov=src/testing/healing --cov-report=term-missing

# Expected: 85%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Test failure analysis accuracy
python scripts/test_failure_analysis_accuracy.py \
  --inject-failures ./test-failures/ \
  --verify-root-cause \
  --measure-accuracy
# Expected: 90%+ root cause identification accuracy

# Test automatic repair
python scripts/test_automatic_repair.py \
  --break-tests ./working-tests/ \
  --heal-tests \
  --verify-fixes
# Expected: 75%+ successful automatic repairs

# Test flaky detection
python scripts/test_flaky_detection.py \
  --inject-flakiness ./stable-tests/ \
  --detect-flaky \
  --verify-accuracy
# Expected: 90%+ flaky test detection accuracy

# Test optimization suggestions
python scripts/test_optimization_suggestions.py \
  --slow-tests ./performance-tests/ \
  --suggest-optimizations \
  --verify-improvements
# Expected: 30%+ execution time reduction

# Test history tracking
python scripts/test_history_tracking.py \
  --record-executions \
  --analyze-trends \
  --verify-predictions
# Expected: Accurate trend analysis and predictions

# Test repair validation
python scripts/test_repair_validation.py \
  --generate-repairs \
  --validate-syntax \
  --run-repaired-tests
# Expected: 95%+ repaired tests pass
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Large Test Suite Healing
python scripts/test_large_suite_healing.py \
  --test-suite ./large-project/tests/ \
  --inject-breaking-changes \
  --measure-healing-time \
  --verify-all-fixed
# Expected: <10s healing time for typical failures

# Multi-Framework Healing
python scripts/test_multi_framework_healing.py \
  --frameworks "pytest,jest,junit" \
  --break-tests \
  --heal-all \
  --verify-framework-specific
# Expected: All frameworks supported correctly

# Complex Refactoring Recovery
python scripts/test_refactoring_recovery.py \
  --apply-refactoring ./src/ \
  --heal-affected-tests \
  --verify-coverage-maintained
# Expected: Tests adapt to major refactoring

# Flaky Test Root Cause Analysis
python scripts/test_flaky_root_cause.py \
  --known-flaky-tests ./flaky/ \
  --identify-causes \
  --verify-fixes
# Expected: Correct root cause identification

# Performance Regression Prevention
python scripts/test_performance_regression.py \
  --monitor-test-performance \
  --detect-regressions \
  --suggest-fixes
# Expected: Detect and prevent performance regressions

# Cascading Failure Recovery
python scripts/test_cascading_failures.py \
  --break-dependency \
  --detect-cascade \
  --heal-all-affected
# Expected: Handle cascading test failures

# Test Maintenance Prediction
python scripts/test_maintenance_prediction.py \
  --analyze-history \
  --predict-failures \
  --verify-predictions
# Expected: 80%+ prediction accuracy

# Continuous Healing Simulation
python scripts/test_continuous_healing.py \
  --simulate-development 100 \
  --apply-continuous-changes \
  --measure-healing-effectiveness
# Expected: Maintain 80%+ test pass rate

# Agent Integration Test
python scripts/test_tester_agent_healing.py \
  --task "Fix failing authentication tests" \
  --verify-agent-healing \
  --check-results
# Expected: Tester agent successfully heals tests

# Memory Efficiency Test
python scripts/test_healing_memory_efficiency.py \
  --large-history 10000 \
  --measure-memory-usage \
  --verify-performance
# Expected: Efficient memory usage with large history
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] Healing tests achieve 85%+ coverage: `pytest src/tests/testing/healing/ --cov=src/testing/healing`
- [ ] No linting errors: `ruff check src/testing/healing/`
- [ ] No type errors: `mypy src/testing/healing/ --strict`
- [ ] All repair strategies working

### Feature Validation

- [ ] 75%+ test failures automatically repaired
- [ ] Flaky test detection 90%+ accurate
- [ ] Test maintenance effort reduced by 70%
- [ ] Repair suggestions 85%+ acceptance rate
- [ ] False positive reduction of 80%+
- [ ] Test execution time optimized by 30%+
- [ ] Historical tracking operational
- [ ] Self-healing completes in <10 seconds

### Code Quality Validation

- [ ] Follows existing testing patterns from PRP-09
- [ ] All operations async-compatible
- [ ] Proper error handling for repair failures
- [ ] Test isolation maintained during analysis
- [ ] Repairs validated before application
- [ ] History efficiently stored and retrieved

### Documentation & Deployment

- [ ] Environment variables documented
- [ ] Repair strategies documented with examples
- [ ] Flaky test detection algorithm explained
- [ ] Optimization recommendations documented
- [ ] API endpoints for healing service
- [ ] Integration guide with CI/CD pipelines

---

## Anti-Patterns to Avoid

- ❌ Don't apply repairs without validation
- ❌ Don't ignore test coverage when repairing
- ❌ Don't mark tests as flaky without sufficient runs
- ❌ Don't optimize tests at the expense of effectiveness
- ❌ Don't store unlimited historical data
- ❌ Don't repair tests that should be deleted
- ❌ Don't ignore framework-specific patterns
- ❌ Don't apply high-risk optimizations automatically
- ❌ Don't skip syntax validation for repairs
- ❌ Don't lose test intent when repairing
