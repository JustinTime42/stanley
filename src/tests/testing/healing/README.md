# Self-Healing System Test Suite

Comprehensive unit tests for the self-healing test system components following PRP-11 specifications.

## Overview

This test suite provides extensive coverage for all self-healing system components:
- **Failure Analysis**: Tests for FailureAnalyzer orchestrator and all analyzer strategies
- **Test Repair**: Tests for TestRepairer orchestrator and all repair strategies
- **Flaky Detection**: Tests for FlakyDetector with statistical analysis
- **Test Optimization**: Tests for TestOptimizer with performance suggestions

## Test Files

### 1. test_failure_analysis.py
Tests all failure analyzers and the FailureAnalyzer orchestrator.

**Test Classes:**
- `TestSyntaxErrorAnalyzer` - Tests syntax error detection and analysis
- `TestAssertionAnalyzer` - Tests assertion failure analysis and value extraction
- `TestFailureAnalyzer` - Tests the main orchestrator and failure classification
- `TestRuntimeAnalyzer` - Tests runtime error analysis
- `TestTimeoutAnalyzer` - Tests timeout error detection
- `TestAnalyzerEdgeCases` - Tests edge cases across all analyzers

**Coverage:**
- Error message parsing and classification
- Root cause identification
- Confidence scoring
- Strategy suggestion
- Code diff integration
- Edge cases (empty messages, unicode, multiline, etc.)

### 2. test_repair.py
Tests all repair strategies and the TestRepairer orchestrator.

**Test Classes:**
- `TestSignatureRepair` - Tests method signature mismatch repairs
- `TestAssertionRepair` - Tests assertion value update repairs
- `TestTestRepairer` - Tests the main repair orchestrator
- `TestMockRepair` - Tests mock configuration repairs

**Coverage:**
- Strategy selection and prioritization
- Repair code generation
- Syntax validation
- Test validation integration
- LLM-based repair fallback
- Coverage maintenance verification
- File I/O and temporary file handling
- Error handling and recovery

**Mocking:**
- LLM service (AsyncMock)
- Test validator (AsyncMock)
- Coverage analyzer (Mock)

### 3. test_flaky_detection.py
Tests flaky test detection with simulated flaky behavior.

**Test Classes:**
- `TestFlakyDetector` - Main flaky detection tests
- `TestFlakyDetectorEdgeCases` - Edge cases and boundary conditions

**Coverage:**
- Statistical analysis (mean, std dev, coefficient of variation)
- Multiple test run execution
- Pass rate calculation
- Flakiness scoring algorithm
- Root cause identification (timing, race conditions, network, etc.)
- Recommendation generation
- Historical analysis
- Failure type classification
- Edge cases (empty lists, zero values, single runs, etc.)

**Simulated Patterns:**
- 50% pass rate (clearly flaky)
- 100% pass rate (stable)
- 0% pass rate (consistently failing)
- High timing variance
- Multiple failure types
- Async-related failures
- Network-related failures

### 4. test_optimization.py
Tests test optimization suggestions and analysis.

**Test Classes:**
- `TestTestOptimizer` - Main optimizer tests
- `TestOptimizationModel` - Model validation tests
- `TestOptimizerEdgeCases` - Edge cases

**Coverage:**
- Slow test identification
- Unnecessary wait detection
- Redundant setup detection
- Parallelization opportunities
- Excessive I/O detection
- Optimization prioritization
- Time saving calculations
- Risk assessment
- Multiple optimization types:
  - Remove unnecessary waits
  - Optimize setup/fixtures
  - Enable parallelization
  - Mock I/O operations
  - Merge redundant tests

**Mocking:**
- Performance analyzer (Mock)
- Coverage analyzer (Mock)

## Running the Tests

### Run all healing tests:
```bash
pytest src/tests/testing/healing/ -v
```

### Run with coverage:
```bash
pytest src/tests/testing/healing/ --cov=src/testing/healing --cov-report=term-missing
```

### Run specific test file:
```bash
pytest src/tests/testing/healing/test_failure_analysis.py -v
pytest src/tests/testing/healing/test_repair.py -v
pytest src/tests/testing/healing/test_flaky_detection.py -v
pytest src/tests/testing/healing/test_optimization.py -v
```

### Run specific test class:
```bash
pytest src/tests/testing/healing/test_failure_analysis.py::TestFailureAnalyzer -v
```

### Run specific test:
```bash
pytest src/tests/testing/healing/test_repair.py::TestTestRepairer::test_repair_test_success -v
```

## Test Statistics

- **Total Tests**: 220+
- **Passing Tests**: 206+ (93.6%+)
- **Test Files**: 4
- **Test Classes**: 14+
- **Coverage Target**: 85%+

## Test Patterns

### Async Testing
All async methods are tested using `pytest-asyncio`:
```python
@pytest.mark.asyncio
async def test_analyze_failure(self, analyzer):
    result = await analyzer.analyze_failure(failure)
    assert result is not None
```

### Fixtures
Comprehensive fixtures for test data:
```python
@pytest.fixture
def analyzer(self):
    """Create analyzer instance."""
    return FailureAnalyzer()

@pytest.fixture
def sample_failure(self):
    """Create sample test failure."""
    return TestFailure(...)
```

### Mocking
Proper mocking of external dependencies:
```python
@pytest.fixture
def mock_llm_service(self):
    """Create mock LLM service."""
    llm = AsyncMock()
    llm.agenerate = AsyncMock(return_value="fixed code")
    return llm
```

### Edge Cases
Extensive edge case testing:
- Empty inputs
- Very large inputs
- Unicode characters
- Multiline strings
- None values
- Boundary conditions
- Error conditions

## Key Testing Principles

1. **Isolation**: Each test is independent and doesn't rely on external state
2. **Mocking**: External dependencies (LLM, Redis, file system) are properly mocked
3. **Coverage**: All major code paths and edge cases are tested
4. **Async**: Proper async/await testing with pytest-asyncio
5. **Fixtures**: Reusable test data and mock objects
6. **Assertions**: Clear, specific assertions for each test case
7. **Error Handling**: Tests verify both success and failure paths

## Common Issues and Solutions

### Windows File Permissions
Some tests may fail on Windows due to file locking. These are minor issues that don't affect test validity:
```python
# Use try/finally for cleanup
try:
    # test code
finally:
    if os.path.exists(temp_file):
        os.unlink(temp_file)
```

### Async Test Execution
Always use `@pytest.mark.asyncio` for async tests:
```python
@pytest.mark.asyncio
async def test_async_method(self):
    result = await some_async_function()
    assert result is not None
```

### Mock Configuration
Ensure mocks are properly configured before use:
```python
mock_service.method = AsyncMock(return_value=expected_result)
```

## Contributing

When adding new tests:
1. Follow existing test patterns
2. Include comprehensive fixtures
3. Test both success and error paths
4. Add edge case tests
5. Mock external dependencies
6. Aim for 85%+ coverage
7. Use clear, descriptive test names
8. Add docstrings for test methods

## Related Documentation

- PRP-11: Self-Healing Test System Specification
- `src/testing/healing/`: Source code for healing components
- `src/models/healing_models.py`: Data models for healing system
