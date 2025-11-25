# Self-Healing Test System - Integration Tests

This directory contains comprehensive integration tests for the self-healing test system (PRP-11). These tests validate end-to-end functionality and measure system performance against defined success criteria.

## Overview

The integration test suite consists of 5 production-quality test scripts that validate different aspects of the healing system:

1. **Failure Analysis Accuracy** - Tests root cause identification (target: 90%+)
2. **Automatic Repair** - Tests automatic test repair success rate (target: 75%+)
3. **Flaky Detection** - Tests flaky test detection accuracy (target: 90%+)
4. **Optimization Suggestions** - Tests optimization time savings (target: 30%+)
5. **History Tracking** - Tests historical tracking and trend analysis

## Quick Start

### Run All Tests

```bash
# Run all integration tests with default settings
python scripts/test_failure_analysis_accuracy.py
python scripts/test_automatic_repair.py
python scripts/test_flaky_detection.py
python scripts/test_optimization_suggestions.py
python scripts/test_history_tracking.py
```

### Run Single Test with Verbose Output

```bash
python scripts/test_failure_analysis_accuracy.py --verbose
```

## Test Scripts

### 1. Test Failure Analysis Accuracy

**File**: `test_failure_analysis_accuracy.py`

**Purpose**: Validates the failure analyzer's ability to correctly identify root causes of test failures.

**Target**: 90%+ accuracy in root cause identification

**Usage**:
```bash
# Basic usage
python scripts/test_failure_analysis_accuracy.py

# With custom sample size
python scripts/test_failure_analysis_accuracy.py --samples 200

# With custom target accuracy
python scripts/test_failure_analysis_accuracy.py --target 0.95

# Verbose mode
python scripts/test_failure_analysis_accuracy.py --verbose

# With real test failures
python scripts/test_failure_analysis_accuracy.py --inject-failures ./test-failures/
```

**Arguments**:
- `--samples N` - Number of test samples to analyze (default: 100)
- `--target FLOAT` - Target accuracy threshold (default: 0.90)
- `--verbose, -v` - Enable verbose output
- `--inject-failures PATH` - Path to directory with real test failures

**Metrics Measured**:
- Failure type classification accuracy
- Root cause identification accuracy
- Strategy suggestion accuracy
- Overall accuracy
- Average confidence score

**Exit Codes**:
- `0` - Success (achieved target accuracy)
- `1` - Failure (did not achieve target)
- `130` - Interrupted by user

**Example Output**:
```
================================================================================
                   FAILURE ANALYSIS ACCURACY TEST
================================================================================

Initializing failure analyzer...
Generating 100 test scenarios...
Created 100 test scenarios across 8 failure types

Analyzing test failures...
[████████████████████████████████████████] 100.0%

================================================================================
                              TEST RESULTS
================================================================================

Overall Results:
  Total Scenarios: 100
  Successful: 92 (92.0%)
  Failed: 8
  Execution Time: 2.34s

Accuracy Metrics:
  Failure Type Classification................  92.0% ✓
  Root Cause Identification..................  91.0% ✓
  Strategy Suggestion........................  94.0% ✓
  Overall Accuracy...........................  92.0% ✓
  Average Confidence.........................  84.5%

✓ SUCCESS: Achieved 92.0% accuracy (target: 90%)
```

---

### 2. Test Automatic Repair

**File**: `test_automatic_repair.py`

**Purpose**: Validates the automatic repair system's ability to fix failing tests.

**Target**: 75%+ repair success rate

**Usage**:
```bash
# Basic usage
python scripts/test_automatic_repair.py

# With custom sample size
python scripts/test_automatic_repair.py --samples 150

# With custom target
python scripts/test_automatic_repair.py --target 0.80

# Verbose mode
python scripts/test_automatic_repair.py --verbose

# With real tests
python scripts/test_automatic_repair.py --break-tests ./working-tests/ --heal-tests
```

**Arguments**:
- `--samples N` - Number of test samples (default: 100)
- `--target FLOAT` - Target success rate (default: 0.75)
- `--verbose, -v` - Enable verbose output
- `--break-tests PATH` - Path to working tests to break and repair
- `--heal-tests` - Heal the broken tests

**Metrics Measured**:
- Repair success rate
- Syntax validity rate
- Average confidence
- Average repair time
- Success rate by repair strategy

**Example Output**:
```
================================================================================
                      AUTOMATIC TEST REPAIR TEST
================================================================================

Initializing test repairer...
Generating 100 repair scenarios...
Created 100 repair scenarios across 8 repair types

Testing automatic repairs...
[████████████████████████████████████████] 100.0%

================================================================================
                              TEST RESULTS
================================================================================

Overall Results:
  Total Scenarios: 100
  Successful Repairs: 78 (78.0%)
  Failed Repairs: 22
  Execution Time: 15.67s

Repair Metrics:
  Repair Success Rate........................  78.0% ✓
  Syntax Validity Rate.......................  96.0% ✓
  Average Confidence.........................  81.2%
  Average Repair Time........................  0.16s

Success Rate by Strategy:
  update_assertion...........................  85.0% (17/20) ✓
  update_import..............................  90.0% (18/20) ✓
  update_signature...........................  75.0% (15/20) ✓
  update_mock................................  70.0% (14/20) ✗

✓ SUCCESS: Achieved 78.0% repair success rate (target: 75%)
```

---

### 3. Test Flaky Detection

**File**: `test_flaky_detection.py`

**Purpose**: Validates the flaky detector's ability to identify non-deterministic tests.

**Target**: 90%+ detection accuracy

**Usage**:
```bash
# Basic usage
python scripts/test_flaky_detection.py

# With custom runs per test
python scripts/test_flaky_detection.py --runs 30

# With custom iterations
python scripts/test_flaky_detection.py --iterations 20

# With custom target
python scripts/test_flaky_detection.py --target 0.92

# Verbose mode
python scripts/test_flaky_detection.py --verbose

# With real tests
python scripts/test_flaky_detection.py --inject-flakiness ./stable-tests/
```

**Arguments**:
- `--runs N` - Number of runs per test (default: 20)
- `--iterations N` - Number of test iterations (default: 10)
- `--target FLOAT` - Target accuracy (default: 0.90)
- `--verbose, -v` - Enable verbose output
- `--inject-flakiness PATH` - Path to stable tests to inject flakiness

**Metrics Measured**:
- Overall detection accuracy
- Precision (TP / (TP + FP))
- Recall (TP / (TP + FN))
- F1 score
- Average flakiness score

**Example Output**:
```
================================================================================
                      FLAKY TEST DETECTION TEST
================================================================================

Initializing flaky detector...
Generating test scenarios...
Created 100 test scenarios (10 unique, 10 iterations)
Runs per test: 20

Running flaky detection tests...
[████████████████████████████████████████] 100.0%

================================================================================
                              TEST RESULTS
================================================================================

Overall Results:
  Total Scenarios: 100
  Correct Detections: 91 (91.0%)
  Incorrect Detections: 9
  Execution Time: 45.23s

Detection Breakdown:
  True Positives (Flaky → Detected):  58
  True Negatives (Stable → Stable):   33
  False Positives (Stable → Flaky):   5
  False Negatives (Flaky → Stable):   4

Detection Metrics:
  Accuracy (Correct / Total)................. 91.0% ✓
  Precision (TP / (TP + FP))................. 92.1% ✓
  Recall (TP / (TP + FN)).................... 93.5% ✓
  F1 Score................................... 92.8% ✓
  Average Flakiness Score.................... 42.3%

✓ SUCCESS: Achieved 91.0% detection accuracy (target: 90%)
```

---

### 4. Test Optimization Suggestions

**File**: `test_optimization_suggestions.py`

**Purpose**: Validates the optimizer's ability to suggest performance improvements.

**Target**: 30%+ time savings

**Usage**:
```bash
# Basic usage
python scripts/test_optimization_suggestions.py

# With custom sample size
python scripts/test_optimization_suggestions.py --samples 75

# With custom target savings
python scripts/test_optimization_suggestions.py --target 35.0

# Verbose mode
python scripts/test_optimization_suggestions.py --verbose

# With real slow tests
python scripts/test_optimization_suggestions.py --slow-tests ./performance-tests/
```

**Arguments**:
- `--samples N` - Number of test samples (default: 50)
- `--target FLOAT` - Target time savings percentage (default: 30.0)
- `--verbose, -v` - Enable verbose output
- `--slow-tests PATH` - Path to slow tests to analyze

**Metrics Measured**:
- Overall time savings percentage
- Average savings for successful optimizations
- Suggestion success rate
- Time savings by optimization type

**Example Output**:
```
================================================================================
                 TEST OPTIMIZATION SUGGESTIONS TEST
================================================================================

Initializing test optimizer...
Generating 50 optimization scenarios...
Created 50 scenarios across 8 optimization types

Generating optimization suggestions...
[████████████████████████████████████████] 100.0%

================================================================================
                              TEST RESULTS
================================================================================

Overall Results:
  Total Scenarios: 50
  Successful: 42 (84.0%)
  Failed: 8
  Execution Time: 8.45s

Performance Impact:
  Total Current Time: 385000ms
  Total Time Savings: 134750ms
  Average Suggestions per Test: 2.3

Optimization Metrics:
  Overall Time Savings....................... 35.0% ✓
  Average Savings (Successful)............... 38.5% ✓
  Suggestion Success Rate.................... 84.0% ✓

Performance by Optimization Type:
  remove_waits........................ 90.0% (9/10) [avg:  4800ms] ✓
  optimize_setup...................... 85.0% (6/7)  [avg:  2400ms] ✓
  parallelize......................... 80.0% (8/10) [avg:  5000ms] ✓
  optimize_io......................... 87.5% (7/8)  [avg:  3375ms] ✓

✓ SUCCESS: Achieved 35.0% time savings (target: 30%)
```

---

### 5. Test History Tracking

**File**: `test_history_tracking.py`

**Purpose**: Validates historical tracking and trend analysis capabilities.

**Target**: 85%+ tracking and prediction accuracy

**Usage**:
```bash
# Basic usage
python scripts/test_history_tracking.py

# With custom iterations
python scripts/test_history_tracking.py --iterations 20

# With custom target
python scripts/test_history_tracking.py --target 0.90

# Verbose mode
python scripts/test_history_tracking.py --verbose

# With specific actions
python scripts/test_history_tracking.py --record-executions --analyze-trends
```

**Arguments**:
- `--iterations N` - Number of test iterations (default: 10)
- `--target FLOAT` - Target accuracy (default: 0.85)
- `--verbose, -v` - Enable verbose output
- `--record-executions` - Record test executions
- `--analyze-trends` - Analyze historical trends

**Metrics Measured**:
- Overall success rate
- Trend detection accuracy
- Failure prediction accuracy
- Failure rate tracking accuracy
- Average executions tracked

**Example Output**:
```
================================================================================
           HISTORICAL TRACKING AND TREND ANALYSIS TEST
================================================================================

Initializing history tracker...
Generating history tracking scenarios...
Created 60 scenarios (6 unique, 10 iterations)

Testing history tracking and trend analysis...
[████████████████████████████████████████] 100.0%

================================================================================
                              TEST RESULTS
================================================================================

Overall Results:
  Total Scenarios: 60
  Successful: 52 (86.7%)
  Failed: 8
  Execution Time: 12.34s

Tracking Metrics:
  Average Executions Tracked: 58.3

Analysis Accuracy:
  Overall Success Rate....................... 86.7% ✓
  Trend Detection Accuracy................... 88.3% ✓
  Failure Prediction Accuracy................ 85.0% ✓
  Failure Rate Tracking Accuracy............. 90.0% ✓

Accuracy by Execution Pattern:
  degrading.............. 90.0% ✓ (trend: 95.0%, prediction: 85.0%)
  flaky.................. 85.0% ✓ (trend: 80.0%, prediction: 90.0%)
  improving.............. 90.0% ✓ (trend: 95.0%, prediction: 85.0%)
  stable................. 85.0% ✓ (trend: 85.0%, prediction: 85.0%)

✓ SUCCESS: Achieved 86.7% accuracy (target: 85%)
```

---

## Running the Full Test Suite

To run all integration tests sequentially:

```bash
#!/bin/bash
# run_all_healing_tests.sh

echo "Running Self-Healing Integration Tests"
echo "======================================="

# Test 1: Failure Analysis
echo -e "\n[1/5] Testing Failure Analysis Accuracy..."
python scripts/test_failure_analysis_accuracy.py
RESULT_1=$?

# Test 2: Automatic Repair
echo -e "\n[2/5] Testing Automatic Repair..."
python scripts/test_automatic_repair.py
RESULT_2=$?

# Test 3: Flaky Detection
echo -e "\n[3/5] Testing Flaky Detection..."
python scripts/test_flaky_detection.py
RESULT_3=$?

# Test 4: Optimization Suggestions
echo -e "\n[4/5] Testing Optimization Suggestions..."
python scripts/test_optimization_suggestions.py
RESULT_4=$?

# Test 5: History Tracking
echo -e "\n[5/5] Testing History Tracking..."
python scripts/test_history_tracking.py
RESULT_5=$?

# Summary
echo -e "\n======================================="
echo "Test Suite Summary"
echo "======================================="
echo "Failure Analysis:        $([ $RESULT_1 -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "Automatic Repair:        $([ $RESULT_2 -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "Flaky Detection:         $([ $RESULT_3 -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "Optimization Suggestions: $([ $RESULT_4 -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "History Tracking:        $([ $RESULT_5 -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"

# Exit with failure if any test failed
if [ $RESULT_1 -ne 0 ] || [ $RESULT_2 -ne 0 ] || [ $RESULT_3 -ne 0 ] || [ $RESULT_4 -ne 0 ] || [ $RESULT_5 -ne 0 ]; then
    echo -e "\n✗ Some tests failed"
    exit 1
else
    echo -e "\n✓ All tests passed"
    exit 0
fi
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Self-Healing Integration Tests

on: [push, pull_request]

jobs:
  healing-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Test Failure Analysis
        run: python scripts/test_failure_analysis_accuracy.py --samples 200

      - name: Test Automatic Repair
        run: python scripts/test_automatic_repair.py --samples 150

      - name: Test Flaky Detection
        run: python scripts/test_flaky_detection.py --runs 30 --iterations 15

      - name: Test Optimization Suggestions
        run: python scripts/test_optimization_suggestions.py --samples 100

      - name: Test History Tracking
        run: python scripts/test_history_tracking.py --iterations 20
```

## Test Features

All integration tests include:

### Progress Indicators
Each test displays a real-time progress bar showing current status:
```
[████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 45.0% Analyzing test 45/100
```

### Color-Coded Output
- **Green (✓)**: Success metrics that meet targets
- **Red (✗)**: Failure metrics that don't meet targets
- **Yellow (⚠)**: Warnings and non-critical issues
- **Blue**: Informational messages
- **Cyan**: Progress indicators

### Verbose Mode
Enable detailed output with `--verbose` or `-v`:
- Individual scenario results
- Detailed metrics for each test
- Error messages and stack traces
- Performance breakdowns

### Configurable Targets
All tests support custom target thresholds:
```bash
# More strict testing
python scripts/test_failure_analysis_accuracy.py --target 0.95

# More lenient testing
python scripts/test_automatic_repair.py --target 0.70
```

### Proper Exit Codes
All scripts follow standard exit code conventions:
- `0` - Success
- `1` - Failure
- `130` - Keyboard interrupt (Ctrl+C)

## Success Criteria

According to PRP-11, the self-healing system must achieve:

| Metric | Target | Test Script |
|--------|--------|-------------|
| Failure root cause accuracy | 90%+ | test_failure_analysis_accuracy.py |
| Automatic repair success | 75%+ | test_automatic_repair.py |
| Flaky test detection | 90%+ | test_flaky_detection.py |
| Test optimization savings | 30%+ | test_optimization_suggestions.py |
| Historical tracking accuracy | 85%+ | test_history_tracking.py |

## Troubleshooting

### Tests Failing

If tests are failing:

1. **Check verbose output**:
   ```bash
   python scripts/test_failure_analysis_accuracy.py --verbose
   ```

2. **Reduce sample size** to isolate issues:
   ```bash
   python scripts/test_automatic_repair.py --samples 10 --verbose
   ```

3. **Check dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Performance Issues

If tests are running slowly:

1. **Reduce sample sizes**:
   ```bash
   python scripts/test_flaky_detection.py --runs 10 --iterations 5
   ```

2. **Run tests individually** instead of all at once

3. **Check system resources** (CPU, memory)

### Import Errors

Ensure you're running from the project root:
```bash
cd /path/to/agent-swarm
python scripts/test_failure_analysis_accuracy.py
```

## Development

### Adding New Scenarios

To add new test scenarios, edit the scenario generation functions:

```python
def generate_test_scenarios() -> List[Scenario]:
    scenarios = []

    # Add your new scenario
    scenarios.append(Scenario(
        name="New Test Scenario",
        # ... scenario parameters
    ))

    return scenarios
```

### Modifying Metrics

To track additional metrics, update the metrics dictionaries:

```python
metrics = {
    "existing_metric": value,
    "new_metric": new_value,  # Add new metric
}
```

## License

Copyright (c) 2024 Agent Swarm Project
