# Quick Start: Self-Healing Integration Tests

## Run All Tests (Recommended)

```bash
# From project root
./scripts/run_all_healing_tests.sh
```

## Run Individual Tests

### Test 1: Failure Analysis (90%+ accuracy)
```bash
python scripts/test_failure_analysis_accuracy.py
```

### Test 2: Automatic Repair (75%+ success)
```bash
python scripts/test_automatic_repair.py
```

### Test 3: Flaky Detection (90%+ accuracy)
```bash
python scripts/test_flaky_detection.py
```

### Test 4: Optimization (30%+ savings)
```bash
python scripts/test_optimization_suggestions.py
```

### Test 5: History Tracking (85%+ accuracy)
```bash
python scripts/test_history_tracking.py
```

## Common Options

```bash
# More samples (slower, more accurate)
python scripts/test_failure_analysis_accuracy.py --samples 200

# Verbose output (detailed results)
python scripts/test_automatic_repair.py --verbose

# Custom target threshold
python scripts/test_flaky_detection.py --target 0.95

# Combine options
python scripts/test_optimization_suggestions.py --samples 100 --target 35 -v
```

## Expected Output

### Success Example
```
================================================================================
                      FAILURE ANALYSIS ACCURACY TEST
================================================================================

Initializing failure analyzer...
Generating 100 test scenarios...

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

✓ SUCCESS: Achieved 92.0% accuracy (target: 90%)
```

## Exit Codes

- **0** = Success (met target)
- **1** = Failure (didn't meet target)
- **130** = Interrupted (Ctrl+C)

## Troubleshooting

### Unicode Error (Windows)
Already fixed with UTF-8 encoding setup. If issues persist:
```bash
# Set environment variable
set PYTHONIOENCODING=utf-8
python scripts/test_failure_analysis_accuracy.py
```

### Import Errors
Run from project root:
```bash
cd /path/to/agent-swarm
python scripts/test_failure_analysis_accuracy.py
```

### Slow Tests
Reduce sample size:
```bash
python scripts/test_flaky_detection.py --runs 10 --iterations 5
```

## Documentation

- **Full Guide**: `scripts/HEALING_INTEGRATION_TESTS.md`
- **Summary**: `scripts/INTEGRATION_TESTS_SUMMARY.md`
- **PRP-11**: `PRPs/PRP-11.md`

## Quick Test

Run a fast test (10 samples):
```bash
python scripts/test_failure_analysis_accuracy.py --samples 10
```

## CI/CD Example

```yaml
# .github/workflows/healing-tests.yml
name: Healing Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: ./scripts/run_all_healing_tests.sh
```

## Success Criteria (PRP-11)

| Test | Target | Measures |
|------|--------|----------|
| Failure Analysis | 90%+ | Root cause identification accuracy |
| Automatic Repair | 75%+ | Test repair success rate |
| Flaky Detection | 90%+ | Flaky test detection accuracy |
| Optimization | 30%+ | Time savings from suggestions |
| History Tracking | 85%+ | Trend analysis accuracy |

**All tests must pass to meet PRP-11 requirements.**
