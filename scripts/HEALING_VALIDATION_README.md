# Self-Healing Test System - Validation Scripts

This directory contains comprehensive validation scripts for testing the self-healing test system (PRP-11). These scripts validate the system's robustness and performance in realistic, production-like scenarios.

## Overview

The validation suite consists of 5 domain-specific scripts that test creative edge cases and realistic scenarios:

1. **test_large_suite_healing.py** - Large test suite performance
2. **test_multi_framework_healing.py** - Multi-framework support
3. **test_refactoring_recovery.py** - Refactoring adaptation
4. **test_cascading_failures.py** - Cascading failure handling
5. **test_continuous_healing.py** - Continuous development simulation

## Success Criteria

Each script validates against specific success criteria aligned with PRP-11:

- **Healing Success Rate**: 75-85%+ of failures automatically repaired
- **Performance**: <10s healing time per test for typical failures
- **Coverage Maintenance**: Test coverage maintained after repairs
- **Framework Support**: pytest, jest, junit handled correctly
- **Response Time**: <30s for continuous healing scenarios

## Scripts

### 1. test_large_suite_healing.py

Tests healing on large test suites (50-500 tests) with performance benchmarks.

**What it tests:**
- Healing 50, 100, 200, and 500 test suites
- Various failure rates (10%, 20%, 30%)
- Concurrent healing of multiple suites
- Incremental healing as failures occur
- Memory usage and throughput

**Success criteria:**
- <10s average healing time per test
- 75%+ repair success rate
- Efficient memory usage (<500MB peak)
- Throughput > 5 tests/second

**Usage:**
```bash
python scripts/test_large_suite_healing.py
```

**Output:**
- JSON results in `test_results/large_suite_healing/`
- Performance metrics per suite size
- Percentile breakdown (p50, p90, p95, p99)
- Memory usage tracking

### 2. test_multi_framework_healing.py

Tests healing across pytest, jest, and junit with framework-specific patterns.

**What it tests:**
- Framework-specific error patterns
- Framework conventions (assertions, fixtures, mocks)
- Cross-framework test suites
- Framework-specific repair strategies

**Success criteria:**
- 75%+ success rate per framework
- Correct framework pattern recognition
- Maintains framework conventions in repairs

**Framework patterns tested:**
- **pytest**: assert rewrite, fixtures, parametrize
- **jest**: matchers, async timeouts, mock assertions
- **junit**: assertEquals, annotations, null pointers

**Usage:**
```bash
python scripts/test_multi_framework_healing.py
```

**Output:**
- JSON results in `test_results/multi_framework_healing/`
- Per-framework success rates
- Pattern handling statistics
- Cross-framework compatibility report

### 3. test_refactoring_recovery.py

Tests healing after major code refactoring operations.

**What it tests:**
- Function/method renames
- Signature changes (parameters added/removed)
- Module reorganization (imports updated)
- Class extraction/inlining
- Parameter restructuring
- Breaking API changes

**Success criteria:**
- 80%+ adaptation success rate
- Coverage maintained after refactoring
- Handle multi-step refactoring

**Refactoring types:**
- `RENAME_FUNCTION`: Method renamed
- `CHANGE_SIGNATURE`: Parameters changed
- `MOVE_MODULE`: Import paths changed
- `EXTRACT_CLASS`: Logic extracted to new class
- `INLINE_FUNCTION`: Function inlined
- `RESTRUCTURE_PARAMS`: Parameters grouped into objects

**Usage:**
```bash
python scripts/test_refactoring_recovery.py
```

**Output:**
- JSON results in `test_results/refactoring_recovery/`
- Per-refactoring-type metrics
- Coverage maintenance tracking
- Complex refactoring handling

### 4. test_cascading_failures.py

Tests handling of cascading failures where one root cause affects multiple tests.

**What it tests:**
- Common cascade patterns (import, base class, config)
- Dependency chain failures (A -> B -> C -> D)
- Shared fixture failures
- Mock configuration cascades
- Multi-level cascades (3+ levels deep)

**Success criteria:**
- 85%+ tests healed after root cause fix
- Root cause detection accuracy
- Healing efficiency (tests healed per repair action)

**Cascade patterns:**
- Import failure affecting 20+ tests
- Base class change breaking child class tests
- Global config error affecting dependent tests
- Dependency chains with 4+ levels
- Broken fixtures affecting multiple test classes

**Usage:**
```bash
python scripts/test_cascading_failures.py
```

**Output:**
- JSON results in `test_results/cascading_failures/`
- Root cause detection accuracy
- Healing efficiency metrics
- Cascade depth analysis

### 5. test_continuous_healing.py

Simulates continuous development with ongoing code changes and test failures.

**What it tests:**
- Normal development pace (1 change/minute)
- Rapid development (multiple changes/minute)
- Burst failures (sudden spike of 50+ failures)
- Long-running sessions (50+ cycles)
- Healing under load (concurrent development)

**Success criteria:**
- 80%+ average test pass rate maintained
- <30s healing response time
- Handle burst failures (50+ tests)
- Stability over long sessions

**Scenarios:**
- **Normal pace**: 10 cycles, 15% failure rate per change
- **Rapid**: 20 cycles, multiple concurrent changes
- **Burst**: 50% of suite fails suddenly
- **Long-run**: 50 cycles with varying failure rates
- **Under load**: 5 concurrent development streams

**Usage:**
```bash
python scripts/test_continuous_healing.py
```

**Output:**
- JSON results in `test_results/continuous_healing/`
- Health snapshots over time
- Pass rate trends
- Healing response time analysis
- Trend report (text format)

## Running All Validation Scripts

Run all scripts sequentially:

```bash
# Run all validation tests
python scripts/test_large_suite_healing.py
python scripts/test_multi_framework_healing.py
python scripts/test_refactoring_recovery.py
python scripts/test_cascading_failures.py
python scripts/test_continuous_healing.py
```

Or create a batch runner:

```bash
# Create runner script
cat > scripts/run_all_healing_tests.sh << 'EOF'
#!/bin/bash
echo "Running all self-healing validation tests..."
python scripts/test_large_suite_healing.py
python scripts/test_multi_framework_healing.py
python scripts/test_refactoring_recovery.py
python scripts/test_cascading_failures.py
python scripts/test_continuous_healing.py
echo "All validation tests completed!"
EOF

chmod +x scripts/run_all_healing_tests.sh
./scripts/run_all_healing_tests.sh
```

## Results and Reports

All scripts save detailed results to `test_results/` directory:

```
test_results/
├── large_suite_healing/
│   └── large_suite_healing_<timestamp>.json
├── multi_framework_healing/
│   └── multi_framework_<timestamp>.json
├── refactoring_recovery/
│   └── refactoring_recovery_<timestamp>.json
├── cascading_failures/
│   └── cascading_failures_<timestamp>.json
└── continuous_healing/
    ├── continuous_healing_<timestamp>.json
    └── trend_report_<timestamp>.txt
```

### Result Format

Each JSON result contains:

```json
{
  "test_suite": "script_name",
  "timestamp": "ISO timestamp",
  "scenarios": [
    {
      "scenario": "scenario_name",
      "type": "scenario_type",
      "metrics": { ... },
      "success": true/false
    }
  ],
  "summary": {
    "total_scenarios": 10,
    "overall_success": true,
    "...": "..."
  }
}
```

## Performance Metrics

Key metrics tracked across all scripts:

| Metric | Target | Description |
|--------|--------|-------------|
| Repair Success Rate | 75-85%+ | Percentage of tests successfully healed |
| Healing Time | <10s | Average time to heal one test |
| Response Time | <30s | Time to respond to failures in continuous mode |
| Pass Rate | 80%+ | Test suite pass rate during development |
| Memory Usage | <500MB | Peak memory during large suite healing |
| Throughput | 5+ tests/s | Tests healed per second |
| Healing Efficiency | 10+ | Tests healed per repair action (cascades) |

## Exit Codes

All scripts use standard exit codes:

- `0`: All tests passed (success criteria met)
- `1`: Some tests failed (success criteria not met)

This allows integration with CI/CD pipelines:

```bash
# CI/CD integration
python scripts/test_large_suite_healing.py
if [ $? -eq 0 ]; then
  echo "✓ Large suite healing validation passed"
else
  echo "✗ Large suite healing validation failed"
  exit 1
fi
```

## Interpreting Results

### Success Indicators

✓ **All tests passed** if:
- All scenarios meet their success criteria
- Overall success rate >= target
- Performance benchmarks met
- No critical failures

✗ **Tests failed** if:
- Any scenario below success threshold
- Performance too slow
- Critical errors occurred

### Common Failure Reasons

1. **Low healing success rate**: Repair strategies need improvement
2. **Slow healing time**: Optimization needed
3. **Low pass rate**: Too many failures or slow healing
4. **High memory usage**: Memory leaks or inefficient storage

## Dependencies

Required packages:
- `asyncio` - Async execution
- `psutil` - Memory tracking (install: `pip install psutil`)
- `statistics` - Statistical analysis

From project:
- `src.testing.healing.*` - Healing components
- `src.models.healing_models` - Data models

## Integration with PRP-11

These scripts validate the complete PRP-11 implementation:

| PRP-11 Component | Validated By |
|------------------|--------------|
| FailureAnalyzer | All scripts |
| TestRepairer | All scripts |
| FlakyDetector | test_continuous_healing.py |
| TestOptimizer | test_large_suite_healing.py |
| HistoryTracker | test_continuous_healing.py |
| RepairStrategies | test_multi_framework_healing.py, test_refactoring_recovery.py |

## Troubleshooting

### Script won't run

```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check dependencies
pip install -r requirements.txt
```

### Memory errors

```bash
# Reduce test suite size
# Edit script and modify suite_sizes or total_tests
```

### Slow execution

```bash
# Scripts use asyncio.sleep() for simulation
# Real execution would be faster
# Reduce sleep times or cycle counts for faster testing
```

## Customization

Each script can be customized by editing parameters:

```python
# test_large_suite_healing.py
self.suite_sizes = [50, 100, 200, 500]  # Modify suite sizes
self.failure_rates = [0.1, 0.2, 0.3]     # Modify failure rates

# test_continuous_healing.py
total_tests = 100                         # Modify test count
development_cycles = 10                   # Modify cycle count
failure_rate_per_change = 0.15           # Modify failure rate
```

## Best Practices

1. **Run regularly**: Include in CI/CD pipeline
2. **Monitor trends**: Track metrics over time
3. **Investigate failures**: Review detailed results when tests fail
4. **Adjust thresholds**: Update success criteria as system improves
5. **Compare results**: Compare across versions/commits

## Future Enhancements

Potential additions:
- Real test execution (not simulation)
- Integration with actual test frameworks
- Performance profiling with flamegraphs
- Automated root cause analysis
- ML-based pattern detection
- Real-time dashboards

## Contributing

When adding new validation scripts:

1. Follow naming: `test_<scenario>_<aspect>.py`
2. Include detailed docstring with success criteria
3. Generate JSON results
4. Use consistent metrics structure
5. Print progress and summaries
6. Return appropriate exit codes
7. Update this README

## Support

For issues or questions:
- Review script output and JSON results
- Check PRP-11 implementation status
- Verify healing components are working
- Check for missing dependencies

---

**Generated for PRP-11: Self-Healing Test System**

These validation scripts ensure the self-healing system meets production-quality standards and can handle realistic, complex scenarios.
