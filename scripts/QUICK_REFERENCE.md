# Self-Healing Validation Scripts - Quick Reference

## Run Commands

```bash
# Individual scripts
python scripts/test_large_suite_healing.py
python scripts/test_multi_framework_healing.py
python scripts/test_refactoring_recovery.py
python scripts/test_cascading_failures.py
python scripts/test_continuous_healing.py

# Run all (bash)
for s in large_suite multi_framework refactoring cascading continuous; do
  python scripts/test_${s}_healing.py
done
```

## What Each Script Tests

| Script | Lines | Tests | Target | Key Metrics |
|--------|-------|-------|--------|-------------|
| **large_suite** | 583 | Large suites (50-500 tests) | <10s/test, 75%+ success | Time, memory, throughput |
| **multi_framework** | 740 | pytest/jest/junit | 75%+ per framework | Framework compatibility |
| **refactoring** | 675 | 6 refactoring types | 80%+ adaptation | Coverage maintenance |
| **cascading** | 722 | Root cause detection | 85%+ heal rate | Healing efficiency |
| **continuous** | 693 | Ongoing development | 80%+ pass rate | Response time, trends |

## Success Criteria Cheat Sheet

```
✓ Repair Success Rate:    75-85%+ (varies by script)
✓ Healing Time:           <10s per test (large_suite)
✓ Response Time:          <30s (continuous)
✓ Pass Rate:              80%+ (continuous)
✓ Memory Usage:           <500MB peak (large_suite)
✓ Throughput:             5+ tests/s (large_suite)
✓ Healing Efficiency:     10+ tests/action (cascading)
✓ Framework Support:      75%+ per framework (multi_framework)
✓ Adaptation Rate:        80%+ (refactoring)
```

## Results Location

```
test_results/
├── large_suite_healing/*.json
├── multi_framework_healing/*.json
├── refactoring_recovery/*.json
├── cascading_failures/*.json
└── continuous_healing/*.json + *.txt
```

## Exit Codes

- `0` = All tests PASSED ✓
- `1` = Some tests FAILED ✗

## Quick Debug

```bash
# Check if script runs
python scripts/test_large_suite_healing.py --help 2>/dev/null || echo "Run directly"

# View results
cat test_results/large_suite_healing/*.json | jq '.summary'

# Check success
python scripts/test_large_suite_healing.py && echo "PASSED" || echo "FAILED"
```

## Common Parameters to Customize

```python
# large_suite_healing.py
self.suite_sizes = [50, 100, 200, 500]
self.failure_rates = [0.1, 0.2, 0.3]

# continuous_healing.py
total_tests = 100
development_cycles = 10
failure_rate_per_change = 0.15

# refactoring_recovery.py
operation.affected_tests = 15

# cascading_failures.py
chain_depth = 4
tests_per_level = [10, 8, 6, 4]
```

## Scenarios Summary

### large_suite_healing.py
- Suite size variations (4 sizes × 3 failure rates)
- Concurrent suite healing (5 parallel)
- Incremental healing (2 batches)

### multi_framework_healing.py
- Per-framework tests (pytest, jest, junit)
- Cross-framework mixed suites
- Framework-specific patterns (27 patterns)

### refactoring_recovery.py
- 6 refactoring types
- Complex multi-step refactoring
- Breaking API changes

### cascading_failures.py
- Common cascades (3 patterns)
- Dependency chains (4 levels)
- Fixture failures (5 classes)
- Mock cascades (30 tests)
- Multi-level cascades (3 levels)

### continuous_healing.py
- Normal development (10 cycles)
- Rapid development (20 cycles)
- Burst failures (50 tests)
- Long session (50 cycles)
- Under load (5 concurrent streams)

## Key Metrics Formula

```python
# Repair Success Rate
success_rate = successful_repairs / total_failures

# Healing Time
avg_time = sum(healing_times) / len(healing_times)

# Throughput
throughput = tests_healed / (total_time_seconds)

# Pass Rate
pass_rate = passing_tests / total_tests

# Healing Efficiency
efficiency = tests_healed / repair_actions

# Response Time (p95)
p95 = sorted(response_times)[int(len(response_times) * 0.95)]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ImportError | `export PYTHONPATH=$PWD` |
| ModuleNotFoundError | `pip install -r requirements.txt` |
| MemoryError | Reduce suite sizes in script |
| Slow execution | Reduce sleep times or cycles |
| No output | Check script has execute permissions |

## CI/CD Integration

```yaml
# Quick CI job
test:
  script:
    - pip install -r requirements.txt
    - python scripts/test_large_suite_healing.py
    - python scripts/test_multi_framework_healing.py
    - python scripts/test_refactoring_recovery.py
    - python scripts/test_cascading_failures.py
    - python scripts/test_continuous_healing.py
  artifacts:
    paths:
      - test_results/
```

---

**Total**: 3,413 lines | 5 scripts | 20+ scenarios | 10+ metrics

For detailed docs: See `HEALING_VALIDATION_README.md` and `HEALING_VALIDATION_SUMMARY.md`
