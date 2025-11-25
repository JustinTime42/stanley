#!/bin/bash
# Run all self-healing integration tests
# Usage: ./scripts/run_all_healing_tests.sh [--verbose]

set -e  # Exit on error

VERBOSE=""
if [ "$1" = "--verbose" ] || [ "$1" = "-v" ]; then
    VERBOSE="--verbose"
fi

echo "================================================================================"
echo "                 Self-Healing Test System - Integration Tests"
echo "================================================================================"
echo ""

# Test 1: Failure Analysis
echo "[1/5] Testing Failure Analysis Accuracy (target: 90%+)..."
python scripts/test_failure_analysis_accuracy.py $VERBOSE
RESULT_1=$?
echo ""

# Test 2: Automatic Repair
echo "[2/5] Testing Automatic Repair (target: 75%+ success rate)..."
python scripts/test_automatic_repair.py $VERBOSE
RESULT_2=$?
echo ""

# Test 3: Flaky Detection
echo "[3/5] Testing Flaky Detection (target: 90%+ accuracy)..."
python scripts/test_flaky_detection.py $VERBOSE
RESULT_3=$?
echo ""

# Test 4: Optimization Suggestions
echo "[4/5] Testing Optimization Suggestions (target: 30%+ time savings)..."
python scripts/test_optimization_suggestions.py $VERBOSE
RESULT_4=$?
echo ""

# Test 5: History Tracking
echo "[5/5] Testing History Tracking (target: 85%+ accuracy)..."
python scripts/test_history_tracking.py $VERBOSE
RESULT_5=$?
echo ""

# Summary
echo "================================================================================"
echo "                           Test Suite Summary"
echo "================================================================================"
echo ""

if [ $RESULT_1 -eq 0 ]; then
    echo "✓ Failure Analysis:         PASS (90%+ accuracy)"
else
    echo "✗ Failure Analysis:         FAIL"
fi

if [ $RESULT_2 -eq 0 ]; then
    echo "✓ Automatic Repair:         PASS (75%+ success rate)"
else
    echo "✗ Automatic Repair:         FAIL"
fi

if [ $RESULT_3 -eq 0 ]; then
    echo "✓ Flaky Detection:          PASS (90%+ accuracy)"
else
    echo "✗ Flaky Detection:          FAIL"
fi

if [ $RESULT_4 -eq 0 ]; then
    echo "✓ Optimization Suggestions: PASS (30%+ time savings)"
else
    echo "✗ Optimization Suggestions: FAIL"
fi

if [ $RESULT_5 -eq 0 ]; then
    echo "✓ History Tracking:         PASS (85%+ accuracy)"
else
    echo "✗ History Tracking:         FAIL"
fi

echo ""
echo "================================================================================"

# Exit with failure if any test failed
if [ $RESULT_1 -ne 0 ] || [ $RESULT_2 -ne 0 ] || [ $RESULT_3 -ne 0 ] || [ $RESULT_4 -ne 0 ] || [ $RESULT_5 -ne 0 ]; then
    echo "✗ Some tests failed"
    exit 1
else
    echo "✓ All tests passed - Self-healing system meets PRP-11 success criteria!"
    exit 0
fi
