#!/usr/bin/env python3
"""Integration test for historical tracking and trend analysis.

Tests the history tracker's ability to record executions, analyze trends,
and make accurate predictions about future test behavior.

Usage:
    python scripts/test_history_tracking.py [--verbose] [--days N]
    python scripts/test_history_tracking.py --record-executions --analyze-trends
"""

import argparse
import asyncio
import logging
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.healing_models import (
    TestPerformanceHistory,
    FailureType,
)
from src.testing.healing import HistoryTracker


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print colored header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")


def print_progress(current: int, total: int, message: str = ""):
    """Print progress indicator."""
    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f"\r{Colors.CYAN}[{bar}] {percent:.1f}% {message}{Colors.ENDC}", end='', flush=True)
    if current == total:
        print()


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")


class MockRedisClient:
    """Mock Redis client for testing."""

    def __init__(self):
        self.storage: Dict[str, str] = {}
        self.ttls: Dict[str, int] = {}

    async def setex(self, key: str, ttl: int, value: str):
        """Set with expiry."""
        self.storage[key] = value
        self.ttls[key] = ttl

    async def get(self, key: str) -> str:
        """Get value."""
        return self.storage.get(key)

    async def set(self, key: str, value: str):
        """Set value."""
        self.storage[key] = value


class HistoryScenario:
    """Test history scenario with known patterns."""

    def __init__(
        self,
        name: str,
        test_id: str,
        execution_pattern: str,  # stable, degrading, improving, flaky
        num_executions: int,
        expected_trend: str,  # stable, increasing, decreasing
        should_predict_failure: bool,
    ):
        self.name = name
        self.test_id = test_id
        self.execution_pattern = execution_pattern
        self.num_executions = num_executions
        self.expected_trend = expected_trend
        self.should_predict_failure = should_predict_failure


def generate_execution_history(
    scenario: HistoryScenario,
    base_time: datetime
) -> List[Dict[str, Any]]:
    """Generate execution history for a scenario.

    Args:
        scenario: History scenario
        base_time: Base timestamp

    Returns:
        List of execution records
    """
    executions = []
    base_execution_time = 100.0

    for i in range(scenario.num_executions):
        timestamp = base_time - timedelta(days=scenario.num_executions - i - 1)

        if scenario.execution_pattern == "stable":
            # Stable performance and reliability
            passed = random.random() < 0.95  # 95% pass rate
            execution_time = base_execution_time + random.uniform(-5, 5)

        elif scenario.execution_pattern == "degrading":
            # Performance degrading over time
            degradation_factor = (i / scenario.num_executions) * 0.5
            execution_time = base_execution_time * (1 + degradation_factor)
            # Also increasing failure rate
            pass_rate = 0.95 - (i / scenario.num_executions) * 0.3
            passed = random.random() < pass_rate

        elif scenario.execution_pattern == "improving":
            # Performance improving over time
            improvement_factor = (i / scenario.num_executions) * 0.3
            execution_time = base_execution_time * (1 - improvement_factor)
            # Also improving pass rate
            pass_rate = 0.70 + (i / scenario.num_executions) * 0.25
            passed = random.random() < pass_rate

        elif scenario.execution_pattern == "flaky":
            # Inconsistent performance
            execution_time = base_execution_time + random.uniform(-30, 30)
            passed = random.random() < 0.6  # 60% pass rate

        else:
            # Default stable
            passed = True
            execution_time = base_execution_time

        execution = {
            "timestamp": timestamp.isoformat(),
            "passed": passed,
            "execution_time_ms": execution_time,
            "failure_type": None if passed else FailureType.RUNTIME_ERROR.value,
            "error_message": None if passed else f"Error in run {i + 1}",
        }

        executions.append(execution)

    return executions


def generate_history_scenarios() -> List[HistoryScenario]:
    """Generate history tracking scenarios.

    Returns:
        List of scenarios with different patterns
    """
    scenarios = []

    # Stable test
    scenarios.append(HistoryScenario(
        name="Stable Test - Consistent Performance",
        test_id="test_stable_001",
        execution_pattern="stable",
        num_executions=50,
        expected_trend="stable",
        should_predict_failure=False,
    ))

    # Degrading performance
    scenarios.append(HistoryScenario(
        name="Degrading Test - Performance Regression",
        test_id="test_degrading_002",
        execution_pattern="degrading",
        num_executions=50,
        expected_trend="increasing",  # Execution time increasing
        should_predict_failure=True,  # Should predict maintenance needed
    ))

    # Improving performance
    scenarios.append(HistoryScenario(
        name="Improving Test - Performance Gains",
        test_id="test_improving_003",
        execution_pattern="improving",
        num_executions=50,
        expected_trend="decreasing",  # Execution time decreasing
        should_predict_failure=False,
    ))

    # Flaky test
    scenarios.append(HistoryScenario(
        name="Flaky Test - Inconsistent Results",
        test_id="test_flaky_004",
        execution_pattern="flaky",
        num_executions=50,
        expected_trend="stable",  # No clear trend
        should_predict_failure=True,  # High failure rate
    ))

    # Another stable test
    scenarios.append(HistoryScenario(
        name="Stable Test - Long History",
        test_id="test_stable_005",
        execution_pattern="stable",
        num_executions=100,
        expected_trend="stable",
        should_predict_failure=False,
    ))

    # Recently degraded
    scenarios.append(HistoryScenario(
        name="Recently Degraded - New Issues",
        test_id="test_recent_degraded_006",
        execution_pattern="degrading",
        num_executions=30,
        expected_trend="increasing",
        should_predict_failure=True,
    ))

    return scenarios


async def test_history_tracking(
    tracker: HistoryTracker,
    scenario: HistoryScenario,
    verbose: bool = False
) -> Tuple[bool, TestPerformanceHistory, Dict[str, Any]]:
    """Test history tracking for a scenario.

    Args:
        tracker: History tracker instance
        scenario: History scenario
        verbose: Whether to print verbose output

    Returns:
        Tuple of (success, history, metrics)
    """
    try:
        # Generate and record execution history
        base_time = datetime.now()
        executions = generate_execution_history(scenario, base_time)

        # Record all executions
        for execution in executions:
            await tracker.record_execution(scenario.test_id, execution)

        # Analyze trends
        history = await tracker.analyze_trends(
            scenario.test_id,
            time_window=scenario.num_executions
        )

        if history is None:
            if verbose:
                print(f"\n  {Colors.RED}No history for: {scenario.name}{Colors.ENDC}")
            return False, None, {"error": "no_history"}

        # Validate trend detection
        trend_correct = history.execution_time_trend == scenario.expected_trend

        # Validate failure prediction
        prediction_correct = (
            history.predicted_maintenance_needed == scenario.should_predict_failure
        )

        # Validate execution count
        count_correct = history.total_executions == scenario.num_executions

        # Calculate actual metrics from executions
        actual_failures = sum(1 for e in executions if not e["passed"])
        actual_failure_rate = actual_failures / len(executions)

        # Check if tracked failure rate is close to actual
        failure_rate_close = abs(history.failure_rate - actual_failure_rate) < 0.1

        success = trend_correct and prediction_correct and count_correct and failure_rate_close

        metrics = {
            "trend_correct": trend_correct,
            "prediction_correct": prediction_correct,
            "count_correct": count_correct,
            "failure_rate_close": failure_rate_close,
            "total_executions": history.total_executions,
            "total_failures": history.total_failures,
            "failure_rate": history.failure_rate,
            "actual_failure_rate": actual_failure_rate,
            "avg_execution_time_ms": history.avg_execution_time_ms,
            "execution_time_trend": history.execution_time_trend,
            "predicted_failure_probability": history.predicted_failure_probability,
            "predicted_maintenance_needed": history.predicted_maintenance_needed,
        }

        if verbose:
            print(f"\n  Scenario: {scenario.name}")
            print(f"    Pattern: {scenario.execution_pattern}")
            print(f"    Total Executions: {history.total_executions} {'✓' if count_correct else '✗'}")
            print(f"    Total Failures: {history.total_failures}")
            print(f"    Failure Rate: {history.failure_rate:.2%} (actual: {actual_failure_rate:.2%}) {'✓' if failure_rate_close else '✗'}")
            print(f"    Avg Execution Time: {history.avg_execution_time_ms:.1f}ms")
            print(f"    Trend: {history.execution_time_trend} (expected: {scenario.expected_trend}) {'✓' if trend_correct else '✗'}")
            print(f"    Predicted Failure Prob: {history.predicted_failure_probability:.2%}")
            print(f"    Maintenance Needed: {history.predicted_maintenance_needed} (expected: {scenario.should_predict_failure}) {'✓' if prediction_correct else '✗'}")

        return success, history, metrics

    except Exception as e:
        if verbose:
            print(f"\n  {Colors.RED}Error tracking history for {scenario.name}: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return False, None, {"error": str(e)}


async def run_history_tracking_test(
    num_iterations: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run history tracking test.

    Args:
        num_iterations: Number of test iterations
        verbose: Whether to print verbose output

    Returns:
        Test results dictionary
    """
    print_header("HISTORICAL TRACKING AND TREND ANALYSIS TEST")

    # Initialize tracker with mock Redis
    print(f"{Colors.BLUE}Initializing history tracker...{Colors.ENDC}")
    mock_redis = MockRedisClient()
    tracker = HistoryTracker(redis_client=mock_redis, retention_days=90)

    # Generate scenarios
    print(f"{Colors.BLUE}Generating history tracking scenarios...{Colors.ENDC}")
    base_scenarios = generate_history_scenarios()

    # Replicate for multiple iterations
    scenarios = base_scenarios * num_iterations

    print(f"{Colors.CYAN}Created {len(scenarios)} scenarios ({len(base_scenarios)} unique, {num_iterations} iterations){Colors.ENDC}\n")

    # Track results
    start_time = time.time()
    results = {
        "total": len(scenarios),
        "successful": 0,
        "failed": 0,
        "by_pattern": defaultdict(lambda: {"total": 0, "correct_trends": 0, "correct_predictions": 0}),
        "analyses": [],
        "metrics": {
            "success_rate": 0.0,
            "trend_accuracy": 0.0,
            "prediction_accuracy": 0.0,
            "failure_rate_accuracy": 0.0,
            "avg_executions_tracked": 0.0,
        }
    }

    # Test each scenario
    print(f"{Colors.BLUE}Testing history tracking and trend analysis...{Colors.ENDC}")

    trend_correct_count = 0
    prediction_correct_count = 0
    failure_rate_close_count = 0
    total_executions_tracked = 0

    for i, scenario in enumerate(scenarios):
        print_progress(i + 1, len(scenarios), f"Tracking test {i + 1}/{len(scenarios)}")

        success, history, metrics = await test_history_tracking(tracker, scenario, verbose)

        if success:
            results["successful"] += 1
        else:
            results["failed"] += 1

        # Track by pattern
        pattern = scenario.execution_pattern
        results["by_pattern"][pattern]["total"] += 1
        if "error" not in metrics:
            if metrics.get("trend_correct"):
                results["by_pattern"][pattern]["correct_trends"] += 1
                trend_correct_count += 1
            if metrics.get("prediction_correct"):
                results["by_pattern"][pattern]["correct_predictions"] += 1
                prediction_correct_count += 1
            if metrics.get("failure_rate_close"):
                failure_rate_close_count += 1

            total_executions_tracked += metrics.get("total_executions", 0)

        results["analyses"].append({
            "scenario": scenario.name,
            "success": success,
            "metrics": metrics,
        })

    # Calculate final metrics
    total = len(scenarios)
    results["metrics"]["success_rate"] = results["successful"] / total
    results["metrics"]["trend_accuracy"] = trend_correct_count / total
    results["metrics"]["prediction_accuracy"] = prediction_correct_count / total
    results["metrics"]["failure_rate_accuracy"] = failure_rate_close_count / total
    results["metrics"]["avg_executions_tracked"] = total_executions_tracked / total
    results["execution_time_seconds"] = time.time() - start_time

    return results


def print_results(results: Dict[str, Any], target_accuracy: float = 0.85):
    """Print test results.

    Args:
        results: Test results
        target_accuracy: Target accuracy threshold
    """
    print_header("TEST RESULTS")

    # Overall metrics
    print(f"{Colors.BOLD}Overall Results:{Colors.ENDC}")
    print(f"  Total Scenarios: {results['total']}")
    print(f"  Successful: {results['successful']} ({results['metrics']['success_rate']:.1%})")
    print(f"  Failed: {results['failed']}")
    print(f"  Execution Time: {results['execution_time_seconds']:.2f}s")
    print()

    # Tracking metrics
    print(f"{Colors.BOLD}Tracking Metrics:{Colors.ENDC}")
    print(f"  Average Executions Tracked: {results['metrics']['avg_executions_tracked']:.1f}")
    print()

    # Detailed metrics
    print(f"{Colors.BOLD}Analysis Accuracy:{Colors.ENDC}")

    metrics = results['metrics']

    def print_metric(name: str, value: float, target: float):
        status = "✓" if value >= target else "✗"
        color = Colors.GREEN if value >= target else Colors.RED
        print(f"  {name:.<40} {color}{value:>6.1%} {status}{Colors.ENDC}")

    print_metric("Overall Success Rate", metrics['success_rate'], target_accuracy)
    print_metric("Trend Detection Accuracy", metrics['trend_accuracy'], target_accuracy)
    print_metric("Failure Prediction Accuracy", metrics['prediction_accuracy'], target_accuracy)
    print_metric("Failure Rate Tracking Accuracy", metrics['failure_rate_accuracy'], target_accuracy)
    print()

    # By execution pattern
    print(f"{Colors.BOLD}Accuracy by Execution Pattern:{Colors.ENDC}")
    for pattern, stats in sorted(results['by_pattern'].items()):
        trend_acc = stats['correct_trends'] / stats['total'] if stats['total'] > 0 else 0.0
        pred_acc = stats['correct_predictions'] / stats['total'] if stats['total'] > 0 else 0.0
        overall_acc = (stats['correct_trends'] + stats['correct_predictions']) / (stats['total'] * 2)

        status = "✓" if overall_acc >= target_accuracy else "✗"
        color = Colors.GREEN if overall_acc >= target_accuracy else Colors.RED
        print(f"  {pattern:.<20} {color}{overall_acc:>6.1%} {status}{Colors.ENDC} "
              f"(trend: {trend_acc:.1%}, prediction: {pred_acc:.1%})")
    print()

    # Final verdict
    overall_success = metrics['success_rate'] >= target_accuracy

    if overall_success:
        print_success(f"SUCCESS: Achieved {metrics['success_rate']:.1%} accuracy (target: {target_accuracy:.0%})")
        return 0
    else:
        print_error(f"FAILURE: Achieved {metrics['success_rate']:.1%} accuracy (target: {target_accuracy:.0%})")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test historical tracking and trend analysis (target: 85%+ accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of test iterations (default: 10)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.85,
        help="Target accuracy threshold (default: 0.85)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--record-executions",
        action="store_true",
        help="Record test executions",
    )
    parser.add_argument(
        "--analyze-trends",
        action="store_true",
        help="Analyze historical trends",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run test
    try:
        results = asyncio.run(run_history_tracking_test(
            num_iterations=args.iterations,
            verbose=args.verbose,
        ))

        exit_code = print_results(results, target_accuracy=args.target)
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Test failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
