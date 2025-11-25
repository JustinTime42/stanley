#!/usr/bin/env python3
"""Integration test for flaky test detection.

Tests the flaky detector's ability to identify non-deterministic tests
with target accuracy of 90%+.

Usage:
    python scripts/test_flaky_detection.py [--verbose] [--runs N]
    python scripts/test_flaky_detection.py --inject-flakiness ./stable-tests/
"""

import argparse
import asyncio
import logging
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.healing_models import FlakyTestResult, FailureType
from src.testing.healing import FlakyDetector


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


class MockTestRunner:
    """Mock test runner for simulating test executions."""

    def __init__(self):
        self.execution_count = defaultdict(int)

    async def run_single_test(self, test_id: str) -> Dict[str, Any]:
        """Run a single test (mock implementation).

        Args:
            test_id: Test identifier

        Returns:
            Test result dictionary
        """
        # Extract test behavior from test_id
        # Format: test_<type>_<behavior>
        parts = test_id.split('_')

        if len(parts) >= 3:
            behavior = parts[2]
        else:
            behavior = "stable"

        self.execution_count[test_id] += 1
        run_num = self.execution_count[test_id]

        # Simulate different test behaviors
        if behavior == "stable":
            # Always passes
            return {
                "passed": True,
                "execution_time_ms": 100 + random.uniform(-5, 5),
                "error_message": None,
            }

        elif behavior == "intermittent":
            # Passes 50% of the time
            passed = random.random() < 0.5
            return {
                "passed": passed,
                "execution_time_ms": 100 + random.uniform(-10, 10),
                "error_message": None if passed else "Intermittent failure",
            }

        elif behavior == "timing":
            # Timing-dependent: high execution time variance
            exec_time = 100 + random.uniform(-50, 100)
            passed = exec_time < 120
            return {
                "passed": passed,
                "execution_time_ms": exec_time,
                "error_message": None if passed else "Timeout",
            }

        elif behavior == "racecondition":
            # Race condition: fails occasionally
            passed = random.random() < 0.85
            return {
                "passed": passed,
                "execution_time_ms": 100 + random.uniform(-10, 30),
                "error_message": None if passed else f"Race condition (run {run_num})",
            }

        elif behavior == "environmental":
            # Environment-dependent: different failure modes
            rand = random.random()
            if rand < 0.7:
                passed = True
                error = None
            elif rand < 0.85:
                passed = False
                error = "Network error"
            else:
                passed = False
                error = "Database connection failed"

            return {
                "passed": passed,
                "execution_time_ms": 100 + random.uniform(-20, 20),
                "error_message": error,
            }

        elif behavior == "unstable":
            # Highly unstable: passes 20% of time
            passed = random.random() < 0.2
            return {
                "passed": passed,
                "execution_time_ms": 100 + random.uniform(-30, 30),
                "error_message": None if passed else "Unstable test failure",
            }

        elif behavior == "failed":
            # Always fails (not flaky)
            return {
                "passed": False,
                "execution_time_ms": 100 + random.uniform(-5, 5),
                "error_message": "Consistent failure",
            }

        else:
            # Default stable behavior
            return {
                "passed": True,
                "execution_time_ms": 100,
                "error_message": None,
            }


class FlakyScenario:
    """Test scenario for flaky detection."""

    def __init__(
        self,
        test_id: str,
        name: str,
        is_flaky: bool,
        expected_pass_rate: float,
        description: str,
    ):
        self.test_id = test_id
        self.name = name
        self.is_flaky = is_flaky
        self.expected_pass_rate = expected_pass_rate
        self.description = description


def generate_flaky_scenarios() -> List[FlakyScenario]:
    """Generate flaky test detection scenarios.

    Returns:
        List of test scenarios with known flakiness
    """
    scenarios = []

    # Stable tests (not flaky)
    scenarios.extend([
        FlakyScenario(
            test_id="test_001_stable_pass",
            name="Stable Test - Always Passes",
            is_flaky=False,
            expected_pass_rate=1.0,
            description="Test that consistently passes",
        ),
        FlakyScenario(
            test_id="test_002_stable_fail",
            name="Stable Test - Always Fails",
            is_flaky=False,
            expected_pass_rate=0.0,
            description="Test that consistently fails (not flaky)",
        ),
    ])

    # Flaky tests - intermittent failures
    scenarios.extend([
        FlakyScenario(
            test_id="test_003_intermittent_50",
            name="Intermittent Flaky - 50% Pass Rate",
            is_flaky=True,
            expected_pass_rate=0.5,
            description="Test with 50% pass rate",
        ),
        FlakyScenario(
            test_id="test_004_intermittent_70",
            name="Intermittent Flaky - 70% Pass Rate",
            is_flaky=True,
            expected_pass_rate=0.7,
            description="Test with 70% pass rate",
        ),
    ])

    # Flaky tests - timing issues
    scenarios.extend([
        FlakyScenario(
            test_id="test_005_timing_variance",
            name="Timing Flaky - High Variance",
            is_flaky=True,
            expected_pass_rate=0.7,
            description="Test with high execution time variance",
        ),
    ])

    # Flaky tests - race conditions
    scenarios.extend([
        FlakyScenario(
            test_id="test_006_racecondition_rare",
            name="Race Condition - Rare Failures",
            is_flaky=True,
            expected_pass_rate=0.85,
            description="Test with occasional race condition failures",
        ),
    ])

    # Flaky tests - environmental
    scenarios.extend([
        FlakyScenario(
            test_id="test_007_environmental_deps",
            name="Environmental Flaky - External Dependencies",
            is_flaky=True,
            expected_pass_rate=0.7,
            description="Test dependent on external resources",
        ),
    ])

    # Flaky tests - highly unstable
    scenarios.extend([
        FlakyScenario(
            test_id="test_008_unstable_20",
            name="Highly Unstable - 20% Pass Rate",
            is_flaky=True,
            expected_pass_rate=0.2,
            description="Test that rarely passes",
        ),
    ])

    # More stable tests
    scenarios.extend([
        FlakyScenario(
            test_id="test_009_stable_consistent",
            name="Stable Test - Consistent",
            is_flaky=False,
            expected_pass_rate=1.0,
            description="Another stable test",
        ),
        FlakyScenario(
            test_id="test_010_stable_broken",
            name="Stable Test - Broken",
            is_flaky=False,
            expected_pass_rate=0.0,
            description="Consistently broken test (not flaky)",
        ),
    ])

    return scenarios


async def test_flaky_detection(
    detector: FlakyDetector,
    scenario: FlakyScenario,
    runs_per_test: int,
    verbose: bool = False
) -> Tuple[bool, FlakyTestResult, Dict[str, Any]]:
    """Test flaky detection for a scenario.

    Args:
        detector: Flaky detector instance
        scenario: Test scenario
        runs_per_test: Number of runs per test
        verbose: Whether to print verbose output

    Returns:
        Tuple of (correct, result, metrics)
    """
    try:
        # Detect flakiness
        results = await detector.detect_flaky_tests([scenario.test_id], runs_per_test)

        if not results:
            if verbose:
                print(f"\n  {Colors.RED}No results for: {scenario.name}{Colors.ENDC}")
            return False, None, {"error": "no_results"}

        result = results[0]

        # Check if detection is correct
        detected_flaky = result.is_flaky
        actual_flaky = scenario.is_flaky

        correct = detected_flaky == actual_flaky

        # Additional validation metrics
        pass_rate_reasonable = abs(result.pass_rate - scenario.expected_pass_rate) < 0.2
        flakiness_score_reasonable = (
            (result.flakiness_score > 0.3 and actual_flaky) or
            (result.flakiness_score < 0.3 and not actual_flaky)
        )

        metrics = {
            "detected_flaky": detected_flaky,
            "actual_flaky": actual_flaky,
            "correct_detection": correct,
            "flakiness_score": result.flakiness_score,
            "pass_rate": result.pass_rate,
            "expected_pass_rate": scenario.expected_pass_rate,
            "pass_rate_reasonable": pass_rate_reasonable,
            "total_runs": result.total_runs,
            "mean_time_ms": result.mean_time_ms,
            "std_dev_time_ms": result.std_dev_time_ms,
        }

        if verbose:
            print(f"\n  Scenario: {scenario.name}")
            print(f"    Expected Flaky: {actual_flaky}")
            print(f"    Detected Flaky: {detected_flaky} {'✓' if correct else '✗'}")
            print(f"    Flakiness Score: {result.flakiness_score:.2%}")
            print(f"    Pass Rate: {result.pass_rate:.2%} (expected: {scenario.expected_pass_rate:.2%})")
            print(f"    Total Runs: {result.total_runs}")
            print(f"    Mean Time: {result.mean_time_ms:.1f}ms ± {result.std_dev_time_ms:.1f}ms")

        return correct, result, metrics

    except Exception as e:
        if verbose:
            print(f"\n  {Colors.RED}Error testing scenario {scenario.name}: {e}{Colors.ENDC}")
        return False, None, {"error": str(e)}


async def run_flaky_detection_test(
    runs_per_test: int = 20,
    num_iterations: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run flaky detection test.

    Args:
        runs_per_test: Number of runs per test
        num_iterations: Number of test iterations
        verbose: Whether to print verbose output

    Returns:
        Test results dictionary
    """
    print_header("FLAKY TEST DETECTION TEST")

    # Initialize detector with mock runner
    print(f"{Colors.BLUE}Initializing flaky detector...{Colors.ENDC}")
    mock_runner = MockTestRunner()
    detector = FlakyDetector(test_runner=mock_runner, min_runs=10)

    # Generate scenarios
    print(f"{Colors.BLUE}Generating test scenarios...{Colors.ENDC}")
    base_scenarios = generate_flaky_scenarios()

    # Replicate for multiple iterations
    scenarios = base_scenarios * num_iterations

    print(f"{Colors.CYAN}Created {len(scenarios)} test scenarios ({len(base_scenarios)} unique, {num_iterations} iterations){Colors.ENDC}")
    print(f"{Colors.CYAN}Runs per test: {runs_per_test}{Colors.ENDC}\n")

    # Track results
    start_time = time.time()
    results = {
        "total": len(scenarios),
        "correct_detections": 0,
        "incorrect_detections": 0,
        "true_positives": 0,  # Correctly identified flaky
        "true_negatives": 0,  # Correctly identified stable
        "false_positives": 0,  # Incorrectly marked as flaky
        "false_negatives": 0,  # Missed flaky test
        "detections": [],
        "metrics": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "avg_flakiness_score": 0.0,
        }
    }

    # Test each scenario
    print(f"{Colors.BLUE}Running flaky detection tests...{Colors.ENDC}")

    total_flakiness_score = 0.0

    for i, scenario in enumerate(scenarios):
        print_progress(i + 1, len(scenarios), f"Testing {i + 1}/{len(scenarios)}")

        correct, result, metrics = await test_flaky_detection(
            detector, scenario, runs_per_test, verbose
        )

        if correct:
            results["correct_detections"] += 1

            # Track true positives and negatives
            if scenario.is_flaky:
                results["true_positives"] += 1
            else:
                results["true_negatives"] += 1
        else:
            results["incorrect_detections"] += 1

            # Track false positives and negatives
            if scenario.is_flaky:
                results["false_negatives"] += 1
            else:
                results["false_positives"] += 1

        if "error" not in metrics and result:
            total_flakiness_score += metrics.get("flakiness_score", 0.0)

        results["detections"].append({
            "scenario": scenario.name,
            "correct": correct,
            "metrics": metrics,
        })

    # Calculate final metrics
    total = len(scenarios)
    tp = results["true_positives"]
    tn = results["true_negatives"]
    fp = results["false_positives"]
    fn = results["false_negatives"]

    results["metrics"]["accuracy"] = (tp + tn) / total if total > 0 else 0.0
    results["metrics"]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    results["metrics"]["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    precision = results["metrics"]["precision"]
    recall = results["metrics"]["recall"]
    results["metrics"]["f1_score"] = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    results["metrics"]["avg_flakiness_score"] = total_flakiness_score / total
    results["execution_time_seconds"] = time.time() - start_time

    return results


def print_results(results: Dict[str, Any], target_accuracy: float = 0.90):
    """Print test results.

    Args:
        results: Test results
        target_accuracy: Target accuracy threshold
    """
    print_header("TEST RESULTS")

    # Overall metrics
    print(f"{Colors.BOLD}Overall Results:{Colors.ENDC}")
    print(f"  Total Scenarios: {results['total']}")
    print(f"  Correct Detections: {results['correct_detections']} ({results['metrics']['accuracy']:.1%})")
    print(f"  Incorrect Detections: {results['incorrect_detections']}")
    print(f"  Execution Time: {results['execution_time_seconds']:.2f}s")
    print()

    # Confusion matrix
    print(f"{Colors.BOLD}Detection Breakdown:{Colors.ENDC}")
    print(f"  True Positives (Flaky → Detected):  {results['true_positives']}")
    print(f"  True Negatives (Stable → Stable):   {results['true_negatives']}")
    print(f"  False Positives (Stable → Flaky):   {results['false_positives']}")
    print(f"  False Negatives (Flaky → Stable):   {results['false_negatives']}")
    print()

    # Detailed metrics
    print(f"{Colors.BOLD}Detection Metrics:{Colors.ENDC}")

    metrics = results['metrics']

    def print_metric(name: str, value: float, target: float):
        status = "✓" if value >= target else "✗"
        color = Colors.GREEN if value >= target else Colors.RED
        print(f"  {name:.<40} {color}{value:>6.1%} {status}{Colors.ENDC}")

    print_metric("Accuracy (Correct / Total)", metrics['accuracy'], target_accuracy)
    print_metric("Precision (TP / (TP + FP))", metrics['precision'], 0.85)
    print_metric("Recall (TP / (TP + FN))", metrics['recall'], 0.85)
    print_metric("F1 Score", metrics['f1_score'], 0.85)
    print(f"  {'Average Flakiness Score':.<40} {metrics['avg_flakiness_score']:>6.1%}")
    print()

    # Final verdict
    overall_success = metrics['accuracy'] >= target_accuracy

    if overall_success:
        print_success(f"SUCCESS: Achieved {metrics['accuracy']:.1%} detection accuracy (target: {target_accuracy:.0%})")
        return 0
    else:
        print_error(f"FAILURE: Achieved {metrics['accuracy']:.1%} detection accuracy (target: {target_accuracy:.0%})")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test flaky test detection (target: 90%+ accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of runs per test (default: 20)",
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
        default=0.90,
        help="Target accuracy threshold (default: 0.90)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--inject-flakiness",
        type=str,
        help="Path to stable tests to inject flakiness",
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
        results = asyncio.run(run_flaky_detection_test(
            runs_per_test=args.runs,
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
