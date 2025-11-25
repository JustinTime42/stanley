#!/usr/bin/env python3
"""Integration test for test optimization suggestions.

Tests the optimizer's ability to suggest improvements that achieve
target time savings of 30%+.

Usage:
    python scripts/test_optimization_suggestions.py [--verbose] [--samples N]
    python scripts/test_optimization_suggestions.py --slow-tests ./performance-tests/
"""

import argparse
import asyncio
import logging
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.healing_models import TestOptimization
from src.testing.healing import TestOptimizer


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


class OptimizationScenario:
    """Test optimization scenario."""

    def __init__(
        self,
        name: str,
        test_id: str,
        current_time_ms: float,
        performance_data: Dict[str, Any],
        expected_optimization_types: List[str],
        expected_min_savings_percent: float,
    ):
        self.name = name
        self.test_id = test_id
        self.current_time_ms = current_time_ms
        self.performance_data = performance_data
        self.expected_optimization_types = expected_optimization_types
        self.expected_min_savings_percent = expected_min_savings_percent


def generate_optimization_scenarios() -> List[OptimizationScenario]:
    """Generate test optimization scenarios.

    Returns:
        List of scenarios with performance issues
    """
    scenarios = []

    # Scenario 1: Test with unnecessary sleeps/waits
    scenarios.append(OptimizationScenario(
        name="Unnecessary Fixed Waits",
        test_id="tests/test_slow_waits.py::test_with_sleeps",
        current_time_ms=8000,
        performance_data={
            "tests": {
                "tests/test_slow_waits.py::test_with_sleeps": {
                    "execution_time_ms": 8000,
                    "test_file": "tests/test_slow_waits.py",
                }
            }
        },
        expected_optimization_types=["remove_waits", "optimize_waits"],
        expected_min_savings_percent=60,  # Can save 60%+ by replacing fixed waits
    ))

    # Scenario 2: Test with redundant setup
    scenarios.append(OptimizationScenario(
        name="Redundant Test Setup",
        test_id="tests/test_redundant_setup.py::test_with_setup",
        current_time_ms=6000,
        performance_data={
            "tests": {
                "tests/test_redundant_setup.py::test_with_setup": {
                    "execution_time_ms": 6000,
                    "test_file": "tests/test_redundant_setup.py",
                }
            }
        },
        expected_optimization_types=["optimize_setup", "cache_setup"],
        expected_min_savings_percent=40,  # Can save 40%+ by caching setup
    ))

    # Scenario 3: Test that can be parallelized
    scenarios.append(OptimizationScenario(
        name="Parallelizable Test Operations",
        test_id="tests/test_sequential.py::test_parallel_ops",
        current_time_ms=10000,
        performance_data={
            "tests": {
                "tests/test_sequential.py::test_parallel_ops": {
                    "execution_time_ms": 10000,
                    "test_file": "tests/test_sequential.py",
                }
            }
        },
        expected_optimization_types=["parallelize", "concurrent"],
        expected_min_savings_percent=50,  # Can save 50%+ with parallelization
    ))

    # Scenario 4: Test with excessive I/O
    scenarios.append(OptimizationScenario(
        name="Excessive File I/O",
        test_id="tests/test_io_heavy.py::test_file_operations",
        current_time_ms=7500,
        performance_data={
            "tests": {
                "tests/test_io_heavy.py::test_file_operations": {
                    "execution_time_ms": 7500,
                    "test_file": "tests/test_io_heavy.py",
                }
            }
        },
        expected_optimization_types=["optimize_io", "mock_io"],
        expected_min_savings_percent=45,  # Can save 45%+ by mocking I/O
    ))

    # Scenario 5: Multiple slow tests (redundancy)
    scenarios.append(OptimizationScenario(
        name="Redundant Test Cases",
        test_id="tests/test_redundant.py::test_duplicate_1",
        current_time_ms=5000,
        performance_data={
            "tests": {
                "tests/test_redundant.py::test_duplicate_1": {
                    "execution_time_ms": 5000,
                    "test_file": "tests/test_redundant.py",
                },
                "tests/test_redundant.py::test_duplicate_2": {
                    "execution_time_ms": 5000,
                    "test_file": "tests/test_redundant.py",
                },
                "tests/test_redundant.py::test_duplicate_3": {
                    "execution_time_ms": 5000,
                    "test_file": "tests/test_redundant.py",
                },
            }
        },
        expected_optimization_types=["merge_redundant", "deduplicate"],
        expected_min_savings_percent=66,  # Can save 66%+ by merging 3→1
    ))

    # Scenario 6: Test with inefficient mocks
    scenarios.append(OptimizationScenario(
        name="Inefficient Mock Setup",
        test_id="tests/test_heavy_mocks.py::test_with_mocks",
        current_time_ms=5500,
        performance_data={
            "tests": {
                "tests/test_heavy_mocks.py::test_with_mocks": {
                    "execution_time_ms": 5500,
                    "test_file": "tests/test_heavy_mocks.py",
                }
            }
        },
        expected_optimization_types=["optimize_mocks", "shared_mocks"],
        expected_min_savings_percent=30,  # Can save 30%+ by sharing mocks
    ))

    # Scenario 7: Test with slow database operations
    scenarios.append(OptimizationScenario(
        name="Slow Database Operations",
        test_id="tests/test_database.py::test_db_queries",
        current_time_ms=9000,
        performance_data={
            "tests": {
                "tests/test_database.py::test_db_queries": {
                    "execution_time_ms": 9000,
                    "test_file": "tests/test_database.py",
                }
            }
        },
        expected_optimization_types=["optimize_io", "mock_database"],
        expected_min_savings_percent=70,  # Can save 70%+ by mocking DB
    ))

    # Scenario 8: Test with network calls
    scenarios.append(OptimizationScenario(
        name="External Network Calls",
        test_id="tests/test_api.py::test_api_calls",
        current_time_ms=12000,
        performance_data={
            "tests": {
                "tests/test_api.py::test_api_calls": {
                    "execution_time_ms": 12000,
                    "test_file": "tests/test_api.py",
                }
            }
        },
        expected_optimization_types=["mock_io", "mock_network"],
        expected_min_savings_percent=80,  # Can save 80%+ by mocking network
    ))

    return scenarios


async def test_optimization_scenario(
    optimizer: TestOptimizer,
    scenario: OptimizationScenario,
    verbose: bool = False
) -> Tuple[bool, List[TestOptimization], Dict[str, Any]]:
    """Test optimization for a scenario.

    Args:
        optimizer: Test optimizer instance
        scenario: Optimization scenario
        verbose: Whether to print verbose output

    Returns:
        Tuple of (success, optimizations, metrics)
    """
    try:
        # Generate optimization suggestions
        optimizations = await optimizer.suggest_optimizations(
            test_suite=scenario.test_id,
            performance_data=scenario.performance_data,
            min_time_saving_ms=100,
        )

        if not optimizations:
            if verbose:
                print(f"\n  {Colors.YELLOW}No optimizations suggested for: {scenario.name}{Colors.ENDC}")
            return False, [], {"reason": "no_suggestions"}

        # Calculate total time savings
        total_savings_ms = sum(opt.time_saving_ms for opt in optimizations)
        savings_percent = (total_savings_ms / scenario.current_time_ms) * 100

        # Check if suggested optimization types match expected
        suggested_types = [opt.optimization_type for opt in optimizations]
        has_expected_type = any(
            any(expected in opt_type for expected in scenario.expected_optimization_types)
            for opt_type in suggested_types
        )

        # Check if savings meet target
        meets_target = savings_percent >= scenario.expected_min_savings_percent

        success = has_expected_type and meets_target

        metrics = {
            "num_suggestions": len(optimizations),
            "total_savings_ms": total_savings_ms,
            "savings_percent": savings_percent,
            "expected_min_savings_percent": scenario.expected_min_savings_percent,
            "meets_target": meets_target,
            "has_expected_type": has_expected_type,
            "suggested_types": suggested_types,
            "optimizations": [
                {
                    "type": opt.optimization_type,
                    "savings_ms": opt.time_saving_ms,
                    "savings_percent": opt.time_saving_percent,
                    "risk": opt.risk_level,
                    "priority": opt.priority,
                }
                for opt in optimizations
            ]
        }

        if verbose:
            print(f"\n  Scenario: {scenario.name}")
            print(f"    Current Time: {scenario.current_time_ms:.0f}ms")
            print(f"    Suggestions: {len(optimizations)}")
            print(f"    Total Savings: {total_savings_ms:.0f}ms ({savings_percent:.1f}%)")
            print(f"    Target Savings: {scenario.expected_min_savings_percent:.1f}% {'✓' if meets_target else '✗'}")
            print(f"    Expected Type Found: {has_expected_type} {'✓' if has_expected_type else '✗'}")
            for i, opt in enumerate(optimizations[:3], 1):  # Show top 3
                print(f"      {i}. {opt.optimization_type}: {opt.time_saving_ms:.0f}ms ({opt.time_saving_percent:.1f}%) - {opt.risk_level} risk")

        return success, optimizations, metrics

    except Exception as e:
        if verbose:
            print(f"\n  {Colors.RED}Error analyzing scenario {scenario.name}: {e}{Colors.ENDC}")
        return False, [], {"error": str(e)}


async def run_optimization_test(
    num_samples: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run optimization suggestions test.

    Args:
        num_samples: Number of test samples
        verbose: Whether to print verbose output

    Returns:
        Test results dictionary
    """
    print_header("TEST OPTIMIZATION SUGGESTIONS TEST")

    # Initialize optimizer
    print(f"{Colors.BLUE}Initializing test optimizer...{Colors.ENDC}")
    optimizer = TestOptimizer(slow_test_threshold_ms=5000)

    # Generate scenarios
    print(f"{Colors.BLUE}Generating {num_samples} optimization scenarios...{Colors.ENDC}")
    base_scenarios = generate_optimization_scenarios()

    # Replicate scenarios to reach num_samples
    scenarios = []
    while len(scenarios) < num_samples:
        scenarios.extend(base_scenarios)
    scenarios = scenarios[:num_samples]

    print(f"{Colors.CYAN}Created {len(scenarios)} scenarios across {len(base_scenarios)} optimization types{Colors.ENDC}\n")

    # Track results
    start_time = time.time()
    results = {
        "total": len(scenarios),
        "successful": 0,
        "failed": 0,
        "by_type": defaultdict(lambda: {"total": 0, "successful": 0, "total_savings": 0}),
        "analyses": [],
        "metrics": {
            "success_rate": 0.0,
            "avg_savings_percent": 0.0,
            "total_time_savings_ms": 0.0,
            "total_current_time_ms": 0.0,
            "overall_savings_percent": 0.0,
            "avg_suggestions_per_test": 0.0,
        }
    }

    # Test each scenario
    print(f"{Colors.BLUE}Generating optimization suggestions...{Colors.ENDC}")

    total_savings_ms = 0.0
    total_current_time_ms = 0.0
    total_suggestions = 0
    savings_percentages = []

    for i, scenario in enumerate(scenarios):
        print_progress(i + 1, len(scenarios), f"Analyzing test {i + 1}/{len(scenarios)}")

        success, optimizations, metrics = await test_optimization_scenario(
            optimizer, scenario, verbose
        )

        if success:
            results["successful"] += 1
        else:
            results["failed"] += 1

        # Track by optimization type
        for opt_type in scenario.expected_optimization_types:
            results["by_type"][opt_type]["total"] += 1
            if success:
                results["by_type"][opt_type]["successful"] += 1
                if "error" not in metrics:
                    results["by_type"][opt_type]["total_savings"] += metrics.get("total_savings_ms", 0)

        # Track metrics
        if "error" not in metrics:
            total_savings_ms += metrics.get("total_savings_ms", 0.0)
            total_current_time_ms += scenario.current_time_ms
            total_suggestions += metrics.get("num_suggestions", 0)
            if metrics.get("meets_target"):
                savings_percentages.append(metrics.get("savings_percent", 0.0))

        results["analyses"].append({
            "scenario": scenario.name,
            "success": success,
            "metrics": metrics,
        })

    # Calculate final metrics
    total = len(scenarios)
    results["metrics"]["success_rate"] = results["successful"] / total
    results["metrics"]["total_time_savings_ms"] = total_savings_ms
    results["metrics"]["total_current_time_ms"] = total_current_time_ms
    results["metrics"]["overall_savings_percent"] = (
        (total_savings_ms / total_current_time_ms * 100)
        if total_current_time_ms > 0 else 0.0
    )
    results["metrics"]["avg_savings_percent"] = (
        sum(savings_percentages) / len(savings_percentages)
        if savings_percentages else 0.0
    )
    results["metrics"]["avg_suggestions_per_test"] = total_suggestions / total
    results["execution_time_seconds"] = time.time() - start_time

    return results


def print_results(results: Dict[str, Any], target_savings: float = 30.0):
    """Print test results.

    Args:
        results: Test results
        target_savings: Target savings percentage
    """
    print_header("TEST RESULTS")

    # Overall metrics
    print(f"{Colors.BOLD}Overall Results:{Colors.ENDC}")
    print(f"  Total Scenarios: {results['total']}")
    print(f"  Successful: {results['successful']} ({results['metrics']['success_rate']:.1%})")
    print(f"  Failed: {results['failed']}")
    print(f"  Execution Time: {results['execution_time_seconds']:.2f}s")
    print()

    # Time savings
    print(f"{Colors.BOLD}Performance Impact:{Colors.ENDC}")
    print(f"  Total Current Time: {results['metrics']['total_current_time_ms']:.0f}ms")
    print(f"  Total Time Savings: {results['metrics']['total_time_savings_ms']:.0f}ms")
    print(f"  Average Suggestions per Test: {results['metrics']['avg_suggestions_per_test']:.1f}")
    print()

    # Detailed metrics
    print(f"{Colors.BOLD}Optimization Metrics:{Colors.ENDC}")

    metrics = results['metrics']

    def print_metric(name: str, value: float, target: float, suffix: str = "%"):
        status = "✓" if value >= target else "✗"
        color = Colors.GREEN if value >= target else Colors.RED
        print(f"  {name:.<40} {color}{value:>6.1f}{suffix} {status}{Colors.ENDC}")

    print_metric("Overall Time Savings", metrics['overall_savings_percent'], target_savings)
    print_metric("Average Savings (Successful)", metrics['avg_savings_percent'], target_savings)
    print_metric("Suggestion Success Rate", metrics['success_rate'] * 100, 80)
    print()

    # By optimization type
    print(f"{Colors.BOLD}Performance by Optimization Type:{Colors.ENDC}")
    for opt_type, stats in sorted(results['by_type'].items()):
        success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0.0
        avg_savings = stats['total_savings'] / stats['successful'] if stats['successful'] > 0 else 0.0
        status = "✓" if success_rate >= 0.8 else "✗"
        color = Colors.GREEN if success_rate >= 0.8 else Colors.RED
        print(f"  {opt_type:.<30} {color}{success_rate:>6.1%} ({stats['successful']}/{stats['total']}) "
              f"[avg: {avg_savings:>6.0f}ms] {status}{Colors.ENDC}")
    print()

    # Final verdict
    overall_success = metrics['overall_savings_percent'] >= target_savings

    if overall_success:
        print_success(
            f"SUCCESS: Achieved {metrics['overall_savings_percent']:.1f}% time savings "
            f"(target: {target_savings:.0f}%)"
        )
        return 0
    else:
        print_error(
            f"FAILURE: Achieved {metrics['overall_savings_percent']:.1f}% time savings "
            f"(target: {target_savings:.0f}%)"
        )
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test optimization suggestions (target: 30%+ time savings)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of test samples (default: 50)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=30.0,
        help="Target time savings percentage (default: 30.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--slow-tests",
        type=str,
        help="Path to slow tests to analyze",
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
        results = asyncio.run(run_optimization_test(
            num_samples=args.samples,
            verbose=args.verbose,
        ))

        exit_code = print_results(results, target_savings=args.target)
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
