"""Test healing on large test suites with performance benchmarks.

This script validates that the self-healing system can efficiently handle
large test suites (100+ tests) and complete healing in <10 seconds for typical failures.

Success Criteria:
- Heal 75%+ of failures in large suites
- Complete healing in <10 seconds per test
- Handle concurrent failures efficiently
- Maintain test suite integrity
- Generate comprehensive performance reports
"""

import asyncio
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Import healing components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.testing.healing.failure_analyzer import FailureAnalyzer
from src.testing.healing.test_repairer import TestRepairer
from src.testing.healing.flaky_detector import FlakyDetector
from src.models.healing_models import (
    TestFailure,
    FailureType,
    HealingRequest,
    HealingResult,
)


@dataclass
class PerformanceMetrics:
    """Performance metrics for large suite healing."""
    total_tests: int
    total_failures: int
    healing_started_at: str
    healing_completed_at: str
    total_healing_time_ms: float
    average_healing_time_per_test_ms: float
    successful_repairs: int
    failed_repairs: int
    repair_success_rate: float
    healing_throughput_tests_per_second: float
    memory_usage_mb: float
    peak_memory_mb: float
    met_performance_target: bool  # <10s per test


@dataclass
class TestSuiteMetrics:
    """Metrics for test suite characteristics."""
    suite_name: str
    total_tests: int
    test_frameworks: List[str]
    average_test_complexity: float
    test_types: Dict[str, int]
    failure_distribution: Dict[str, int]


class MockTestRunner:
    """Mock test runner for simulated large suite."""

    async def run_single_test(self, test_id: str) -> Dict[str, Any]:
        """Simulate running a single test."""
        await asyncio.sleep(0.01)  # Simulate test execution
        return {
            "passed": True,
            "execution_time_ms": 50,
            "error_message": None,
        }


class LargeSuiteHealingTester:
    """Test healing on large test suites."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize tester."""
        self.output_dir = output_dir or Path("test_results/large_suite_healing")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize healing components
        self.analyzer = FailureAnalyzer()
        self.repairer = TestRepairer()
        self.flaky_detector = FlakyDetector(test_runner=MockTestRunner())

        # Test suite configurations
        self.suite_sizes = [50, 100, 200, 500]  # Number of tests per suite
        self.failure_rates = [0.1, 0.2, 0.3]  # Percentage of failing tests

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all large suite healing tests."""
        print("=" * 80)
        print("LARGE TEST SUITE HEALING - VALIDATION SUITE")
        print("=" * 80)
        print(f"Target: <10s healing time per test")
        print(f"Target: 75%+ repair success rate")
        print()

        all_results = {
            "test_suite": "large_suite_healing",
            "timestamp": datetime.now().isoformat(),
            "scenarios": [],
            "summary": {},
        }

        # Test different suite sizes with different failure rates
        for suite_size in self.suite_sizes:
            for failure_rate in self.failure_rates:
                print(f"\n{'=' * 80}")
                print(f"Testing: {suite_size} tests with {failure_rate*100}% failure rate")
                print(f"{'=' * 80}\n")

                result = await self.test_large_suite_healing(suite_size, failure_rate)
                all_results["scenarios"].append(result)

                self._print_scenario_summary(result)

        # Test concurrent healing
        print(f"\n{'=' * 80}")
        print("Testing: Concurrent healing of multiple test suites")
        print(f"{'=' * 80}\n")

        concurrent_result = await self.test_concurrent_healing()
        all_results["scenarios"].append(concurrent_result)
        self._print_scenario_summary(concurrent_result)

        # Test incremental healing
        print(f"\n{'=' * 80}")
        print("Testing: Incremental healing (heal as failures occur)")
        print(f"{'=' * 80}\n")

        incremental_result = await self.test_incremental_healing()
        all_results["scenarios"].append(incremental_result)
        self._print_scenario_summary(incremental_result)

        # Generate summary
        summary = self._generate_summary(all_results["scenarios"])
        all_results["summary"] = summary

        # Print final summary
        self._print_final_summary(summary)

        # Save results
        self._save_results(all_results)

        return all_results

    async def test_large_suite_healing(
        self, suite_size: int, failure_rate: float
    ) -> Dict[str, Any]:
        """Test healing on a large test suite."""
        scenario_name = f"suite_{suite_size}_failures_{int(failure_rate*100)}pct"

        # Generate test suite
        test_suite = self._generate_test_suite(suite_size)
        suite_metrics = self._calculate_suite_metrics(test_suite)

        # Inject failures
        failing_tests = self._inject_failures(test_suite, failure_rate)

        print(f"Generated suite: {suite_size} tests")
        print(f"Injected failures: {len(failing_tests)} tests")
        print(f"Failure types: {self._count_failure_types(failing_tests)}")
        print()

        # Track memory
        import psutil
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        peak_memory_mb = initial_memory_mb

        # Start healing
        start_time = time.time()
        healing_start = datetime.now()

        successful_repairs = 0
        failed_repairs = 0
        repair_times = []

        print("Starting healing process...")

        # Heal each failing test
        for i, failure in enumerate(failing_tests, 1):
            test_start = time.time()

            try:
                # Analyze failure
                analysis = await self.analyzer.analyze_failure(failure)

                # Attempt repair
                repair = await self.repairer.repair_test(analysis)

                test_time_ms = (time.time() - test_start) * 1000
                repair_times.append(test_time_ms)

                if repair and repair.test_passes:
                    successful_repairs += 1
                    status = "SUCCESS"
                else:
                    failed_repairs += 1
                    status = "FAILED"

                # Track memory
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                peak_memory_mb = max(peak_memory_mb, current_memory_mb)

                if i % 10 == 0 or i == len(failing_tests):
                    print(f"  Progress: {i}/{len(failing_tests)} - "
                          f"{status} in {test_time_ms:.0f}ms - "
                          f"Success rate: {successful_repairs/i:.1%}")

            except Exception as e:
                print(f"  Error healing test {failure.test_id}: {e}")
                failed_repairs += 1
                repair_times.append((time.time() - test_start) * 1000)

        # Calculate metrics
        total_time_ms = (time.time() - start_time) * 1000
        healing_end = datetime.now()

        avg_time_per_test = statistics.mean(repair_times) if repair_times else 0
        repair_success_rate = successful_repairs / len(failing_tests) if failing_tests else 0
        throughput = len(failing_tests) / (total_time_ms / 1000) if total_time_ms > 0 else 0
        met_target = avg_time_per_test < 10000  # <10s per test

        metrics = PerformanceMetrics(
            total_tests=suite_size,
            total_failures=len(failing_tests),
            healing_started_at=healing_start.isoformat(),
            healing_completed_at=healing_end.isoformat(),
            total_healing_time_ms=total_time_ms,
            average_healing_time_per_test_ms=avg_time_per_test,
            successful_repairs=successful_repairs,
            failed_repairs=failed_repairs,
            repair_success_rate=repair_success_rate,
            healing_throughput_tests_per_second=throughput,
            memory_usage_mb=process.memory_info().rss / 1024 / 1024,
            peak_memory_mb=peak_memory_mb,
            met_performance_target=met_target,
        )

        return {
            "scenario": scenario_name,
            "type": "large_suite",
            "suite_metrics": asdict(suite_metrics),
            "performance_metrics": asdict(metrics),
            "repair_times_ms": repair_times,
            "repair_time_percentiles": {
                "p50": statistics.median(repair_times) if repair_times else 0,
                "p90": self._percentile(repair_times, 0.90) if repair_times else 0,
                "p95": self._percentile(repair_times, 0.95) if repair_times else 0,
                "p99": self._percentile(repair_times, 0.99) if repair_times else 0,
            },
            "success": repair_success_rate >= 0.75 and met_target,
        }

    async def test_concurrent_healing(self) -> Dict[str, Any]:
        """Test concurrent healing of multiple test suites."""
        num_suites = 5
        suite_size = 50
        failure_rate = 0.2

        print(f"Healing {num_suites} suites concurrently ({suite_size} tests each)")

        start_time = time.time()

        # Create healing tasks
        tasks = [
            self.test_large_suite_healing(suite_size, failure_rate)
            for _ in range(num_suites)
        ]

        # Run concurrently
        results = await asyncio.gather(*tasks)

        total_time_ms = (time.time() - start_time) * 1000

        # Aggregate results
        total_repairs = sum(r["performance_metrics"]["successful_repairs"] for r in results)
        total_failures = sum(r["performance_metrics"]["total_failures"] for r in results)
        avg_success_rate = statistics.mean(
            r["performance_metrics"]["repair_success_rate"] for r in results
        )

        print(f"\nConcurrent healing completed in {total_time_ms/1000:.2f}s")
        print(f"Total repairs: {total_repairs}/{total_failures} ({avg_success_rate:.1%})")

        return {
            "scenario": "concurrent_healing",
            "type": "concurrent",
            "num_suites": num_suites,
            "suite_size": suite_size,
            "total_healing_time_ms": total_time_ms,
            "total_successful_repairs": total_repairs,
            "total_failures": total_failures,
            "average_success_rate": avg_success_rate,
            "individual_results": results,
            "success": avg_success_rate >= 0.75,
        }

    async def test_incremental_healing(self) -> Dict[str, Any]:
        """Test incremental healing as failures occur."""
        suite_size = 100

        print(f"Testing incremental healing on {suite_size} tests")

        # Generate test suite
        test_suite = self._generate_test_suite(suite_size)

        # Simulate incremental failures (10% fail on first run)
        first_failures = self._inject_failures(test_suite, 0.1)

        # Heal first batch
        start_time = time.time()
        first_batch_healed = await self._heal_batch(first_failures)
        first_batch_time = time.time() - start_time

        print(f"First batch: {first_batch_healed}/{len(first_failures)} healed "
              f"in {first_batch_time:.2f}s")

        # Simulate second batch of failures (5% additional)
        second_failures = self._inject_failures(test_suite, 0.05)

        start_time = time.time()
        second_batch_healed = await self._heal_batch(second_failures)
        second_batch_time = time.time() - start_time

        print(f"Second batch: {second_batch_healed}/{len(second_failures)} healed "
              f"in {second_batch_time:.2f}s")

        total_healed = first_batch_healed + second_batch_healed
        total_failures = len(first_failures) + len(second_failures)
        total_time = first_batch_time + second_batch_time

        success_rate = total_healed / total_failures if total_failures > 0 else 0

        return {
            "scenario": "incremental_healing",
            "type": "incremental",
            "suite_size": suite_size,
            "total_failures": total_failures,
            "total_healed": total_healed,
            "success_rate": success_rate,
            "total_time_seconds": total_time,
            "batches": [
                {
                    "batch": 1,
                    "failures": len(first_failures),
                    "healed": first_batch_healed,
                    "time_seconds": first_batch_time,
                },
                {
                    "batch": 2,
                    "failures": len(second_failures),
                    "healed": second_batch_healed,
                    "time_seconds": second_batch_time,
                },
            ],
            "success": success_rate >= 0.75,
        }

    async def _heal_batch(self, failures: List[TestFailure]) -> int:
        """Heal a batch of failures."""
        healed = 0

        for failure in failures:
            try:
                analysis = await self.analyzer.analyze_failure(failure)
                repair = await self.repairer.repair_test(analysis)

                if repair and repair.test_passes:
                    healed += 1
            except:
                pass

        return healed

    def _generate_test_suite(self, size: int) -> List[Dict[str, Any]]:
        """Generate a synthetic test suite."""
        tests = []

        for i in range(size):
            test = {
                "id": f"test_{i:04d}",
                "name": f"test_feature_{i}",
                "file": f"tests/test_module_{i//10}.py",
                "framework": "pytest",
                "complexity": (i % 5) + 1,  # 1-5
                "type": self._get_test_type(i),
            }
            tests.append(test)

        return tests

    def _get_test_type(self, index: int) -> str:
        """Get test type based on index."""
        types = ["unit", "integration", "functional", "e2e", "performance"]
        return types[index % len(types)]

    def _inject_failures(
        self, test_suite: List[Dict[str, Any]], failure_rate: float
    ) -> List[TestFailure]:
        """Inject failures into test suite."""
        import random

        num_failures = int(len(test_suite) * failure_rate)
        failing_tests = random.sample(test_suite, num_failures)

        failures = []
        for test in failing_tests:
            failure_type = self._random_failure_type()

            failure = TestFailure(
                test_id=test["id"],
                test_name=test["name"],
                test_file=test["file"],
                failure_type=failure_type,
                error_message=self._generate_error_message(failure_type),
                stack_trace=None,
                line_number=random.randint(10, 100),
                target_file=test["file"].replace("test_", "").replace("tests/", "src/"),
                target_function=test["name"].replace("test_", ""),
                test_framework=test["framework"],
                execution_time_ms=random.randint(50, 500),
            )
            failures.append(failure)

        return failures

    def _random_failure_type(self) -> FailureType:
        """Get random failure type."""
        import random

        types = [
            FailureType.ASSERTION_FAILED,
            FailureType.ATTRIBUTE_ERROR,
            FailureType.IMPORT_ERROR,
            FailureType.TIMEOUT,
            FailureType.TYPE_ERROR,
        ]

        weights = [0.4, 0.25, 0.15, 0.1, 0.1]  # More assertion failures
        return random.choices(types, weights=weights)[0]

    def _generate_error_message(self, failure_type: FailureType) -> str:
        """Generate error message for failure type."""
        messages = {
            FailureType.ASSERTION_FAILED: "AssertionError: assert 42 == 43",
            FailureType.ATTRIBUTE_ERROR: "AttributeError: 'NoneType' object has no attribute 'value'",
            FailureType.IMPORT_ERROR: "ImportError: cannot import name 'foo' from 'module'",
            FailureType.TIMEOUT: "TimeoutError: Test exceeded 5.0 second timeout",
            FailureType.TYPE_ERROR: "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        }
        return messages.get(failure_type, "Unknown error")

    def _calculate_suite_metrics(self, test_suite: List[Dict[str, Any]]) -> TestSuiteMetrics:
        """Calculate metrics for test suite."""
        frameworks = list(set(t["framework"] for t in test_suite))
        avg_complexity = statistics.mean(t["complexity"] for t in test_suite)

        test_types = {}
        for test in test_suite:
            test_type = test["type"]
            test_types[test_type] = test_types.get(test_type, 0) + 1

        return TestSuiteMetrics(
            suite_name="large_test_suite",
            total_tests=len(test_suite),
            test_frameworks=frameworks,
            average_test_complexity=avg_complexity,
            test_types=test_types,
            failure_distribution={},
        )

    def _count_failure_types(self, failures: List[TestFailure]) -> Dict[str, int]:
        """Count failure types."""
        counts = {}
        for failure in failures:
            failure_type = failure.failure_type.value
            counts[failure_type] = counts.get(failure_type, 0) + 1
        return counts

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _generate_summary(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of all scenarios."""
        large_suite_scenarios = [s for s in scenarios if s["type"] == "large_suite"]

        if not large_suite_scenarios:
            return {}

        avg_success_rate = statistics.mean(
            s["performance_metrics"]["repair_success_rate"] for s in large_suite_scenarios
        )

        avg_healing_time = statistics.mean(
            s["performance_metrics"]["average_healing_time_per_test_ms"]
            for s in large_suite_scenarios
        )

        met_target = all(
            s["performance_metrics"]["met_performance_target"] for s in large_suite_scenarios
        )

        met_success_rate = avg_success_rate >= 0.75

        return {
            "total_scenarios": len(scenarios),
            "large_suite_scenarios": len(large_suite_scenarios),
            "average_repair_success_rate": avg_success_rate,
            "average_healing_time_per_test_ms": avg_healing_time,
            "met_performance_target": met_target,
            "met_success_rate_target": met_success_rate,
            "overall_success": met_target and met_success_rate,
        }

    def _print_scenario_summary(self, result: Dict[str, Any]):
        """Print summary of a scenario."""
        if result["type"] == "large_suite":
            metrics = result["performance_metrics"]
            print(f"\nResults:")
            print(f"  Total healing time: {metrics['total_healing_time_ms']/1000:.2f}s")
            print(f"  Average time per test: {metrics['average_healing_time_per_test_ms']:.0f}ms")
            print(f"  Successful repairs: {metrics['successful_repairs']}/{metrics['total_failures']}")
            print(f"  Success rate: {metrics['repair_success_rate']:.1%}")
            print(f"  Throughput: {metrics['healing_throughput_tests_per_second']:.2f} tests/s")
            print(f"  Memory usage: {metrics['memory_usage_mb']:.1f}MB (peak: {metrics['peak_memory_mb']:.1f}MB)")
            print(f"  Met <10s target: {'YES' if metrics['met_performance_target'] else 'NO'}")
            print(f"  Status: {'PASS' if result['success'] else 'FAIL'}")

    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final summary."""
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}\n")

        print(f"Total scenarios tested: {summary['total_scenarios']}")
        print(f"Average repair success rate: {summary['average_repair_success_rate']:.1%}")
        print(f"Average healing time per test: {summary['average_healing_time_per_test_ms']:.0f}ms")
        print(f"Met <10s performance target: {'YES' if summary['met_performance_target'] else 'NO'}")
        print(f"Met 75%+ success rate target: {'YES' if summary['met_success_rate_target'] else 'NO'}")
        print(f"\nOverall Status: {'PASS' if summary['overall_success'] else 'FAIL'}")

    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        output_file = self.output_dir / f"large_suite_healing_{int(time.time())}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


async def main():
    """Main entry point."""
    tester = LargeSuiteHealingTester()
    results = await tester.run_all_tests()

    # Exit with appropriate code
    if results["summary"].get("overall_success", False):
        print("\n✓ All tests PASSED")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
