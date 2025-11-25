"""Test continuous healing during active development.

This script simulates continuous development where code changes constantly
introduce test failures, and validates that the self-healing system can
maintain high test pass rates in this dynamic environment.

Success Criteria:
- Maintain 80%+ test pass rate during continuous changes
- Heal failures quickly as they occur (<30s response time)
- Handle burst failures (many failures at once)
- Track healing effectiveness over time
- Generate trend reports
"""

import asyncio
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque

# Import healing components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.testing.healing.failure_analyzer import FailureAnalyzer
from src.testing.healing.test_repairer import TestRepairer
from src.testing.healing.history_tracker import HistoryTracker
from src.models.healing_models import (
    TestFailure,
    FailureType,
)


@dataclass
class DevelopmentEvent:
    """Represents a development event."""
    timestamp: datetime
    event_type: str  # "code_change", "test_failure", "test_healed"
    affected_tests: List[str]
    description: str


@dataclass
class ContinuousMetrics:
    """Metrics for continuous healing."""
    total_development_cycles: int
    total_code_changes: int
    total_test_failures: int
    total_tests_healed: int
    average_pass_rate: float
    min_pass_rate: float
    max_pass_rate: float
    average_healing_time_seconds: float
    burst_failures_handled: int
    healing_response_time_p95_seconds: float


@dataclass
class HealthSnapshot:
    """Health snapshot at a point in time."""
    timestamp: datetime
    total_tests: int
    passing_tests: int
    failing_tests: int
    pass_rate: float
    healing_in_progress: int


class ContinuousHealingTester:
    """Test continuous healing during development."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize tester."""
        self.output_dir = output_dir or Path("test_results/continuous_healing")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize healing components
        self.analyzer = FailureAnalyzer()
        self.repairer = TestRepairer()
        self.history_tracker = HistoryTracker()

        # Tracking
        self.events: List[DevelopmentEvent] = []
        self.health_snapshots: List[HealthSnapshot] = []
        self.healing_times: List[float] = []

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all continuous healing tests."""
        print("=" * 80)
        print("CONTINUOUS HEALING - VALIDATION SUITE")
        print("=" * 80)
        print("Target: 80%+ test pass rate during continuous development")
        print("Target: <30s healing response time")
        print()

        all_results = {
            "test_suite": "continuous_healing",
            "timestamp": datetime.now().isoformat(),
            "scenarios": [],
            "summary": {},
        }

        # Test normal development pace
        print(f"\n{'=' * 80}")
        print("Testing: Normal Development Pace (1 change per minute)")
        print(f"{'=' * 80}\n")

        normal_result = await self.test_normal_development_pace()
        all_results["scenarios"].append(normal_result)
        self._print_scenario_summary(normal_result)

        # Test rapid development
        print(f"\n{'=' * 80}")
        print("Testing: Rapid Development (multiple changes per minute)")
        print(f"{'=' * 80}\n")

        rapid_result = await self.test_rapid_development()
        all_results["scenarios"].append(rapid_result)
        self._print_scenario_summary(rapid_result)

        # Test burst failures
        print(f"\n{'=' * 80}")
        print("Testing: Burst Failures (sudden spike in failures)")
        print(f"{'=' * 80}\n")

        burst_result = await self.test_burst_failures()
        all_results["scenarios"].append(burst_result)
        self._print_scenario_summary(burst_result)

        # Test long-running session
        print(f"\n{'=' * 80}")
        print("Testing: Long-Running Development Session")
        print(f"{'=' * 80}\n")

        longrun_result = await self.test_long_running_session()
        all_results["scenarios"].append(longrun_result)
        self._print_scenario_summary(longrun_result)

        # Test healing under load
        print(f"\n{'=' * 80}")
        print("Testing: Healing Under Load (high concurrency)")
        print(f"{'=' * 80}\n")

        load_result = await self.test_healing_under_load()
        all_results["scenarios"].append(load_result)
        self._print_scenario_summary(load_result)

        # Generate summary
        summary = self._generate_summary(all_results["scenarios"])
        all_results["summary"] = summary

        # Print final summary
        self._print_final_summary(summary)

        # Save results
        self._save_results(all_results)

        # Generate trend report
        self._generate_trend_report(all_results)

        return all_results

    async def test_normal_development_pace(self) -> Dict[str, Any]:
        """Test healing at normal development pace."""
        total_tests = 100
        development_cycles = 10
        changes_per_cycle = 1
        failure_rate_per_change = 0.15  # 15% of tests fail per change

        print(f"Simulating {development_cycles} development cycles")
        print(f"Test suite: {total_tests} tests")
        print(f"Expected failures per cycle: ~{int(total_tests * failure_rate_per_change)}")
        print()

        start_time = time.time()
        self._reset_tracking()

        # Initialize test suite
        passing_tests = set(f"test_{i:03d}" for i in range(total_tests))
        failing_tests = set()

        print("Development cycle:")

        for cycle in range(1, development_cycles + 1):
            cycle_start = time.time()

            # Simulate code change
            change_event = DevelopmentEvent(
                timestamp=datetime.now(),
                event_type="code_change",
                affected_tests=[],
                description=f"Developer makes change {cycle}",
            )
            self.events.append(change_event)

            # Some tests fail due to change
            num_failures = int(total_tests * failure_rate_per_change)
            newly_failing = set(random.sample(list(passing_tests), min(num_failures, len(passing_tests))))

            passing_tests -= newly_failing
            failing_tests |= newly_failing

            # Record snapshot
            self._record_snapshot(len(passing_tests) + len(failing_tests), len(passing_tests), len(failing_tests))

            print(f"  Cycle {cycle}: {len(newly_failing)} tests failed")

            # Heal failures
            healed = await self._heal_failures(list(failing_tests))

            healing_time = time.time() - cycle_start
            self.healing_times.append(healing_time)

            # Update test states
            healed_tests = set(random.sample(list(failing_tests), healed))
            failing_tests -= healed_tests
            passing_tests |= healed_tests

            # Record snapshot after healing
            self._record_snapshot(len(passing_tests) + len(failing_tests), len(passing_tests), len(failing_tests))

            print(f"           Healed {healed}/{len(newly_failing)} in {healing_time:.1f}s")
            print(f"           Pass rate: {len(passing_tests)/total_tests:.1%}")

            # Simulate time between changes
            await asyncio.sleep(0.1)

        total_time_ms = (time.time() - start_time) * 1000

        # Calculate metrics
        metrics = self._calculate_metrics(development_cycles)

        return {
            "scenario": "normal_development_pace",
            "type": "normal",
            "development_cycles": development_cycles,
            "total_tests": total_tests,
            "metrics": asdict(metrics),
            "health_snapshots": [asdict(s) for s in self.health_snapshots[-5:]],  # Last 5
            "total_time_ms": total_time_ms,
            "success": metrics.average_pass_rate >= 0.80 and metrics.average_healing_time_seconds < 30,
        }

    async def test_rapid_development(self) -> Dict[str, Any]:
        """Test healing during rapid development."""
        total_tests = 100
        development_cycles = 20  # More cycles
        changes_per_minute = 5  # Faster pace

        print(f"Simulating {development_cycles} rapid development cycles")
        print(f"Multiple changes per minute")
        print()

        start_time = time.time()
        self._reset_tracking()

        passing_tests = set(f"test_{i:03d}" for i in range(total_tests))
        failing_tests = set()

        for cycle in range(1, development_cycles + 1):
            # Multiple changes in quick succession
            for change in range(random.randint(1, 3)):
                # Introduce failures
                num_failures = random.randint(5, 15)
                if passing_tests:
                    newly_failing = set(random.sample(list(passing_tests), min(num_failures, len(passing_tests))))
                    passing_tests -= newly_failing
                    failing_tests |= newly_failing

            self._record_snapshot(total_tests, len(passing_tests), len(failing_tests))

            # Heal in batches
            if failing_tests:
                healed_count = await self._heal_failures(list(failing_tests))
                healed_tests = set(random.sample(list(failing_tests), min(healed_count, len(failing_tests))))
                failing_tests -= healed_tests
                passing_tests |= healed_tests

            if cycle % 5 == 0:
                print(f"  Cycle {cycle}: Pass rate {len(passing_tests)/total_tests:.1%}, "
                      f"{len(failing_tests)} failing")

            await asyncio.sleep(0.05)  # Faster pace

        total_time_ms = (time.time() - start_time) * 1000
        metrics = self._calculate_metrics(development_cycles)

        return {
            "scenario": "rapid_development",
            "type": "rapid",
            "development_cycles": development_cycles,
            "changes_per_minute": changes_per_minute,
            "total_tests": total_tests,
            "metrics": asdict(metrics),
            "total_time_ms": total_time_ms,
            "success": metrics.average_pass_rate >= 0.75,  # Slightly lower for rapid
        }

    async def test_burst_failures(self) -> Dict[str, Any]:
        """Test handling sudden burst of failures."""
        total_tests = 100

        print("Simulating sudden burst of failures")
        print()

        self._reset_tracking()

        # Start healthy
        passing_tests = set(f"test_{i:03d}" for i in range(total_tests))
        failing_tests = set()

        self._record_snapshot(total_tests, len(passing_tests), 0)
        print(f"Initial state: {len(passing_tests)} tests passing")

        # Sudden burst: 50% of tests fail at once
        print("\nBurst event: Major breaking change!")
        burst_failures = 50
        newly_failing = set(random.sample(list(passing_tests), burst_failures))

        passing_tests -= newly_failing
        failing_tests |= newly_failing

        self._record_snapshot(total_tests, len(passing_tests), len(failing_tests))
        print(f"After burst: {len(failing_tests)} tests failing")

        # Heal burst
        print("\nHealing burst failures...")
        start_time = time.time()

        healed_count = await self._heal_failures(list(failing_tests))

        healing_time = time.time() - start_time

        healed_tests = set(random.sample(list(failing_tests), min(healed_count, len(failing_tests))))
        failing_tests -= healed_tests
        passing_tests |= healed_tests

        self._record_snapshot(total_tests, len(passing_tests), len(failing_tests))

        print(f"Healed {healed_count}/{burst_failures} in {healing_time:.1f}s")
        print(f"Final pass rate: {len(passing_tests)/total_tests:.1%}")

        heal_rate = healed_count / burst_failures
        response_time_ok = healing_time < 60  # Allow 60s for burst

        return {
            "scenario": "burst_failures",
            "type": "burst",
            "burst_size": burst_failures,
            "total_tests": total_tests,
            "healed": healed_count,
            "heal_rate": heal_rate,
            "healing_time_seconds": healing_time,
            "final_pass_rate": len(passing_tests) / total_tests,
            "success": heal_rate >= 0.75 and response_time_ok,
        }

    async def test_long_running_session(self) -> Dict[str, Any]:
        """Test long-running development session."""
        total_tests = 150
        session_duration_cycles = 50  # Longer session

        print(f"Simulating {session_duration_cycles}-cycle development session")
        print(f"Test suite: {total_tests} tests")
        print()

        start_time = time.time()
        self._reset_tracking()

        passing_tests = set(f"test_{i:03d}" for i in range(total_tests))
        failing_tests = set()

        pass_rate_history = []

        for cycle in range(1, session_duration_cycles + 1):
            # Vary failure rate over time (some cycles worse than others)
            failure_rate = 0.10 + 0.05 * random.random()  # 10-15%

            # Introduce failures
            num_failures = int(total_tests * failure_rate)
            if passing_tests:
                newly_failing = set(random.sample(list(passing_tests), min(num_failures, len(passing_tests))))
                passing_tests -= newly_failing
                failing_tests |= newly_failing

            # Heal
            if failing_tests:
                healed_count = await self._heal_failures(list(failing_tests))
                healed_tests = set(random.sample(list(failing_tests), min(healed_count, len(failing_tests))))
                failing_tests -= healed_tests
                passing_tests |= healed_tests

            # Track pass rate
            pass_rate = len(passing_tests) / total_tests
            pass_rate_history.append(pass_rate)
            self._record_snapshot(total_tests, len(passing_tests), len(failing_tests))

            if cycle % 10 == 0:
                import statistics
                recent_avg = statistics.mean(pass_rate_history[-10:])
                print(f"  Cycle {cycle}: Pass rate {pass_rate:.1%} "
                      f"(10-cycle avg: {recent_avg:.1%})")

            await asyncio.sleep(0.05)

        total_time_ms = (time.time() - start_time) * 1000

        # Calculate trend
        import statistics
        avg_pass_rate = statistics.mean(pass_rate_history)
        min_pass_rate = min(pass_rate_history)
        stability = statistics.stdev(pass_rate_history) if len(pass_rate_history) > 1 else 0

        print(f"\nSession summary:")
        print(f"  Average pass rate: {avg_pass_rate:.1%}")
        print(f"  Minimum pass rate: {min_pass_rate:.1%}")
        print(f"  Stability (lower is better): {stability:.3f}")

        return {
            "scenario": "long_running_session",
            "type": "longrun",
            "session_cycles": session_duration_cycles,
            "total_tests": total_tests,
            "average_pass_rate": avg_pass_rate,
            "min_pass_rate": min_pass_rate,
            "max_pass_rate": max(pass_rate_history),
            "stability": stability,
            "pass_rate_trend": pass_rate_history,
            "total_time_ms": total_time_ms,
            "success": avg_pass_rate >= 0.80 and min_pass_rate >= 0.70,
        }

    async def test_healing_under_load(self) -> Dict[str, Any]:
        """Test healing under high concurrent load."""
        total_tests = 200
        concurrent_changes = 5  # Multiple developers

        print(f"Simulating {concurrent_changes} concurrent development streams")
        print()

        start_time = time.time()

        # Simulate concurrent development
        tasks = []
        for dev_id in range(concurrent_changes):
            task = self._simulate_developer(dev_id, total_tests // concurrent_changes, 10)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        total_time_ms = (time.time() - start_time) * 1000

        # Aggregate results
        total_failures = sum(r["failures"] for r in results)
        total_healed = sum(r["healed"] for r in results)
        heal_rate = total_healed / total_failures if total_failures > 0 else 0

        print(f"\nConcurrent development results:")
        print(f"  Total failures: {total_failures}")
        print(f"  Total healed: {total_healed}")
        print(f"  Heal rate: {heal_rate:.1%}")
        print(f"  Total time: {total_time_ms/1000:.1f}s")

        return {
            "scenario": "healing_under_load",
            "type": "load",
            "concurrent_streams": concurrent_changes,
            "total_tests": total_tests,
            "total_failures": total_failures,
            "total_healed": total_healed,
            "heal_rate": heal_rate,
            "total_time_ms": total_time_ms,
            "success": heal_rate >= 0.75,
        }

    async def _simulate_developer(
        self, dev_id: int, tests_per_dev: int, cycles: int
    ) -> Dict[str, Any]:
        """Simulate a developer making changes."""
        failures = 0
        healed = 0

        for cycle in range(cycles):
            # Introduce failures
            num_failures = random.randint(2, 5)
            failures += num_failures

            # Generate failures
            test_failures = []
            for i in range(num_failures):
                failure = TestFailure(
                    test_id=f"dev{dev_id}_test_{i}",
                    test_name=f"test_feature_{i}",
                    test_file=f"tests/dev{dev_id}/test_{i}.py",
                    failure_type=FailureType.ASSERTION_FAILED,
                    error_message="AssertionError: test failed",
                    test_framework="pytest",
                    target_file=f"src/module_{i}.py",
                    target_function="function",
                    execution_time_ms=100,
                )
                test_failures.append(failure)

            # Heal
            healed_count = await self._heal_failures(test_failures)
            healed += healed_count

            await asyncio.sleep(0.05)

        return {"dev_id": dev_id, "failures": failures, "healed": healed}

    async def _heal_failures(self, failures: List[Any]) -> int:
        """Heal a list of failures."""
        healed = 0

        # Convert to TestFailure if needed
        test_failures = []
        for f in failures:
            if isinstance(f, str):
                # Create failure from test ID
                failure = TestFailure(
                    test_id=f,
                    test_name=f,
                    test_file=f"tests/{f}.py",
                    failure_type=FailureType.ASSERTION_FAILED,
                    error_message="Test failed",
                    test_framework="pytest",
                    target_file="src/module.py",
                    target_function="function",
                    execution_time_ms=100,
                )
                test_failures.append(failure)
            else:
                test_failures.append(f)

        for failure in test_failures:
            try:
                analysis = await self.analyzer.analyze_failure(failure)
                repair = await self.repairer.repair_test(analysis, max_attempts=1)

                if repair and repair.test_passes:
                    healed += 1
            except:
                pass

        return healed

    def _reset_tracking(self):
        """Reset tracking state."""
        self.events = []
        self.health_snapshots = []
        self.healing_times = []

    def _record_snapshot(self, total: int, passing: int, failing: int):
        """Record health snapshot."""
        snapshot = HealthSnapshot(
            timestamp=datetime.now(),
            total_tests=total,
            passing_tests=passing,
            failing_tests=failing,
            pass_rate=passing / total if total > 0 else 0,
            healing_in_progress=0,
        )
        self.health_snapshots.append(snapshot)

    def _calculate_metrics(self, cycles: int) -> ContinuousMetrics:
        """Calculate continuous metrics."""
        import statistics

        pass_rates = [s.pass_rate for s in self.health_snapshots]
        avg_pass_rate = statistics.mean(pass_rates) if pass_rates else 0
        min_pass_rate = min(pass_rates) if pass_rates else 0
        max_pass_rate = max(pass_rates) if pass_rates else 0

        avg_healing_time = statistics.mean(self.healing_times) if self.healing_times else 0

        healing_times_sorted = sorted(self.healing_times)
        p95_index = int(len(healing_times_sorted) * 0.95)
        p95_healing_time = healing_times_sorted[p95_index] if healing_times_sorted else 0

        return ContinuousMetrics(
            total_development_cycles=cycles,
            total_code_changes=cycles,
            total_test_failures=len(self.events),
            total_tests_healed=sum(1 for e in self.events if e.event_type == "test_healed"),
            average_pass_rate=avg_pass_rate,
            min_pass_rate=min_pass_rate,
            max_pass_rate=max_pass_rate,
            average_healing_time_seconds=avg_healing_time,
            burst_failures_handled=0,
            healing_response_time_p95_seconds=p95_healing_time,
        )

    def _generate_summary(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary."""
        all_passed = all(s.get("success", False) for s in scenarios)

        # Extract pass rates
        pass_rates = []
        for s in scenarios:
            if "metrics" in s:
                pass_rates.append(s["metrics"]["average_pass_rate"])
            elif "average_pass_rate" in s:
                pass_rates.append(s["average_pass_rate"])
            elif "final_pass_rate" in s:
                pass_rates.append(s["final_pass_rate"])

        import statistics
        overall_avg_pass_rate = statistics.mean(pass_rates) if pass_rates else 0

        return {
            "total_scenarios": len(scenarios),
            "all_scenarios_passed": all_passed,
            "overall_average_pass_rate": overall_avg_pass_rate,
            "overall_success": all_passed and overall_avg_pass_rate >= 0.80,
        }

    def _print_scenario_summary(self, result: Dict[str, Any]):
        """Print scenario summary."""
        print(f"\nScenario: {result['scenario']}")

        if "metrics" in result:
            metrics = result["metrics"]
            print(f"  Average pass rate: {metrics['average_pass_rate']:.1%}")
            print(f"  Min pass rate: {metrics['min_pass_rate']:.1%}")
            print(f"  Avg healing time: {metrics['average_healing_time_seconds']:.1f}s")
        elif "heal_rate" in result:
            print(f"  Heal rate: {result['heal_rate']:.1%}")
            print(f"  Healing time: {result.get('healing_time_seconds', 0):.1f}s")

        print(f"  Status: {'PASS' if result.get('success', False) else 'FAIL'}")

    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final summary."""
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}\n")

        print(f"Total scenarios: {summary['total_scenarios']}")
        print(f"Overall average pass rate: {summary['overall_average_pass_rate']:.1%}")
        print(f"All scenarios passed: {'YES' if summary['all_scenarios_passed'] else 'NO'}")
        print(f"\nOverall Status: {'PASS' if summary['overall_success'] else 'FAIL'}")

    def _save_results(self, results: Dict[str, Any]):
        """Save results."""
        output_file = self.output_dir / f"continuous_healing_{int(time.time())}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

    def _generate_trend_report(self, results: Dict[str, Any]):
        """Generate trend report."""
        report_file = self.output_dir / f"trend_report_{int(time.time())}.txt"

        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("CONTINUOUS HEALING - TREND REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("Test Suite Health Over Time\n")
            f.write("-" * 40 + "\n\n")

            for snapshot in self.health_snapshots[-20:]:  # Last 20 snapshots
                f.write(f"{snapshot.timestamp.strftime('%H:%M:%S')}: "
                       f"Pass rate {snapshot.pass_rate:.1%} "
                       f"({snapshot.passing_tests}/{snapshot.total_tests})\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Trend report saved to: {report_file}")


async def main():
    """Main entry point."""
    tester = ContinuousHealingTester()
    results = await tester.run_all_tests()

    if results["summary"].get("overall_success", False):
        print("\n✓ All tests PASSED")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
