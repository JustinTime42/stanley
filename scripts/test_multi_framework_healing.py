"""Test healing across multiple test frameworks (pytest, jest, junit).

This script validates that the self-healing system can correctly handle
tests from different frameworks with framework-specific error patterns,
syntax, and repair strategies.

Success Criteria:
- Support pytest, jest, and junit frameworks
- Correctly parse framework-specific errors
- Apply framework-appropriate repairs
- Maintain framework conventions
- 75%+ repair success rate per framework
"""

import asyncio
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Import healing components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.testing.healing.failure_analyzer import FailureAnalyzer
from src.testing.healing.test_repairer import TestRepairer
from src.models.healing_models import (
    TestFailure,
    FailureType,
    RepairStrategy,
)


class TestFramework(str, Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    JEST = "jest"
    JUNIT = "junit"


@dataclass
class FrameworkMetrics:
    """Metrics per framework."""
    framework: str
    total_tests: int
    total_failures: int
    successful_repairs: int
    failed_repairs: int
    repair_success_rate: float
    average_repair_time_ms: float
    failure_types_handled: List[str]
    framework_specific_patterns: List[str]


class MultiFrameworkHealingTester:
    """Test healing across multiple frameworks."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize tester."""
        self.output_dir = output_dir or Path("test_results/multi_framework_healing")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize healing components
        self.analyzer = FailureAnalyzer()
        self.repairer = TestRepairer()

        # Framework-specific configurations
        self.framework_configs = {
            TestFramework.PYTEST: {
                "file_extension": ".py",
                "test_prefix": "test_",
                "assert_keyword": "assert",
                "import_style": "import",
            },
            TestFramework.JEST: {
                "file_extension": ".test.js",
                "test_prefix": "test(",
                "assert_keyword": "expect(",
                "import_style": "import",
            },
            TestFramework.JUNIT: {
                "file_extension": ".java",
                "test_prefix": "@Test",
                "assert_keyword": "assertEquals",
                "import_style": "import",
            },
        }

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all multi-framework healing tests."""
        print("=" * 80)
        print("MULTI-FRAMEWORK HEALING - VALIDATION SUITE")
        print("=" * 80)
        print("Frameworks: pytest, jest, junit")
        print("Target: 75%+ repair success rate per framework")
        print()

        all_results = {
            "test_suite": "multi_framework_healing",
            "timestamp": datetime.now().isoformat(),
            "frameworks_tested": [f.value for f in TestFramework],
            "scenarios": [],
            "framework_metrics": {},
            "summary": {},
        }

        # Test each framework
        for framework in TestFramework:
            print(f"\n{'=' * 80}")
            print(f"Testing Framework: {framework.value.upper()}")
            print(f"{'=' * 80}\n")

            result = await self.test_framework_healing(framework)
            all_results["scenarios"].append(result)
            all_results["framework_metrics"][framework.value] = result["metrics"]

            self._print_framework_summary(result)

        # Test cross-framework scenarios
        print(f"\n{'=' * 80}")
        print("Testing: Cross-Framework Healing (mixed test suite)")
        print(f"{'=' * 80}\n")

        cross_result = await self.test_cross_framework_healing()
        all_results["scenarios"].append(cross_result)
        self._print_cross_framework_summary(cross_result)

        # Test framework-specific patterns
        print(f"\n{'=' * 80}")
        print("Testing: Framework-Specific Error Patterns")
        print(f"{'=' * 80}\n")

        patterns_result = await self.test_framework_specific_patterns()
        all_results["scenarios"].append(patterns_result)

        # Generate summary
        summary = self._generate_summary(all_results)
        all_results["summary"] = summary

        # Print final summary
        self._print_final_summary(summary)

        # Save results
        self._save_results(all_results)

        return all_results

    async def test_framework_healing(self, framework: TestFramework) -> Dict[str, Any]:
        """Test healing for a specific framework."""
        # Generate framework-specific test failures
        failures = self._generate_framework_failures(framework, count=30)

        print(f"Generated {len(failures)} {framework.value} test failures")
        print(f"Failure types: {self._count_failure_types(failures)}")
        print()

        start_time = time.time()
        successful_repairs = 0
        failed_repairs = 0
        repair_times = []
        handled_patterns = set()

        print("Healing tests...")

        for i, failure in enumerate(failures, 1):
            test_start = time.time()

            try:
                # Analyze failure with framework context
                analysis = await self.analyzer.analyze_failure(failure)

                # Track framework-specific patterns
                pattern = self._identify_framework_pattern(failure, framework)
                if pattern:
                    handled_patterns.add(pattern)

                # Attempt repair
                repair = await self.repairer.repair_test(analysis)

                test_time_ms = (time.time() - test_start) * 1000
                repair_times.append(test_time_ms)

                if repair and self._validate_framework_repair(repair, framework):
                    successful_repairs += 1
                    status = "SUCCESS"
                else:
                    failed_repairs += 1
                    status = "FAILED"

                if i % 5 == 0:
                    print(f"  Progress: {i}/{len(failures)} - {status} - "
                          f"Success: {successful_repairs/i:.1%}")

            except Exception as e:
                print(f"  Error: {e}")
                failed_repairs += 1
                repair_times.append((time.time() - test_start) * 1000)

        total_time_ms = (time.time() - start_time) * 1000

        # Calculate metrics
        import statistics
        avg_repair_time = statistics.mean(repair_times) if repair_times else 0
        success_rate = successful_repairs / len(failures) if failures else 0

        metrics = FrameworkMetrics(
            framework=framework.value,
            total_tests=len(failures),
            total_failures=len(failures),
            successful_repairs=successful_repairs,
            failed_repairs=failed_repairs,
            repair_success_rate=success_rate,
            average_repair_time_ms=avg_repair_time,
            failure_types_handled=list(self._count_failure_types(failures).keys()),
            framework_specific_patterns=list(handled_patterns),
        )

        return {
            "scenario": f"{framework.value}_healing",
            "framework": framework.value,
            "type": "single_framework",
            "metrics": asdict(metrics),
            "total_time_ms": total_time_ms,
            "success": success_rate >= 0.75,
        }

    async def test_cross_framework_healing(self) -> Dict[str, Any]:
        """Test healing on mixed framework test suite."""
        # Generate mixed failures
        all_failures = []
        for framework in TestFramework:
            failures = self._generate_framework_failures(framework, count=15)
            all_failures.extend(failures)

        # Shuffle to simulate real mixed suite
        import random
        random.shuffle(all_failures)

        print(f"Generated mixed suite: {len(all_failures)} tests across 3 frameworks")
        print(f"Distribution: {self._count_frameworks(all_failures)}")
        print()

        start_time = time.time()
        successful_repairs = 0
        framework_results = {f.value: {"success": 0, "total": 0} for f in TestFramework}

        print("Healing mixed suite...")

        for i, failure in enumerate(all_failures, 1):
            framework = failure.test_framework

            try:
                analysis = await self.analyzer.analyze_failure(failure)
                repair = await self.repairer.repair_test(analysis)

                framework_results[framework]["total"] += 1

                if repair and repair.test_passes:
                    successful_repairs += 1
                    framework_results[framework]["success"] += 1

                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(all_failures)} - "
                          f"Success: {successful_repairs/i:.1%}")

            except:
                framework_results[framework]["total"] += 1

        total_time_ms = (time.time() - start_time) * 1000

        # Calculate per-framework success rates
        framework_success_rates = {}
        for framework, results in framework_results.items():
            if results["total"] > 0:
                framework_success_rates[framework] = results["success"] / results["total"]
            else:
                framework_success_rates[framework] = 0.0

        overall_success_rate = successful_repairs / len(all_failures) if all_failures else 0

        return {
            "scenario": "cross_framework_healing",
            "type": "cross_framework",
            "total_tests": len(all_failures),
            "successful_repairs": successful_repairs,
            "overall_success_rate": overall_success_rate,
            "framework_success_rates": framework_success_rates,
            "total_time_ms": total_time_ms,
            "success": overall_success_rate >= 0.75 and all(
                rate >= 0.7 for rate in framework_success_rates.values()
            ),
        }

    async def test_framework_specific_patterns(self) -> Dict[str, Any]:
        """Test handling of framework-specific error patterns."""
        patterns_tested = {}

        # Pytest-specific patterns
        pytest_patterns = await self._test_pytest_patterns()
        patterns_tested["pytest"] = pytest_patterns

        # Jest-specific patterns
        jest_patterns = await self._test_jest_patterns()
        patterns_tested["jest"] = jest_patterns

        # JUnit-specific patterns
        junit_patterns = await self._test_junit_patterns()
        patterns_tested["junit"] = junit_patterns

        # Calculate overall success
        all_success_rates = []
        for framework, patterns in patterns_tested.items():
            for pattern_result in patterns:
                all_success_rates.append(pattern_result["success_rate"])

        import statistics
        overall_success = statistics.mean(all_success_rates) if all_success_rates else 0

        return {
            "scenario": "framework_specific_patterns",
            "type": "patterns",
            "patterns_by_framework": patterns_tested,
            "overall_pattern_handling": overall_success,
            "success": overall_success >= 0.75,
        }

    async def _test_pytest_patterns(self) -> List[Dict[str, Any]]:
        """Test pytest-specific patterns."""
        patterns = [
            {
                "name": "assert_rewrite",
                "error": "AssertionError: assert response.status_code == 200",
                "pattern_type": "assertion_introspection",
            },
            {
                "name": "fixture_error",
                "error": "fixture 'db_session' not found",
                "pattern_type": "fixture_missing",
            },
            {
                "name": "parametrize_error",
                "error": "ValueError in test_with_params[case1]",
                "pattern_type": "parametrized_test",
            },
        ]

        results = []
        for pattern in patterns:
            # Test healing this pattern
            failures = [self._create_pytest_failure(pattern["error"]) for _ in range(5)]

            healed = 0
            for failure in failures:
                try:
                    analysis = await self.analyzer.analyze_failure(failure)
                    repair = await self.repairer.repair_test(analysis)
                    if repair and repair.test_passes:
                        healed += 1
                except:
                    pass

            results.append({
                "pattern": pattern["name"],
                "pattern_type": pattern["pattern_type"],
                "tests": len(failures),
                "healed": healed,
                "success_rate": healed / len(failures) if failures else 0,
            })

        return results

    async def _test_jest_patterns(self) -> List[Dict[str, Any]]:
        """Test jest-specific patterns."""
        patterns = [
            {
                "name": "expect_matcher",
                "error": "expect(received).toBe(expected)\n\nExpected: 42\nReceived: 43",
                "pattern_type": "jest_matcher",
            },
            {
                "name": "async_timeout",
                "error": "Timeout - Async callback was not invoked within 5000ms",
                "pattern_type": "async_timeout",
            },
            {
                "name": "mock_clear",
                "error": "Expected mock function to be called but it was not",
                "pattern_type": "mock_assertion",
            },
        ]

        results = []
        for pattern in patterns:
            failures = [self._create_jest_failure(pattern["error"]) for _ in range(5)]

            healed = 0
            for failure in failures:
                try:
                    analysis = await self.analyzer.analyze_failure(failure)
                    repair = await self.repairer.repair_test(analysis)
                    if repair and repair.test_passes:
                        healed += 1
                except:
                    pass

            results.append({
                "pattern": pattern["name"],
                "pattern_type": pattern["pattern_type"],
                "tests": len(failures),
                "healed": healed,
                "success_rate": healed / len(failures) if failures else 0,
            })

        return results

    async def _test_junit_patterns(self) -> List[Dict[str, Any]]:
        """Test junit-specific patterns."""
        patterns = [
            {
                "name": "assertEquals_fail",
                "error": "org.junit.ComparisonFailure: expected:<42> but was:<43>",
                "pattern_type": "junit_assertion",
            },
            {
                "name": "null_pointer",
                "error": "java.lang.NullPointerException at TestClass.testMethod(TestClass.java:42)",
                "pattern_type": "null_pointer",
            },
            {
                "name": "annotation_missing",
                "error": "Method testExample() should be marked with @Test",
                "pattern_type": "annotation",
            },
        ]

        results = []
        for pattern in patterns:
            failures = [self._create_junit_failure(pattern["error"]) for _ in range(5)]

            healed = 0
            for failure in failures:
                try:
                    analysis = await self.analyzer.analyze_failure(failure)
                    repair = await self.repairer.repair_test(analysis)
                    if repair and repair.test_passes:
                        healed += 1
                except:
                    pass

            results.append({
                "pattern": pattern["name"],
                "pattern_type": pattern["pattern_type"],
                "tests": len(failures),
                "healed": healed,
                "success_rate": healed / len(failures) if failures else 0,
            })

        return results

    def _generate_framework_failures(
        self, framework: TestFramework, count: int
    ) -> List[TestFailure]:
        """Generate framework-specific test failures."""
        import random

        failures = []
        failure_types = [
            FailureType.ASSERTION_FAILED,
            FailureType.ATTRIBUTE_ERROR,
            FailureType.IMPORT_ERROR,
            FailureType.TIMEOUT,
            FailureType.TYPE_ERROR,
        ]

        config = self.framework_configs[framework]

        for i in range(count):
            failure_type = random.choice(failure_types)

            failure = TestFailure(
                test_id=f"{framework.value}_test_{i:03d}",
                test_name=f"{config['test_prefix']}feature_{i}",
                test_file=f"tests/test_module_{i//10}{config['file_extension']}",
                failure_type=failure_type,
                error_message=self._generate_framework_error(framework, failure_type),
                stack_trace=self._generate_framework_stack_trace(framework),
                line_number=random.randint(10, 100),
                target_file=f"src/module_{i//10}{config['file_extension']}",
                target_function=f"feature_{i}",
                test_framework=framework.value,
                execution_time_ms=random.randint(50, 500),
            )
            failures.append(failure)

        return failures

    def _generate_framework_error(
        self, framework: TestFramework, failure_type: FailureType
    ) -> str:
        """Generate framework-specific error message."""
        if framework == TestFramework.PYTEST:
            if failure_type == FailureType.ASSERTION_FAILED:
                return "AssertionError: assert 42 == 43\n +  where 42 = obj.value"
            elif failure_type == FailureType.ATTRIBUTE_ERROR:
                return "AttributeError: 'NoneType' object has no attribute 'value'"
            elif failure_type == FailureType.IMPORT_ERROR:
                return "ImportError: cannot import name 'foo' from 'module'"

        elif framework == TestFramework.JEST:
            if failure_type == FailureType.ASSERTION_FAILED:
                return "expect(received).toBe(expected)\n\nExpected: 42\nReceived: 43"
            elif failure_type == FailureType.ATTRIBUTE_ERROR:
                return "TypeError: Cannot read property 'value' of undefined"
            elif failure_type == FailureType.TIMEOUT:
                return "Timeout - Async callback was not invoked within 5000ms"

        elif framework == TestFramework.JUNIT:
            if failure_type == FailureType.ASSERTION_FAILED:
                return "org.junit.ComparisonFailure: expected:<42> but was:<43>"
            elif failure_type == FailureType.ATTRIBUTE_ERROR:
                return "java.lang.NullPointerException at TestClass.testMethod"
            elif failure_type == FailureType.TYPE_ERROR:
                return "java.lang.ClassCastException: Cannot cast Integer to String"

        return "Generic error message"

    def _generate_framework_stack_trace(self, framework: TestFramework) -> str:
        """Generate framework-specific stack trace."""
        if framework == TestFramework.PYTEST:
            return "test_module.py:42: in test_function\n    assert result == expected"
        elif framework == TestFramework.JEST:
            return "at Object.<anonymous> (test_module.test.js:42:5)"
        elif framework == TestFramework.JUNIT:
            return "at com.example.TestClass.testMethod(TestClass.java:42)"
        return ""

    def _create_pytest_failure(self, error: str) -> TestFailure:
        """Create a pytest failure."""
        return TestFailure(
            test_id="pytest_test_001",
            test_name="test_feature",
            test_file="tests/test_module.py",
            failure_type=FailureType.ASSERTION_FAILED,
            error_message=error,
            test_framework="pytest",
            target_file="src/module.py",
            target_function="feature",
            execution_time_ms=100,
        )

    def _create_jest_failure(self, error: str) -> TestFailure:
        """Create a jest failure."""
        return TestFailure(
            test_id="jest_test_001",
            test_name="test('feature')",
            test_file="tests/module.test.js",
            failure_type=FailureType.ASSERTION_FAILED,
            error_message=error,
            test_framework="jest",
            target_file="src/module.js",
            target_function="feature",
            execution_time_ms=100,
        )

    def _create_junit_failure(self, error: str) -> TestFailure:
        """Create a junit failure."""
        return TestFailure(
            test_id="junit_test_001",
            test_name="testFeature",
            test_file="src/test/java/TestClass.java",
            failure_type=FailureType.ASSERTION_FAILED,
            error_message=error,
            test_framework="junit",
            target_file="src/main/java/Module.java",
            target_function="feature",
            execution_time_ms=100,
        )

    def _identify_framework_pattern(
        self, failure: TestFailure, framework: TestFramework
    ) -> Optional[str]:
        """Identify framework-specific pattern."""
        error = failure.error_message.lower()

        if framework == TestFramework.PYTEST:
            if "fixture" in error:
                return "pytest_fixture"
            elif "assert" in error and "where" in error:
                return "pytest_assert_introspection"
            elif "parametrize" in error:
                return "pytest_parametrize"

        elif framework == TestFramework.JEST:
            if "expect(" in error and ".tobe(" in error:
                return "jest_matcher"
            elif "timeout" in error and "async" in error:
                return "jest_async_timeout"
            elif "mock" in error:
                return "jest_mock"

        elif framework == TestFramework.JUNIT:
            if "comparisonfailure" in error:
                return "junit_comparison"
            elif "nullpointerexception" in error:
                return "junit_null_pointer"
            elif "@test" in error:
                return "junit_annotation"

        return None

    def _validate_framework_repair(
        self, repair: Any, framework: TestFramework
    ) -> bool:
        """Validate repair maintains framework conventions."""
        # Check if repair code follows framework patterns
        # This is simplified - real implementation would do AST analysis

        if not repair.repaired_code:
            return False

        code = repair.repaired_code.lower()
        config = self.framework_configs[framework]

        # Check for framework-specific patterns
        if framework == TestFramework.PYTEST:
            return "assert" in code or "def test_" in code

        elif framework == TestFramework.JEST:
            return "expect(" in code or "test(" in code

        elif framework == TestFramework.JUNIT:
            return "@test" in code or "assertequals" in code

        return True

    def _count_failure_types(self, failures: List[TestFailure]) -> Dict[str, int]:
        """Count failure types."""
        counts = {}
        for failure in failures:
            failure_type = failure.failure_type.value
            counts[failure_type] = counts.get(failure_type, 0) + 1
        return counts

    def _count_frameworks(self, failures: List[TestFailure]) -> Dict[str, int]:
        """Count frameworks."""
        counts = {}
        for failure in failures:
            framework = failure.test_framework
            counts[framework] = counts.get(framework, 0) + 1
        return counts

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary."""
        framework_metrics = results["framework_metrics"]

        avg_success_rate = sum(
            m["repair_success_rate"] for m in framework_metrics.values()
        ) / len(framework_metrics) if framework_metrics else 0

        all_meet_target = all(
            m["repair_success_rate"] >= 0.75 for m in framework_metrics.values()
        )

        return {
            "frameworks_tested": len(framework_metrics),
            "average_success_rate_across_frameworks": avg_success_rate,
            "all_frameworks_meet_target": all_meet_target,
            "framework_success_rates": {
                f: m["repair_success_rate"] for f, m in framework_metrics.items()
            },
            "overall_success": all_meet_target and avg_success_rate >= 0.75,
        }

    def _print_framework_summary(self, result: Dict[str, Any]):
        """Print framework summary."""
        metrics = result["metrics"]
        print(f"\nResults for {result['framework'].upper()}:")
        print(f"  Total tests: {metrics['total_tests']}")
        print(f"  Successful repairs: {metrics['successful_repairs']}/{metrics['total_failures']}")
        print(f"  Success rate: {metrics['repair_success_rate']:.1%}")
        print(f"  Average repair time: {metrics['average_repair_time_ms']:.0f}ms")
        print(f"  Framework patterns handled: {len(metrics['framework_specific_patterns'])}")
        print(f"  Status: {'PASS' if result['success'] else 'FAIL'}")

    def _print_cross_framework_summary(self, result: Dict[str, Any]):
        """Print cross-framework summary."""
        print(f"\nCross-Framework Results:")
        print(f"  Total tests: {result['total_tests']}")
        print(f"  Successful repairs: {result['successful_repairs']}")
        print(f"  Overall success rate: {result['overall_success_rate']:.1%}")
        print(f"  Per-framework success rates:")
        for framework, rate in result['framework_success_rates'].items():
            print(f"    {framework}: {rate:.1%}")
        print(f"  Status: {'PASS' if result['success'] else 'FAIL'}")

    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final summary."""
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}\n")

        print(f"Frameworks tested: {summary['frameworks_tested']}")
        print(f"Average success rate: {summary['average_success_rate_across_frameworks']:.1%}")
        print(f"All frameworks meet 75% target: {'YES' if summary['all_frameworks_meet_target'] else 'NO'}")
        print(f"\nPer-framework success rates:")
        for framework, rate in summary['framework_success_rates'].items():
            status = "✓" if rate >= 0.75 else "✗"
            print(f"  {status} {framework}: {rate:.1%}")
        print(f"\nOverall Status: {'PASS' if summary['overall_success'] else 'FAIL'}")

    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        output_file = self.output_dir / f"multi_framework_{int(time.time())}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


async def main():
    """Main entry point."""
    tester = MultiFrameworkHealingTester()
    results = await tester.run_all_tests()

    # Exit with appropriate code
    if results["summary"].get("overall_success", False):
        print("\n✓ All tests PASSED")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
