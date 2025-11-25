"""Test healing after major code refactoring.

This script validates that the self-healing system can adapt tests to
major refactoring operations like function renames, signature changes,
module reorganization, and class restructuring.

Success Criteria:
- Adapt tests to renamed functions/methods
- Handle parameter signature changes
- Update imports after module reorganization
- Maintain test coverage after refactoring
- 80%+ successful adaptation rate
"""

import asyncio
import time
import json
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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


class RefactoringType(str, Enum):
    """Types of refactoring operations."""
    RENAME_FUNCTION = "rename_function"
    CHANGE_SIGNATURE = "change_signature"
    MOVE_MODULE = "move_module"
    EXTRACT_CLASS = "extract_class"
    INLINE_FUNCTION = "inline_function"
    RESTRUCTURE_PARAMS = "restructure_params"


@dataclass
class RefactoringOperation:
    """Represents a refactoring operation."""
    refactoring_type: RefactoringType
    before_code: str
    after_code: str
    affected_tests: int
    description: str


@dataclass
class RefactoringMetrics:
    """Metrics for refactoring recovery."""
    refactoring_type: str
    total_affected_tests: int
    tests_healed: int
    tests_failed: int
    healing_success_rate: float
    coverage_maintained: bool
    average_healing_time_ms: float


class RefactoringRecoveryTester:
    """Test healing after major refactoring."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize tester."""
        self.output_dir = output_dir or Path("test_results/refactoring_recovery")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize healing components
        self.analyzer = FailureAnalyzer()
        self.repairer = TestRepairer()

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all refactoring recovery tests."""
        print("=" * 80)
        print("REFACTORING RECOVERY - VALIDATION SUITE")
        print("=" * 80)
        print("Target: 80%+ adaptation success rate")
        print("Target: Maintain test coverage after refactoring")
        print()

        all_results = {
            "test_suite": "refactoring_recovery",
            "timestamp": datetime.now().isoformat(),
            "scenarios": [],
            "summary": {},
        }

        # Test each refactoring type
        refactoring_types = [
            RefactoringType.RENAME_FUNCTION,
            RefactoringType.CHANGE_SIGNATURE,
            RefactoringType.MOVE_MODULE,
            RefactoringType.EXTRACT_CLASS,
            RefactoringType.INLINE_FUNCTION,
            RefactoringType.RESTRUCTURE_PARAMS,
        ]

        for refactor_type in refactoring_types:
            print(f"\n{'=' * 80}")
            print(f"Testing: {refactor_type.value.replace('_', ' ').title()}")
            print(f"{'=' * 80}\n")

            result = await self.test_refactoring_recovery(refactor_type)
            all_results["scenarios"].append(result)

            self._print_scenario_summary(result)

        # Test complex multi-step refactoring
        print(f"\n{'=' * 80}")
        print("Testing: Complex Multi-Step Refactoring")
        print(f"{'=' * 80}\n")

        complex_result = await self.test_complex_refactoring()
        all_results["scenarios"].append(complex_result)
        self._print_scenario_summary(complex_result)

        # Test breaking API changes
        print(f"\n{'=' * 80}")
        print("Testing: Breaking API Changes")
        print(f"{'=' * 80}\n")

        breaking_result = await self.test_breaking_api_changes()
        all_results["scenarios"].append(breaking_result)
        self._print_scenario_summary(breaking_result)

        # Generate summary
        summary = self._generate_summary(all_results["scenarios"])
        all_results["summary"] = summary

        # Print final summary
        self._print_final_summary(summary)

        # Save results
        self._save_results(all_results)

        return all_results

    async def test_refactoring_recovery(
        self, refactor_type: RefactoringType
    ) -> Dict[str, Any]:
        """Test recovery from a specific refactoring type."""
        # Generate refactoring operation
        operation = self._create_refactoring_operation(refactor_type)

        print(f"Refactoring: {operation.description}")
        print(f"Affected tests: {operation.affected_tests}")
        print()

        # Generate test failures from refactoring
        failures = self._generate_refactoring_failures(operation)

        print(f"Generated {len(failures)} failing tests")
        print()

        start_time = time.time()
        healed = 0
        failed = 0
        healing_times = []
        coverage_checks = []

        print("Healing tests...")

        for i, failure in enumerate(failures, 1):
            test_start = time.time()

            try:
                # Provide refactoring context
                code_diff = self._create_code_diff(operation)

                # Analyze with context
                analysis = await self.analyzer.analyze_failure(failure, code_diff)

                # Repair test
                repair = await self.repairer.repair_test(analysis)

                test_time_ms = (time.time() - test_start) * 1000
                healing_times.append(test_time_ms)

                if repair and repair.test_passes:
                    healed += 1
                    coverage_checks.append(repair.coverage_maintained)
                    status = "SUCCESS"
                else:
                    failed += 1
                    status = "FAILED"

                if i % 5 == 0:
                    print(f"  Progress: {i}/{len(failures)} - {status} - "
                          f"Success: {healed/i:.1%}")

            except Exception as e:
                print(f"  Error: {e}")
                failed += 1
                healing_times.append((time.time() - test_start) * 1000)

        total_time_ms = (time.time() - start_time) * 1000

        # Calculate metrics
        import statistics
        avg_healing_time = statistics.mean(healing_times) if healing_times else 0
        success_rate = healed / len(failures) if failures else 0
        coverage_maintained = (
            sum(coverage_checks) / len(coverage_checks)
            if coverage_checks
            else False
        ) > 0.9

        metrics = RefactoringMetrics(
            refactoring_type=refactor_type.value,
            total_affected_tests=len(failures),
            tests_healed=healed,
            tests_failed=failed,
            healing_success_rate=success_rate,
            coverage_maintained=coverage_maintained,
            average_healing_time_ms=avg_healing_time,
        )

        return {
            "scenario": refactor_type.value,
            "refactoring_operation": {
                "type": operation.refactoring_type.value,
                "description": operation.description,
                "before_code": operation.before_code[:200],  # Truncate for readability
                "after_code": operation.after_code[:200],
            },
            "metrics": asdict(metrics),
            "total_time_ms": total_time_ms,
            "success": success_rate >= 0.8 and coverage_maintained,
        }

    async def test_complex_refactoring(self) -> Dict[str, Any]:
        """Test recovery from complex multi-step refactoring."""
        print("Applying multi-step refactoring:")
        print("  1. Rename multiple functions")
        print("  2. Reorganize module structure")
        print("  3. Change function signatures")
        print()

        # Simulate complex refactoring
        operations = [
            self._create_refactoring_operation(RefactoringType.RENAME_FUNCTION),
            self._create_refactoring_operation(RefactoringType.MOVE_MODULE),
            self._create_refactoring_operation(RefactoringType.CHANGE_SIGNATURE),
        ]

        all_failures = []
        for op in operations:
            failures = self._generate_refactoring_failures(op)
            all_failures.extend(failures)

        print(f"Total affected tests: {len(all_failures)}")
        print()

        start_time = time.time()
        healed = 0

        print("Healing tests...")

        for i, failure in enumerate(all_failures, 1):
            try:
                # Analyze failure
                analysis = await self.analyzer.analyze_failure(failure)

                # Repair with multiple strategies
                repair = await self.repairer.repair_test(analysis, max_attempts=5)

                if repair and repair.test_passes:
                    healed += 1

                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(all_failures)} - "
                          f"Success: {healed/i:.1%}")

            except:
                pass

        total_time_ms = (time.time() - start_time) * 1000
        success_rate = healed / len(all_failures) if all_failures else 0

        return {
            "scenario": "complex_multi_step_refactoring",
            "type": "complex",
            "refactoring_steps": len(operations),
            "total_affected_tests": len(all_failures),
            "tests_healed": healed,
            "success_rate": success_rate,
            "total_time_ms": total_time_ms,
            "success": success_rate >= 0.75,  # Slightly lower threshold for complex
        }

    async def test_breaking_api_changes(self) -> Dict[str, Any]:
        """Test recovery from breaking API changes."""
        print("Testing breaking API changes:")
        print("  - Return type changes")
        print("  - Parameter removals")
        print("  - Method deletions")
        print()

        # Generate breaking changes
        breaking_changes = [
            {
                "type": "return_type_change",
                "before": "def get_user() -> User:",
                "after": "def get_user() -> Optional[User]:",
                "affected_tests": 10,
            },
            {
                "type": "param_removal",
                "before": "def process(data, format, validate):",
                "after": "def process(data, format):",  # validate removed
                "affected_tests": 8,
            },
            {
                "type": "method_deletion",
                "before": "class Service:\n    def deprecated_method():",
                "after": "class Service:\n    # deprecated_method removed",
                "affected_tests": 5,
            },
        ]

        all_failures = []
        for change in breaking_changes:
            failures = self._generate_breaking_change_failures(change)
            all_failures.extend(failures)

        print(f"Total tests affected by breaking changes: {len(all_failures)}")
        print()

        start_time = time.time()
        healed = 0
        healed_by_type = {}

        print("Healing tests...")

        for failure in all_failures:
            try:
                analysis = await self.analyzer.analyze_failure(failure)
                repair = await self.repairer.repair_test(analysis)

                if repair and repair.test_passes:
                    healed += 1

                    # Track by change type
                    change_type = failure.error_message.split(":")[0]
                    healed_by_type[change_type] = healed_by_type.get(change_type, 0) + 1

            except:
                pass

        total_time_ms = (time.time() - start_time) * 1000
        success_rate = healed / len(all_failures) if all_failures else 0

        return {
            "scenario": "breaking_api_changes",
            "type": "breaking",
            "breaking_changes": len(breaking_changes),
            "total_affected_tests": len(all_failures),
            "tests_healed": healed,
            "success_rate": success_rate,
            "healed_by_change_type": healed_by_type,
            "total_time_ms": total_time_ms,
            "success": success_rate >= 0.70,  # Lower threshold for breaking changes
        }

    def _create_refactoring_operation(
        self, refactor_type: RefactoringType
    ) -> RefactoringOperation:
        """Create a refactoring operation."""
        if refactor_type == RefactoringType.RENAME_FUNCTION:
            return RefactoringOperation(
                refactoring_type=refactor_type,
                before_code="""
def calculate_total(items):
    return sum(item.price for item in items)
                """.strip(),
                after_code="""
def compute_order_total(items):
    return sum(item.price for item in items)
                """.strip(),
                affected_tests=15,
                description="Renamed calculate_total() to compute_order_total()",
            )

        elif refactor_type == RefactoringType.CHANGE_SIGNATURE:
            return RefactoringOperation(
                refactoring_type=refactor_type,
                before_code="""
def process_payment(amount, currency):
    return Payment(amount, currency)
                """.strip(),
                after_code="""
def process_payment(amount, currency, payment_method="card"):
    return Payment(amount, currency, payment_method)
                """.strip(),
                affected_tests=12,
                description="Added payment_method parameter with default value",
            )

        elif refactor_type == RefactoringType.MOVE_MODULE:
            return RefactoringOperation(
                refactoring_type=refactor_type,
                before_code="from utils.helpers import format_date",
                after_code="from core.formatting.dates import format_date",
                affected_tests=20,
                description="Moved format_date from utils.helpers to core.formatting.dates",
            )

        elif refactor_type == RefactoringType.EXTRACT_CLASS:
            return RefactoringOperation(
                refactoring_type=refactor_type,
                before_code="""
class Order:
    def calculate_shipping(self):
        return self.weight * 2.5
                """.strip(),
                after_code="""
class ShippingCalculator:
    def calculate(self, weight):
        return weight * 2.5

class Order:
    def calculate_shipping(self):
        return ShippingCalculator().calculate(self.weight)
                """.strip(),
                affected_tests=10,
                description="Extracted shipping logic into ShippingCalculator class",
            )

        elif refactor_type == RefactoringType.INLINE_FUNCTION:
            return RefactoringOperation(
                refactoring_type=refactor_type,
                before_code="""
def is_valid(value):
    return _check_validity(value)

def _check_validity(value):
    return value is not None and value > 0
                """.strip(),
                after_code="""
def is_valid(value):
    return value is not None and value > 0
                """.strip(),
                affected_tests=8,
                description="Inlined _check_validity() into is_valid()",
            )

        elif refactor_type == RefactoringType.RESTRUCTURE_PARAMS:
            return RefactoringOperation(
                refactoring_type=refactor_type,
                before_code="""
def create_user(name, email, age, country, city):
    return User(name, email, age, country, city)
                """.strip(),
                after_code="""
def create_user(name, email, address: Address):
    return User(name, email, address)
                """.strip(),
                affected_tests=14,
                description="Grouped address parameters into Address object",
            )

        return RefactoringOperation(
            refactoring_type=refactor_type,
            before_code="",
            after_code="",
            affected_tests=10,
            description="Generic refactoring",
        )

    def _generate_refactoring_failures(
        self, operation: RefactoringOperation
    ) -> List[TestFailure]:
        """Generate test failures from refactoring."""
        failures = []

        for i in range(operation.affected_tests):
            # Determine failure type based on refactoring
            if operation.refactoring_type == RefactoringType.RENAME_FUNCTION:
                failure_type = FailureType.ATTRIBUTE_ERROR
                error_msg = f"AttributeError: module has no attribute 'calculate_total'"

            elif operation.refactoring_type == RefactoringType.CHANGE_SIGNATURE:
                failure_type = FailureType.TYPE_ERROR
                error_msg = "TypeError: process_payment() missing 1 required positional argument: 'payment_method'"

            elif operation.refactoring_type == RefactoringType.MOVE_MODULE:
                failure_type = FailureType.IMPORT_ERROR
                error_msg = "ImportError: cannot import name 'format_date' from 'utils.helpers'"

            elif operation.refactoring_type == RefactoringType.EXTRACT_CLASS:
                failure_type = FailureType.ATTRIBUTE_ERROR
                error_msg = "AttributeError: 'Order' object has no attribute 'calculate_shipping'"

            elif operation.refactoring_type == RefactoringType.INLINE_FUNCTION:
                failure_type = FailureType.ATTRIBUTE_ERROR
                error_msg = "AttributeError: module has no attribute '_check_validity'"

            elif operation.refactoring_type == RefactoringType.RESTRUCTURE_PARAMS:
                failure_type = FailureType.TYPE_ERROR
                error_msg = "TypeError: create_user() takes 3 positional arguments but 5 were given"

            else:
                failure_type = FailureType.RUNTIME_ERROR
                error_msg = "RuntimeError: refactoring broke test"

            failure = TestFailure(
                test_id=f"test_{operation.refactoring_type.value}_{i:03d}",
                test_name=f"test_after_{operation.refactoring_type.value}_{i}",
                test_file=f"tests/test_refactored_{i//5}.py",
                failure_type=failure_type,
                error_message=error_msg,
                test_framework="pytest",
                target_file="src/refactored_module.py",
                target_function="refactored_function",
                execution_time_ms=100,
            )
            failures.append(failure)

        return failures

    def _generate_breaking_change_failures(
        self, change: Dict[str, Any]
    ) -> List[TestFailure]:
        """Generate failures from breaking API changes."""
        failures = []

        for i in range(change["affected_tests"]):
            if change["type"] == "return_type_change":
                failure_type = FailureType.TYPE_ERROR
                error_msg = "return_type_change: TypeError: 'NoneType' object is not iterable"

            elif change["type"] == "param_removal":
                failure_type = FailureType.TYPE_ERROR
                error_msg = "param_removal: TypeError: got unexpected keyword argument 'validate'"

            elif change["type"] == "method_deletion":
                failure_type = FailureType.ATTRIBUTE_ERROR
                error_msg = "method_deletion: AttributeError: 'Service' has no attribute 'deprecated_method'"

            else:
                failure_type = FailureType.RUNTIME_ERROR
                error_msg = "breaking_change: RuntimeError"

            failure = TestFailure(
                test_id=f"test_breaking_{change['type']}_{i:03d}",
                test_name=f"test_{change['type']}_{i}",
                test_file=f"tests/test_api_{i//3}.py",
                failure_type=failure_type,
                error_message=error_msg,
                test_framework="pytest",
                target_file="src/api.py",
                target_function="api_method",
                execution_time_ms=100,
            )
            failures.append(failure)

        return failures

    def _create_code_diff(self, operation: RefactoringOperation) -> Dict[str, Any]:
        """Create code diff for refactoring operation."""
        return {
            "functions": [
                {
                    "before": operation.before_code,
                    "after": operation.after_code,
                    "change_type": operation.refactoring_type.value,
                }
            ],
            "files": ["src/refactored_module.py"],
        }

    def _generate_summary(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary."""
        refactoring_scenarios = [
            s for s in scenarios
            if s.get("metrics") and "refactoring_type" in s["metrics"]
        ]

        if not refactoring_scenarios:
            return {"overall_success": False}

        import statistics

        avg_success_rate = statistics.mean(
            s["metrics"]["healing_success_rate"] for s in refactoring_scenarios
        )

        coverage_maintained_count = sum(
            1 for s in refactoring_scenarios
            if s["metrics"]["coverage_maintained"]
        )

        coverage_maintained_rate = (
            coverage_maintained_count / len(refactoring_scenarios)
            if refactoring_scenarios
            else 0
        )

        all_meet_target = all(s["success"] for s in scenarios)

        return {
            "total_scenarios": len(scenarios),
            "refactoring_scenarios": len(refactoring_scenarios),
            "average_adaptation_rate": avg_success_rate,
            "coverage_maintained_rate": coverage_maintained_rate,
            "all_scenarios_passed": all_meet_target,
            "overall_success": avg_success_rate >= 0.8 and all_meet_target,
        }

    def _print_scenario_summary(self, result: Dict[str, Any]):
        """Print scenario summary."""
        if "metrics" in result:
            metrics = result["metrics"]
            print(f"\nResults:")
            print(f"  Affected tests: {metrics['total_affected_tests']}")
            print(f"  Tests healed: {metrics['tests_healed']}/{metrics['total_affected_tests']}")
            print(f"  Success rate: {metrics['healing_success_rate']:.1%}")
            print(f"  Coverage maintained: {'YES' if metrics['coverage_maintained'] else 'NO'}")
            print(f"  Average healing time: {metrics['average_healing_time_ms']:.0f}ms")
            print(f"  Status: {'PASS' if result['success'] else 'FAIL'}")
        else:
            # Complex scenarios
            print(f"\nResults:")
            print(f"  Affected tests: {result.get('total_affected_tests', 0)}")
            print(f"  Tests healed: {result.get('tests_healed', 0)}")
            print(f"  Success rate: {result.get('success_rate', 0):.1%}")
            print(f"  Status: {'PASS' if result['success'] else 'FAIL'}")

    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final summary."""
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}\n")

        print(f"Total scenarios: {summary['total_scenarios']}")
        print(f"Average adaptation rate: {summary['average_adaptation_rate']:.1%}")
        print(f"Coverage maintained rate: {summary['coverage_maintained_rate']:.1%}")
        print(f"All scenarios passed: {'YES' if summary['all_scenarios_passed'] else 'NO'}")
        print(f"\nOverall Status: {'PASS' if summary['overall_success'] else 'FAIL'}")

    def _save_results(self, results: Dict[str, Any]):
        """Save results."""
        output_file = self.output_dir / f"refactoring_recovery_{int(time.time())}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


async def main():
    """Main entry point."""
    tester = RefactoringRecoveryTester()
    results = await tester.run_all_tests()

    if results["summary"].get("overall_success", False):
        print("\n✓ All tests PASSED")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
