#!/usr/bin/env python3
"""Integration test for failure analysis accuracy.

Tests the failure analyzer's ability to correctly identify root causes
of test failures with target accuracy of 90%+.

Usage:
    python scripts/test_failure_analysis_accuracy.py [--verbose] [--samples N]
    python scripts/test_failure_analysis_accuracy.py --inject-failures ./test-failures/
"""

import argparse
import asyncio
import logging
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.healing_models import (
    TestFailure,
    FailureType,
    FailureAnalysis,
)
from src.testing.healing import FailureAnalyzer
from src.services.healing_service import HealingOrchestrator


# ANSI color codes for terminal output
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

    # Use ASCII characters for Windows compatibility
    try:
        bar = '█' * filled + '░' * (bar_length - filled)
    except UnicodeEncodeError:
        bar = '#' * filled + '-' * (bar_length - filled)

    try:
        print(f"\r{Colors.CYAN}[{bar}] {percent:.1f}% {message}{Colors.ENDC}", end='', flush=True)
    except UnicodeEncodeError:
        # Fallback for Windows console
        print(f"\r[{bar}] {percent:.1f}% {message}", end='', flush=True)

    if current == total:
        print()  # New line when complete


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")


class FailureScenario:
    """Represents a test failure scenario with known root cause."""

    def __init__(
        self,
        name: str,
        failure: TestFailure,
        expected_root_cause: str,
        expected_failure_type: FailureType,
        expected_strategies: List[str],
    ):
        self.name = name
        self.failure = failure
        self.expected_root_cause = expected_root_cause
        self.expected_failure_type = expected_failure_type
        self.expected_strategies = expected_strategies


def generate_test_scenarios() -> List[FailureScenario]:
    """Generate realistic test failure scenarios.

    Returns:
        List of failure scenarios with known root causes
    """
    scenarios = []

    # Scenario 1: Syntax error
    scenarios.append(FailureScenario(
        name="Syntax Error - Missing Colon",
        failure=TestFailure(
            test_id="test_001",
            test_name="test_user_login",
            test_file="tests/test_auth.py",
            failure_type=FailureType.SYNTAX_ERROR,
            error_message="SyntaxError: invalid syntax (test_auth.py, line 42)",
            stack_trace="  File 'tests/test_auth.py', line 42\n    def test_user_login()\n                         ^\nSyntaxError: invalid syntax",
            line_number=42,
            target_file="src/auth/login.py",
            target_function="user_login",
            test_framework="pytest",
            execution_time_ms=50,
        ),
        expected_root_cause="syntax error",
        expected_failure_type=FailureType.SYNTAX_ERROR,
        expected_strategies=["regenerate"],
    ))

    # Scenario 2: Assertion failure
    scenarios.append(FailureScenario(
        name="Assertion Failure - Expected Value Changed",
        failure=TestFailure(
            test_id="test_002",
            test_name="test_calculate_total",
            test_file="tests/test_billing.py",
            failure_type=FailureType.ASSERTION_FAILED,
            error_message="AssertionError: assert 150 == 120\n  Expected: 120\n  Actual: 150",
            stack_trace="tests/test_billing.py:25: AssertionError",
            line_number=25,
            target_file="src/billing/calculator.py",
            target_function="calculate_total",
            test_framework="pytest",
            execution_time_ms=100,
        ),
        expected_root_cause="assertion",
        expected_failure_type=FailureType.ASSERTION_FAILED,
        expected_strategies=["update_assertion", "regenerate"],
    ))

    # Scenario 3: Import error
    scenarios.append(FailureScenario(
        name="Import Error - Module Renamed",
        failure=TestFailure(
            test_id="test_003",
            test_name="test_data_processing",
            test_file="tests/test_processor.py",
            failure_type=FailureType.IMPORT_ERROR,
            error_message="ImportError: cannot import name 'DataProcessor' from 'src.processing'",
            stack_trace="  File 'tests/test_processor.py', line 5\n    from src.processing import DataProcessor\nImportError: cannot import name 'DataProcessor'",
            line_number=5,
            target_file="src/processing/__init__.py",
            target_function=None,
            test_framework="pytest",
            execution_time_ms=30,
        ),
        expected_root_cause="import",
        expected_failure_type=FailureType.IMPORT_ERROR,
        expected_strategies=["update_import"],
    ))

    # Scenario 4: Attribute error
    scenarios.append(FailureScenario(
        name="Attribute Error - Method Renamed",
        failure=TestFailure(
            test_id="test_004",
            test_name="test_user_profile",
            test_file="tests/test_user.py",
            failure_type=FailureType.ATTRIBUTE_ERROR,
            error_message="AttributeError: 'User' object has no attribute 'getName'",
            stack_trace="  File 'tests/test_user.py', line 15\n    name = user.getName()\nAttributeError: 'User' object has no attribute 'getName'",
            line_number=15,
            target_file="src/models/user.py",
            target_function="getName",
            test_framework="pytest",
            execution_time_ms=80,
        ),
        expected_root_cause="attribute",
        expected_failure_type=FailureType.ATTRIBUTE_ERROR,
        expected_strategies=["update_signature", "regenerate"],
    ))

    # Scenario 5: Timeout
    scenarios.append(FailureScenario(
        name="Timeout - Test Exceeded Time Limit",
        failure=TestFailure(
            test_id="test_005",
            test_name="test_api_call",
            test_file="tests/test_api.py",
            failure_type=FailureType.TIMEOUT,
            error_message="Test timed out after 5 seconds",
            stack_trace="tests/test_api.py:30: TimeoutError: Test exceeded 5s limit",
            line_number=30,
            target_file="src/api/client.py",
            target_function="fetch_data",
            test_framework="pytest",
            execution_time_ms=5000,
        ),
        expected_root_cause="timeout",
        expected_failure_type=FailureType.TIMEOUT,
        expected_strategies=["add_wait", "regenerate"],
    ))

    # Scenario 6: Type error
    scenarios.append(FailureScenario(
        name="Type Error - Wrong Argument Type",
        failure=TestFailure(
            test_id="test_006",
            test_name="test_format_date",
            test_file="tests/test_utils.py",
            failure_type=FailureType.TYPE_ERROR,
            error_message="TypeError: format_date() argument must be datetime, not str",
            stack_trace="  File 'tests/test_utils.py', line 20\n    result = format_date('2024-01-01')\nTypeError: format_date() argument must be datetime, not str",
            line_number=20,
            target_file="src/utils/date.py",
            target_function="format_date",
            test_framework="pytest",
            execution_time_ms=60,
        ),
        expected_root_cause="type",
        expected_failure_type=FailureType.TYPE_ERROR,
        expected_strategies=["regenerate"],
    ))

    # Scenario 7: Mock error
    scenarios.append(FailureScenario(
        name="Mock Error - Mock Configuration Invalid",
        failure=TestFailure(
            test_id="test_007",
            test_name="test_email_sender",
            test_file="tests/test_email.py",
            failure_type=FailureType.MOCK_ERROR,
            error_message="MockError: mock.send_email() was called with unexpected arguments",
            stack_trace="tests/test_email.py:18: MockError",
            line_number=18,
            target_file="src/email/sender.py",
            target_function="send_email",
            test_framework="pytest",
            execution_time_ms=90,
        ),
        expected_root_cause="mock",
        expected_failure_type=FailureType.MOCK_ERROR,
        expected_strategies=["update_mock", "regenerate"],
    ))

    # Scenario 8: Runtime error
    scenarios.append(FailureScenario(
        name="Runtime Error - Null Reference",
        failure=TestFailure(
            test_id="test_008",
            test_name="test_order_processing",
            test_file="tests/test_orders.py",
            failure_type=FailureType.RUNTIME_ERROR,
            error_message="NoneType object has no attribute 'total'",
            stack_trace="  File 'tests/test_orders.py', line 28\n    total = order.total\nAttributeError: 'NoneType' object has no attribute 'total'",
            line_number=28,
            target_file="src/orders/processor.py",
            target_function="process_order",
            test_framework="pytest",
            execution_time_ms=120,
        ),
        expected_root_cause="runtime",
        expected_failure_type=FailureType.RUNTIME_ERROR,
        expected_strategies=["regenerate"],
    ))

    return scenarios


async def analyze_scenario(
    analyzer: FailureAnalyzer,
    scenario: FailureScenario,
    verbose: bool = False
) -> Tuple[bool, FailureAnalysis, Dict[str, Any]]:
    """Analyze a failure scenario and check accuracy.

    Args:
        analyzer: Failure analyzer instance
        scenario: Test scenario
        verbose: Whether to print verbose output

    Returns:
        Tuple of (success, analysis, metrics)
    """
    try:
        # Perform analysis
        analysis = await analyzer.analyze_failure(scenario.failure)

        # Check if failure type correctly identified
        type_correct = analysis.failure.failure_type == scenario.expected_failure_type

        # Check if root cause contains expected keywords
        root_cause_correct = any(
            keyword in analysis.root_cause.lower()
            for keyword in scenario.expected_root_cause.lower().split()
        )

        # Check if suggested strategies are reasonable
        suggested_strategy_names = [s.value for s in analysis.suggested_strategies]
        strategies_correct = any(
            expected in suggested_strategy_names
            for expected in scenario.expected_strategies
        )

        # Overall success
        success = type_correct and root_cause_correct and strategies_correct

        metrics = {
            "type_correct": type_correct,
            "root_cause_correct": root_cause_correct,
            "strategies_correct": strategies_correct,
            "confidence": analysis.confidence,
            "identified_type": analysis.failure.failure_type.value,
            "suggested_strategies": suggested_strategy_names,
        }

        if verbose:
            print(f"\n  Scenario: {scenario.name}")
            print(f"    Expected Type: {scenario.expected_failure_type.value}")
            print(f"    Identified Type: {analysis.failure.failure_type.value} {'✓' if type_correct else '✗'}")
            print(f"    Expected Root Cause: {scenario.expected_root_cause}")
            print(f"    Identified Root Cause: {analysis.root_cause} {'✓' if root_cause_correct else '✗'}")
            print(f"    Confidence: {analysis.confidence:.2%}")
            print(f"    Suggested Strategies: {', '.join(suggested_strategy_names)} {'✓' if strategies_correct else '✗'}")

        return success, analysis, metrics

    except Exception as e:
        if verbose:
            print(f"\n  {Colors.RED}Error analyzing scenario {scenario.name}: {e}{Colors.ENDC}")
        return False, None, {"error": str(e)}


async def run_accuracy_test(
    num_samples: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run failure analysis accuracy test.

    Args:
        num_samples: Number of test samples
        verbose: Whether to print verbose output

    Returns:
        Test results dictionary
    """
    print_header("FAILURE ANALYSIS ACCURACY TEST")

    # Initialize analyzer
    print(f"{Colors.BLUE}Initializing failure analyzer...{Colors.ENDC}")
    analyzer = FailureAnalyzer()

    # Generate scenarios
    print(f"{Colors.BLUE}Generating {num_samples} test scenarios...{Colors.ENDC}")
    base_scenarios = generate_test_scenarios()

    # Replicate scenarios to reach num_samples
    scenarios = []
    while len(scenarios) < num_samples:
        scenarios.extend(base_scenarios)
    scenarios = scenarios[:num_samples]

    print(f"{Colors.CYAN}Created {len(scenarios)} test scenarios across {len(base_scenarios)} failure types{Colors.ENDC}\n")

    # Track results
    start_time = time.time()
    results = {
        "total": len(scenarios),
        "successful": 0,
        "failed": 0,
        "by_type": defaultdict(lambda: {"total": 0, "correct": 0}),
        "analyses": [],
        "metrics": {
            "type_accuracy": 0.0,
            "root_cause_accuracy": 0.0,
            "strategy_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "avg_confidence": 0.0,
        }
    }

    # Analyze each scenario
    print(f"{Colors.BLUE}Analyzing test failures...{Colors.ENDC}")

    type_correct_count = 0
    root_cause_correct_count = 0
    strategy_correct_count = 0
    total_confidence = 0.0

    for i, scenario in enumerate(scenarios):
        print_progress(i + 1, len(scenarios), f"Analyzing scenario {i + 1}/{len(scenarios)}")

        success, analysis, metrics = await analyze_scenario(analyzer, scenario, verbose)

        if success:
            results["successful"] += 1
        else:
            results["failed"] += 1

        # Track by type
        failure_type = scenario.expected_failure_type.value
        results["by_type"][failure_type]["total"] += 1
        if success:
            results["by_type"][failure_type]["correct"] += 1

        # Track individual metrics
        if "error" not in metrics:
            if metrics.get("type_correct"):
                type_correct_count += 1
            if metrics.get("root_cause_correct"):
                root_cause_correct_count += 1
            if metrics.get("strategies_correct"):
                strategy_correct_count += 1
            total_confidence += metrics.get("confidence", 0.0)

        results["analyses"].append({
            "scenario": scenario.name,
            "success": success,
            "metrics": metrics,
        })

    # Calculate final metrics
    total = len(scenarios)
    results["metrics"]["type_accuracy"] = type_correct_count / total
    results["metrics"]["root_cause_accuracy"] = root_cause_correct_count / total
    results["metrics"]["strategy_accuracy"] = strategy_correct_count / total
    results["metrics"]["overall_accuracy"] = results["successful"] / total
    results["metrics"]["avg_confidence"] = total_confidence / total
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
    print(f"  Successful: {results['successful']} ({results['metrics']['overall_accuracy']:.1%})")
    print(f"  Failed: {results['failed']}")
    print(f"  Execution Time: {results['execution_time_seconds']:.2f}s")
    print()

    # Detailed metrics
    print(f"{Colors.BOLD}Accuracy Metrics:{Colors.ENDC}")

    metrics = results['metrics']

    def print_metric(name: str, value: float, target: float):
        status = "✓" if value >= target else "✗"
        color = Colors.GREEN if value >= target else Colors.RED
        print(f"  {name:.<40} {color}{value:>6.1%} {status}{Colors.ENDC}")

    print_metric("Failure Type Classification", metrics['type_accuracy'], target_accuracy)
    print_metric("Root Cause Identification", metrics['root_cause_accuracy'], target_accuracy)
    print_metric("Strategy Suggestion", metrics['strategy_accuracy'], target_accuracy)
    print_metric("Overall Accuracy", metrics['overall_accuracy'], target_accuracy)
    print(f"  {'Average Confidence':.<40} {metrics['avg_confidence']:>6.1%}")
    print()

    # By failure type
    print(f"{Colors.BOLD}Accuracy by Failure Type:{Colors.ENDC}")
    for failure_type, stats in sorted(results['by_type'].items()):
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        status = "✓" if accuracy >= target_accuracy else "✗"
        color = Colors.GREEN if accuracy >= target_accuracy else Colors.RED
        print(f"  {failure_type:.<40} {color}{accuracy:>6.1%} ({stats['correct']}/{stats['total']}) {status}{Colors.ENDC}")
    print()

    # Final verdict
    overall_success = metrics['overall_accuracy'] >= target_accuracy

    if overall_success:
        print_success(f"SUCCESS: Achieved {metrics['overall_accuracy']:.1%} accuracy (target: {target_accuracy:.0%})")
        return 0
    else:
        print_error(f"FAILURE: Achieved {metrics['overall_accuracy']:.1%} accuracy (target: {target_accuracy:.0%})")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test failure analysis accuracy (target: 90%+)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of test samples to analyze (default: 100)",
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
        "--inject-failures",
        type=str,
        help="Path to directory with real test failures to analyze",
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
        results = asyncio.run(run_accuracy_test(
            num_samples=args.samples,
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
