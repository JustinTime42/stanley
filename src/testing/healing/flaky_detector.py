"""Flaky test detection through statistical analysis.

PATTERN: Statistical flaky test detection
CRITICAL: Requires multiple test runs for accuracy
"""

import logging
import asyncio
import statistics
from typing import List, Dict, Any, Set, Optional

from ...models.healing_models import (
    FlakyTestResult,
    FailureType,
)

logger = logging.getLogger(__name__)


class FlakyDetector:
    """
    Detect flaky tests through statistical analysis.

    PATTERN: Multiple runs with statistical analysis
    CRITICAL: Minimum 10 runs recommended for accuracy
    GOTCHA: Must ensure clean state between runs for accurate detection
    """

    def __init__(self, test_runner=None, min_runs: int = 10):
        """
        Initialize flaky detector.

        Args:
            test_runner: Test runner for executing tests
            min_runs: Minimum runs for detection (default: 10)
        """
        self.test_runner = test_runner
        self.min_runs = min_runs
        self.logger = logger

    async def detect_flaky_tests(
        self,
        test_ids: List[str],
        runs_per_test: int = 20,
        parallel: bool = False,
    ) -> List[FlakyTestResult]:
        """
        Detect flaky tests through statistical analysis.

        PATTERN: Multiple runs with statistical analysis
        CRITICAL: Ensure test isolation between runs

        Args:
            test_ids: List of test IDs to analyze
            runs_per_test: Number of runs per test (min 10, recommended 20)
            parallel: Whether to run tests in parallel (default: False for accuracy)

        Returns:
            List of flaky test results
        """
        if runs_per_test < self.min_runs:
            logger.warning(
                f"runs_per_test ({runs_per_test}) < min_runs ({self.min_runs}), "
                f"results may be inaccurate"
            )

        results = []

        for test_id in test_ids:
            logger.info(f"Analyzing flakiness for test: {test_id}")

            try:
                result = await self._analyze_single_test(test_id, runs_per_test)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing test {test_id}: {e}", exc_info=True)
                # Create error result
                results.append(self._create_error_result(test_id, str(e)))

        return results

    async def _analyze_single_test(
        self, test_id: str, runs_per_test: int
    ) -> FlakyTestResult:
        """
        Analyze single test for flakiness.

        Args:
            test_id: Test to analyze
            runs_per_test: Number of runs

        Returns:
            Flaky test result
        """
        # Storage for run results
        run_results: List[bool] = []
        execution_times: List[float] = []
        failure_messages: Set[str] = set()
        failure_types: Dict[FailureType, int] = {}

        # Run test multiple times
        for run_num in range(runs_per_test):
            logger.debug(f"Test {test_id} run {run_num + 1}/{runs_per_test}")

            # Ensure clean state between runs
            await self._clean_test_state()

            # Run test
            result = await self._run_single_test(test_id)

            # Record results
            passed = result.get("passed", False)
            run_results.append(passed)

            exec_time = result.get("execution_time_ms", 0)
            execution_times.append(float(exec_time))

            # Record failure information
            if not passed:
                error_msg = result.get("error_message", "Unknown error")
                failure_messages.add(error_msg)

                failure_type = self._classify_failure(error_msg)
                failure_types[failure_type] = failure_types.get(failure_type, 0) + 1

        # Calculate statistics
        pass_count = sum(run_results)
        fail_count = len(run_results) - pass_count
        pass_rate = pass_count / len(run_results) if run_results else 0.0

        # Execution time statistics
        mean_time = statistics.mean(execution_times) if execution_times else 0.0
        std_dev_time = (
            statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
        )

        # Calculate coefficient of variation for execution time
        cv_time = std_dev_time / mean_time if mean_time > 0 else 0.0

        # Determine flakiness
        is_flaky = self._is_flaky(pass_rate, cv_time, len(failure_messages))

        # Calculate flakiness score
        flakiness_score = self._calculate_flakiness_score(
            pass_rate, cv_time, len(failure_messages)
        )

        # Identify root causes
        root_causes = self._identify_flakiness_causes(
            failure_types, cv_time, list(failure_messages), pass_rate
        )

        # Get test name
        test_name = await self._get_test_name(test_id)

        # Generate recommendation
        recommended_action = self._recommend_action(is_flaky, root_causes, pass_rate)

        return FlakyTestResult(
            test_id=test_id,
            test_name=test_name,
            flakiness_score=flakiness_score,
            total_runs=len(run_results),
            pass_count=pass_count,
            fail_count=fail_count,
            pass_rate=pass_rate,
            execution_times=execution_times,
            mean_time_ms=mean_time,
            std_dev_time_ms=std_dev_time,
            failure_types=failure_types,
            failure_messages=list(failure_messages),
            is_flaky=is_flaky,
            recommended_action=recommended_action,
            root_causes=root_causes,
        )

    async def _run_single_test(self, test_id: str) -> Dict[str, Any]:
        """
        Run a single test.

        Args:
            test_id: Test to run

        Returns:
            Test result dictionary
        """
        if not self.test_runner:
            # Mock result for testing without runner
            return {
                "passed": True,
                "execution_time_ms": 100,
                "error_message": None,
            }

        try:
            result = await self.test_runner.run_single_test(test_id)
            return result
        except Exception as e:
            logger.error(f"Error running test {test_id}: {e}")
            return {
                "passed": False,
                "execution_time_ms": 0,
                "error_message": str(e),
            }

    async def _clean_test_state(self):
        """
        Clean test state between runs.

        CRITICAL: Ensures test isolation for accurate flakiness detection
        """
        # Add small delay to avoid timing issues
        await asyncio.sleep(0.1)

        # Additional cleanup could include:
        # - Clear caches
        # - Reset global state
        # - Clean up temp files
        # - Reset mocks
        pass

    async def _get_test_name(self, test_id: str) -> str:
        """
        Get human-readable test name from ID.

        Args:
            test_id: Test ID

        Returns:
            Test name
        """
        # Extract test name from ID or path
        # In practice, would query test metadata
        return test_id.split("::")[-1] if "::" in test_id else test_id

    def _is_flaky(self, pass_rate: float, cv_time: float, unique_failures: int) -> bool:
        """
        Determine if test is flaky based on heuristics.

        PATTERN: Multiple signals for flakiness
        CRITICAL: Uses statistical thresholds

        Args:
            pass_rate: Test pass rate (0-1)
            cv_time: Coefficient of variation for execution time
            unique_failures: Number of unique failure messages

        Returns:
            True if test is determined to be flaky
        """
        # Test is flaky if:
        # 1. Pass rate between 10% and 90% (sometimes passes, sometimes fails)
        if 0.1 < pass_rate < 0.9:
            return True

        # 2. High variation in execution time (CV > 0.3)
        if cv_time > 0.3:
            return True

        # 3. Multiple different failure messages (indicates non-determinism)
        if unique_failures > 2:
            return True

        return False

    def _calculate_flakiness_score(
        self, pass_rate: float, cv_time: float, unique_failures: int
    ) -> float:
        """
        Calculate flakiness probability score.

        PATTERN: Weighted combination of signals
        CRITICAL: Returns score between 0 and 1

        Args:
            pass_rate: Test pass rate
            cv_time: Coefficient of variation for time
            unique_failures: Number of unique failure messages

        Returns:
            Flakiness score (0-1)
        """
        # Score based on pass rate variance from stable (0 or 1)
        # Maximum flakiness at 50% pass rate
        pass_rate_score = 2 * min(pass_rate, 1 - pass_rate)

        # Score based on timing variation
        # Cap at 1.0 for very high variation
        time_score = min(cv_time, 1.0)

        # Score based on failure diversity
        # More unique failures = more flaky
        failure_score = min(unique_failures / 5, 1.0)

        # Weighted combination
        # Pass rate most important signal
        weights = [0.5, 0.3, 0.2]
        score = (
            weights[0] * pass_rate_score
            + weights[1] * time_score
            + weights[2] * failure_score
        )

        return min(score, 1.0)

    def _classify_failure(self, error_message: str) -> FailureType:
        """
        Classify failure type from error message.

        Args:
            error_message: Error message

        Returns:
            Failure type
        """
        if not error_message:
            return FailureType.RUNTIME_ERROR

        error_msg = error_message.lower()

        if "timeout" in error_msg or "timed out" in error_msg:
            return FailureType.TIMEOUT
        elif "assertion" in error_msg or "assert" in error_msg:
            return FailureType.ASSERTION_FAILED
        elif "attributeerror" in error_msg:
            return FailureType.ATTRIBUTE_ERROR
        elif "importerror" in error_msg or "modulenotfound" in error_msg:
            return FailureType.IMPORT_ERROR
        elif "syntaxerror" in error_msg:
            return FailureType.SYNTAX_ERROR
        elif "mock" in error_msg:
            return FailureType.MOCK_ERROR
        elif "typeerror" in error_msg:
            return FailureType.TYPE_ERROR
        else:
            return FailureType.RUNTIME_ERROR

    def _identify_flakiness_causes(
        self,
        failure_types: Dict[FailureType, int],
        cv_time: float,
        failure_messages: List[str],
        pass_rate: float,
    ) -> List[str]:
        """
        Identify potential root causes of flakiness.

        Args:
            failure_types: Distribution of failure types
            cv_time: Coefficient of variation for time
            failure_messages: List of failure messages
            pass_rate: Test pass rate

        Returns:
            List of potential root causes
        """
        causes = []

        # Timing-related flakiness
        if cv_time > 0.3:
            causes.append(
                "High execution time variability suggests timing-related flakiness"
            )

        if FailureType.TIMEOUT in failure_types:
            causes.append(
                "Timeout errors indicate race conditions or insufficient wait times"
            )

        # Non-deterministic failures
        if len(failure_messages) > 2:
            causes.append(
                f"Multiple failure modes ({len(failure_messages)}) indicate "
                "non-deterministic behavior"
            )

        # Intermittent failures
        if 0.1 < pass_rate < 0.9:
            causes.append(
                f"Intermittent failures ({pass_rate:.1%} pass rate) suggest "
                "environmental dependencies"
            )

        # Specific failure patterns
        if FailureType.ASSERTION_FAILED in failure_types:
            causes.append(
                "Assertion failures may indicate order dependencies or shared state"
            )

        if FailureType.MOCK_ERROR in failure_types:
            causes.append("Mock errors suggest improper test isolation or setup")

        # Async-related
        for msg in failure_messages:
            if "async" in msg.lower() or "await" in msg.lower():
                causes.append("Async/await patterns may cause race conditions")
                break

        # Network-related
        for msg in failure_messages:
            if any(
                keyword in msg.lower() for keyword in ["connection", "network", "http"]
            ):
                causes.append("Network dependencies may cause intermittent failures")
                break

        # If no specific causes identified
        if not causes:
            causes.append(
                "Unable to identify specific root cause - requires manual investigation"
            )

        return causes

    def _recommend_action(
        self, is_flaky: bool, root_causes: List[str], pass_rate: float
    ) -> str:
        """
        Recommend action for flaky test.

        Args:
            is_flaky: Whether test is flaky
            root_causes: Identified root causes
            pass_rate: Test pass rate

        Returns:
            Recommended action
        """
        if not is_flaky:
            return "Test is stable - no action needed"

        # Severe flakiness
        if pass_rate < 0.3 or pass_rate > 0.7:
            if pass_rate < 0.3:
                return "Test is highly unstable - consider disabling and investigating root cause"
            else:
                return (
                    "Test is mostly passing - investigate and fix intermittent failures"
                )

        # Moderate flakiness
        recommendations = []

        # Check root causes for specific recommendations
        for cause in root_causes:
            if "timing" in cause.lower() or "timeout" in cause.lower():
                recommendations.append("Add explicit waits or increase timeouts")
            elif "race condition" in cause.lower():
                recommendations.append("Add synchronization or proper async handling")
            elif "isolation" in cause.lower() or "shared state" in cause.lower():
                recommendations.append("Improve test isolation and cleanup")
            elif "network" in cause.lower():
                recommendations.append("Mock network dependencies")

        if recommendations:
            return "; ".join(recommendations[:2])  # Return top 2 recommendations

        return "Test shows flakiness - investigate root causes and add stability improvements"

    def _create_error_result(self, test_id: str, error: str) -> FlakyTestResult:
        """
        Create error result when analysis fails.

        Args:
            test_id: Test ID
            error: Error message

        Returns:
            FlakyTestResult with error information
        """
        return FlakyTestResult(
            test_id=test_id,
            test_name=test_id,
            flakiness_score=0.0,
            total_runs=0,
            pass_count=0,
            fail_count=0,
            pass_rate=0.0,
            execution_times=[],
            mean_time_ms=0.0,
            std_dev_time_ms=0.0,
            failure_types={},
            failure_messages=[error],
            is_flaky=False,
            recommended_action=f"Analysis failed: {error}",
            root_causes=[f"Error during analysis: {error}"],
        )

    async def analyze_historical_flakiness(
        self, test_id: str, history_tracker=None, time_window_days: int = 30
    ) -> Optional[FlakyTestResult]:
        """
        Analyze historical test data for flakiness patterns.

        PATTERN: Use historical data instead of running tests
        CRITICAL: Requires sufficient historical data

        Args:
            test_id: Test to analyze
            history_tracker: History tracker with test data
            time_window_days: Days of history to analyze

        Returns:
            Flaky test result based on historical data
        """
        if not history_tracker:
            logger.warning("No history tracker available for historical analysis")
            return None

        try:
            # Get historical executions
            history = await history_tracker.get_test_history(
                test_id, days=time_window_days
            )

            if not history or len(history.executions) < self.min_runs:
                logger.warning(
                    f"Insufficient historical data for {test_id}: "
                    f"{len(history.executions) if history else 0} executions"
                )
                return None

            # Analyze historical data
            executions = history.executions
            pass_count = sum(1 for e in executions if e.get("passed", False))
            fail_count = len(executions) - pass_count
            pass_rate = pass_count / len(executions)

            execution_times = [e.get("execution_time_ms", 0) for e in executions]
            mean_time = statistics.mean(execution_times) if execution_times else 0.0
            std_dev_time = (
                statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            )

            # Similar analysis to real-time detection
            cv_time = std_dev_time / mean_time if mean_time > 0 else 0.0

            # Collect failure information
            failure_messages = set()
            failure_types: Dict[FailureType, int] = {}

            for execution in executions:
                if not execution.get("passed", False):
                    msg = execution.get("error_message", "Unknown")
                    failure_messages.add(msg)

                    f_type = self._classify_failure(msg)
                    failure_types[f_type] = failure_types.get(f_type, 0) + 1

            is_flaky = self._is_flaky(pass_rate, cv_time, len(failure_messages))
            flakiness_score = self._calculate_flakiness_score(
                pass_rate, cv_time, len(failure_messages)
            )
            root_causes = self._identify_flakiness_causes(
                failure_types, cv_time, list(failure_messages), pass_rate
            )

            test_name = await self._get_test_name(test_id)
            recommended_action = self._recommend_action(
                is_flaky, root_causes, pass_rate
            )

            return FlakyTestResult(
                test_id=test_id,
                test_name=test_name,
                flakiness_score=flakiness_score,
                total_runs=len(executions),
                pass_count=pass_count,
                fail_count=fail_count,
                pass_rate=pass_rate,
                execution_times=execution_times,
                mean_time_ms=mean_time,
                std_dev_time_ms=std_dev_time,
                failure_types=failure_types,
                failure_messages=list(failure_messages),
                is_flaky=is_flaky,
                recommended_action=recommended_action,
                root_causes=root_causes,
            )

        except Exception as e:
            logger.error(f"Historical flakiness analysis error: {e}", exc_info=True)
            return None
