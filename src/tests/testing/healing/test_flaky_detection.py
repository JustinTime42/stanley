"""Comprehensive tests for flaky test detection.

Tests FlakyDetector with simulated flaky behavior according to PRP-11.
Includes statistical analysis, edge cases, and various flakiness patterns.
"""

import pytest
import pytest_asyncio
import statistics
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.testing.healing.flaky_detector import FlakyDetector
from src.models.healing_models import (
    FlakyTestResult,
    FailureType,
)


class TestFlakyDetector:
    """Test FlakyDetector component."""

    @pytest.fixture
    def mock_test_runner(self):
        """Create mock test runner."""
        runner = AsyncMock()
        return runner

    @pytest.fixture
    def detector(self, mock_test_runner):
        """Create flaky detector instance."""
        return FlakyDetector(test_runner=mock_test_runner, min_runs=10)

    @pytest.fixture
    def detector_no_runner(self):
        """Create flaky detector without test runner."""
        return FlakyDetector(test_runner=None, min_runs=10)

    @pytest.mark.asyncio
    async def test_detect_flaky_tests_basic(self, detector, mock_test_runner):
        """Test basic flaky test detection."""
        # Simulate 50% pass rate (flaky)
        mock_test_runner.run_single_test = AsyncMock(
            side_effect=[
                {"passed": True, "execution_time_ms": 100, "error_message": None},
                {"passed": False, "execution_time_ms": 100, "error_message": "Error"},
            ] * 5
        )

        results = await detector.detect_flaky_tests(["test_001"], runs_per_test=10)

        assert len(results) == 1
        result = results[0]
        assert result.test_id == "test_001"
        assert result.total_runs == 10
        assert result.pass_count == 5
        assert result.fail_count == 5
        assert result.pass_rate == 0.5
        assert result.is_flaky is True

    @pytest.mark.asyncio
    async def test_detect_flaky_tests_stable_passing(self, detector, mock_test_runner):
        """Test detection of stable passing test."""
        # Simulate 100% pass rate (stable)
        mock_test_runner.run_single_test = AsyncMock(
            return_value={"passed": True, "execution_time_ms": 100, "error_message": None}
        )

        results = await detector.detect_flaky_tests(["test_stable"], runs_per_test=10)

        assert len(results) == 1
        result = results[0]
        assert result.pass_rate == 1.0
        assert result.is_flaky is False
        assert result.flakiness_score < 0.3

    @pytest.mark.asyncio
    async def test_detect_flaky_tests_stable_failing(self, detector, mock_test_runner):
        """Test detection of stable failing test."""
        # Simulate 0% pass rate (consistently failing)
        mock_test_runner.run_single_test = AsyncMock(
            return_value={
                "passed": False,
                "execution_time_ms": 100,
                "error_message": "Consistent error"
            }
        )

        results = await detector.detect_flaky_tests(["test_failing"], runs_per_test=10)

        assert len(results) == 1
        result = results[0]
        assert result.pass_rate == 0.0
        assert result.is_flaky is False

    @pytest.mark.asyncio
    async def test_detect_flaky_tests_timing_variance(self, detector, mock_test_runner):
        """Test detection of flaky test with timing variance."""
        # Simulate high timing variance
        times = [100, 500, 150, 600, 200, 550, 180, 580, 190, 570]
        call_count = 0

        async def variable_timing_run(test_id):
            nonlocal call_count
            time = times[call_count % len(times)]
            call_count += 1
            return {"passed": True, "execution_time_ms": time, "error_message": None}

        mock_test_runner.run_single_test = variable_timing_run

        results = await detector.detect_flaky_tests(["test_timing"], runs_per_test=10)

        assert len(results) == 1
        result = results[0]
        # High coefficient of variation should indicate flakiness
        cv = result.std_dev_time_ms / result.mean_time_ms if result.mean_time_ms > 0 else 0
        assert cv > 0.3 or result.is_flaky

    @pytest.mark.asyncio
    async def test_detect_flaky_tests_multiple_failure_types(self, detector, mock_test_runner):
        """Test detection with multiple different failure types."""
        # Simulate different failure messages (non-deterministic)
        failures = [
            "TimeoutError",
            "AssertionError",
            "RuntimeError",
            "TimeoutError",
            "AssertionError",
        ]
        passes = [None] * 5
        all_results = [
            {"passed": False, "execution_time_ms": 100, "error_message": msg}
            for msg in failures
        ] + [
            {"passed": True, "execution_time_ms": 100, "error_message": msg}
            for msg in passes
        ]

        call_count = 0
        async def multiple_failure_run(test_id):
            nonlocal call_count
            result = all_results[call_count % len(all_results)]
            call_count += 1
            return result

        mock_test_runner.run_single_test = multiple_failure_run

        results = await detector.detect_flaky_tests(["test_multi"], runs_per_test=10)

        assert len(results) == 1
        result = results[0]
        assert len(result.failure_messages) > 1
        assert result.is_flaky is True

    @pytest.mark.asyncio
    async def test_detect_flaky_tests_multiple_tests(self, detector, mock_test_runner):
        """Test detection of multiple tests at once."""
        async def multi_test_run(test_id):
            if "flaky" in test_id:
                return {
                    "passed": call_count % 2 == 0,
                    "execution_time_ms": 100,
                    "error_message": "Intermittent" if call_count % 2 else None
                }
            else:
                return {"passed": True, "execution_time_ms": 100, "error_message": None}

        call_count = 0
        async def counting_run(test_id):
            nonlocal call_count
            result = await multi_test_run(test_id)
            call_count += 1
            return result

        mock_test_runner.run_single_test = counting_run

        results = await detector.detect_flaky_tests(
            ["test_flaky", "test_stable"],
            runs_per_test=10
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_detect_flaky_tests_error_handling(self, detector, mock_test_runner):
        """Test error handling during detection."""
        mock_test_runner.run_single_test = AsyncMock(
            side_effect=Exception("Test runner error")
        )

        results = await detector.detect_flaky_tests(["test_error"], runs_per_test=10)

        assert len(results) == 1
        result = results[0]
        # Should return error result
        assert "Error during analysis" in result.root_causes[0]
        assert result.is_flaky is False

    @pytest.mark.asyncio
    async def test_detect_flaky_tests_min_runs_warning(self, detector, mock_test_runner):
        """Test warning when runs less than minimum."""
        mock_test_runner.run_single_test = AsyncMock(
            return_value={"passed": True, "execution_time_ms": 100, "error_message": None}
        )

        # Should log warning but still execute
        results = await detector.detect_flaky_tests(["test_low_runs"], runs_per_test=5)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_detect_flaky_tests_without_runner(self, detector_no_runner):
        """Test detection without test runner (uses mock results)."""
        results = await detector_no_runner.detect_flaky_tests(
            ["test_no_runner"],
            runs_per_test=10
        )

        assert len(results) == 1
        # Should use mock passing results
        result = results[0]
        assert result.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_analyze_single_test_statistics(self, detector, mock_test_runner):
        """Test statistical calculations in single test analysis."""
        # Controlled execution times
        times = [100, 110, 95, 105, 100, 98, 102, 100, 105, 95]

        call_count = 0
        async def timed_run(test_id):
            nonlocal call_count
            time = times[call_count]
            call_count += 1
            return {"passed": True, "execution_time_ms": time, "error_message": None}

        mock_test_runner.run_single_test = timed_run

        results = await detector.detect_flaky_tests(["test_stats"], runs_per_test=10)

        result = results[0]
        expected_mean = statistics.mean(times)
        expected_std = statistics.stdev(times)

        assert abs(result.mean_time_ms - expected_mean) < 0.1
        assert abs(result.std_dev_time_ms - expected_std) < 0.1

    @pytest.mark.asyncio
    async def test_clean_test_state(self, detector):
        """Test test state cleaning between runs."""
        # Should add small delay for test isolation
        await detector._clean_test_state()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_get_test_name_from_id(self, detector):
        """Test test name extraction from test ID."""
        test_name = await detector._get_test_name("path/to/test.py::TestClass::test_method")
        assert test_name == "test_method"

    @pytest.mark.asyncio
    async def test_get_test_name_simple_id(self, detector):
        """Test test name extraction from simple ID."""
        test_name = await detector._get_test_name("test_simple")
        assert test_name == "test_simple"

    def test_is_flaky_by_pass_rate(self, detector):
        """Test flakiness detection by pass rate."""
        # 50% pass rate - clearly flaky
        assert detector._is_flaky(0.5, 0.1, 1) is True

        # 20% pass rate - flaky
        assert detector._is_flaky(0.2, 0.1, 1) is True

        # 80% pass rate - flaky
        assert detector._is_flaky(0.8, 0.1, 1) is True

        # 100% pass rate - stable
        assert detector._is_flaky(1.0, 0.1, 1) is False

        # 0% pass rate - stable (consistently failing)
        assert detector._is_flaky(0.0, 0.1, 1) is False

    def test_is_flaky_by_timing_variance(self, detector):
        """Test flakiness detection by timing variance."""
        # High CV - flaky
        assert detector._is_flaky(1.0, 0.5, 1) is True

        # Low CV - stable
        assert detector._is_flaky(1.0, 0.1, 1) is False

    def test_is_flaky_by_failure_diversity(self, detector):
        """Test flakiness detection by failure diversity."""
        # Multiple unique failures - flaky
        assert detector._is_flaky(0.5, 0.1, 3) is True

        # Single failure type - less likely flaky
        assert detector._is_flaky(0.5, 0.1, 1) is True  # Still flaky due to pass rate

    def test_calculate_flakiness_score_perfect_pass(self, detector):
        """Test flakiness score for perfectly passing test."""
        score = detector._calculate_flakiness_score(1.0, 0.05, 0)
        # Should be very low
        assert score < 0.3

    def test_calculate_flakiness_score_perfect_fail(self, detector):
        """Test flakiness score for perfectly failing test."""
        score = detector._calculate_flakiness_score(0.0, 0.05, 1)
        # Should be low (consistent behavior)
        assert score < 0.5

    def test_calculate_flakiness_score_fifty_percent(self, detector):
        """Test flakiness score for 50% pass rate."""
        score = detector._calculate_flakiness_score(0.5, 0.1, 1)
        # Should be high (maximum pass rate variance)
        assert score > 0.4

    def test_calculate_flakiness_score_high_timing_variance(self, detector):
        """Test flakiness score with high timing variance."""
        score = detector._calculate_flakiness_score(1.0, 0.8, 0)
        # Should be elevated due to timing
        assert score > 0.2

    def test_calculate_flakiness_score_multiple_failures(self, detector):
        """Test flakiness score with multiple failure types."""
        score = detector._calculate_flakiness_score(0.5, 0.2, 5)
        # Should be very high
        assert score > 0.5

    def test_classify_failure_timeout(self, detector):
        """Test classification of timeout errors."""
        failure_type = detector._classify_failure("TimeoutError: test timed out")
        assert failure_type == FailureType.TIMEOUT

    def test_classify_failure_assertion(self, detector):
        """Test classification of assertion errors."""
        failure_type = detector._classify_failure("AssertionError: expected True")
        assert failure_type == FailureType.ASSERTION_FAILED

    def test_classify_failure_attribute_error(self, detector):
        """Test classification of attribute errors."""
        failure_type = detector._classify_failure("AttributeError: no attribute 'foo'")
        assert failure_type == FailureType.ATTRIBUTE_ERROR

    def test_classify_failure_import_error(self, detector):
        """Test classification of import errors."""
        failure_type = detector._classify_failure("ImportError: no module named 'foo'")
        assert failure_type == FailureType.IMPORT_ERROR

        failure_type = detector._classify_failure("ModuleNotFoundError: No module 'bar'")
        assert failure_type == FailureType.IMPORT_ERROR

    def test_classify_failure_syntax_error(self, detector):
        """Test classification of syntax errors."""
        failure_type = detector._classify_failure("SyntaxError: invalid syntax")
        assert failure_type == FailureType.SYNTAX_ERROR

    def test_classify_failure_mock_error(self, detector):
        """Test classification of mock errors."""
        failure_type = detector._classify_failure("MockError: mock.assert_called failed")
        assert failure_type == FailureType.MOCK_ERROR

    def test_classify_failure_type_error(self, detector):
        """Test classification of type errors."""
        failure_type = detector._classify_failure("TypeError: unsupported type")
        assert failure_type == FailureType.TYPE_ERROR

    def test_classify_failure_unknown(self, detector):
        """Test classification of unknown errors."""
        failure_type = detector._classify_failure("WeirdError: something happened")
        assert failure_type == FailureType.RUNTIME_ERROR

    def test_identify_flakiness_causes_timing(self, detector):
        """Test identification of timing-related flakiness."""
        causes = detector._identify_flakiness_causes(
            {FailureType.TIMEOUT: 5},
            0.5,  # High CV
            ["Timeout"],
            0.5
        )

        assert any("timing" in c.lower() or "timeout" in c.lower() for c in causes)

    def test_identify_flakiness_causes_multiple_failures(self, detector):
        """Test identification of non-deterministic behavior."""
        causes = detector._identify_flakiness_causes(
            {FailureType.ASSERTION_FAILED: 3, FailureType.RUNTIME_ERROR: 2},
            0.2,
            ["Error 1", "Error 2", "Error 3"],
            0.5
        )

        assert any("multiple failure modes" in c.lower() for c in causes)

    def test_identify_flakiness_causes_intermittent(self, detector):
        """Test identification of intermittent failures."""
        causes = detector._identify_flakiness_causes(
            {FailureType.ASSERTION_FAILED: 5},
            0.1,
            ["Error"],
            0.4
        )

        assert any("intermittent" in c.lower() for c in causes)

    def test_identify_flakiness_causes_async(self, detector):
        """Test identification of async-related flakiness."""
        causes = detector._identify_flakiness_causes(
            {FailureType.RUNTIME_ERROR: 5},
            0.2,
            ["Error with await keyword"],
            0.5
        )

        assert any("async" in c.lower() for c in causes)

    def test_identify_flakiness_causes_network(self, detector):
        """Test identification of network-related flakiness."""
        causes = detector._identify_flakiness_causes(
            {FailureType.RUNTIME_ERROR: 5},
            0.2,
            ["Connection error occurred"],
            0.5
        )

        assert any("network" in c.lower() for c in causes)

    def test_identify_flakiness_causes_mock_isolation(self, detector):
        """Test identification of mock/isolation issues."""
        causes = detector._identify_flakiness_causes(
            {FailureType.MOCK_ERROR: 5},
            0.2,
            ["Mock assertion failed"],
            0.5
        )

        assert any("isolation" in c.lower() or "mock" in c.lower() for c in causes)

    def test_identify_flakiness_causes_none_found(self, detector):
        """Test when no specific causes identified."""
        causes = detector._identify_flakiness_causes(
            {FailureType.RUNTIME_ERROR: 1},
            0.1,
            ["Error"],
            1.0
        )

        assert any("manual investigation" in c.lower() for c in causes)

    def test_recommend_action_stable_test(self, detector):
        """Test recommendation for stable test."""
        action = detector._recommend_action(False, [], 1.0)
        assert "no action needed" in action.lower()

    def test_recommend_action_highly_unstable(self, detector):
        """Test recommendation for highly unstable test."""
        action = detector._recommend_action(True, [], 0.2)
        assert "disabling" in action.lower() or "unstable" in action.lower()

    def test_recommend_action_mostly_passing(self, detector):
        """Test recommendation for mostly passing flaky test."""
        action = detector._recommend_action(True, [], 0.8)
        assert "investigate" in action.lower() or "fix" in action.lower()

    def test_recommend_action_timing_issues(self, detector):
        """Test recommendation for timing-related flakiness."""
        causes = ["High execution time variability suggests timing-related flakiness"]
        action = detector._recommend_action(True, causes, 0.5)

        assert "wait" in action.lower() or "timeout" in action.lower()

    def test_recommend_action_race_conditions(self, detector):
        """Test recommendation for race conditions."""
        causes = ["Async patterns may cause race conditions"]
        action = detector._recommend_action(True, causes, 0.5)

        assert "synchronization" in action.lower() or "async" in action.lower()

    def test_recommend_action_isolation_issues(self, detector):
        """Test recommendation for isolation issues."""
        causes = ["Tests show improper isolation or shared state"]
        action = detector._recommend_action(True, causes, 0.5)

        assert "isolation" in action.lower() or "cleanup" in action.lower()

    def test_recommend_action_network_issues(self, detector):
        """Test recommendation for network dependencies."""
        causes = ["Network dependencies may cause intermittent failures"]
        action = detector._recommend_action(True, causes, 0.5)

        assert "mock" in action.lower() or "network" in action.lower()

    def test_recommend_action_generic(self, detector):
        """Test generic recommendation when no specific cause."""
        causes = ["Unknown issue"]
        action = detector._recommend_action(True, causes, 0.5)

        assert "investigate" in action.lower() or "stability" in action.lower()

    def test_create_error_result(self, detector):
        """Test creation of error result."""
        result = detector._create_error_result("test_error", "Something went wrong")

        assert result.test_id == "test_error"
        assert result.flakiness_score == 0.0
        assert result.is_flaky is False
        assert "Something went wrong" in result.failure_messages[0]
        assert "Analysis failed" in result.recommended_action

    @pytest.mark.asyncio
    async def test_analyze_historical_flakiness_no_tracker(self, detector):
        """Test historical analysis without history tracker."""
        result = await detector.analyze_historical_flakiness("test_001", None, 30)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_historical_flakiness_insufficient_data(self, detector):
        """Test historical analysis with insufficient data."""
        mock_tracker = AsyncMock()
        mock_history = Mock()
        mock_history.executions = [{"passed": True}] * 5  # Less than min_runs

        mock_tracker.get_test_history = AsyncMock(return_value=mock_history)

        result = await detector.analyze_historical_flakiness(
            "test_001", mock_tracker, 30
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_historical_flakiness_success(self, detector):
        """Test successful historical analysis."""
        mock_tracker = AsyncMock()
        mock_history = Mock()

        # Create execution history with flaky pattern
        mock_history.executions = [
            {"passed": i % 2 == 0, "execution_time_ms": 100 + (i * 10), "error_message": "Error" if i % 2 else None}
            for i in range(20)
        ]

        mock_tracker.get_test_history = AsyncMock(return_value=mock_history)

        result = await detector.analyze_historical_flakiness(
            "test_historical", mock_tracker, 30
        )

        assert result is not None
        assert result.test_id == "test_historical"
        assert result.total_runs == 20
        assert result.pass_rate == 0.5
        assert result.is_flaky is True

    @pytest.mark.asyncio
    async def test_analyze_historical_flakiness_error(self, detector):
        """Test historical analysis error handling."""
        mock_tracker = AsyncMock()
        mock_tracker.get_test_history = AsyncMock(side_effect=Exception("Tracker error"))

        result = await detector.analyze_historical_flakiness(
            "test_error", mock_tracker, 30
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_historical_flakiness_stable_test(self, detector):
        """Test historical analysis of stable test."""
        mock_tracker = AsyncMock()
        mock_history = Mock()

        # All passing
        mock_history.executions = [
            {"passed": True, "execution_time_ms": 100, "error_message": None}
            for _ in range(20)
        ]

        mock_tracker.get_test_history = AsyncMock(return_value=mock_history)

        result = await detector.analyze_historical_flakiness(
            "test_stable", mock_tracker, 30
        )

        assert result is not None
        assert result.pass_rate == 1.0
        assert result.is_flaky is False
        assert result.flakiness_score < 0.3


class TestFlakyDetectorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def detector(self):
        """Create detector without runner for edge case testing."""
        return FlakyDetector(test_runner=None, min_runs=10)

    @pytest.mark.asyncio
    async def test_empty_test_list(self, detector):
        """Test detection with empty test list."""
        results = await detector.detect_flaky_tests([], runs_per_test=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_single_run(self, detector):
        """Test detection with single run (insufficient)."""
        results = await detector.detect_flaky_tests(["test_single"], runs_per_test=1)
        # Should still run but results may be unreliable
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_very_large_runs(self, detector):
        """Test detection with very large number of runs."""
        results = await detector.detect_flaky_tests(["test_large"], runs_per_test=100)
        assert len(results) == 1
        assert results[0].total_runs == 100

    def test_calculate_flakiness_zero_values(self, detector):
        """Test flakiness calculation with zero values."""
        score = detector._calculate_flakiness_score(0.0, 0.0, 0)
        assert 0.0 <= score <= 1.0

    def test_identify_causes_empty_failures(self, detector):
        """Test cause identification with no failures."""
        causes = detector._identify_flakiness_causes({}, 0.0, [], 1.0)
        assert len(causes) > 0  # Should still provide some guidance

    def test_is_flaky_edge_values(self, detector):
        """Test flakiness detection with edge values."""
        # Exactly at threshold
        assert detector._is_flaky(0.1, 0.0, 0) is False
        assert detector._is_flaky(0.9, 0.0, 0) is False

        # Just inside threshold
        assert detector._is_flaky(0.11, 0.0, 0) is True
        assert detector._is_flaky(0.89, 0.0, 0) is True

    @pytest.mark.asyncio
    async def test_run_single_test_error_handling(self, detector):
        """Test error handling in single test run."""
        result = await detector._run_single_test("test_no_runner")
        # Should return mock passing result
        assert result["passed"] is True
