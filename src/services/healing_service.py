"""High-level self-healing test service orchestrator.

This service provides a facade for coordinating all test healing components:
- Failure analysis and root cause detection
- Automatic test repair with validation
- Flaky test detection through statistical analysis
- Test optimization suggestions
- Historical performance tracking

PATTERN: Service facade pattern for orchestrating healing workflow
CRITICAL: Must validate repairs before applying to prevent regression
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..models.healing_models import (
    TestFailure,
    FailureAnalysis,
    TestRepair,
    FlakyTestResult,
    TestOptimization,
    TestPerformanceHistory,
    HealingRequest,
    HealingResult,
    RepairStrategy,
)
from ..testing.healing import (
    FailureAnalyzer,
    TestRepairer,
    FlakyDetector,
    TestOptimizer,
    HistoryTracker,
)

logger = logging.getLogger(__name__)


class HealingOrchestrator:
    """
    High-level self-healing test service.

    PATTERN: Service facade for test healing operations
    CRITICAL: Orchestrates analyze → repair → validate → optimize workflow
    GOTCHA: Must ensure repairs maintain test validity and coverage

    This orchestrator coordinates all healing components to provide
    comprehensive test maintenance automation. It handles failure analysis,
    automatic repairs, flaky test detection, optimization suggestions, and
    maintains historical performance tracking.
    """

    def __init__(
        self,
        failure_analyzer: Optional[FailureAnalyzer] = None,
        test_repairer: Optional[TestRepairer] = None,
        flaky_detector: Optional[FlakyDetector] = None,
        test_optimizer: Optional[TestOptimizer] = None,
        history_tracker: Optional[HistoryTracker] = None,
        ast_parser: Optional[Any] = None,
        llm_service: Optional[Any] = None,
        memory_service: Optional[Any] = None,
    ):
        """
        Initialize healing orchestrator.

        Args:
            failure_analyzer: Component for analyzing test failures
            test_repairer: Component for repairing failing tests
            flaky_detector: Component for detecting flaky tests
            test_optimizer: Component for suggesting optimizations
            history_tracker: Component for tracking test history
            ast_parser: Optional AST parser for code analysis
            llm_service: Optional LLM service for complex repairs
            memory_service: Optional memory service for historical data
        """
        self.logger = logger

        # Initialize healing components
        self.failure_analyzer = failure_analyzer or FailureAnalyzer(
            ast_parser=ast_parser
        )
        self.test_repairer = test_repairer or TestRepairer(llm_service=llm_service)
        self.flaky_detector = flaky_detector or FlakyDetector()
        self.test_optimizer = test_optimizer or TestOptimizer()
        self.history_tracker = history_tracker or HistoryTracker(
            memory_service=memory_service
        )

        # Store services for potential use
        self.ast_parser = ast_parser
        self.llm_service = llm_service
        self.memory_service = memory_service

        self.logger.info("HealingOrchestrator initialized")

    async def heal_tests(self, request: HealingRequest) -> HealingResult:
        """
        Execute complete test healing workflow.

        This is the main entry point for test healing. It orchestrates the full
        workflow: analyze failures → repair tests → validate repairs →
        detect flaky tests → suggest optimizations.

        Args:
            request: Healing request with failed tests and configuration

        Returns:
            HealingResult with all repairs, flaky tests, and optimizations

        Workflow:
            1. Analyze each failure to identify root cause
            2. Attempt repairs using appropriate strategies
            3. Validate repairs maintain test validity
            4. Detect flaky tests if requested
            5. Suggest optimizations if requested
            6. Update historical tracking
        """
        start_time = datetime.now()
        self.logger.info(
            f"Starting healing process for {len(request.test_failures)} failures"
        )

        try:
            # Step 1: Analyze all failures
            self.logger.info("Step 1: Analyzing test failures")
            failure_analyses = await self._analyze_failures(request.test_failures)

            # Step 2: Repair tests based on analysis
            self.logger.info("Step 2: Repairing failing tests")
            repairs = await self._repair_tests(
                failure_analyses,
                auto_repair=request.auto_repair,
                max_attempts=request.max_repair_attempts,
                confidence_threshold=request.confidence_threshold,
            )

            # Step 3: Detect flaky tests if requested
            flaky_tests = []
            if request.detect_flaky:
                self.logger.info("Step 3: Detecting flaky tests")
                flaky_tests = await self._detect_flaky_tests(
                    request.test_failures, repairs
                )

            # Step 4: Generate optimization suggestions if requested
            optimizations = []
            if request.optimize:
                self.logger.info("Step 4: Generating optimization suggestions")
                optimizations = await self._optimize_tests(
                    request.test_failures, repairs
                )

            # Step 5: Update historical tracking
            self.logger.info("Step 5: Updating historical tracking")
            await self._update_history(request.test_failures, repairs)

            # Calculate metrics
            successful_repairs = sum(1 for r in repairs if r.test_passes)
            failed_repairs = len(repairs) - successful_repairs
            repair_success_rate = (
                successful_repairs / len(repairs) if repairs else 0.0
            )
            total_time_saved = sum(opt.time_saving_ms for opt in optimizations)

            # Calculate total healing time
            healing_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            result = HealingResult(
                total_failures=len(request.test_failures),
                successful_repairs=successful_repairs,
                failed_repairs=failed_repairs,
                flaky_tests_detected=len(flaky_tests),
                repairs=repairs,
                flaky_tests=flaky_tests,
                optimizations=optimizations,
                repair_success_rate=repair_success_rate,
                total_time_saved_ms=total_time_saved,
                healing_time_ms=healing_time_ms,
            )

            self.logger.info(
                f"Healing complete: {successful_repairs}/{len(request.test_failures)} "
                f"tests repaired in {healing_time_ms}ms"
            )

            return result

        except Exception as e:
            self.logger.error(f"Healing process failed: {e}", exc_info=True)
            # Return empty result on catastrophic failure
            healing_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return HealingResult(
                total_failures=len(request.test_failures),
                successful_repairs=0,
                failed_repairs=len(request.test_failures),
                flaky_tests_detected=0,
                repairs=[],
                flaky_tests=[],
                optimizations=[],
                repair_success_rate=0.0,
                total_time_saved_ms=0.0,
                healing_time_ms=healing_time_ms,
            )

    async def _analyze_failures(
        self, failures: List[TestFailure]
    ) -> List[FailureAnalysis]:
        """
        Analyze all test failures to identify root causes.

        PATTERN: Parallel analysis for performance
        CRITICAL: Must handle analysis failures gracefully

        Args:
            failures: List of test failures to analyze

        Returns:
            List of failure analyses with root causes and suggested strategies
        """
        analyses = []

        # Analyze failures in parallel for better performance
        analysis_tasks = [
            self.failure_analyzer.analyze_failure(failure) for failure in failures
        ]

        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        for failure, result in zip(failures, results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Failed to analyze {failure.test_id}: {result}"
                )
                # Create fallback analysis
                analyses.append(
                    FailureAnalysis(
                        failure=failure,
                        root_cause="Analysis failed",
                        confidence=0.3,
                        suggested_strategies=[RepairStrategy.REGENERATE],
                    )
                )
            elif isinstance(result, FailureAnalysis):
                analyses.append(result)

        return analyses

    async def _repair_tests(
        self,
        analyses: List[FailureAnalysis],
        auto_repair: bool,
        max_attempts: int,
        confidence_threshold: float,
    ) -> List[TestRepair]:
        """
        Repair tests based on failure analyses.

        PATTERN: Filtered repair based on confidence threshold
        CRITICAL: Must validate repairs before applying

        Args:
            analyses: Failure analyses with repair strategies
            auto_repair: Whether to automatically apply repairs
            max_attempts: Maximum repair attempts per test
            confidence_threshold: Minimum confidence for auto-repair

        Returns:
            List of test repairs (successful and failed)
        """
        repairs = []

        for analysis in analyses:
            try:
                # Skip low-confidence repairs unless forced
                if not auto_repair and analysis.confidence < confidence_threshold:
                    self.logger.warning(
                        f"Skipping repair for {analysis.failure.test_id}: "
                        f"confidence {analysis.confidence} below threshold"
                    )
                    continue

                # Attempt repair
                repair = await self.test_repairer.repair_test(
                    analysis, max_attempts=max_attempts
                )

                # Only add successful repairs
                if repair is not None:
                    repairs.append(repair)

                    if repair.test_passes:
                        self.logger.info(
                            f"Successfully repaired {analysis.failure.test_id} "
                            f"using {repair.strategy}"
                        )
                    else:
                        self.logger.warning(
                            f"Repair failed for {analysis.failure.test_id}"
                        )
                else:
                    self.logger.warning(
                        f"No repair generated for {analysis.failure.test_id}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error repairing {analysis.failure.test_id}: {e}",
                    exc_info=True,
                )

        return repairs

    async def _detect_flaky_tests(
        self, failures: List[TestFailure], repairs: List[TestRepair]
    ) -> List[FlakyTestResult]:
        """
        Detect flaky tests through statistical analysis.

        PATTERN: Run tests multiple times to detect non-determinism
        CRITICAL: Requires sufficient test runs for statistical significance

        Args:
            failures: Original test failures
            repairs: Test repairs that were attempted

        Returns:
            List of flaky test detection results
        """
        flaky_tests = []

        # Identify tests that failed intermittently or after repair
        suspect_test_ids = set()

        # Add tests with inconsistent repair results
        for repair in repairs:
            if not repair.test_passes and repair.confidence > 0.7:
                # High confidence repair failed - might be flaky
                suspect_test_ids.add(repair.failure_analysis.failure.test_id)

        # Add tests that were repaired but have low coverage
        for repair in repairs:
            if repair.test_passes and not repair.coverage_maintained:
                suspect_test_ids.add(repair.failure_analysis.failure.test_id)

        if suspect_test_ids:
            try:
                flaky_results = await self.flaky_detector.detect_flaky_tests(
                    list(suspect_test_ids)
                )
                flaky_tests.extend(flaky_results)

                self.logger.info(
                    f"Detected {len([f for f in flaky_results if f.is_flaky])} "
                    f"flaky tests out of {len(suspect_test_ids)} suspects"
                )

            except Exception as e:
                self.logger.error(f"Flaky detection failed: {e}", exc_info=True)

        return flaky_tests

    async def _optimize_tests(
        self, failures: List[TestFailure], repairs: List[TestRepair]
    ) -> List[TestOptimization]:
        """
        Generate test optimization suggestions.

        PATTERN: Analyze performance patterns and suggest improvements
        CRITICAL: Balance speed with test effectiveness

        Args:
            failures: Original test failures
            repairs: Test repairs that were performed

        Returns:
            List of optimization suggestions
        """
        optimizations = []

        try:
            # Collect test IDs and performance data
            test_performance = {}

            for failure in failures:
                if failure.execution_time_ms:
                    test_performance[failure.test_id] = {
                        "execution_time_ms": failure.execution_time_ms,
                        "test_file": failure.test_file,
                    }

            # Generate optimization suggestions
            if test_performance:
                # Use first test file as suite identifier
                test_suite = next(iter(failures)).test_file if failures else "test_suite"
                opts = await self.test_optimizer.suggest_optimizations(
                    test_suite=test_suite,
                    performance_data=test_performance
                )
                optimizations.extend(opts)

                total_savings = sum(opt.time_saving_ms for opt in opts)
                self.logger.info(
                    f"Generated {len(opts)} optimization suggestions "
                    f"with potential {total_savings}ms savings"
                )

        except Exception as e:
            self.logger.error(f"Optimization analysis failed: {e}", exc_info=True)

        return optimizations

    async def _update_history(
        self, failures: List[TestFailure], repairs: List[TestRepair]
    ) -> None:
        """
        Update historical performance tracking.

        PATTERN: Record all executions for trend analysis
        CRITICAL: Efficient storage to prevent data bloat

        Args:
            failures: Test failures to record
            repairs: Test repairs to record
        """
        try:
            # Record test failures
            for failure in failures:
                execution_data = {
                    "timestamp": failure.timestamp.isoformat(),
                    "passed": False,
                    "failure_type": failure.failure_type.value,
                    "execution_time_ms": failure.execution_time_ms,
                    "error_message": failure.error_message,
                }

                await self.history_tracker.record_execution(
                    failure.test_id, execution_data
                )

            # Record repair attempts
            for repair in repairs:
                execution_data = {
                    "timestamp": repair.created_at.isoformat(),
                    "passed": repair.test_passes,
                    "repair_strategy": repair.strategy.value,
                    "repair_time_ms": repair.repair_time_ms,
                    "confidence": repair.confidence,
                }

                await self.history_tracker.record_execution(
                    repair.failure_analysis.failure.test_id, execution_data
                )

            self.logger.debug("Historical tracking updated")

        except Exception as e:
            # Don't fail healing if history tracking fails
            self.logger.error(f"Failed to update history: {e}", exc_info=True)

    async def analyze_test_health(
        self, test_ids: List[str], time_window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze overall test health and trends.

        This method provides insights into test performance over time,
        helping predict future failures and maintenance needs.

        Args:
            test_ids: Tests to analyze
            time_window_days: Historical time window in days

        Returns:
            Dictionary with health metrics and predictions
        """
        self.logger.info(f"Analyzing health for {len(test_ids)} tests")

        # Use typed list for test histories
        test_histories: List[TestPerformanceHistory] = []
        health_report: Dict[str, Any] = {
            "total_tests": len(test_ids),
            "test_histories": test_histories,
            "overall_metrics": {},
            "predictions": [],
        }

        try:
            # Analyze each test's history
            for test_id in test_ids:
                history = await self.history_tracker.analyze_trends(
                    test_id, time_window=time_window_days
                )
                if history is not None:
                    test_histories.append(history)

            # Calculate overall metrics
            overall_metrics: Dict[str, Any] = {}
            predictions: List[Dict[str, Any]] = []

            if test_histories:
                total_executions = sum(h.total_executions for h in test_histories)
                total_failures = sum(h.total_failures for h in test_histories)
                avg_failure_rate = (
                    sum(h.failure_rate for h in test_histories)
                    / len(test_histories)
                )

                overall_metrics = {
                    "total_executions": total_executions,
                    "total_failures": total_failures,
                    "average_failure_rate": avg_failure_rate,
                    "tests_needing_maintenance": sum(
                        1
                        for h in test_histories
                        if h.predicted_maintenance_needed
                    ),
                }

                # Identify tests likely to fail soon
                predictions = [
                    {
                        "test_id": h.test_id,
                        "failure_probability": h.predicted_failure_probability,
                        "maintenance_needed": h.predicted_maintenance_needed,
                    }
                    for h in test_histories
                    if h.predicted_failure_probability > 0.3
                ]

                # Update health report with calculated values
                health_report["overall_metrics"] = overall_metrics
                health_report["predictions"] = predictions

            self.logger.info(
                f"Health analysis complete: "
                f"{health_report['overall_metrics'].get('tests_needing_maintenance', 0)} "
                f"tests need maintenance"
            )

        except Exception as e:
            self.logger.error(f"Health analysis failed: {e}", exc_info=True)
            health_report["error"] = str(e)

        return health_report

    async def get_repair_recommendations(
        self, failure: TestFailure, code_diff: Optional[Dict[str, Any]] = None
    ) -> FailureAnalysis:
        """
        Get repair recommendations for a single test failure.

        This is a convenience method for analyzing a single failure without
        performing the full healing workflow.

        Args:
            failure: Test failure to analyze
            code_diff: Optional code changes that may have caused the failure

        Returns:
            Failure analysis with repair recommendations
        """
        self.logger.info(f"Generating repair recommendations for {failure.test_id}")

        try:
            analysis = await self.failure_analyzer.analyze_failure(
                failure, code_diff=code_diff
            )

            self.logger.info(
                f"Identified root cause: {analysis.root_cause} "
                f"(confidence: {analysis.confidence})"
            )

            return analysis

        except Exception as e:
            self.logger.error(
                f"Failed to analyze {failure.test_id}: {e}", exc_info=True
            )
            # Return low-confidence fallback
            return FailureAnalysis(
                failure=failure,
                root_cause="Analysis failed",
                confidence=0.3,
                suggested_strategies=[RepairStrategy.REGENERATE],
                evidence=[f"Error: {str(e)}"],
            )
