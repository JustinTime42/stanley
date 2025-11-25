"""Historical test performance tracking.

PATTERN: Track test performance over time
CRITICAL: Efficient storage and retrieval of time-series data
"""

import logging
import json
import statistics
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ...models.healing_models import (
    TestPerformanceHistory,
    TestRepair,
    FailureType,
)

logger = logging.getLogger(__name__)


class HistoryTracker:
    """
    Track test performance over time.

    PATTERN: Time-series data with aggregation
    CRITICAL: Efficient storage for large history
    GOTCHA: Implement data retention policies to prevent unlimited growth
    """

    def __init__(
        self,
        memory_service=None,
        redis_client=None,
        retention_days: int = 90,
    ):
        """
        Initialize history tracker.

        Args:
            memory_service: Memory service for long-term storage
            redis_client: Redis client for recent data
            retention_days: Number of days to retain detailed history
        """
        self.memory_service = memory_service
        self.redis = redis_client
        self.retention_days = retention_days
        self.logger = logger

    async def record_execution(self, test_id: str, execution_data: Dict[str, Any]):
        """
        Record test execution for historical tracking.

        PATTERN: Time-series data with TTL
        CRITICAL: Store efficiently to handle large volumes

        Args:
            test_id: Test identifier
            execution_data: Execution details (passed, time, error, etc.)
        """
        try:
            # Ensure timestamp
            if "timestamp" not in execution_data:
                execution_data["timestamp"] = datetime.now().isoformat()

            # Store in Redis with TTL for recent data
            if self.redis:
                await self._store_in_redis(test_id, execution_data)

            # Update aggregated metrics
            await self._update_aggregates(test_id, execution_data)

            # Store in long-term memory if significant
            if self._is_significant_execution(execution_data):
                await self._store_in_memory(test_id, execution_data)

            logger.debug(f"Recorded execution for test: {test_id}")

        except Exception as e:
            logger.error(f"Error recording execution for {test_id}: {e}", exc_info=True)

    async def _store_in_redis(self, test_id: str, execution_data: Dict[str, Any]):
        """
        Store execution data in Redis.

        Args:
            test_id: Test ID
            execution_data: Execution data
        """
        if not self.redis:
            return

        try:
            timestamp = execution_data.get("timestamp", datetime.now().isoformat())
            key = f"test:execution:{test_id}:{timestamp}"

            # Store with TTL
            ttl_seconds = self.retention_days * 86400
            await self.redis.setex(key, ttl_seconds, json.dumps(execution_data))
        except Exception as e:
            logger.warning(f"Redis storage error: {e}")

    async def _store_in_memory(self, test_id: str, execution_data: Dict[str, Any]):
        """
        Store significant execution in long-term memory.

        Args:
            test_id: Test ID
            execution_data: Execution data
        """
        if not self.memory_service:
            return

        try:
            content = f"Test execution: {test_id}"
            metadata = {"test_id": test_id, "type": "test_execution", **execution_data}

            await self.memory_service.store_memory(
                content=content, metadata=metadata, memory_type="project"
            )
        except Exception as e:
            logger.warning(f"Memory storage error: {e}")

    async def _update_aggregates(self, test_id: str, execution_data: Dict[str, Any]):
        """
        Update aggregated metrics for test.

        Args:
            test_id: Test ID
            execution_data: Execution data
        """
        if not self.redis:
            return

        try:
            # Get or create aggregate key
            agg_key = f"test:aggregate:{test_id}"

            # Get existing aggregates
            agg_data = await self.redis.get(agg_key)
            if agg_data:
                aggregates = json.loads(agg_data)
            else:
                aggregates = {
                    "total_executions": 0,
                    "total_failures": 0,
                    "total_execution_time_ms": 0,
                    "execution_times": [],
                    "failure_types": {},
                }

            # Update aggregates
            aggregates["total_executions"] += 1

            if not execution_data.get("passed", False):
                aggregates["total_failures"] += 1

                # Track failure type
                failure_type = execution_data.get("failure_type", "unknown")
                aggregates["failure_types"][failure_type] = (
                    aggregates["failure_types"].get(failure_type, 0) + 1
                )

            exec_time = execution_data.get("execution_time_ms", 0)
            aggregates["total_execution_time_ms"] += exec_time

            # Keep rolling window of execution times (last 100)
            aggregates["execution_times"].append(exec_time)
            if len(aggregates["execution_times"]) > 100:
                aggregates["execution_times"] = aggregates["execution_times"][-100:]

            # Store updated aggregates
            await self.redis.set(agg_key, json.dumps(aggregates))

        except Exception as e:
            logger.warning(f"Aggregate update error: {e}")

    def _is_significant_execution(self, execution_data: Dict[str, Any]) -> bool:
        """
        Determine if execution is significant enough for long-term storage.

        Args:
            execution_data: Execution data

        Returns:
            True if significant
        """
        # Store failures
        if not execution_data.get("passed", True):
            return True

        # Store performance regressions
        exec_time = execution_data.get("execution_time_ms", 0)
        baseline_time = execution_data.get("baseline_time_ms", 0)
        if baseline_time > 0 and exec_time > baseline_time * 1.5:
            return True

        # Store first execution
        if execution_data.get("is_first_run", False):
            return True

        # Otherwise, not significant
        return False

    async def get_test_history(
        self, test_id: str, days: int = 30
    ) -> Optional[TestPerformanceHistory]:
        """
        Get test execution history.

        Args:
            test_id: Test ID
            days: Days of history to retrieve

        Returns:
            Test performance history or None
        """
        try:
            # Get executions from Redis
            executions = await self._get_executions(test_id, days)

            if not executions:
                return None

            # Calculate aggregated metrics
            total_executions = len(executions)
            failures = [e for e in executions if not e.get("passed", False)]
            total_failures = len(failures)
            failure_rate = (
                total_failures / total_executions if total_executions > 0 else 0.0
            )

            # Execution time statistics
            times = [e.get("execution_time_ms", 0) for e in executions]
            avg_time = statistics.mean(times) if times else 0.0

            # Analyze trend
            time_trend = self._calculate_trend(times)

            # Failure type distribution
            failure_types: Dict[FailureType, int] = {}
            for failure in failures:
                f_type_str = failure.get("failure_type", "unknown")
                try:
                    f_type = FailureType(f_type_str)
                except ValueError:
                    f_type = FailureType.RUNTIME_ERROR
                failure_types[f_type] = failure_types.get(f_type, 0) + 1

            # Get repair history
            repair_history = await self._get_repair_history(test_id)

            # Predict future failures
            failure_probability = self._predict_failure_probability(
                executions, failure_rate, time_trend
            )

            # Determine if maintenance needed
            maintenance_needed = failure_probability > 0.3 or failure_rate > 0.2

            # Get first and last timestamps
            first_seen = datetime.fromisoformat(executions[0]["timestamp"])
            last_updated = datetime.now()

            return TestPerformanceHistory(
                test_id=test_id,
                executions=executions,
                total_executions=total_executions,
                total_failures=total_failures,
                failure_rate=failure_rate,
                avg_execution_time_ms=avg_time,
                execution_time_trend=time_trend,
                common_failure_types=failure_types,
                repair_history=repair_history,
                predicted_failure_probability=failure_probability,
                predicted_maintenance_needed=maintenance_needed,
                first_seen=first_seen,
                last_updated=last_updated,
            )

        except Exception as e:
            logger.error(
                f"Error getting test history for {test_id}: {e}", exc_info=True
            )
            return None

    async def _get_executions(self, test_id: str, days: int) -> List[Dict[str, Any]]:
        """
        Get test executions from Redis.

        Args:
            test_id: Test ID
            days: Days of history

        Returns:
            List of executions
        """
        if not self.redis:
            return []

        try:
            # Get all execution keys for test
            pattern = f"test:execution:{test_id}:*"
            keys = await self.redis.keys(pattern)

            executions = []
            cutoff_date = datetime.now() - timedelta(days=days)

            for key in keys:
                data = await self.redis.get(key)
                if data:
                    execution = json.loads(data)

                    # Check if within time window
                    timestamp_str = execution.get("timestamp")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp >= cutoff_date:
                            executions.append(execution)

            # Sort by timestamp
            executions.sort(key=lambda e: e.get("timestamp", ""))

            return executions

        except Exception as e:
            logger.warning(f"Error getting executions: {e}")
            return []

    async def _get_repair_history(self, test_id: str) -> List[TestRepair]:
        """
        Get repair history for test.

        Args:
            test_id: Test ID

        Returns:
            List of repairs
        """
        # This would query repair records from storage
        # For now, return empty list
        return []

    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend from time series data.

        Args:
            values: Time series values

        Returns:
            Trend: "increasing", "decreasing", or "stable"
        """
        if len(values) < 3:
            return "stable"

        try:
            # Simple linear regression to detect trend
            n = len(values)
            x = list(range(n))
            y = values

            # Calculate slope
            x_mean = sum(x) / n
            y_mean = sum(y) / n

            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                return "stable"

            slope = numerator / denominator

            # Determine trend based on slope and magnitude
            avg_value = statistics.mean(values)
            relative_slope = slope / avg_value if avg_value > 0 else 0

            if relative_slope > 0.1:
                return "increasing"
            elif relative_slope < -0.1:
                return "decreasing"
            else:
                return "stable"

        except Exception as e:
            logger.debug(f"Trend calculation error: {e}")
            return "stable"

    def _predict_failure_probability(
        self,
        executions: List[Dict[str, Any]],
        current_failure_rate: float,
        time_trend: str,
    ) -> float:
        """
        Predict probability of future failures.

        Args:
            executions: Historical executions
            current_failure_rate: Current failure rate
            time_trend: Execution time trend

        Returns:
            Predicted failure probability (0-1)
        """
        # Base prediction on current failure rate
        prediction = current_failure_rate

        # Adjust based on recent trend
        if len(executions) >= 10:
            recent_executions = executions[-10:]
            recent_failures = sum(
                1 for e in recent_executions if not e.get("passed", False)
            )
            recent_failure_rate = recent_failures / len(recent_executions)

            # Weight recent more heavily
            prediction = 0.6 * recent_failure_rate + 0.4 * current_failure_rate

        # Adjust based on performance trend
        if time_trend == "increasing":
            prediction *= 1.2  # Performance degradation increases failure risk

        # Cap at 1.0
        return min(prediction, 1.0)

    async def analyze_trends(
        self, test_id: str, time_window: int = 30
    ) -> Optional[TestPerformanceHistory]:
        """
        Analyze historical trends for test.

        PATTERN: Time-series analysis with predictions
        CRITICAL: Identify performance regressions and failure patterns

        Args:
            test_id: Test ID
            time_window: Days to analyze

        Returns:
            Test performance history with trends
        """
        # This is an alias for get_test_history with trend analysis
        return await self.get_test_history(test_id, days=time_window)

    async def get_suite_statistics(
        self, test_suite: str, days: int = 7
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics for entire test suite.

        Args:
            test_suite: Test suite identifier
            days: Days to analyze

        Returns:
            Suite statistics
        """
        try:
            # This would aggregate data across all tests in suite
            # For now, return basic structure
            stats = {
                "suite": test_suite,
                "time_window_days": days,
                "total_tests": 0,
                "total_executions": 0,
                "total_failures": 0,
                "average_failure_rate": 0.0,
                "average_execution_time_ms": 0.0,
                "flaky_tests": [],
                "slow_tests": [],
                "frequently_failing_tests": [],
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting suite statistics: {e}", exc_info=True)
            return {}

    async def cleanup_old_data(self, days: int = None):
        """
        Clean up old historical data.

        CRITICAL: Prevent unlimited data growth

        Args:
            days: Days to retain (default: retention_days)
        """
        if days is None:
            days = self.retention_days

        try:
            if self.redis:
                # Redis TTL handles this automatically
                logger.info(f"Redis TTL manages cleanup automatically ({days} days)")

            # Clean up memory service if needed
            if self.memory_service:
                cutoff_date = datetime.now() - timedelta(days=days)
                logger.info(f"Would clean up memory data older than {cutoff_date}")
                # Implementation would query and delete old memories

        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)

    async def export_history(self, test_id: str, format: str = "json") -> Optional[str]:
        """
        Export test history for analysis or archival.

        Args:
            test_id: Test ID
            format: Export format (json, csv)

        Returns:
            Exported data as string
        """
        try:
            history = await self.get_test_history(test_id, days=self.retention_days)

            if not history:
                return None

            if format == "json":
                # Export as JSON
                return json.dumps(history.model_dump(), indent=2, default=str)
            elif format == "csv":
                # Export as CSV
                lines = [
                    "timestamp,passed,execution_time_ms,failure_type,error_message"
                ]

                for execution in history.executions:
                    line = ",".join(
                        [
                            execution.get("timestamp", ""),
                            str(execution.get("passed", False)),
                            str(execution.get("execution_time_ms", 0)),
                            execution.get("failure_type", ""),
                            execution.get("error_message", "").replace(",", ";"),
                        ]
                    )
                    lines.append(line)

                return "\n".join(lines)
            else:
                logger.error(f"Unsupported export format: {format}")
                return None

        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            return None

    async def get_failure_timeline(
        self, test_id: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of test failures.

        Args:
            test_id: Test ID
            days: Days of history

        Returns:
            List of failure events with timeline
        """
        try:
            executions = await self._get_executions(test_id, days)

            failures = [
                {
                    "timestamp": e.get("timestamp"),
                    "failure_type": e.get("failure_type"),
                    "error_message": e.get("error_message"),
                    "execution_time_ms": e.get("execution_time_ms"),
                }
                for e in executions
                if not e.get("passed", False)
            ]

            return failures

        except Exception as e:
            logger.error(f"Error getting failure timeline: {e}", exc_info=True)
            return []
