"""Performance analyzer for regression detection.

PATTERN: Baseline comparison with statistical significance
CRITICAL: Environment normalization required for accurate comparisons
GOTCHA: CPU/memory differences affect measurements
"""

import asyncio
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import platform
import psutil

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Performance analyzer for detecting regressions.

    PATTERN: Statistical baseline comparison with environment normalization
    CRITICAL: Must account for hardware and environment differences
    GOTCHA: Performance varies across machines and system load
    """

    def __init__(
        self,
        baseline_path: Optional[str] = None,
        significance_level: float = 0.05
    ):
        """
        Initialize performance analyzer.

        Args:
            baseline_path: Path to baseline performance data
            significance_level: P-value threshold for significance (default 0.05)
        """
        self.logger = logger
        self.baseline_path = baseline_path
        self.significance_level = significance_level
        self.environment_factor = None

    async def measure_performance(
        self,
        test_command: str,
        iterations: int = 10,
        warmup_runs: int = 2
    ) -> Dict[str, Any]:
        """
        Measure performance metrics for tests.

        PATTERN: Multiple iterations with warmup for stable measurements
        CRITICAL: Collect both timing and resource usage

        Args:
            test_command: Command to run performance tests
            iterations: Number of measurement iterations
            warmup_runs: Number of warmup runs to discard

        Returns:
            Performance measurements
        """
        measurements = {
            "execution_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "timestamp": datetime.now().isoformat(),
            "environment": await self._get_environment_info(),
        }

        # Warmup runs
        self.logger.info(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            await self._run_single_measurement(test_command, collect_data=False)

        # Actual measurements
        self.logger.info(f"Running {iterations} measurement iterations...")
        for i in range(iterations):
            result = await self._run_single_measurement(test_command)

            measurements["execution_times"].append(result["execution_time"])
            measurements["memory_usage"].append(result["memory_mb"])
            measurements["cpu_usage"].append(result["cpu_percent"])

            # Brief pause between iterations
            await asyncio.sleep(0.5)

        # Calculate statistics
        measurements["statistics"] = self._calculate_statistics(measurements)

        return measurements

    async def detect_regression(
        self,
        current_metrics: Dict[str, Any],
        baseline_metrics: Optional[Dict[str, Any]] = None,
        threshold_percentage: float = 10.0
    ) -> Dict[str, Any]:
        """
        Detect performance regressions against baseline.

        PATTERN: Statistical comparison with environment normalization
        CRITICAL: Use p-values for significance testing
        GOTCHA: Threshold should account for normal variance

        Args:
            current_metrics: Current performance measurements
            baseline_metrics: Baseline to compare against
            threshold_percentage: Regression threshold (default 10%)

        Returns:
            Regression detection results
        """
        # Load baseline if not provided
        if baseline_metrics is None and self.baseline_path:
            baseline_metrics = await self._load_baseline()

        if baseline_metrics is None:
            return {
                "regression_detected": False,
                "message": "No baseline available for comparison",
                "regressions": [],
            }

        # Normalize for environment differences
        normalized_current = await self._normalize_metrics(
            current_metrics,
            baseline_metrics.get("environment", {})
        )

        # Compare metrics
        regressions = []
        metric_comparisons = []

        # Compare execution time
        time_regression = self._compare_metric(
            "execution_time",
            normalized_current["statistics"]["execution_time"],
            baseline_metrics["statistics"]["execution_time"],
            threshold_percentage
        )
        if time_regression:
            regressions.append(time_regression)
        metric_comparisons.append(time_regression or {
            "metric": "execution_time",
            "regression_detected": False
        })

        # Compare memory usage
        memory_regression = self._compare_metric(
            "memory_usage",
            normalized_current["statistics"]["memory_usage"],
            baseline_metrics["statistics"]["memory_usage"],
            threshold_percentage
        )
        if memory_regression:
            regressions.append(memory_regression)
        metric_comparisons.append(memory_regression or {
            "metric": "memory_usage",
            "regression_detected": False
        })

        # Compare CPU usage
        cpu_regression = self._compare_metric(
            "cpu_usage",
            normalized_current["statistics"]["cpu_usage"],
            baseline_metrics["statistics"]["cpu_usage"],
            threshold_percentage
        )
        if cpu_regression:
            regressions.append(cpu_regression)
        metric_comparisons.append(cpu_regression or {
            "metric": "cpu_usage",
            "regression_detected": False
        })

        return {
            "regression_detected": len(regressions) > 0,
            "regressions": regressions,
            "regression_count": len(regressions),
            "comparisons": metric_comparisons,
            "threshold_percentage": threshold_percentage,
            "baseline_timestamp": baseline_metrics.get("timestamp"),
            "current_timestamp": current_metrics.get("timestamp"),
        }

    async def _run_single_measurement(
        self,
        test_command: str,
        collect_data: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single performance measurement.

        Args:
            test_command: Command to measure
            collect_data: Whether to collect measurement data

        Returns:
            Measurement results
        """
        import time

        # Start resource monitoring
        process = psutil.Process()
        start_time = time.perf_counter()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            # Run command
            proc = await asyncio.create_subprocess_shell(
                test_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            # End resource monitoring
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()

            execution_time = end_time - start_time
            memory_used = max(end_memory - start_memory, 0)

            if collect_data:
                return {
                    "execution_time": execution_time,
                    "memory_mb": memory_used,
                    "cpu_percent": cpu_percent,
                    "success": proc.returncode == 0,
                }
            else:
                return {"success": proc.returncode == 0}

        except Exception as e:
            self.logger.error(f"Measurement failed: {e}")
            return {
                "execution_time": 0,
                "memory_mb": 0,
                "cpu_percent": 0,
                "success": False,
                "error": str(e),
            }

    async def _get_environment_info(self) -> Dict[str, Any]:
        """
        Collect environment information for normalization.

        PATTERN: Capture hardware and system details
        CRITICAL: Used for normalizing performance across environments

        Returns:
            Environment information
        """
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "total_memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "available_memory_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
        }

    def _calculate_statistics(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistical summaries of measurements.

        Args:
            measurements: Raw measurement data

        Returns:
            Statistical summaries
        """
        stats = {}

        for metric in ["execution_times", "memory_usage", "cpu_usage"]:
            values = measurements.get(metric, [])
            if values:
                metric_name = metric.replace("execution_times", "execution_time").replace("_usage", "_usage")

                stats[metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

                # Calculate confidence interval (95%)
                if len(values) > 1:
                    margin = 1.96 * (stats[metric_name]["stdev"] / (len(values) ** 0.5))
                    stats[metric_name]["confidence_interval"] = (
                        stats[metric_name]["mean"] - margin,
                        stats[metric_name]["mean"] + margin
                    )

        return stats

    async def _normalize_metrics(
        self,
        current_metrics: Dict[str, Any],
        baseline_environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Normalize metrics for environment differences.

        PATTERN: Apply environment factors to current metrics
        CRITICAL: Accounts for hardware performance differences
        GOTCHA: Normalization is approximate, not exact

        Args:
            current_metrics: Current measurements
            baseline_environment: Baseline environment info

        Returns:
            Normalized metrics
        """
        current_env = current_metrics.get("environment", {})

        # Calculate environment normalization factor
        factor = self._calculate_environment_factor(current_env, baseline_environment)

        # Apply normalization to execution times
        normalized = current_metrics.copy()

        if "execution_times" in normalized:
            normalized["execution_times"] = [
                t * factor for t in normalized["execution_times"]
            ]

        if "statistics" in normalized and "execution_time" in normalized["statistics"]:
            for key in ["mean", "median", "min", "max"]:
                if key in normalized["statistics"]["execution_time"]:
                    normalized["statistics"]["execution_time"][key] *= factor

        return normalized

    def _calculate_environment_factor(
        self,
        current_env: Dict[str, Any],
        baseline_env: Dict[str, Any]
    ) -> float:
        """
        Calculate environment normalization factor.

        PATTERN: Compare CPU and memory specs
        CRITICAL: Factor adjusts current metrics to baseline equivalent

        Args:
            current_env: Current environment info
            baseline_env: Baseline environment info

        Returns:
            Normalization factor
        """
        # If environments are similar, no adjustment needed
        if not baseline_env:
            return 1.0

        # Compare CPU frequency
        current_freq = current_env.get("cpu_freq_mhz", 1000)
        baseline_freq = baseline_env.get("cpu_freq_mhz", 1000)

        if current_freq > 0 and baseline_freq > 0:
            cpu_factor = baseline_freq / current_freq
        else:
            cpu_factor = 1.0

        # Compare CPU count (parallel execution factor)
        current_cpus = current_env.get("cpu_count", 1)
        baseline_cpus = baseline_env.get("cpu_count", 1)

        if current_cpus > 0 and baseline_cpus > 0:
            # Less adjustment for CPU count (diminishing returns)
            cpu_count_factor = (baseline_cpus / current_cpus) ** 0.5
        else:
            cpu_count_factor = 1.0

        # Combine factors (weighted average)
        factor = (cpu_factor * 0.7) + (cpu_count_factor * 0.3)

        # Limit factor to reasonable range (0.5x to 2.0x)
        factor = max(0.5, min(2.0, factor))

        return factor

    def _compare_metric(
        self,
        metric_name: str,
        current_stats: Dict[str, Any],
        baseline_stats: Dict[str, Any],
        threshold_percentage: float
    ) -> Optional[Dict[str, Any]]:
        """
        Compare a metric against baseline with statistical significance.

        PATTERN: Use t-test for significance, percentage for threshold
        CRITICAL: Both significance and threshold must be met

        Args:
            metric_name: Name of metric being compared
            current_stats: Current metric statistics
            baseline_stats: Baseline metric statistics
            threshold_percentage: Regression threshold

        Returns:
            Regression details if detected, None otherwise
        """
        current_mean = current_stats.get("mean", 0)
        baseline_mean = baseline_stats.get("mean", 0)

        if baseline_mean == 0:
            return None

        # Calculate percentage change
        change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100

        # Check if exceeds threshold
        if abs(change_pct) < threshold_percentage:
            return None

        # Calculate statistical significance
        p_value = self._calculate_p_value(current_stats, baseline_stats)

        # Only report if statistically significant
        if p_value > self.significance_level:
            return None

        return {
            "metric": metric_name,
            "regression_detected": change_pct > 0,  # Positive change is regression
            "current_value": current_mean,
            "baseline_value": baseline_mean,
            "change_percentage": round(change_pct, 2),
            "p_value": round(p_value, 4),
            "significant": True,
            "confidence_interval": current_stats.get("confidence_interval"),
        }

    def _calculate_p_value(
        self,
        current_stats: Dict[str, Any],
        baseline_stats: Dict[str, Any]
    ) -> float:
        """
        Calculate p-value for difference between metrics.

        PATTERN: Simplified t-test approximation
        GOTCHA: Assumes normal distribution

        Args:
            current_stats: Current statistics
            baseline_stats: Baseline statistics

        Returns:
            P-value
        """
        # Simplified p-value calculation
        # In production, would use scipy.stats.ttest_ind

        current_mean = current_stats.get("mean", 0)
        baseline_mean = baseline_stats.get("mean", 0)
        current_stdev = current_stats.get("stdev", 0)
        baseline_stdev = baseline_stats.get("stdev", 0)
        current_n = current_stats.get("count", 1)
        baseline_n = baseline_stats.get("count", 1)

        # Avoid division by zero
        if current_stdev == 0 or baseline_stdev == 0:
            return 1.0 if current_mean == baseline_mean else 0.0

        # Calculate pooled standard deviation
        pooled_stdev = ((current_stdev ** 2 / current_n) + (baseline_stdev ** 2 / baseline_n)) ** 0.5

        if pooled_stdev == 0:
            return 1.0

        # Calculate t-statistic
        t_stat = abs(current_mean - baseline_mean) / pooled_stdev

        # Approximate p-value from t-statistic
        # This is a simplified approximation
        if t_stat < 1.96:  # 95% confidence
            p_value = 0.05 + (1.96 - t_stat) * 0.05
        else:
            p_value = 0.05 * (1.96 / t_stat)

        return min(1.0, max(0.0, p_value))

    async def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """
        Load baseline performance data from file.

        Args:
            None (uses self.baseline_path)

        Returns:
            Baseline data or None
        """
        if not self.baseline_path:
            return None

        try:
            baseline_file = Path(self.baseline_path)
            if not baseline_file.exists():
                self.logger.warning(f"Baseline file not found: {self.baseline_path}")
                return None

            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)

            return baseline_data

        except Exception as e:
            self.logger.error(f"Failed to load baseline: {e}")
            return None

    async def save_baseline(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> bool:
        """
        Save performance metrics as new baseline.

        Args:
            metrics: Metrics to save
            output_path: Path to save baseline (uses self.baseline_path if None)

        Returns:
            Success status
        """
        path = output_path or self.baseline_path

        if not path:
            self.logger.error("No baseline path specified")
            return False

        try:
            output_file = Path(path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            self.logger.info(f"Baseline saved to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save baseline: {e}")
            return False
