"""Performance reporter for Core Web Vitals and metrics analysis.

This module provides the PerformanceReporter class for generating performance
reports with Core Web Vitals, metric trends, threshold validation, and visual
representations of performance data.
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from datetime import datetime
import json
import statistics

from src.models.browser_models import PerformanceMetrics

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """Generate performance reports with Core Web Vitals and trends.

    This class creates comprehensive performance reports showing Core Web Vitals
    metrics, pass/fail status against thresholds, trends over time, and visual
    charts for performance analysis.

    PATTERN: Track metrics over time for trend analysis and regression detection.
    """

    # Core Web Vitals thresholds (good vs needs improvement vs poor)
    THRESHOLDS = {
        "lcp": {"good": 2500, "poor": 4000},  # Largest Contentful Paint (ms)
        "fid": {"good": 100, "poor": 300},  # First Input Delay (ms)
        "cls": {"good": 0.1, "poor": 0.25},  # Cumulative Layout Shift
        "ttfb": {"good": 800, "poor": 1800},  # Time to First Byte (ms)
        "fcp": {"good": 1800, "poor": 3000},  # First Contentful Paint (ms)
        "tti": {"good": 3800, "poor": 7300},  # Time to Interactive (ms)
    }

    def __init__(self, output_dir: str = "performance-reports"):
        """Initialize the performance reporter.

        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[PerformanceMetrics] = []
        logger.info(f"Performance reporter initialized: {self.output_dir}")

    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add performance metrics to history for trend tracking.

        Args:
            metrics: Performance metrics to add

        Example:
            reporter = PerformanceReporter()
            reporter.add_metrics(metrics1)
            reporter.add_metrics(metrics2)
            # Now trends can be analyzed
        """
        self.metrics_history.append(metrics)
        logger.debug(f"Added metrics to history (total: {len(self.metrics_history)})")

    def generate_report(
        self,
        metrics: PerformanceMetrics,
        format: Literal["html", "json"] = "html",
        report_name: Optional[str] = None,
        include_trends: bool = True,
    ) -> str:
        """Generate a performance report for a single metrics snapshot.

        Args:
            metrics: Performance metrics to report
            format: Output format (html or json)
            report_name: Custom report name (auto-generated if None)
            include_trends: Include trend analysis if metrics history exists

        Returns:
            Path to the generated report file

        Example:
            reporter = PerformanceReporter()
            metrics = await monitor.collect_metrics(page, url)
            report_path = reporter.generate_report(
                metrics,
                format="html",
                include_trends=True
            )
        """
        logger.info(f"Generating performance report for {metrics.url}")

        try:
            # Add to history
            if include_trends:
                self.add_metrics(metrics)

            # Generate report name if not provided
            if not report_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_url = metrics.url.replace("://", "_").replace("/", "_")[:50]
                report_name = f"perf_{safe_url}_{timestamp}"

            # Generate report based on format
            if format == "json":
                report_path = self._generate_json_report(
                    metrics, report_name, include_trends
                )
            else:
                report_path = self._generate_html_report(
                    metrics, report_name, include_trends
                )

            logger.info(f"Performance report generated: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            raise

    def _generate_json_report(
        self,
        metrics: PerformanceMetrics,
        report_name: str,
        include_trends: bool,
    ) -> Path:
        """Generate a JSON performance report.

        Args:
            metrics: Performance metrics
            report_name: Report name
            include_trends: Include trend analysis

        Returns:
            Path to JSON report
        """
        # Evaluate metrics against thresholds
        evaluation = self.evaluate_metrics(metrics)

        # Build report data
        report_data = {
            "metadata": {
                "url": metrics.url,
                "timestamp": metrics.timestamp.isoformat(),
                "browser": metrics.browser.value,
            },
            "core_web_vitals": {
                "lcp": {
                    "value": metrics.lcp,
                    "unit": "ms",
                    "status": evaluation["lcp"]["status"],
                    "passes": evaluation["lcp"]["passes"],
                },
                "fid": {
                    "value": metrics.fid,
                    "unit": "ms",
                    "status": evaluation["fid"]["status"],
                    "passes": evaluation["fid"]["passes"],
                },
                "cls": {
                    "value": metrics.cls,
                    "unit": "score",
                    "status": evaluation["cls"]["status"],
                    "passes": evaluation["cls"]["passes"],
                },
                "overall_passes": metrics.passes_cwv,
            },
            "additional_metrics": {
                "ttfb": {"value": metrics.ttfb, "unit": "ms"},
                "fcp": {"value": metrics.fcp, "unit": "ms"},
                "tti": {"value": metrics.tti, "unit": "ms"},
                "speed_index": {"value": metrics.speed_index, "unit": "score"},
                "dom_content_loaded": {
                    "value": metrics.dom_content_loaded,
                    "unit": "ms",
                },
                "load_complete": {"value": metrics.load_complete, "unit": "ms"},
            },
            "resources": {
                "total_requests": metrics.total_requests,
                "total_size_kb": metrics.total_size_kb,
            },
            "memory": {
                "js_heap_size_mb": metrics.js_heap_size_mb,
            },
        }

        # Add trends if requested
        if include_trends and len(self.metrics_history) > 1:
            report_data["trends"] = self.calculate_trends(metrics.url)

        # Save JSON report
        report_path = self.output_dir / f"{report_name}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return report_path

    def _generate_html_report(
        self,
        metrics: PerformanceMetrics,
        report_name: str,
        include_trends: bool,
    ) -> Path:
        """Generate an HTML performance report.

        Args:
            metrics: Performance metrics
            report_name: Report name
            include_trends: Include trend analysis

        Returns:
            Path to HTML report
        """
        # Evaluate metrics
        evaluation = self.evaluate_metrics(metrics)

        # Calculate trends if available
        trends = None
        if include_trends and len(self.metrics_history) > 1:
            trends = self.calculate_trends(metrics.url)

        # Build HTML
        html_content = self._build_html_structure(metrics, evaluation, trends)

        # Save HTML report
        report_path = self.output_dir / f"{report_name}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_path

    def _build_html_structure(
        self,
        metrics: PerformanceMetrics,
        evaluation: Dict[str, Any],
        trends: Optional[Dict[str, Any]],
    ) -> str:
        """Build complete HTML structure for performance report.

        Args:
            metrics: Performance metrics
            evaluation: Metrics evaluation
            trends: Trend analysis (optional)

        Returns:
            Complete HTML content
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Report - {metrics.url}</title>
    {self._get_styles()}
</head>
<body>
    <div class="container">
        <header>
            <h1>Performance Report</h1>
            <p class="url">{metrics.url}</p>
            <p class="meta">Browser: {metrics.browser.value} | Generated: {metrics.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>

        {self._build_cwv_section(metrics, evaluation)}
        {self._build_additional_metrics_section(metrics, evaluation)}
        {self._build_resources_section(metrics)}
        {self._build_trends_section(trends) if trends else ""}

        <footer>
            <p>Performance Report - Generated by Agent Swarm</p>
        </footer>
    </div>
</body>
</html>"""
        return html

    def _get_styles(self) -> str:
        """Get inline CSS styles for the report.

        Returns:
            HTML style tag with CSS
        """
        return """<style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        h1 {
            color: #2c3e50;
            font-size: 2em;
            margin-bottom: 10px;
        }

        h2 {
            color: #34495e;
            font-size: 1.5em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }

        h3 {
            color: #2c3e50;
            font-size: 1.2em;
            margin: 15px 0 10px;
        }

        .url {
            color: #3498db;
            font-size: 1.1em;
            margin: 10px 0;
            word-break: break-all;
        }

        .meta {
            color: #7f8c8d;
            font-size: 0.9em;
        }

        .section {
            background: #fff;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            position: relative;
        }

        .metric-card.good {
            border-left-color: #27ae60;
            background: #f0fdf4;
        }

        .metric-card.needs-improvement {
            border-left-color: #f39c12;
            background: #fffbeb;
        }

        .metric-card.poor {
            border-left-color: #e74c3c;
            background: #fef2f2;
        }

        .metric-label {
            font-size: 0.85em;
            color: #7f8c8d;
            text-transform: uppercase;
            margin-bottom: 5px;
            font-weight: 600;
        }

        .metric-value {
            font-size: 2.2em;
            font-weight: bold;
            margin: 10px 0;
        }

        .metric-card.good .metric-value {
            color: #27ae60;
        }

        .metric-card.needs-improvement .metric-value {
            color: #f39c12;
        }

        .metric-card.poor .metric-value {
            color: #e74c3c;
        }

        .metric-status {
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
        }

        .metric-threshold {
            font-size: 0.75em;
            color: #95a5a6;
            margin-top: 8px;
        }

        .cwv-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 20px;
        }

        .cwv-summary.passed {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        }

        .cwv-summary.failed {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }

        .cwv-summary h3 {
            color: white;
            margin: 0 0 10px 0;
            font-size: 1.5em;
        }

        .cwv-summary p {
            margin: 5px 0;
            font-size: 1.1em;
        }

        .resources-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .resource-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }

        .resource-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }

        .resource-label {
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
        }

        .trend-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .trend-label {
            font-weight: 600;
        }

        .trend-value {
            font-size: 1.2em;
            font-weight: bold;
        }

        .trend-value.improving {
            color: #27ae60;
        }

        .trend-value.degrading {
            color: #e74c3c;
        }

        .trend-value.stable {
            color: #95a5a6;
        }

        .trend-arrow {
            font-size: 1.5em;
            margin-left: 10px;
        }

        .chart-placeholder {
            background: #f0f0f0;
            border: 2px dashed #ccc;
            border-radius: 6px;
            padding: 40px;
            text-align: center;
            color: #999;
            margin-top: 15px;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>"""

    def _build_cwv_section(
        self, metrics: PerformanceMetrics, evaluation: Dict[str, Any]
    ) -> str:
        """Build the Core Web Vitals section.

        Args:
            metrics: Performance metrics
            evaluation: Metrics evaluation

        Returns:
            HTML for CWV section
        """
        cwv_class = "passed" if metrics.passes_cwv else "failed"
        cwv_status = "PASSED" if metrics.passes_cwv else "NEEDS IMPROVEMENT"

        return f"""
        <div class="section">
            <div class="cwv-summary {cwv_class}">
                <h3>Core Web Vitals</h3>
                <p>Status: <strong>{cwv_status}</strong></p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card {evaluation["lcp"]["status"]}">
                    <div class="metric-label">LCP - Largest Contentful Paint</div>
                    <div class="metric-value">{metrics.lcp:.0f}<span style="font-size: 0.5em;">ms</span></div>
                    <div class="metric-status">{evaluation["lcp"]["status"].replace("-", " ")}</div>
                    <div class="metric-threshold">Good: &lt; 2.5s | Poor: &gt; 4.0s</div>
                </div>

                <div class="metric-card {evaluation["fid"]["status"]}">
                    <div class="metric-label">FID - First Input Delay</div>
                    <div class="metric-value">{metrics.fid:.0f}<span style="font-size: 0.5em;">ms</span></div>
                    <div class="metric-status">{evaluation["fid"]["status"].replace("-", " ")}</div>
                    <div class="metric-threshold">Good: &lt; 100ms | Poor: &gt; 300ms</div>
                </div>

                <div class="metric-card {evaluation["cls"]["status"]}">
                    <div class="metric-label">CLS - Cumulative Layout Shift</div>
                    <div class="metric-value">{metrics.cls:.3f}</div>
                    <div class="metric-status">{evaluation["cls"]["status"].replace("-", " ")}</div>
                    <div class="metric-threshold">Good: &lt; 0.1 | Poor: &gt; 0.25</div>
                </div>
            </div>
        </div>"""

    def _build_additional_metrics_section(
        self, metrics: PerformanceMetrics, evaluation: Dict[str, Any]
    ) -> str:
        """Build the additional metrics section.

        Args:
            metrics: Performance metrics
            evaluation: Metrics evaluation

        Returns:
            HTML for additional metrics section
        """
        return f"""
        <div class="section">
            <h2>Additional Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card {evaluation["ttfb"]["status"]}">
                    <div class="metric-label">TTFB - Time to First Byte</div>
                    <div class="metric-value">{metrics.ttfb:.0f}<span style="font-size: 0.5em;">ms</span></div>
                    <div class="metric-status">{evaluation["ttfb"]["status"].replace("-", " ")}</div>
                </div>

                <div class="metric-card {evaluation["fcp"]["status"]}">
                    <div class="metric-label">FCP - First Contentful Paint</div>
                    <div class="metric-value">{metrics.fcp:.0f}<span style="font-size: 0.5em;">ms</span></div>
                    <div class="metric-status">{evaluation["fcp"]["status"].replace("-", " ")}</div>
                </div>

                <div class="metric-card {evaluation["tti"]["status"]}">
                    <div class="metric-label">TTI - Time to Interactive</div>
                    <div class="metric-value">{metrics.tti:.0f}<span style="font-size: 0.5em;">ms</span></div>
                    <div class="metric-status">{evaluation["tti"]["status"].replace("-", " ")}</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Speed Index</div>
                    <div class="metric-value">{metrics.speed_index:.0f}</div>
                    <div class="metric-status">Score</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">DOM Content Loaded</div>
                    <div class="metric-value">{metrics.dom_content_loaded:.0f}<span style="font-size: 0.5em;">ms</span></div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Load Complete</div>
                    <div class="metric-value">{metrics.load_complete:.0f}<span style="font-size: 0.5em;">ms</span></div>
                </div>
            </div>
        </div>"""

    def _build_resources_section(self, metrics: PerformanceMetrics) -> str:
        """Build the resources section.

        Args:
            metrics: Performance metrics

        Returns:
            HTML for resources section
        """
        return f"""
        <div class="section">
            <h2>Resource Metrics</h2>
            <div class="resources-grid">
                <div class="resource-card">
                    <div class="resource-label">Total Requests</div>
                    <div class="resource-value">{metrics.total_requests}</div>
                </div>

                <div class="resource-card">
                    <div class="resource-label">Total Size</div>
                    <div class="resource-value">{metrics.total_size_kb:.2f}<span style="font-size: 0.6em;">KB</span></div>
                </div>

                <div class="resource-card">
                    <div class="resource-label">JS Heap Size</div>
                    <div class="resource-value">{metrics.js_heap_size_mb:.2f}<span style="font-size: 0.6em;">MB</span></div>
                </div>
            </div>
        </div>"""

    def _build_trends_section(self, trends: Dict[str, Any]) -> str:
        """Build the trends section.

        Args:
            trends: Trend analysis data

        Returns:
            HTML for trends section
        """
        if not trends or not trends.get("metrics"):
            return ""

        section_html = """
        <div class="section">
            <h2>Performance Trends</h2>
            <p style="color: #666; margin-bottom: 15px;">
                Based on the last {0} measurements
            </p>""".format(trends.get("sample_size", 0))

        for metric_name, trend_data in trends["metrics"].items():
            if not trend_data:
                continue

            direction = trend_data.get("direction", "stable")
            change_pct = trend_data.get("change_percent", 0)
            arrow = (
                "↓"
                if direction == "improving"
                else ("↑" if direction == "degrading" else "→")
            )

            section_html += f"""
            <div class="trend-item">
                <div class="trend-label">{metric_name.upper()}</div>
                <div class="trend-value {direction}">
                    {change_pct:+.1f}%
                    <span class="trend-arrow">{arrow}</span>
                </div>
            </div>"""

        section_html += """
            <div class="chart-placeholder">
                Chart visualization would be rendered here with a charting library
            </div>
        </div>"""

        return section_html

    def evaluate_metrics(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Evaluate metrics against thresholds.

        Args:
            metrics: Performance metrics to evaluate

        Returns:
            Dictionary with evaluation results for each metric

        Example:
            evaluation = reporter.evaluate_metrics(metrics)
            if evaluation["lcp"]["passes"]:
                print("LCP is good!")
        """
        evaluation = {}

        for metric_name, thresholds in self.THRESHOLDS.items():
            value = getattr(metrics, metric_name)
            good_threshold = thresholds["good"]
            poor_threshold = thresholds["poor"]

            # CLS is "lower is better", others too
            if value <= good_threshold:
                status = "good"
                passes = True
            elif value <= poor_threshold:
                status = "needs-improvement"
                passes = False
            else:
                status = "poor"
                passes = False

            evaluation[metric_name] = {
                "value": value,
                "status": status,
                "passes": passes,
                "threshold_good": good_threshold,
                "threshold_poor": poor_threshold,
            }

        return evaluation

    def calculate_trends(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Calculate performance trends from metrics history.

        Args:
            url: Filter history by URL (None for all URLs)

        Returns:
            Dictionary with trend analysis

        Example:
            reporter.add_metrics(metrics1)
            reporter.add_metrics(metrics2)
            reporter.add_metrics(metrics3)
            trends = reporter.calculate_trends()
            print(f"LCP trend: {trends['metrics']['lcp']['direction']}")
        """
        if len(self.metrics_history) < 2:
            logger.warning(
                "Not enough metrics data for trend analysis (need at least 2)"
            )
            return {}

        # Filter by URL if specified
        filtered_metrics = (
            [m for m in self.metrics_history if m.url == url]
            if url
            else self.metrics_history
        )

        if len(filtered_metrics) < 2:
            return {}

        trends = {
            "sample_size": len(filtered_metrics),
            "time_range": {
                "start": filtered_metrics[0].timestamp.isoformat(),
                "end": filtered_metrics[-1].timestamp.isoformat(),
            },
            "metrics": {},
        }

        # Analyze each metric
        metric_names = ["lcp", "fid", "cls", "ttfb", "fcp", "tti"]

        for metric_name in metric_names:
            values = [getattr(m, metric_name) for m in filtered_metrics]

            if len(values) < 2:
                continue

            # Calculate statistics
            first_value = values[0]
            last_value = values[-1]
            change = last_value - first_value
            change_percent = (change / first_value * 100) if first_value != 0 else 0

            # Determine direction (for performance, lower is better)
            if abs(change_percent) < 5:  # Less than 5% change is stable
                direction = "stable"
            elif change < 0:  # Value decreased
                direction = "improving"
            else:  # Value increased
                direction = "degrading"

            trends["metrics"][metric_name] = {
                "current": last_value,
                "previous": first_value,
                "change": change,
                "change_percent": change_percent,
                "direction": direction,
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }

        return trends

    def compare_metrics(
        self, metrics1: PerformanceMetrics, metrics2: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Compare two performance metrics snapshots.

        Args:
            metrics1: First metrics snapshot (baseline)
            metrics2: Second metrics snapshot (comparison)

        Returns:
            Dictionary with comparison results

        Example:
            comparison = reporter.compare_metrics(baseline_metrics, new_metrics)
            if comparison["lcp"]["is_regression"]:
                print("LCP regressed!")
        """
        comparison = {}

        metric_names = ["lcp", "fid", "cls", "ttfb", "fcp", "tti"]

        for metric_name in metric_names:
            value1 = getattr(metrics1, metric_name)
            value2 = getattr(metrics2, metric_name)

            change = value2 - value1
            change_percent = (change / value1 * 100) if value1 != 0 else 0

            # For performance metrics, increase is regression
            is_regression = change > 0 and abs(change_percent) > 5  # 5% threshold

            comparison[metric_name] = {
                "baseline": value1,
                "current": value2,
                "change": change,
                "change_percent": change_percent,
                "is_regression": is_regression,
                "is_improvement": change < 0 and abs(change_percent) > 5,
            }

        return comparison

    def clear_history(self) -> None:
        """Clear the metrics history.

        Example:
            reporter.clear_history()  # Start fresh tracking
        """
        self.metrics_history.clear()
        logger.info("Metrics history cleared")
