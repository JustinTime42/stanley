"""Performance monitoring for browser automation.

This module provides the PerformanceMonitor class for collecting and analyzing
web performance metrics, including Core Web Vitals, navigation timing, resource
metrics, and memory usage.
"""

from typing import Dict, Any
import logging
from datetime import datetime

from playwright.async_api import Page

from src.models.browser_models import PerformanceMetrics, BrowserType

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and collect web performance metrics.

    This class collects comprehensive performance metrics from a web page,
    including Core Web Vitals (LCP, FID, CLS), additional metrics (TTFB, FCP,
    TTI, Speed Index), resource timing, network metrics, and memory usage.

    PATTERN: Use Performance Observer API via page.evaluate() for accurate
    real-user-monitoring metrics.
    """

    # Core Web Vitals thresholds (good performance)
    LCP_THRESHOLD = 2500  # 2.5 seconds
    FID_THRESHOLD = 100  # 100 milliseconds
    CLS_THRESHOLD = 0.1  # Cumulative Layout Shift

    def __init__(self, browser_type: BrowserType = BrowserType.CHROMIUM):
        """Initialize the performance monitor.

        Args:
            browser_type: Browser type for metadata
        """
        self.browser_type = browser_type

    async def collect_metrics(self, page: Page, url: str) -> PerformanceMetrics:
        """Collect comprehensive performance metrics from a page.

        This method uses the Performance Observer API and Navigation Timing API
        to collect real-user-monitoring metrics.

        Args:
            page: Playwright page instance
            url: Page URL being measured

        Returns:
            PerformanceMetrics object with all collected metrics

        Raises:
            Exception: If metric collection fails
        """
        logger.info(f"Collecting performance metrics for: {url}")

        try:
            # Collect Core Web Vitals and other metrics
            metrics_data = await self._collect_web_vitals(page)

            # Collect navigation timing metrics
            nav_timing = await self._collect_navigation_timing(page)
            metrics_data.update(nav_timing)

            # Collect resource metrics
            resource_metrics = await self._collect_resource_metrics(page)
            metrics_data.update(resource_metrics)

            # Collect memory metrics
            memory_metrics = await self._collect_memory_metrics(page)
            metrics_data.update(memory_metrics)

            # Check Core Web Vitals thresholds
            passes_cwv = self._check_cwv_thresholds(metrics_data)

            # Create PerformanceMetrics object
            performance_metrics = PerformanceMetrics(
                url=url,
                lcp=metrics_data.get("lcp", 0.0),
                fid=metrics_data.get("fid", 0.0),
                cls=metrics_data.get("cls", 0.0),
                ttfb=metrics_data.get("ttfb", 0.0),
                fcp=metrics_data.get("fcp", 0.0),
                tti=metrics_data.get("tti", 0.0),
                speed_index=metrics_data.get("speed_index", 0.0),
                dom_content_loaded=metrics_data.get("dom_content_loaded", 0.0),
                load_complete=metrics_data.get("load_complete", 0.0),
                total_requests=metrics_data.get("total_requests", 0),
                total_size_kb=metrics_data.get("total_size_kb", 0.0),
                js_heap_size_mb=metrics_data.get("js_heap_size_mb", 0.0),
                passes_cwv=passes_cwv,
                browser=self.browser_type,
                timestamp=datetime.now(),
            )

            logger.info(
                f"Metrics collected - LCP: {performance_metrics.lcp:.2f}ms, "
                f"FID: {performance_metrics.fid:.2f}ms, CLS: {performance_metrics.cls:.3f}, "
                f"Passes CWV: {passes_cwv}"
            )

            return performance_metrics

        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            raise

    async def _collect_web_vitals(self, page: Page) -> Dict[str, Any]:
        """Collect Core Web Vitals using Performance Observer API.

        This method collects:
        - LCP (Largest Contentful Paint)
        - FID (First Input Delay) - estimated using FCP as fallback
        - CLS (Cumulative Layout Shift)
        - FCP (First Contentful Paint)

        Args:
            page: Playwright page instance

        Returns:
            Dictionary with web vitals metrics
        """
        try:
            metrics = await page.evaluate("""
                () => {
                    return new Promise((resolve) => {
                        const metrics = {
                            lcp: 0,
                            fid: 0,
                            cls: 0,
                            fcp: 0
                        };

                        // Get paint timing (FCP)
                        const paintEntries = performance.getEntriesByType('paint');
                        const fcpEntry = paintEntries.find(entry => entry.name === 'first-contentful-paint');
                        if (fcpEntry) {
                            metrics.fcp = fcpEntry.startTime;
                        }

                        // Track LCP
                        let lcpObserved = false;
                        const lcpObserver = new PerformanceObserver((list) => {
                            const entries = list.getEntries();
                            const lastEntry = entries[entries.length - 1];
                            metrics.lcp = lastEntry.renderTime || lastEntry.loadTime;
                            lcpObserved = true;
                        });

                        try {
                            lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
                        } catch (e) {
                            // LCP not supported in this browser
                            console.warn('LCP not supported:', e);
                        }

                        // Track CLS
                        let clsValue = 0;
                        const clsObserver = new PerformanceObserver((list) => {
                            for (const entry of list.getEntries()) {
                                if (!entry.hadRecentInput) {
                                    clsValue += entry.value;
                                }
                            }
                            metrics.cls = clsValue;
                        });

                        try {
                            clsObserver.observe({ entryTypes: ['layout-shift'] });
                        } catch (e) {
                            // Layout shift not supported
                            console.warn('Layout shift not supported:', e);
                        }

                        // Track FID (First Input Delay)
                        const fidObserver = new PerformanceObserver((list) => {
                            const entries = list.getEntries();
                            const firstInput = entries[0];
                            if (firstInput) {
                                metrics.fid = firstInput.processingStart - firstInput.startTime;
                            }
                        });

                        try {
                            fidObserver.observe({ entryTypes: ['first-input'] });
                        } catch (e) {
                            // FID not supported
                            console.warn('FID not supported:', e);
                        }

                        // Wait for metrics to be collected
                        setTimeout(() => {
                            lcpObserver.disconnect();
                            clsObserver.disconnect();
                            fidObserver.disconnect();

                            // If LCP wasn't observed, try to get it from existing entries
                            if (!lcpObserved) {
                                const lcpEntries = performance.getEntriesByType('largest-contentful-paint');
                                if (lcpEntries.length > 0) {
                                    const lastLcp = lcpEntries[lcpEntries.length - 1];
                                    metrics.lcp = lastLcp.renderTime || lastLcp.loadTime;
                                }
                            }

                            resolve(metrics);
                        }, 1000);
                    });
                }
            """)

            logger.debug(f"Web vitals collected: {metrics}")
            return metrics

        except Exception as e:
            logger.warning(f"Failed to collect web vitals: {e}")
            return {"lcp": 0.0, "fid": 0.0, "cls": 0.0, "fcp": 0.0}

    async def _collect_navigation_timing(self, page: Page) -> Dict[str, Any]:
        """Collect navigation timing metrics.

        This method collects:
        - TTFB (Time to First Byte)
        - TTI (Time to Interactive) - estimated
        - DOM Content Loaded
        - Load Complete
        - Speed Index (approximation)

        Args:
            page: Playwright page instance

        Returns:
            Dictionary with navigation timing metrics
        """
        try:
            timing = await page.evaluate("""
                () => {
                    const perfData = performance.getEntriesByType('navigation')[0];
                    if (!perfData) {
                        return {
                            ttfb: 0,
                            tti: 0,
                            dom_content_loaded: 0,
                            load_complete: 0,
                            speed_index: 0
                        };
                    }

                    const timing = {
                        ttfb: perfData.responseStart - perfData.requestStart,
                        dom_content_loaded: perfData.domContentLoadedEventEnd - perfData.fetchStart,
                        load_complete: perfData.loadEventEnd - perfData.fetchStart,
                        // TTI approximation: DOMContentLoaded + (LoadComplete - DOMContentLoaded) * 0.8
                        tti: 0,
                        // Speed Index approximation based on FCP and load time
                        speed_index: 0
                    };

                    // Estimate TTI (Time to Interactive)
                    const domContentLoaded = timing.dom_content_loaded;
                    const loadComplete = timing.load_complete;
                    timing.tti = domContentLoaded + (loadComplete - domContentLoaded) * 0.8;

                    // Estimate Speed Index (simplified calculation)
                    // Speed Index ≈ (FCP + TTI) / 2
                    const paintEntries = performance.getEntriesByType('paint');
                    const fcpEntry = paintEntries.find(entry => entry.name === 'first-contentful-paint');
                    const fcp = fcpEntry ? fcpEntry.startTime : domContentLoaded * 0.5;
                    timing.speed_index = (fcp + timing.tti) / 2;

                    return timing;
                }
            """)

            logger.debug(f"Navigation timing collected: {timing}")
            return timing

        except Exception as e:
            logger.warning(f"Failed to collect navigation timing: {e}")
            return {
                "ttfb": 0.0,
                "tti": 0.0,
                "dom_content_loaded": 0.0,
                "load_complete": 0.0,
                "speed_index": 0.0,
            }

    async def _collect_resource_metrics(self, page: Page) -> Dict[str, Any]:
        """Collect resource timing and network metrics.

        This method collects:
        - Total number of network requests
        - Total transfer size in KB

        Args:
            page: Playwright page instance

        Returns:
            Dictionary with resource metrics
        """
        try:
            resources = await page.evaluate("""
                () => {
                    const resources = performance.getEntriesByType('resource');
                    const totalRequests = resources.length;

                    // Calculate total transfer size
                    const totalSize = resources.reduce((total, resource) => {
                        return total + (resource.transferSize || 0);
                    }, 0);

                    return {
                        total_requests: totalRequests,
                        total_size_kb: totalSize / 1024
                    };
                }
            """)

            logger.debug(
                f"Resource metrics collected: {resources['total_requests']} requests, "
                f"{resources['total_size_kb']:.2f} KB"
            )
            return resources

        except Exception as e:
            logger.warning(f"Failed to collect resource metrics: {e}")
            return {"total_requests": 0, "total_size_kb": 0.0}

    async def _collect_memory_metrics(self, page: Page) -> Dict[str, Any]:
        """Collect JavaScript heap size metrics.

        This method collects JS heap size if available (Chromium-based browsers).

        Args:
            page: Playwright page instance

        Returns:
            Dictionary with memory metrics
        """
        try:
            memory = await page.evaluate("""
                () => {
                    // performance.memory is only available in Chromium-based browsers
                    if (performance.memory) {
                        return {
                            js_heap_size_mb: performance.memory.usedJSHeapSize / (1024 * 1024)
                        };
                    }
                    return {
                        js_heap_size_mb: 0
                    };
                }
            """)

            if memory["js_heap_size_mb"] > 0:
                logger.debug(
                    f"Memory metrics collected: {memory['js_heap_size_mb']:.2f} MB"
                )
            else:
                logger.debug(
                    "Memory metrics not available (browser may not support performance.memory)"
                )

            return memory

        except Exception as e:
            logger.warning(f"Failed to collect memory metrics: {e}")
            return {"js_heap_size_mb": 0.0}

    def _check_cwv_thresholds(self, metrics: Dict[str, Any]) -> bool:
        """Validate Core Web Vitals against thresholds.

        Checks if metrics meet the "good" thresholds:
        - LCP < 2.5s (2500ms)
        - FID < 100ms
        - CLS < 0.1

        Args:
            metrics: Dictionary with collected metrics

        Returns:
            True if all Core Web Vitals pass thresholds, False otherwise
        """
        lcp = metrics.get("lcp", float("inf"))
        fid = metrics.get("fid", float("inf"))
        cls = metrics.get("cls", float("inf"))

        lcp_passes = lcp <= self.LCP_THRESHOLD
        fid_passes = fid <= self.FID_THRESHOLD
        cls_passes = cls <= self.CLS_THRESHOLD

        logger.debug(
            f"CWV threshold check - LCP: {lcp_passes} ({lcp:.2f}ms <= {self.LCP_THRESHOLD}ms), "
            f"FID: {fid_passes} ({fid:.2f}ms <= {self.FID_THRESHOLD}ms), "
            f"CLS: {cls_passes} ({cls:.3f} <= {self.CLS_THRESHOLD})"
        )

        return lcp_passes and fid_passes and cls_passes

    async def wait_for_metrics_stable(self, page: Page, timeout_ms: int = 5000) -> None:
        """Wait for performance metrics to stabilize.

        This method waits for the page to be fully loaded and for performance
        metrics to be available. Useful before collecting metrics.

        Args:
            page: Playwright page instance
            timeout_ms: Maximum wait time in milliseconds

        Raises:
            TimeoutError: If metrics don't stabilize within timeout
        """
        try:
            logger.debug("Waiting for performance metrics to stabilize...")

            await page.evaluate(
                f"""
                () => {{
                    return new Promise((resolve, reject) => {{
                        const timeout = setTimeout(() => {{
                            reject(new Error('Timeout waiting for metrics'));
                        }}, {timeout_ms});

                        // Wait for load event
                        if (document.readyState === 'complete') {{
                            clearTimeout(timeout);
                            resolve();
                        }} else {{
                            window.addEventListener('load', () => {{
                                clearTimeout(timeout);
                                // Give a bit more time for observers to fire
                                setTimeout(resolve, 500);
                            }});
                        }}
                    }});
                }}
            """
            )

            logger.debug("Performance metrics stabilized")

        except Exception as e:
            logger.warning(f"Failed to wait for metrics stability: {e}")
            # Don't raise - we'll attempt to collect metrics anyway

    def get_performance_summary(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Get a human-readable summary of performance metrics.

        Args:
            metrics: PerformanceMetrics object

        Returns:
            Dictionary with formatted performance summary
        """
        return {
            "url": metrics.url,
            "core_web_vitals": {
                "lcp": f"{metrics.lcp:.2f}ms",
                "fid": f"{metrics.fid:.2f}ms",
                "cls": f"{metrics.cls:.3f}",
                "passes": metrics.passes_cwv,
            },
            "load_metrics": {
                "ttfb": f"{metrics.ttfb:.2f}ms",
                "fcp": f"{metrics.fcp:.2f}ms",
                "tti": f"{metrics.tti:.2f}ms",
                "speed_index": f"{metrics.speed_index:.2f}",
                "dom_content_loaded": f"{metrics.dom_content_loaded:.2f}ms",
                "load_complete": f"{metrics.load_complete:.2f}ms",
            },
            "resources": {
                "total_requests": metrics.total_requests,
                "total_size": f"{metrics.total_size_kb:.2f} KB",
            },
            "memory": {
                "js_heap_size": f"{metrics.js_heap_size_mb:.2f} MB",
            },
            "browser": metrics.browser.value,
            "timestamp": metrics.timestamp.isoformat(),
        }
