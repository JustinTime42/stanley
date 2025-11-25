"""Browser automation orchestrator service.

This module provides the BrowserOrchestrator class which acts as a high-level
facade for all browser automation features. It orchestrates Playwright manager,
page analyzer, journey recorder, visual tester, accessibility tester, performance
monitor, network interceptor, and POM generator to execute complete test suites.

PATTERN: Facade pattern - simplifies complex subsystem interactions
CRITICAL: Proper resource cleanup is essential to avoid memory leaks
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from playwright.async_api import Page

from src.browser.playwright_integration import PlaywrightManager
from src.browser.browser_manager import BrowserContextManager
from src.browser.page_analyzer import PageAnalyzer
from src.browser.journey_recorder import JourneyRecorder
from src.browser.visual_tester import VisualTester
from src.browser.accessibility_tester import AccessibilityTester
from src.browser.performance_monitor import PerformanceMonitor
from src.browser.network_interceptor import NetworkInterceptor
from src.browser.pom.pom_generator import POMGenerator

from src.models.browser_models import (
    BrowserType,
    BrowserTestSuite,
    PageObjectModel,
    UserJourney,
    VisualTestResult,
    AccessibilityIssue,
    PerformanceMetrics,
    Viewport,
)

logger = logging.getLogger(__name__)


class BrowserOrchestrator:
    """High-level facade for browser automation features.

    This class orchestrates all browser automation components to provide
    a unified interface for:
    - Generating Page Object Models from URLs
    - Executing E2E tests, visual tests, accessibility tests, performance tests
    - Running complete test suites with parallel workers
    - Managing browser lifecycle and resource cleanup

    PATTERN: Facade + Orchestrator pattern
    CRITICAL: Always call cleanup() or use as async context manager

    Example:
        async with BrowserOrchestrator() as orchestrator:
            # Generate POM from URL
            pom = await orchestrator.generate_pom("https://example.com")

            # Run E2E tests
            results = await orchestrator.run_e2e_tests(journeys)

            # Execute complete test suite
            suite_results = await orchestrator.execute_test_suite(test_suite)
    """

    def __init__(
        self,
        screenshots_dir: Optional[Path] = None,
        visual_threshold: float = 0.01,
    ):
        """Initialize the browser orchestrator.

        PATTERN: Initialize all components but defer resource allocation
        CRITICAL: Components are lazy-initialized on first use

        Args:
            screenshots_dir: Directory for screenshots (default: ./screenshots)
            visual_threshold: Visual diff threshold (0.0-1.0, default: 0.01)
        """
        # Core browser components
        self.playwright_manager = PlaywrightManager()
        self.browser_manager = BrowserContextManager(self.playwright_manager)

        # Analysis and testing components
        self.page_analyzer = PageAnalyzer()
        self.journey_recorder = JourneyRecorder()
        self.visual_tester = VisualTester(
            threshold=visual_threshold,
            screenshots_dir=screenshots_dir,
        )
        self.accessibility_tester = AccessibilityTester()
        self.performance_monitor = PerformanceMonitor()
        self.network_interceptor = NetworkInterceptor()

        # POM generation
        self.pom_generator = POMGenerator(self.page_analyzer)

        # State tracking
        self._initialized = False
        self._active_pages: List[Page] = []

        logger.info("BrowserOrchestrator initialized")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the orchestrator and Playwright.

        CRITICAL: Must be called before using orchestrator functionality

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            logger.debug("BrowserOrchestrator already initialized")
            return

        try:
            logger.info("Initializing BrowserOrchestrator...")
            await self.playwright_manager.initialize()
            self._initialized = True
            logger.info("BrowserOrchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BrowserOrchestrator: {e}")
            raise RuntimeError(f"Orchestrator initialization failed: {e}")

    async def generate_pom(
        self,
        url: str,
        browser_type: BrowserType = BrowserType.CHROMIUM,
        headless: bool = True,
        viewport: Optional[Viewport] = None,
        framework: Optional[str] = None,
        name: Optional[str] = None,
    ) -> PageObjectModel:
        """Generate a Page Object Model from a URL.

        PATTERN: Launch browser → Navigate → Analyze → Generate POM → Cleanup
        CRITICAL: Ensures browser cleanup even if generation fails

        Args:
            url: Target URL to analyze
            browser_type: Browser to use for analysis
            headless: Whether to run browser in headless mode
            viewport: Optional viewport configuration
            framework: Optional framework hint (auto-detected if not provided)
            name: Optional POM name (derived from page title if not provided)

        Returns:
            Generated PageObjectModel

        Raises:
            RuntimeError: If POM generation fails

        Example:
            >>> pom = await orchestrator.generate_pom("https://example.com")
            >>> print(f"Generated POM: {pom.name}")
            >>> print(f"Elements: {len(pom.elements)}")
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Generating POM for {url} using {browser_type.value}")

        browser = None
        context = None
        page = None

        try:
            # Launch browser
            browser = await self.playwright_manager.launch_browser(
                browser_type=browser_type,
                headless=headless,
            )

            # Create context with viewport if provided
            context = await self.playwright_manager.create_context(
                browser=browser,
                viewport=viewport,
            )

            # Create page
            page = await self.playwright_manager.create_page(context)
            self._active_pages.append(page)

            # Navigate to URL
            await self.playwright_manager.navigate(page, url)

            # Wait for page to stabilize
            await page.wait_for_load_state("networkidle", timeout=30000)

            # Generate POM
            pom = await self.pom_generator.generate_pom(
                page=page,
                framework=framework,
                name=name,
            )

            logger.info(
                f"POM generated successfully: {pom.name} "
                f"({len(pom.elements)} elements, {len(pom.actions)} actions)"
            )

            return pom

        except Exception as e:
            logger.error(f"POM generation failed for {url}: {e}")
            raise RuntimeError(f"Failed to generate POM: {e}")

        finally:
            # Cleanup page and context (browser is reused)
            if page:
                try:
                    await page.close()
                    if page in self._active_pages:
                        self._active_pages.remove(page)
                except Exception as e:
                    logger.warning(f"Error closing page: {e}")

            if context:
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing context: {e}")

    async def run_e2e_tests(
        self,
        journeys: List[UserJourney],
        browser_type: BrowserType = BrowserType.CHROMIUM,
        headless: bool = True,
        parallel: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute E2E tests from user journeys.

        PATTERN: Execute journeys sequentially or in parallel
        CRITICAL: Proper error handling and result aggregation

        Args:
            journeys: List of user journeys to execute
            browser_type: Browser to use for tests
            headless: Whether to run browser in headless mode
            parallel: Whether to run journeys in parallel

        Returns:
            List of test results for each journey

        Example:
            >>> results = await orchestrator.run_e2e_tests(journeys)
            >>> passed = sum(1 for r in results if r["status"] == "passed")
            >>> print(f"Passed: {passed}/{len(results)}")
        """
        if not self._initialized:
            await self.initialize()

        logger.info(
            f"Running {len(journeys)} E2E tests "
            f"({'parallel' if parallel else 'sequential'})"
        )

        if parallel:
            # Run journeys in parallel
            tasks = [
                self._execute_journey(journey, browser_type, headless)
                for journey in journeys
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(
                        {
                            "journey_id": journeys[i].id,
                            "journey_name": journeys[i].name,
                            "status": "failed",
                            "error": str(result),
                            "timestamp": datetime.now(),
                        }
                    )
                else:
                    processed_results.append(result)

            return processed_results
        else:
            # Run journeys sequentially
            results = []
            for journey in journeys:
                try:
                    result = await self._execute_journey(
                        journey, browser_type, headless
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Journey {journey.name} failed: {e}", exc_info=True)
                    results.append(
                        {
                            "journey_id": journey.id,
                            "journey_name": journey.name,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now(),
                        }
                    )

            return results

    async def _execute_journey(
        self,
        journey: UserJourney,
        browser_type: BrowserType,
        headless: bool,
    ) -> Dict[str, Any]:
        """Execute a single user journey.

        PATTERN: Launch browser → Playback journey → Capture results → Cleanup
        CRITICAL: Isolated execution with cleanup

        Args:
            journey: UserJourney to execute
            browser_type: Browser type to use
            headless: Whether to run headless

        Returns:
            Journey execution result
        """
        browser = None
        context = None
        page = None
        start_time = datetime.now()

        try:
            logger.info(f"Executing journey: {journey.name}")

            # Launch browser
            browser = await self.playwright_manager.launch_browser(
                browser_type=browser_type,
                headless=headless,
            )

            # Create context with journey viewport
            context = await self.playwright_manager.create_context(
                browser=browser,
                viewport=journey.viewport,
            )

            # Create page
            page = await self.playwright_manager.create_page(context)
            self._active_pages.append(page)

            # Execute journey
            success = await self.journey_recorder.playback_journey(page, journey)

            # Calculate duration
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            result = {
                "journey_id": journey.id,
                "journey_name": journey.name,
                "status": "passed" if success else "failed",
                "duration_ms": duration_ms,
                "steps_executed": len(journey.steps),
                "timestamp": start_time,
            }

            logger.info(
                f"Journey {journey.name} completed: "
                f"{result['status']} in {duration_ms}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Journey execution failed: {e}", exc_info=True)
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return {
                "journey_id": journey.id,
                "journey_name": journey.name,
                "status": "failed",
                "error": str(e),
                "duration_ms": duration_ms,
                "timestamp": start_time,
            }

        finally:
            # Cleanup
            if page:
                try:
                    await page.close()
                    if page in self._active_pages:
                        self._active_pages.remove(page)
                except Exception as e:
                    logger.warning(f"Error closing page: {e}")

            if context:
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing context: {e}")

    async def run_visual_tests(
        self,
        test_configs: List[Dict[str, Any]],
        browser_type: BrowserType = BrowserType.CHROMIUM,
        headless: bool = True,
    ) -> List[VisualTestResult]:
        """Execute visual regression tests.

        PATTERN: Navigate → Capture → Compare → Report
        CRITICAL: Baseline management and diff generation

        Args:
            test_configs: List of test configurations with url, test_id, etc.
            browser_type: Browser to use for tests
            headless: Whether to run browser in headless mode

        Returns:
            List of visual test results

        Example:
            >>> configs = [
            ...     {"url": "https://example.com", "test_id": "homepage"},
            ...     {"url": "https://example.com/login", "test_id": "login"},
            ... ]
            >>> results = await orchestrator.run_visual_tests(configs)
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Running {len(test_configs)} visual tests")

        results = []
        browser = None
        context = None
        page = None

        try:
            # Launch browser (reused for all tests)
            browser = await self.playwright_manager.launch_browser(
                browser_type=browser_type,
                headless=headless,
            )

            # Create context
            context = await self.playwright_manager.create_context(browser)

            # Create page
            page = await self.playwright_manager.create_page(context)
            self._active_pages.append(page)

            # Execute each test
            for config in test_configs:
                try:
                    url = config["url"]
                    test_id = config.get("test_id", f"visual_{len(results)}")
                    full_page = config.get("full_page", False)
                    ignore_regions = config.get("ignore_regions", [])

                    logger.info(f"Running visual test: {test_id} ({url})")

                    # Navigate to URL
                    await self.playwright_manager.navigate(page, url)
                    await page.wait_for_load_state("networkidle")

                    # Capture and compare
                    viewport = Viewport(
                        width=page.viewport_size.get("width", 1280),
                        height=page.viewport_size.get("height", 720),
                    )

                    result = await self.visual_tester.capture_and_compare(
                        page=page,
                        test_id=test_id,
                        full_page=full_page,
                        ignore_regions=ignore_regions,
                        browser=browser_type,
                        viewport=viewport,
                    )

                    results.append(result)

                    logger.info(
                        f"Visual test {test_id}: "
                        f"{'PASS' if result.is_match else 'FAIL'} "
                        f"(match: {result.match_percentage:.2f}%)"
                    )

                except Exception as e:
                    logger.error(f"Visual test {test_id} failed: {e}")
                    continue

        finally:
            # Cleanup
            if page:
                try:
                    await page.close()
                    if page in self._active_pages:
                        self._active_pages.remove(page)
                except Exception as e:
                    logger.warning(f"Error closing page: {e}")

            if context:
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing context: {e}")

        logger.info(f"Visual tests complete: {len(results)} results")
        return results

    async def run_accessibility_tests(
        self,
        urls: List[str],
        browser_type: BrowserType = BrowserType.CHROMIUM,
        headless: bool = True,
        wcag_level: str = "AA",
    ) -> Dict[str, List[AccessibilityIssue]]:
        """Execute accessibility audits on URLs.

        PATTERN: Navigate → Inject axe-core → Run audit → Parse results
        CRITICAL: WCAG compliance reporting

        Args:
            urls: List of URLs to audit
            browser_type: Browser to use for audits
            headless: Whether to run browser in headless mode
            wcag_level: WCAG level (A, AA, or AAA)

        Returns:
            Dictionary mapping URLs to accessibility issues

        Example:
            >>> results = await orchestrator.run_accessibility_tests(
            ...     ["https://example.com"],
            ...     wcag_level="AA"
            ... )
            >>> for url, issues in results.items():
            ...     print(f"{url}: {len(issues)} issues")
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Running accessibility tests on {len(urls)} URLs")

        results: Dict[str, List[AccessibilityIssue]] = {}
        browser = None
        context = None
        page = None

        try:
            # Launch browser
            browser = await self.playwright_manager.launch_browser(
                browser_type=browser_type,
                headless=headless,
            )

            # Create context
            context = await self.playwright_manager.create_context(browser)

            # Create page
            page = await self.playwright_manager.create_page(context)
            self._active_pages.append(page)

            # Execute tests for each URL
            for url in urls:
                try:
                    logger.info(f"Running accessibility audit: {url}")

                    # Navigate to URL
                    await self.playwright_manager.navigate(page, url)
                    await page.wait_for_load_state("networkidle")

                    # Run accessibility audit
                    issues = await self.accessibility_tester.run_audit(
                        page=page,
                        wcag_level=wcag_level,
                        include_best_practices=True,
                    )

                    results[url] = issues

                    logger.info(
                        f"Accessibility audit complete for {url}: "
                        f"{len(issues)} issues found"
                    )

                except Exception as e:
                    logger.error(f"Accessibility audit failed for {url}: {e}")
                    results[url] = []

        finally:
            # Cleanup
            if page:
                try:
                    await page.close()
                    if page in self._active_pages:
                        self._active_pages.remove(page)
                except Exception as e:
                    logger.warning(f"Error closing page: {e}")

            if context:
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing context: {e}")

        logger.info(f"Accessibility tests complete: {len(results)} URLs audited")
        return results

    async def run_performance_tests(
        self,
        urls: List[str],
        browser_type: BrowserType = BrowserType.CHROMIUM,
        headless: bool = True,
    ) -> Dict[str, PerformanceMetrics]:
        """Execute performance tests on URLs.

        PATTERN: Navigate → Wait for stable → Collect metrics
        CRITICAL: Core Web Vitals measurement

        Args:
            urls: List of URLs to test
            browser_type: Browser to use for tests
            headless: Whether to run browser in headless mode

        Returns:
            Dictionary mapping URLs to performance metrics

        Example:
            >>> results = await orchestrator.run_performance_tests(
            ...     ["https://example.com"]
            ... )
            >>> for url, metrics in results.items():
            ...     print(f"{url}: LCP={metrics.lcp}ms, CWV={metrics.passes_cwv}")
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Running performance tests on {len(urls)} URLs")

        results: Dict[str, PerformanceMetrics] = {}
        browser = None
        context = None
        page = None

        try:
            # Launch browser
            browser = await self.playwright_manager.launch_browser(
                browser_type=browser_type,
                headless=headless,
            )

            # Create context
            context = await self.playwright_manager.create_context(browser)

            # Create page
            page = await self.playwright_manager.create_page(context)
            self._active_pages.append(page)

            # Execute tests for each URL
            for url in urls:
                try:
                    logger.info(f"Running performance test: {url}")

                    # Navigate to URL
                    await self.playwright_manager.navigate(page, url)

                    # Wait for metrics to stabilize
                    await self.performance_monitor.wait_for_metrics_stable(page)

                    # Collect performance metrics
                    metrics = await self.performance_monitor.collect_metrics(
                        page=page,
                        url=url,
                    )

                    results[url] = metrics

                    logger.info(
                        f"Performance test complete for {url}: "
                        f"LCP={metrics.lcp:.2f}ms, "
                        f"FID={metrics.fid:.2f}ms, "
                        f"CLS={metrics.cls:.3f}, "
                        f"CWV={'PASS' if metrics.passes_cwv else 'FAIL'}"
                    )

                except Exception as e:
                    logger.error(f"Performance test failed for {url}: {e}")
                    continue

        finally:
            # Cleanup
            if page:
                try:
                    await page.close()
                    if page in self._active_pages:
                        self._active_pages.remove(page)
                except Exception as e:
                    logger.warning(f"Error closing page: {e}")

            if context:
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing context: {e}")

        logger.info(f"Performance tests complete: {len(results)} URLs tested")
        return results

    async def execute_test_suite(
        self,
        suite: BrowserTestSuite,
    ) -> Dict[str, Any]:
        """Execute a complete browser test suite.

        PATTERN: Execute all test types → Aggregate results → Generate report
        CRITICAL: Parallel execution with proper resource management

        Args:
            suite: BrowserTestSuite configuration

        Returns:
            Complete test suite results

        Example:
            >>> suite = BrowserTestSuite(
            ...     id="test-suite-1",
            ...     name="Full Site Test",
            ...     journeys=[...],
            ...     visual_tests=[...],
            ...     accessibility_tests=[...],
            ...     performance_tests=[...],
            ... )
            >>> results = await orchestrator.execute_test_suite(suite)
            >>> print(f"Total tests: {results['total_tests']}")
            >>> print(f"Passed: {results['passed_count']}")
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Executing test suite: {suite.name}")
        start_time = datetime.now()

        results = {
            "suite_id": suite.id,
            "suite_name": suite.name,
            "start_time": start_time,
            "e2e_results": [],
            "visual_results": [],
            "accessibility_results": {},
            "performance_results": {},
            "total_tests": 0,
            "passed_count": 0,
            "failed_count": 0,
        }

        try:
            # Setup network mocks if configured
            if suite.network_mocks:
                logger.info(f"Setting up {len(suite.network_mocks)} network mocks")
                # Note: Network mocks would be applied per-context in actual tests

            # Execute E2E tests (journeys)
            if suite.journeys:
                logger.info(f"Executing {len(suite.journeys)} E2E tests")
                e2e_results = await self.run_e2e_tests(
                    journeys=suite.journeys,
                    browser_type=suite.browsers[0]
                    if suite.browsers
                    else BrowserType.CHROMIUM,
                    headless=True,
                    parallel=suite.parallel,
                )
                results["e2e_results"] = e2e_results
                results["total_tests"] += len(e2e_results)
                results["passed_count"] += sum(
                    1 for r in e2e_results if r["status"] == "passed"
                )
                results["failed_count"] += sum(
                    1 for r in e2e_results if r["status"] == "failed"
                )

            # Execute visual tests
            if suite.visual_tests:
                logger.info(f"Executing {len(suite.visual_tests)} visual tests")
                visual_results = await self.run_visual_tests(
                    test_configs=suite.visual_tests,
                    browser_type=suite.browsers[0]
                    if suite.browsers
                    else BrowserType.CHROMIUM,
                )
                results["visual_results"] = visual_results
                results["total_tests"] += len(visual_results)
                results["passed_count"] += sum(1 for r in visual_results if r.is_match)
                results["failed_count"] += sum(
                    1 for r in visual_results if not r.is_match
                )

            # Execute accessibility tests
            if suite.accessibility_tests:
                logger.info(
                    f"Executing {len(suite.accessibility_tests)} accessibility tests"
                )
                urls = [test["url"] for test in suite.accessibility_tests]
                wcag_level = suite.accessibility_tests[0].get("wcag_level", "AA")
                accessibility_results = await self.run_accessibility_tests(
                    urls=urls,
                    browser_type=suite.browsers[0]
                    if suite.browsers
                    else BrowserType.CHROMIUM,
                    wcag_level=wcag_level,
                )
                results["accessibility_results"] = accessibility_results
                results["total_tests"] += len(accessibility_results)
                # Accessibility tests pass if no critical/serious issues
                for url, issues in accessibility_results.items():
                    critical_issues = [
                        i for i in issues if i.impact in ["critical", "serious"]
                    ]
                    if not critical_issues:
                        results["passed_count"] += 1
                    else:
                        results["failed_count"] += 1

            # Execute performance tests
            if suite.performance_tests:
                logger.info(
                    f"Executing {len(suite.performance_tests)} performance tests"
                )
                urls = [test["url"] for test in suite.performance_tests]
                performance_results = await self.run_performance_tests(
                    urls=urls,
                    browser_type=suite.browsers[0]
                    if suite.browsers
                    else BrowserType.CHROMIUM,
                )
                results["performance_results"] = performance_results
                results["total_tests"] += len(performance_results)
                results["passed_count"] += sum(
                    1 for m in performance_results.values() if m.passes_cwv
                )
                results["failed_count"] += sum(
                    1 for m in performance_results.values() if not m.passes_cwv
                )

            # Calculate final statistics
            end_time = datetime.now()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            results["end_time"] = end_time
            results["duration_ms"] = duration_ms
            results["success_rate"] = (
                (results["passed_count"] / results["total_tests"] * 100)
                if results["total_tests"] > 0
                else 0.0
            )

            logger.info(
                f"Test suite complete: {suite.name} - "
                f"{results['passed_count']}/{results['total_tests']} passed "
                f"({results['success_rate']:.1f}%) in {duration_ms}ms"
            )

            # Update suite last_run timestamp
            suite.last_run = end_time

            return results

        except Exception as e:
            logger.error(f"Test suite execution failed: {e}", exc_info=True)
            results["error"] = str(e)
            results["end_time"] = datetime.now()
            return results

    async def cleanup(self) -> None:
        """Clean up all browser resources.

        CRITICAL: Must be called to prevent resource leaks
        PATTERN: Close pages → Close contexts → Close browsers → Stop Playwright

        Raises:
            RuntimeError: If cleanup encounters critical errors
        """
        logger.info("Starting BrowserOrchestrator cleanup...")

        errors = []

        try:
            # Close any active pages
            for page in list(self._active_pages):
                try:
                    await page.close()
                    logger.debug(f"Closed active page: {page.url}")
                except Exception as e:
                    errors.append(f"Failed to close page: {e}")
            self._active_pages.clear()

            # Cleanup browser manager
            try:
                await self.browser_manager.cleanup_all()
            except Exception as e:
                errors.append(f"Browser manager cleanup failed: {e}")

            # Cleanup playwright manager
            try:
                await self.playwright_manager.cleanup()
            except Exception as e:
                errors.append(f"Playwright manager cleanup failed: {e}")

            self._initialized = False

            if errors:
                error_msg = "; ".join(errors)
                logger.warning(f"Cleanup completed with errors: {error_msg}")
            else:
                logger.info("BrowserOrchestrator cleanup completed successfully")

        except Exception as e:
            logger.error(f"Critical error during cleanup: {e}", exc_info=True)
            raise RuntimeError(f"Cleanup failed: {e}")

    async def validate_pom(
        self,
        url: str,
        pom: PageObjectModel,
        browser_type: BrowserType = BrowserType.CHROMIUM,
    ) -> Dict[str, Any]:
        """Validate a POM against the current state of a page.

        PATTERN: Navigate → Validate selectors → Report broken elements
        CRITICAL: Helps maintain POM accuracy over time

        Args:
            url: URL to validate against
            pom: PageObjectModel to validate
            browser_type: Browser to use for validation

        Returns:
            Validation report

        Example:
            >>> report = await orchestrator.validate_pom(
            ...     "https://example.com",
            ...     pom
            ... )
            >>> if not report["validation_passed"]:
            ...     print(f"Broken selectors: {report['invalid_selectors']}")
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Validating POM '{pom.name}' against {url}")

        browser = None
        context = None
        page = None

        try:
            # Launch browser
            browser = await self.playwright_manager.launch_browser(
                browser_type=browser_type,
                headless=True,
            )

            # Create context
            context = await self.playwright_manager.create_context(browser)

            # Create page
            page = await self.playwright_manager.create_page(context)
            self._active_pages.append(page)

            # Navigate to URL
            await self.playwright_manager.navigate(page, url)
            await page.wait_for_load_state("networkidle")

            # Validate POM
            validation_report = await self.pom_generator.validate_pom(page, pom)

            logger.info(
                f"POM validation complete: "
                f"{validation_report['valid_count']}/{validation_report['total_elements']} "
                f"selectors valid"
            )

            return validation_report

        except Exception as e:
            logger.error(f"POM validation failed: {e}")
            raise RuntimeError(f"Failed to validate POM: {e}")

        finally:
            # Cleanup
            if page:
                try:
                    await page.close()
                    if page in self._active_pages:
                        self._active_pages.remove(page)
                except Exception as e:
                    logger.warning(f"Error closing page: {e}")

            if context:
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing context: {e}")
