"""Browser automation tools for agents.

This module provides browser automation tools that leverage Playwright for
E2E testing, visual regression, accessibility audits, performance monitoring,
and Page Object Model generation.

PATTERN: Tools with browser service integration
CRITICAL: All browser operations must be async and properly cleaned up
GOTCHA: Browser instances must be managed carefully to prevent resource leaks
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from ..base import BaseTool
from ...models.tool_models import (
    ToolCategory,
    ToolSchema,
    ToolParameter,
    ToolResult,
)
from ...models.browser_models import (
    BrowserType,
    Viewport,
)
from ...browser.playwright_integration import PlaywrightManager
from ...browser.accessibility_tester import AccessibilityTester
from ...browser.visual_tester import VisualTester
from ...browser.performance_monitor import PerformanceMonitor
from ...browser.page_analyzer import PageAnalyzer

logger = logging.getLogger(__name__)


class RunE2ETestTool(BaseTool):
    """
    Tool for executing E2E tests using recorded user journeys.

    PATTERN: Browser automation with journey playback
    CRITICAL: Must manage browser lifecycle properly
    GOTCHA: Ensure browser is cleaned up even on failure
    """

    def __init__(self):
        """Initialize E2E test tool."""
        super().__init__(
            name="run_e2e_test",
            category=ToolCategory.TESTING,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Execute E2E test using a user journey specification",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="test_spec",
                    type="object",
                    description="Test specification containing URL, steps, and assertions",
                    required=True,
                ),
                ToolParameter(
                    name="browser_type",
                    type="string",
                    description="Browser to use (chromium, firefox, webkit)",
                    required=False,
                    default="chromium",
                    enum=["chromium", "firefox", "webkit"],
                ),
                ToolParameter(
                    name="headless",
                    type="boolean",
                    description="Run browser in headless mode",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="viewport",
                    type="object",
                    description="Viewport configuration (width, height)",
                    required=False,
                ),
                ToolParameter(
                    name="timeout_ms",
                    type="integer",
                    description="Test timeout in milliseconds",
                    required=False,
                    default=30000,
                ),
            ],
            returns="E2E test results with step outcomes and screenshots",
            timeout_seconds=120,
        )

    async def execute(
        self,
        test_spec: Dict[str, Any],
        browser_type: str = "chromium",
        headless: bool = True,
        viewport: Optional[Dict[str, int]] = None,
        timeout_ms: int = 30000,
        **kwargs,
    ) -> ToolResult:
        """
        Execute E2E test.

        Args:
            test_spec: Test specification with URL, steps, assertions
            browser_type: Browser to use
            headless: Run in headless mode
            viewport: Viewport configuration
            timeout_ms: Test timeout

        Returns:
            ToolResult with test execution results
        """
        start_time = datetime.now()
        playwright_manager = None
        browser = None
        context = None
        page = None

        try:
            # Validate test spec
            if "url" not in test_spec or "steps" not in test_spec:
                raise ValueError("test_spec must contain 'url' and 'steps' fields")

            # Parse browser type
            browser_enum = BrowserType(browser_type)

            # Create viewport if provided
            viewport_obj = None
            if viewport:
                viewport_obj = Viewport(**viewport)

            # Initialize Playwright
            playwright_manager = PlaywrightManager()
            await playwright_manager.initialize()

            # Launch browser
            browser = await playwright_manager.launch_browser(
                browser_type=browser_enum,
                headless=headless,
            )

            # Create context
            context = await playwright_manager.create_context(
                browser=browser,
                viewport=viewport_obj,
            )

            # Create page
            page = await playwright_manager.create_page(context)

            # Navigate to URL
            await playwright_manager.navigate(page, test_spec["url"])

            # Execute steps
            step_results = []
            for i, step in enumerate(test_spec.get("steps", [])):
                try:
                    step_result = await self._execute_step(
                        playwright_manager, page, step
                    )
                    step_results.append(
                        {
                            "step": i + 1,
                            "action": step.get("action"),
                            "status": "success",
                            "result": step_result,
                        }
                    )
                except Exception as e:
                    logger.error(f"Step {i + 1} failed: {e}")
                    step_results.append(
                        {
                            "step": i + 1,
                            "action": step.get("action"),
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    break

            # Check assertions if present
            assertion_results = []
            for assertion in test_spec.get("assertions", []):
                try:
                    assertion_result = await self._check_assertion(page, assertion)
                    assertion_results.append(
                        {
                            "assertion": assertion,
                            "status": "passed" if assertion_result else "failed",
                        }
                    )
                except Exception as e:
                    assertion_results.append(
                        {
                            "assertion": assertion,
                            "status": "error",
                            "error": str(e),
                        }
                    )

            # Calculate execution time
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            # Determine overall success
            all_steps_passed = all(s["status"] == "success" for s in step_results)
            all_assertions_passed = all(
                a["status"] == "passed" for a in assertion_results
            )
            test_passed = all_steps_passed and (
                all_assertions_passed or not assertion_results
            )

            result = {
                "test_passed": test_passed,
                "url": test_spec["url"],
                "steps_executed": len(step_results),
                "steps_passed": sum(
                    1 for s in step_results if s["status"] == "success"
                ),
                "step_results": step_results,
                "assertion_results": assertion_results,
                "execution_time_ms": execution_time_ms,
            }

            return self._create_success_result(
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            logger.error(f"E2E test execution failed: {e}")
            return self._create_error_result(
                error=f"E2E test failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

        finally:
            # Clean up resources
            try:
                if page:
                    await page.close()
                if context:
                    await context.close()
                if browser:
                    await browser.close()
                if playwright_manager:
                    await playwright_manager.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    async def _execute_step(
        self,
        playwright_manager: PlaywrightManager,
        page: Any,
        step: Dict[str, Any],
    ) -> Any:
        """Execute a single test step."""
        action = step.get("action")
        target = step.get("target")
        value = step.get("value")

        if action == "click":
            await playwright_manager.click(page, target)
        elif action == "fill":
            await playwright_manager.fill(page, target, value)
        elif action == "navigate":
            await playwright_manager.navigate(page, value)
        elif action == "wait_for_selector":
            await playwright_manager.wait_for_selector(page, target)
        else:
            raise ValueError(f"Unknown action: {action}")

        return {"action": action, "target": target}

    async def _check_assertion(self, page: Any, assertion: str) -> bool:
        """Check an assertion on the page."""
        # Simple assertion checking - can be extended
        try:
            result = await page.evaluate(f"() => {{ return {assertion}; }}")
            return bool(result)
        except Exception as e:
            logger.error(f"Assertion check failed: {e}")
            return False


class CaptureScreenshotTool(BaseTool):
    """
    Tool for capturing screenshots of web pages.

    PATTERN: Simple browser automation
    CRITICAL: Ensure proper cleanup of browser resources
    """

    def __init__(self):
        """Initialize screenshot tool."""
        super().__init__(
            name="capture_screenshot",
            category=ToolCategory.TESTING,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Capture screenshot of a web page",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL of page to screenshot",
                    required=True,
                ),
                ToolParameter(
                    name="full_page",
                    type="boolean",
                    description="Capture full scrollable page",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to save screenshot",
                    required=False,
                ),
                ToolParameter(
                    name="browser_type",
                    type="string",
                    description="Browser to use",
                    required=False,
                    default="chromium",
                    enum=["chromium", "firefox", "webkit"],
                ),
            ],
            returns="Screenshot path",
            timeout_seconds=60,
        )

    async def execute(
        self,
        url: str,
        full_page: bool = False,
        path: Optional[str] = None,
        browser_type: str = "chromium",
        **kwargs,
    ) -> ToolResult:
        """
        Capture screenshot.

        Args:
            url: Page URL
            full_page: Capture full page
            path: Save path
            browser_type: Browser to use

        Returns:
            ToolResult with screenshot path
        """
        start_time = datetime.now()
        playwright_manager = None
        browser = None
        context = None
        page = None

        try:
            # Generate path if not provided
            if not path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"screenshots/screenshot_{timestamp}.png"

            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Parse browser type
            browser_enum = BrowserType(browser_type)

            # Initialize Playwright
            playwright_manager = PlaywrightManager()
            await playwright_manager.initialize()

            # Launch browser
            browser = await playwright_manager.launch_browser(
                browser_type=browser_enum,
                headless=True,
            )

            # Create context and page
            context = await playwright_manager.create_context(browser)
            page = await playwright_manager.create_page(context)

            # Navigate and capture
            await playwright_manager.navigate(page, url)
            await playwright_manager.screenshot(page, path, full_page=full_page)

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            result = {
                "screenshot_path": str(Path(path).absolute()),
                "url": url,
                "full_page": full_page,
            }

            return self._create_success_result(
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            logger.error(f"Screenshot capture failed: {e}")
            return self._create_error_result(
                error=f"Screenshot failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

        finally:
            # Clean up resources
            try:
                if page:
                    await page.close()
                if context:
                    await context.close()
                if browser:
                    await browser.close()
                if playwright_manager:
                    await playwright_manager.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


class CheckAccessibilityTool(BaseTool):
    """
    Tool for running accessibility audits on web pages.

    PATTERN: Browser automation with specialized testing library
    CRITICAL: Inject axe-core library for WCAG compliance testing
    """

    def __init__(self):
        """Initialize accessibility tool."""
        super().__init__(
            name="check_accessibility",
            category=ToolCategory.TESTING,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Run accessibility audit using axe-core for WCAG compliance",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL of page to audit",
                    required=True,
                ),
                ToolParameter(
                    name="wcag_level",
                    type="string",
                    description="WCAG conformance level",
                    required=False,
                    default="AA",
                    enum=["A", "AA", "AAA"],
                ),
                ToolParameter(
                    name="include_best_practices",
                    type="boolean",
                    description="Include best practice rules",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="browser_type",
                    type="string",
                    description="Browser to use",
                    required=False,
                    default="chromium",
                    enum=["chromium", "firefox", "webkit"],
                ),
            ],
            returns="List of accessibility issues with severity and fix suggestions",
            timeout_seconds=60,
        )

    async def execute(
        self,
        url: str,
        wcag_level: str = "AA",
        include_best_practices: bool = True,
        browser_type: str = "chromium",
        **kwargs,
    ) -> ToolResult:
        """
        Run accessibility audit.

        Args:
            url: Page URL
            wcag_level: WCAG level (A, AA, AAA)
            include_best_practices: Include best practices
            browser_type: Browser to use

        Returns:
            ToolResult with accessibility issues
        """
        start_time = datetime.now()
        playwright_manager = None
        browser = None
        context = None
        page = None

        try:
            # Parse browser type
            browser_enum = BrowserType(browser_type)

            # Initialize Playwright
            playwright_manager = PlaywrightManager()
            await playwright_manager.initialize()

            # Launch browser
            browser = await playwright_manager.launch_browser(
                browser_type=browser_enum,
                headless=True,
            )

            # Create context and page
            context = await playwright_manager.create_context(browser)
            page = await playwright_manager.create_page(context)

            # Navigate to URL
            await playwright_manager.navigate(page, url)

            # Run accessibility audit
            tester = AccessibilityTester()
            issues = await tester.run_audit(
                page,
                wcag_level=wcag_level,
                include_best_practices=include_best_practices,
            )

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            # Convert issues to JSON-serializable format
            issues_data = [
                {
                    "id": issue.id,
                    "impact": issue.impact,
                    "rule_id": issue.rule_id,
                    "description": issue.description,
                    "help_text": issue.help_text,
                    "selector": issue.selector,
                    "html": issue.html,
                    "wcag_criteria": issue.wcag_criteria,
                    "wcag_level": issue.wcag_level,
                    "fix_suggestion": issue.fix_suggestion,
                }
                for issue in issues
            ]

            # Count issues by impact
            impact_counts = {
                "critical": sum(1 for i in issues if i.impact == "critical"),
                "serious": sum(1 for i in issues if i.impact == "serious"),
                "moderate": sum(1 for i in issues if i.impact == "moderate"),
                "minor": sum(1 for i in issues if i.impact == "minor"),
            }

            result = {
                "url": url,
                "wcag_level": wcag_level,
                "total_issues": len(issues),
                "impact_counts": impact_counts,
                "issues": issues_data,
                "passed": len(issues) == 0,
            }

            return self._create_success_result(
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            logger.error(f"Accessibility audit failed: {e}")
            return self._create_error_result(
                error=f"Accessibility audit failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

        finally:
            # Clean up resources
            try:
                if page:
                    await page.close()
                if context:
                    await context.close()
                if browser:
                    await browser.close()
                if playwright_manager:
                    await playwright_manager.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


class GeneratePOMTool(BaseTool):
    """
    Tool for generating Page Object Models from web pages.

    PATTERN: Browser automation with page analysis
    CRITICAL: Extract element data and framework information
    GOTCHA: Framework detection may not work for all frameworks
    """

    def __init__(self):
        """Initialize POM generation tool."""
        super().__init__(
            name="generate_pom",
            category=ToolCategory.TESTING,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Generate Page Object Model from a web page",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL of page to analyze",
                    required=True,
                ),
                ToolParameter(
                    name="framework",
                    type="string",
                    description="Expected frontend framework (optional)",
                    required=False,
                    enum=["react", "vue", "angular", "svelte"],
                ),
                ToolParameter(
                    name="include_invisible",
                    type="boolean",
                    description="Include invisible elements",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="browser_type",
                    type="string",
                    description="Browser to use",
                    required=False,
                    default="chromium",
                    enum=["chromium", "firefox", "webkit"],
                ),
            ],
            returns="Page Object Model with elements, framework info, and structure",
            timeout_seconds=60,
        )

    async def execute(
        self,
        url: str,
        framework: Optional[str] = None,
        include_invisible: bool = False,
        browser_type: str = "chromium",
        **kwargs,
    ) -> ToolResult:
        """
        Generate Page Object Model.

        Args:
            url: Page URL
            framework: Expected framework
            include_invisible: Include hidden elements
            browser_type: Browser to use

        Returns:
            ToolResult with POM data
        """
        start_time = datetime.now()
        playwright_manager = None
        browser = None
        context = None
        page = None

        try:
            # Parse browser type
            browser_enum = BrowserType(browser_type)

            # Initialize Playwright
            playwright_manager = PlaywrightManager()
            await playwright_manager.initialize()

            # Launch browser
            browser = await playwright_manager.launch_browser(
                browser_type=browser_enum,
                headless=True,
            )

            # Create context and page
            context = await playwright_manager.create_context(browser)
            page = await playwright_manager.create_page(context)

            # Navigate to URL
            await playwright_manager.navigate(page, url)

            # Analyze page
            analyzer = PageAnalyzer()
            page_data = await analyzer.analyze_page(
                page,
                include_invisible=include_invisible,
            )

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            # Build POM structure
            pom_id = f"pom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Convert elements to PageElement format
            elements = {}
            for elem in page_data.get("elements", []):
                elem_name = elem.get("name", "unknown")
                elements[elem_name] = {
                    "name": elem.get("name"),
                    "selector": elem.get("selector"),
                    "element_type": elem.get("element_type"),
                    "attributes": elem.get("attributes", {}),
                    "text_content": elem.get("text_content"),
                    "is_visible": elem.get("is_visible", True),
                    "is_enabled": elem.get("is_enabled", True),
                    "aria_label": elem.get("aria_label"),
                    "data_testid": elem.get("data_testid"),
                }

            # Extract actions from interactive elements
            actions = []
            for category, elems in page_data.get("interactive_elements", {}).items():
                for elem in elems:
                    action_name = f"{category}_{elem.get('name', 'element')}"
                    actions.append(action_name)

            result = {
                "id": pom_id,
                "url": url,
                "name": page_data.get("title", "Page"),
                "framework": page_data.get("framework"),
                "elements": elements,
                "actions": actions,
                "components": page_data.get("components", []),
                "interactive_elements": page_data.get("interactive_elements", {}),
                "metadata": page_data.get("metadata", {}),
                "element_count": len(elements),
                "interactive_count": sum(
                    len(v) for v in page_data.get("interactive_elements", {}).values()
                ),
            }

            return self._create_success_result(
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            logger.error(f"POM generation failed: {e}")
            return self._create_error_result(
                error=f"POM generation failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

        finally:
            # Clean up resources
            try:
                if page:
                    await page.close()
                if context:
                    await context.close()
                if browser:
                    await browser.close()
                if playwright_manager:
                    await playwright_manager.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


class RunVisualTestTool(BaseTool):
    """
    Tool for running visual regression tests.

    PATTERN: Browser automation with image comparison
    CRITICAL: Manage baseline images and diff generation
    GOTCHA: Image comparison threshold must be tuned per use case
    """

    def __init__(self):
        """Initialize visual test tool."""
        super().__init__(
            name="run_visual_test",
            category=ToolCategory.TESTING,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Run visual regression test comparing with baseline",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL of page to test",
                    required=True,
                ),
                ToolParameter(
                    name="baseline_path",
                    type="string",
                    description="Path to baseline screenshot",
                    required=False,
                ),
                ToolParameter(
                    name="threshold",
                    type="number",
                    description="Difference threshold (0.0-1.0)",
                    required=False,
                    default=0.01,
                ),
                ToolParameter(
                    name="full_page",
                    type="boolean",
                    description="Capture full page",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="test_id",
                    type="string",
                    description="Test identifier",
                    required=False,
                ),
                ToolParameter(
                    name="browser_type",
                    type="string",
                    description="Browser to use",
                    required=False,
                    default="chromium",
                    enum=["chromium", "firefox", "webkit"],
                ),
            ],
            returns="Visual test result with match percentage and diff image",
            timeout_seconds=60,
        )

    async def execute(
        self,
        url: str,
        baseline_path: Optional[str] = None,
        threshold: float = 0.01,
        full_page: bool = False,
        test_id: Optional[str] = None,
        browser_type: str = "chromium",
        **kwargs,
    ) -> ToolResult:
        """
        Run visual regression test.

        Args:
            url: Page URL
            baseline_path: Baseline screenshot path
            threshold: Difference threshold
            full_page: Capture full page
            test_id: Test identifier
            browser_type: Browser to use

        Returns:
            ToolResult with visual test results
        """
        start_time = datetime.now()
        playwright_manager = None
        browser = None
        context = None
        page = None

        try:
            # Generate test ID if not provided
            if not test_id:
                test_id = f"visual_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Parse browser type
            browser_enum = BrowserType(browser_type)

            # Initialize Playwright
            playwright_manager = PlaywrightManager()
            await playwright_manager.initialize()

            # Launch browser
            browser = await playwright_manager.launch_browser(
                browser_type=browser_enum,
                headless=True,
            )

            # Create context and page
            context = await playwright_manager.create_context(browser)
            page = await playwright_manager.create_page(context)

            # Navigate to URL
            await playwright_manager.navigate(page, url)

            # Run visual test
            tester = VisualTester(threshold=threshold)
            visual_result = await tester.capture_and_compare(
                page=page,
                test_id=test_id,
                full_page=full_page,
                browser=browser_enum,
            )

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            result = {
                "test_id": visual_result.test_id,
                "url": url,
                "is_match": visual_result.is_match,
                "match_percentage": visual_result.match_percentage,
                "pixel_difference": visual_result.pixel_difference,
                "baseline_path": visual_result.baseline_path,
                "actual_path": visual_result.actual_path,
                "diff_path": visual_result.diff_path,
                "threshold": visual_result.threshold,
                "diff_regions": visual_result.diff_regions,
                "passed": visual_result.is_match,
            }

            return self._create_success_result(
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            logger.error(f"Visual test failed: {e}")
            return self._create_error_result(
                error=f"Visual test failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

        finally:
            # Clean up resources
            try:
                if page:
                    await page.close()
                if context:
                    await context.close()
                if browser:
                    await browser.close()
                if playwright_manager:
                    await playwright_manager.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


class CollectPerformanceTool(BaseTool):
    """
    Tool for collecting web performance metrics.

    PATTERN: Browser automation with performance monitoring
    CRITICAL: Collect Core Web Vitals and resource timing
    GOTCHA: Metrics may vary between runs - collect multiple samples
    """

    def __init__(self):
        """Initialize performance collection tool."""
        super().__init__(
            name="collect_performance",
            category=ToolCategory.TESTING,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Collect performance metrics including Core Web Vitals",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL of page to measure",
                    required=True,
                ),
                ToolParameter(
                    name="browser_type",
                    type="string",
                    description="Browser to use",
                    required=False,
                    default="chromium",
                    enum=["chromium", "firefox", "webkit"],
                ),
                ToolParameter(
                    name="wait_for_stable",
                    type="boolean",
                    description="Wait for metrics to stabilize",
                    required=False,
                    default=True,
                ),
            ],
            returns="Performance metrics including LCP, FID, CLS, TTFB, and more",
            timeout_seconds=60,
        )

    async def execute(
        self,
        url: str,
        browser_type: str = "chromium",
        wait_for_stable: bool = True,
        **kwargs,
    ) -> ToolResult:
        """
        Collect performance metrics.

        Args:
            url: Page URL
            browser_type: Browser to use
            wait_for_stable: Wait for metrics to stabilize

        Returns:
            ToolResult with performance metrics
        """
        start_time = datetime.now()
        playwright_manager = None
        browser = None
        context = None
        page = None

        try:
            # Parse browser type
            browser_enum = BrowserType(browser_type)

            # Initialize Playwright
            playwright_manager = PlaywrightManager()
            await playwright_manager.initialize()

            # Launch browser
            browser = await playwright_manager.launch_browser(
                browser_type=browser_enum,
                headless=True,
            )

            # Create context and page
            context = await playwright_manager.create_context(browser)
            page = await playwright_manager.create_page(context)

            # Navigate to URL
            await playwright_manager.navigate(page, url, wait_until="networkidle")

            # Collect performance metrics
            monitor = PerformanceMonitor(browser_type=browser_enum)

            if wait_for_stable:
                await monitor.wait_for_metrics_stable(page)

            metrics = await monitor.collect_metrics(page, url)

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            result = {
                "url": metrics.url,
                "core_web_vitals": {
                    "lcp": metrics.lcp,
                    "fid": metrics.fid,
                    "cls": metrics.cls,
                    "passes_cwv": metrics.passes_cwv,
                },
                "load_metrics": {
                    "ttfb": metrics.ttfb,
                    "fcp": metrics.fcp,
                    "tti": metrics.tti,
                    "speed_index": metrics.speed_index,
                    "dom_content_loaded": metrics.dom_content_loaded,
                    "load_complete": metrics.load_complete,
                },
                "resources": {
                    "total_requests": metrics.total_requests,
                    "total_size_kb": metrics.total_size_kb,
                },
                "memory": {
                    "js_heap_size_mb": metrics.js_heap_size_mb,
                },
                "browser": metrics.browser.value,
                "timestamp": metrics.timestamp.isoformat(),
                "passes_cwv": metrics.passes_cwv,
            }

            return self._create_success_result(
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            logger.error(f"Performance collection failed: {e}")
            return self._create_error_result(
                error=f"Performance collection failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

        finally:
            # Clean up resources
            try:
                if page:
                    await page.close()
                if context:
                    await context.close()
                if browser:
                    await browser.close()
                if playwright_manager:
                    await playwright_manager.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
