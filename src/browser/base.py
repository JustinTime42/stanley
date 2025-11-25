"""Base abstract class for browser automation.

This module defines the abstract interface that all browser automation
implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from src.models.browser_models import (
    BrowserType,
    Viewport,
    UserJourney,
    PageObjectModel,
    VisualTestResult,
    AccessibilityIssue,
    PerformanceMetrics,
    BrowserTestSuite,
)


class BaseBrowserAutomation(ABC):
    """Abstract base class for browser automation implementations.

    Defines the interface that all browser automation implementations
    must follow, ensuring consistency across different browser engines
    and automation frameworks.
    """

    @abstractmethod
    async def launch_browser(
        self,
        browser_type: BrowserType = BrowserType.CHROMIUM,
        headless: bool = True,
        **options: Any,
    ) -> Any:
        """Launch a browser instance.

        Args:
            browser_type: Type of browser to launch
            headless: Whether to run in headless mode
            **options: Additional browser launch options

        Returns:
            Browser instance

        Raises:
            RuntimeError: If browser fails to launch
        """
        pass

    @abstractmethod
    async def create_page(
        self, viewport: Optional[Viewport] = None, **options: Any
    ) -> Any:
        """Create a new browser page with optional viewport configuration.

        Args:
            viewport: Viewport configuration
            **options: Additional page options

        Returns:
            Page instance

        Raises:
            RuntimeError: If page creation fails
        """
        pass

    @abstractmethod
    async def navigate(self, url: str, wait_until: str = "load") -> None:
        """Navigate to a URL.

        Args:
            url: Target URL
            wait_until: Wait condition (load, domcontentloaded, networkidle)

        Raises:
            RuntimeError: If navigation fails
        """
        pass

    @abstractmethod
    async def execute_test(self, test_suite: BrowserTestSuite) -> Dict[str, Any]:
        """Execute a browser test suite.

        Args:
            test_suite: Test suite specification

        Returns:
            Test results dictionary

        Raises:
            RuntimeError: If test execution fails
        """
        pass

    @abstractmethod
    async def generate_pom(
        self, url: str, framework: Optional[str] = None
    ) -> PageObjectModel:
        """Generate Page Object Model from a URL.

        Args:
            url: Target URL to analyze
            framework: Frontend framework (react, vue, angular)

        Returns:
            Generated Page Object Model

        Raises:
            RuntimeError: If POM generation fails
        """
        pass

    @abstractmethod
    async def record_journey(self, start_url: str, journey_name: str) -> UserJourney:
        """Record a user journey.

        Args:
            start_url: Starting URL
            journey_name: Name for the journey

        Returns:
            Recorded user journey

        Raises:
            RuntimeError: If journey recording fails
        """
        pass

    @abstractmethod
    async def run_visual_test(
        self,
        url: str,
        baseline_path: str,
        threshold: float = 0.01,
        ignore_regions: Optional[List[Dict[str, int]]] = None,
    ) -> VisualTestResult:
        """Run visual regression test.

        Args:
            url: Page URL to test
            baseline_path: Path to baseline screenshot
            threshold: Acceptable difference threshold
            ignore_regions: Regions to ignore in comparison

        Returns:
            Visual test result

        Raises:
            RuntimeError: If visual test fails
        """
        pass

    @abstractmethod
    async def run_accessibility_test(
        self, url: str, wcag_level: str = "AA"
    ) -> List[AccessibilityIssue]:
        """Run accessibility audit.

        Args:
            url: Page URL to test
            wcag_level: WCAG compliance level (A, AA, AAA)

        Returns:
            List of accessibility issues

        Raises:
            RuntimeError: If accessibility test fails
        """
        pass

    @abstractmethod
    async def collect_performance_metrics(self, url: str) -> PerformanceMetrics:
        """Collect performance metrics.

        Args:
            url: Page URL to measure

        Returns:
            Performance metrics

        Raises:
            RuntimeError: If metrics collection fails
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up browser resources.

        CRITICAL: Must be called to prevent resource leaks.

        Raises:
            RuntimeError: If cleanup fails
        """
        pass
