"""Playwright browser automation integration.

This module provides the PlaywrightManager class which manages Playwright
browser instances, contexts, and pages. It handles browser lifecycle,
resource management, and provides the core browser control functionality.

CRITICAL: Proper cleanup is essential to avoid resource leaks.
"""

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
)
from typing import Optional, Dict, Any
import logging

from src.models.browser_models import BrowserType, Viewport

logger = logging.getLogger(__name__)


class PlaywrightManager:
    """Manage Playwright browser instances and contexts.

    This class handles the lifecycle of Playwright browsers, contexts, and pages.
    It ensures proper resource management and cleanup to prevent memory leaks.

    PATTERN: Reuse browser instances when possible, but create isolated contexts
    for each test to prevent interference.

    CRITICAL: Always call cleanup() or use as async context manager to ensure
    proper resource cleanup.
    """

    def __init__(self):
        """Initialize the Playwright manager."""
        self.playwright: Optional[Playwright] = None
        self.browsers: Dict[str, Browser] = {}
        self.contexts: Dict[str, BrowserContext] = {}
        self.pages: Dict[str, Page] = {}
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize Playwright instance.

        This must be called before using any browser functionality.

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return

        try:
            self.playwright = await async_playwright().start()
            self._initialized = True
            logger.info("Playwright initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            raise RuntimeError(f"Playwright initialization failed: {e}")

    async def launch_browser(
        self,
        browser_type: BrowserType = BrowserType.CHROMIUM,
        headless: bool = True,
        **options: Any,
    ) -> Browser:
        """Launch a browser instance.

        PATTERN: Reuse browser instances when possible to save resources.
        The same browser instance can be used to create multiple contexts.

        Args:
            browser_type: Type of browser to launch
            headless: Whether to run in headless mode
            **options: Additional browser launch options

        Returns:
            Browser instance

        Raises:
            RuntimeError: If browser fails to launch
        """
        if not self._initialized:
            await self.initialize()

        # Check if browser is already running
        browser_key = browser_type.value
        if browser_key in self.browsers:
            logger.debug(f"Reusing existing {browser_type.value} browser")
            return self.browsers[browser_key]

        try:
            # Get the browser launcher for the specified type
            browser_launcher = getattr(self.playwright, browser_type.value)

            # Launch the browser
            browser = await browser_launcher.launch(headless=headless, **options)

            # Store browser instance
            self.browsers[browser_key] = browser
            logger.info(f"Launched {browser_type.value} browser (headless={headless})")

            return browser
        except Exception as e:
            logger.error(f"Failed to launch {browser_type.value} browser: {e}")
            raise RuntimeError(f"Browser launch failed: {e}")

    async def create_context(
        self,
        browser: Browser,
        viewport: Optional[Viewport] = None,
        **options: Any,
    ) -> BrowserContext:
        """Create an isolated browser context.

        CRITICAL: Each test should use its own context to prevent interference.
        Contexts provide isolation similar to incognito mode.

        Args:
            browser: Browser instance to create context in
            viewport: Viewport configuration
            **options: Additional context options (e.g., geolocation, permissions)

        Returns:
            Browser context

        Raises:
            RuntimeError: If context creation fails
        """
        try:
            context_options: Dict[str, Any] = {}

            # Configure viewport if specified
            if viewport:
                context_options["viewport"] = {
                    "width": viewport.width,
                    "height": viewport.height,
                }
                context_options["device_scale_factor"] = viewport.device_scale_factor
                context_options["is_mobile"] = viewport.is_mobile
                context_options["has_touch"] = viewport.has_touch

            # Merge with additional options
            context_options.update(options)

            # Create context
            context = await browser.new_context(**context_options)

            # Generate context ID
            context_id = f"context_{id(context)}"
            self.contexts[context_id] = context

            logger.debug(f"Created browser context: {context_id}")
            return context
        except Exception as e:
            logger.error(f"Failed to create browser context: {e}")
            raise RuntimeError(f"Context creation failed: {e}")

    async def create_page(self, context: BrowserContext, **options: Any) -> Page:
        """Create a new page in the specified context.

        Args:
            context: Browser context to create page in
            **options: Additional page options

        Returns:
            Page instance

        Raises:
            RuntimeError: If page creation fails
        """
        try:
            page = await context.new_page()

            # Generate page ID
            page_id = f"page_{id(page)}"
            self.pages[page_id] = page

            logger.debug(f"Created page: {page_id}")
            return page
        except Exception as e:
            logger.error(f"Failed to create page: {e}")
            raise RuntimeError(f"Page creation failed: {e}")

    async def navigate(
        self, page: Page, url: str, wait_until: str = "load", timeout: int = 30000
    ) -> None:
        """Navigate page to URL.

        Args:
            page: Page instance
            url: Target URL
            wait_until: Wait condition (load, domcontentloaded, networkidle)
            timeout: Navigation timeout in milliseconds

        Raises:
            RuntimeError: If navigation fails
        """
        try:
            await page.goto(url, wait_until=wait_until, timeout=timeout)
            logger.debug(f"Navigated to {url}")
        except Exception as e:
            logger.error(f"Navigation to {url} failed: {e}")
            raise RuntimeError(f"Navigation failed: {e}")

    async def click(self, page: Page, selector: str, timeout: int = 5000) -> None:
        """Click an element.

        Args:
            page: Page instance
            selector: Element selector
            timeout: Timeout in milliseconds

        Raises:
            RuntimeError: If click fails
        """
        try:
            await page.click(selector, timeout=timeout)
            logger.debug(f"Clicked element: {selector}")
        except Exception as e:
            logger.error(f"Click on {selector} failed: {e}")
            raise RuntimeError(f"Click failed: {e}")

    async def fill(
        self, page: Page, selector: str, value: str, timeout: int = 5000
    ) -> None:
        """Fill an input element.

        Args:
            page: Page instance
            selector: Element selector
            value: Value to fill
            timeout: Timeout in milliseconds

        Raises:
            RuntimeError: If fill fails
        """
        try:
            await page.fill(selector, value, timeout=timeout)
            logger.debug(f"Filled {selector} with value")
        except Exception as e:
            logger.error(f"Fill on {selector} failed: {e}")
            raise RuntimeError(f"Fill failed: {e}")

    async def wait_for_selector(
        self,
        page: Page,
        selector: str,
        state: str = "visible",
        timeout: int = 5000,
    ) -> None:
        """Wait for selector to reach specified state.

        Args:
            page: Page instance
            selector: Element selector
            state: Element state (visible, hidden, attached, detached)
            timeout: Timeout in milliseconds

        Raises:
            RuntimeError: If wait fails
        """
        try:
            await page.wait_for_selector(selector, state=state, timeout=timeout)
            logger.debug(f"Element {selector} reached state: {state}")
        except Exception as e:
            logger.error(f"Wait for {selector} failed: {e}")
            raise RuntimeError(f"Wait failed: {e}")

    async def screenshot(
        self,
        page: Page,
        path: str,
        full_page: bool = False,
        **options: Any,
    ) -> bytes:
        """Capture page screenshot.

        Args:
            page: Page instance
            path: Path to save screenshot
            full_page: Whether to capture full scrollable page
            **options: Additional screenshot options

        Returns:
            Screenshot bytes

        Raises:
            RuntimeError: If screenshot fails
        """
        try:
            screenshot_bytes = await page.screenshot(
                path=path, full_page=full_page, **options
            )
            logger.debug(f"Captured screenshot to {path}")
            return screenshot_bytes
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            raise RuntimeError(f"Screenshot failed: {e}")

    async def evaluate(self, page: Page, expression: str, arg: Any = None) -> Any:
        """Evaluate JavaScript expression in page context.

        Args:
            page: Page instance
            expression: JavaScript expression to evaluate
            arg: Optional argument to pass to expression

        Returns:
            Evaluation result

        Raises:
            RuntimeError: If evaluation fails
        """
        try:
            if arg is not None:
                result = await page.evaluate(expression, arg)
            else:
                result = await page.evaluate(expression)
            return result
        except Exception as e:
            logger.error(f"JavaScript evaluation failed: {e}")
            raise RuntimeError(f"Evaluation failed: {e}")

    async def cleanup(self) -> None:
        """Clean up all browser resources.

        CRITICAL: Must be called to prevent resource leaks.
        Closes all pages, contexts, and browsers in the correct order.

        Raises:
            RuntimeError: If cleanup fails
        """
        errors = []

        try:
            # Close pages
            for page_id, page in list(self.pages.items()):
                try:
                    await page.close()
                    logger.debug(f"Closed page: {page_id}")
                except Exception as e:
                    errors.append(f"Failed to close page {page_id}: {e}")
            self.pages.clear()

            # Close contexts
            for context_id, context in list(self.contexts.items()):
                try:
                    await context.close()
                    logger.debug(f"Closed context: {context_id}")
                except Exception as e:
                    errors.append(f"Failed to close context {context_id}: {e}")
            self.contexts.clear()

            # Close browsers
            for browser_type, browser in list(self.browsers.items()):
                try:
                    await browser.close()
                    logger.debug(f"Closed browser: {browser_type}")
                except Exception as e:
                    errors.append(f"Failed to close browser {browser_type}: {e}")
            self.browsers.clear()

            # Stop Playwright
            if self.playwright:
                try:
                    await self.playwright.stop()
                    logger.info("Playwright stopped successfully")
                except Exception as e:
                    errors.append(f"Failed to stop Playwright: {e}")
                self.playwright = None

            self._initialized = False

            if errors:
                error_msg = "; ".join(errors)
                logger.warning(f"Cleanup completed with errors: {error_msg}")
                raise RuntimeError(f"Cleanup errors: {error_msg}")
            else:
                logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise
