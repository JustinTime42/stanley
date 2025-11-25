"""Browser context and lifecycle management.

This module provides the BrowserContextManager for managing browser contexts
with proper resource cleanup and lifecycle management.
"""

from typing import Optional, Dict, Any
import logging
from contextlib import asynccontextmanager

from playwright.async_api import Browser, BrowserContext, Page

from src.browser.playwright_integration import PlaywrightManager
from src.models.browser_models import BrowserType, Viewport

logger = logging.getLogger(__name__)


class BrowserContextManager:
    """Manage browser contexts with automatic cleanup.

    This class provides context manager functionality for browser contexts,
    ensuring proper resource cleanup even if errors occur.

    PATTERN: Use context managers for automatic resource cleanup.
    """

    def __init__(self, playwright_manager: PlaywrightManager):
        """Initialize the browser context manager.

        Args:
            playwright_manager: Playwright manager instance
        """
        self.playwright_manager = playwright_manager
        self._active_contexts: Dict[str, BrowserContext] = {}
        self._active_pages: Dict[str, Page] = {}

    @asynccontextmanager
    async def create_browser(
        self,
        browser_type: BrowserType = BrowserType.CHROMIUM,
        headless: bool = True,
        **options: Any,
    ):
        """Create and manage a browser instance.

        Args:
            browser_type: Type of browser to launch
            headless: Whether to run in headless mode
            **options: Additional browser options

        Yields:
            Browser instance

        Example:
            async with browser_manager.create_browser() as browser:
                # Use browser
                pass
            # Browser automatically cleaned up
        """
        browser = None
        try:
            browser = await self.playwright_manager.launch_browser(
                browser_type=browser_type, headless=headless, **options
            )
            yield browser
        finally:
            if browser:
                try:
                    await browser.close()
                    logger.debug(f"Closed browser: {browser_type.value}")
                except Exception as e:
                    logger.error(f"Error closing browser: {e}")

    @asynccontextmanager
    async def create_context(
        self,
        browser: Browser,
        viewport: Optional[Viewport] = None,
        **options: Any,
    ):
        """Create and manage a browser context.

        Args:
            browser: Browser instance
            viewport: Viewport configuration
            **options: Additional context options

        Yields:
            Browser context

        Example:
            async with browser_manager.create_context(browser) as context:
                # Use context
                pass
            # Context automatically cleaned up
        """
        context = None
        context_id = None
        try:
            context = await self.playwright_manager.create_context(
                browser=browser, viewport=viewport, **options
            )
            context_id = f"context_{id(context)}"
            self._active_contexts[context_id] = context
            yield context
        finally:
            if context:
                try:
                    await context.close()
                    if context_id and context_id in self._active_contexts:
                        del self._active_contexts[context_id]
                    logger.debug(f"Closed context: {context_id}")
                except Exception as e:
                    logger.error(f"Error closing context: {e}")

    @asynccontextmanager
    async def create_page(
        self,
        context: BrowserContext,
        **options: Any,
    ):
        """Create and manage a page.

        Args:
            context: Browser context
            **options: Additional page options

        Yields:
            Page instance

        Example:
            async with browser_manager.create_page(context) as page:
                await page.goto("https://example.com")
                # Use page
            # Page automatically cleaned up
        """
        page = None
        page_id = None
        try:
            page = await self.playwright_manager.create_page(context, **options)
            page_id = f"page_{id(page)}"
            self._active_pages[page_id] = page
            yield page
        finally:
            if page:
                try:
                    await page.close()
                    if page_id and page_id in self._active_pages:
                        del self._active_pages[page_id]
                    logger.debug(f"Closed page: {page_id}")
                except Exception as e:
                    logger.error(f"Error closing page: {e}")

    async def cleanup_all(self) -> None:
        """Clean up all active contexts and pages.

        This should be called when shutting down the browser manager.
        """
        # Close all pages
        for page_id, page in list(self._active_pages.items()):
            try:
                await page.close()
                logger.debug(f"Cleaned up page: {page_id}")
            except Exception as e:
                logger.error(f"Error cleaning up page {page_id}: {e}")
        self._active_pages.clear()

        # Close all contexts
        for context_id, context in list(self._active_contexts.items()):
            try:
                await context.close()
                logger.debug(f"Cleaned up context: {context_id}")
            except Exception as e:
                logger.error(f"Error cleaning up context {context_id}: {e}")
        self._active_contexts.clear()

        logger.info("Browser context manager cleanup completed")
