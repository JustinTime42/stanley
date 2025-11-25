"""Tests for PlaywrightManager class.

This module contains comprehensive tests for the Playwright browser automation
integration, including browser launch, context creation, page operations, and
resource cleanup.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from playwright.async_api import Browser, BrowserContext, Page

from src.browser.playwright_integration import PlaywrightManager
from src.models.browser_models import BrowserType, Viewport


@pytest.fixture
def manager():
    """Create a PlaywrightManager instance for testing."""
    return PlaywrightManager()


@pytest.fixture
def mock_playwright():
    """Create a mock Playwright instance."""
    playwright = AsyncMock()
    playwright.chromium = AsyncMock()
    playwright.firefox = AsyncMock()
    playwright.webkit = AsyncMock()
    return playwright


@pytest.fixture
def mock_browser():
    """Create a mock Browser instance."""
    browser = AsyncMock(spec=Browser)
    browser.new_context = AsyncMock()
    browser.close = AsyncMock()
    return browser


@pytest.fixture
def mock_context():
    """Create a mock BrowserContext instance."""
    context = AsyncMock(spec=BrowserContext)
    context.new_page = AsyncMock()
    context.close = AsyncMock()
    return context


@pytest.fixture
def mock_page():
    """Create a mock Page instance."""
    page = AsyncMock(spec=Page)
    page.goto = AsyncMock()
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"screenshot_data")
    page.evaluate = AsyncMock()
    page.close = AsyncMock()
    return page


class TestPlaywrightManagerInitialization:
    """Tests for PlaywrightManager initialization."""

    def test_init(self):
        """Test PlaywrightManager initialization."""
        manager = PlaywrightManager()
        assert manager.playwright is None
        assert manager.browsers == {}
        assert manager.contexts == {}
        assert manager.pages == {}
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, manager):
        """Test successful Playwright initialization."""
        with patch("src.browser.playwright_integration.async_playwright") as mock_async_pw:
            mock_pw_instance = AsyncMock()
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)

            await manager.initialize()

            assert manager.playwright is not None
            assert manager._initialized is True
            mock_async_pw.return_value.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, manager):
        """Test that initialize is idempotent."""
        with patch("src.browser.playwright_integration.async_playwright") as mock_async_pw:
            mock_pw_instance = AsyncMock()
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)

            await manager.initialize()
            await manager.initialize()

            # Should only call start once
            mock_async_pw.return_value.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, manager):
        """Test initialization failure handling."""
        with patch("src.browser.playwright_integration.async_playwright") as mock_async_pw:
            mock_async_pw.return_value.start = AsyncMock(
                side_effect=Exception("Initialization failed")
            )

            with pytest.raises(RuntimeError, match="Playwright initialization failed"):
                await manager.initialize()

            assert manager._initialized is False


class TestBrowserLaunch:
    """Tests for browser launch functionality."""

    @pytest.mark.asyncio
    async def test_launch_chromium(self, manager, mock_playwright, mock_browser):
        """Test launching Chromium browser."""
        manager.playwright = mock_playwright
        manager._initialized = True
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)

        browser = await manager.launch_browser(browser_type=BrowserType.CHROMIUM)

        assert browser is mock_browser
        assert "chromium" in manager.browsers
        mock_playwright.chromium.launch.assert_called_once_with(headless=True)

    @pytest.mark.asyncio
    async def test_launch_firefox(self, manager, mock_playwright, mock_browser):
        """Test launching Firefox browser."""
        manager.playwright = mock_playwright
        manager._initialized = True
        mock_playwright.firefox.launch = AsyncMock(return_value=mock_browser)

        browser = await manager.launch_browser(browser_type=BrowserType.FIREFOX)

        assert browser is mock_browser
        assert "firefox" in manager.browsers
        mock_playwright.firefox.launch.assert_called_once_with(headless=True)

    @pytest.mark.asyncio
    async def test_launch_webkit(self, manager, mock_playwright, mock_browser):
        """Test launching WebKit browser."""
        manager.playwright = mock_playwright
        manager._initialized = True
        mock_playwright.webkit.launch = AsyncMock(return_value=mock_browser)

        browser = await manager.launch_browser(browser_type=BrowserType.WEBKIT)

        assert browser is mock_browser
        assert "webkit" in manager.browsers
        mock_playwright.webkit.launch.assert_called_once_with(headless=True)

    @pytest.mark.asyncio
    async def test_launch_with_options(self, manager, mock_playwright, mock_browser):
        """Test launching browser with custom options."""
        manager.playwright = mock_playwright
        manager._initialized = True
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)

        await manager.launch_browser(
            browser_type=BrowserType.CHROMIUM,
            headless=False,
            slow_mo=100,
            devtools=True,
        )

        mock_playwright.chromium.launch.assert_called_once_with(
            headless=False, slow_mo=100, devtools=True
        )

    @pytest.mark.asyncio
    async def test_launch_browser_reuse(self, manager, mock_playwright, mock_browser):
        """Test that browsers are reused when already launched."""
        manager.playwright = mock_playwright
        manager._initialized = True
        manager.browsers["chromium"] = mock_browser
        mock_playwright.chromium.launch = AsyncMock()

        browser = await manager.launch_browser(browser_type=BrowserType.CHROMIUM)

        assert browser is mock_browser
        # Should not call launch again
        mock_playwright.chromium.launch.assert_not_called()

    @pytest.mark.asyncio
    async def test_launch_browser_not_initialized(self, manager, mock_playwright, mock_browser):
        """Test that launch_browser initializes Playwright if needed."""
        with patch("src.browser.playwright_integration.async_playwright") as mock_async_pw:
            mock_pw_instance = mock_playwright
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)
            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)

            browser = await manager.launch_browser()

            assert manager._initialized is True
            assert browser is mock_browser

    @pytest.mark.asyncio
    async def test_launch_browser_failure(self, manager, mock_playwright):
        """Test browser launch failure handling."""
        manager.playwright = mock_playwright
        manager._initialized = True
        mock_playwright.chromium.launch = AsyncMock(
            side_effect=Exception("Launch failed")
        )

        with pytest.raises(RuntimeError, match="Browser launch failed"):
            await manager.launch_browser()


class TestContextCreation:
    """Tests for browser context creation."""

    @pytest.mark.asyncio
    async def test_create_context_basic(self, manager, mock_browser, mock_context):
        """Test creating a basic browser context."""
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        context = await manager.create_context(mock_browser)

        assert context is mock_context
        assert len(manager.contexts) == 1
        mock_browser.new_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_context_with_viewport(self, manager, mock_browser, mock_context):
        """Test creating context with custom viewport."""
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        viewport = Viewport(width=1920, height=1080, device_scale_factor=2.0)

        context = await manager.create_context(mock_browser, viewport=viewport)

        assert context is mock_context
        call_args = mock_browser.new_context.call_args[1]
        assert call_args["viewport"]["width"] == 1920
        assert call_args["viewport"]["height"] == 1080
        assert call_args["device_scale_factor"] == 2.0

    @pytest.mark.asyncio
    async def test_create_context_mobile(self, manager, mock_browser, mock_context):
        """Test creating a mobile context."""
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        viewport = Viewport(
            width=375,
            height=667,
            is_mobile=True,
            has_touch=True,
        )

        await manager.create_context(mock_browser, viewport=viewport)

        call_args = mock_browser.new_context.call_args[1]
        assert call_args["is_mobile"] is True
        assert call_args["has_touch"] is True

    @pytest.mark.asyncio
    async def test_create_context_with_options(self, manager, mock_browser, mock_context):
        """Test creating context with additional options."""
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        await manager.create_context(
            mock_browser,
            locale="en-US",
            timezone_id="America/New_York",
            geolocation={"latitude": 40.7128, "longitude": -74.0060},
            permissions=["geolocation"],
        )

        call_args = mock_browser.new_context.call_args[1]
        assert call_args["locale"] == "en-US"
        assert call_args["timezone_id"] == "America/New_York"
        assert "geolocation" in call_args

    @pytest.mark.asyncio
    async def test_create_context_failure(self, manager, mock_browser):
        """Test context creation failure handling."""
        mock_browser.new_context = AsyncMock(
            side_effect=Exception("Context creation failed")
        )

        with pytest.raises(RuntimeError, match="Context creation failed"):
            await manager.create_context(mock_browser)


class TestPageCreation:
    """Tests for page creation."""

    @pytest.mark.asyncio
    async def test_create_page_basic(self, manager, mock_context, mock_page):
        """Test creating a page."""
        mock_context.new_page = AsyncMock(return_value=mock_page)

        page = await manager.create_page(mock_context)

        assert page is mock_page
        assert len(manager.pages) == 1
        mock_context.new_page.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_page_failure(self, manager, mock_context):
        """Test page creation failure handling."""
        mock_context.new_page = AsyncMock(side_effect=Exception("Page creation failed"))

        with pytest.raises(RuntimeError, match="Page creation failed"):
            await manager.create_page(mock_context)


class TestPageActions:
    """Tests for page interaction methods."""

    @pytest.mark.asyncio
    async def test_navigate_success(self, manager, mock_page):
        """Test successful page navigation."""
        mock_page.goto = AsyncMock()

        await manager.navigate(mock_page, "https://example.com")

        mock_page.goto.assert_called_once_with(
            "https://example.com", wait_until="load", timeout=30000
        )

    @pytest.mark.asyncio
    async def test_navigate_with_options(self, manager, mock_page):
        """Test navigation with custom options."""
        mock_page.goto = AsyncMock()

        await manager.navigate(
            mock_page, "https://example.com", wait_until="networkidle", timeout=60000
        )

        mock_page.goto.assert_called_once_with(
            "https://example.com", wait_until="networkidle", timeout=60000
        )

    @pytest.mark.asyncio
    async def test_navigate_failure(self, manager, mock_page):
        """Test navigation failure handling."""
        mock_page.goto = AsyncMock(side_effect=Exception("Navigation failed"))

        with pytest.raises(RuntimeError, match="Navigation failed"):
            await manager.navigate(mock_page, "https://example.com")

    @pytest.mark.asyncio
    async def test_click_success(self, manager, mock_page):
        """Test clicking an element."""
        mock_page.click = AsyncMock()

        await manager.click(mock_page, "#submit-button")

        mock_page.click.assert_called_once_with("#submit-button", timeout=5000)

    @pytest.mark.asyncio
    async def test_click_failure(self, manager, mock_page):
        """Test click failure handling."""
        mock_page.click = AsyncMock(side_effect=Exception("Click failed"))

        with pytest.raises(RuntimeError, match="Click failed"):
            await manager.click(mock_page, "#button")

    @pytest.mark.asyncio
    async def test_fill_success(self, manager, mock_page):
        """Test filling an input field."""
        mock_page.fill = AsyncMock()

        await manager.fill(mock_page, "#username", "testuser")

        mock_page.fill.assert_called_once_with("#username", "testuser", timeout=5000)

    @pytest.mark.asyncio
    async def test_fill_failure(self, manager, mock_page):
        """Test fill failure handling."""
        mock_page.fill = AsyncMock(side_effect=Exception("Fill failed"))

        with pytest.raises(RuntimeError, match="Fill failed"):
            await manager.fill(mock_page, "#input", "value")

    @pytest.mark.asyncio
    async def test_wait_for_selector_success(self, manager, mock_page):
        """Test waiting for selector."""
        mock_page.wait_for_selector = AsyncMock()

        await manager.wait_for_selector(mock_page, "#element", state="visible")

        mock_page.wait_for_selector.assert_called_once_with(
            "#element", state="visible", timeout=5000
        )

    @pytest.mark.asyncio
    async def test_wait_for_selector_hidden(self, manager, mock_page):
        """Test waiting for element to be hidden."""
        mock_page.wait_for_selector = AsyncMock()

        await manager.wait_for_selector(
            mock_page, "#spinner", state="hidden", timeout=10000
        )

        mock_page.wait_for_selector.assert_called_once_with(
            "#spinner", state="hidden", timeout=10000
        )

    @pytest.mark.asyncio
    async def test_wait_for_selector_failure(self, manager, mock_page):
        """Test wait for selector timeout."""
        mock_page.wait_for_selector = AsyncMock(
            side_effect=Exception("Timeout waiting for selector")
        )

        with pytest.raises(RuntimeError, match="Wait failed"):
            await manager.wait_for_selector(mock_page, "#element")

    @pytest.mark.asyncio
    async def test_screenshot_success(self, manager, mock_page):
        """Test capturing screenshot."""
        mock_page.screenshot = AsyncMock(return_value=b"screenshot_bytes")

        screenshot = await manager.screenshot(mock_page, "/tmp/test.png")

        assert screenshot == b"screenshot_bytes"
        mock_page.screenshot.assert_called_once_with(
            path="/tmp/test.png", full_page=False
        )

    @pytest.mark.asyncio
    async def test_screenshot_full_page(self, manager, mock_page):
        """Test capturing full page screenshot."""
        mock_page.screenshot = AsyncMock(return_value=b"screenshot_bytes")

        await manager.screenshot(mock_page, "/tmp/test.png", full_page=True)

        call_args = mock_page.screenshot.call_args[1]
        assert call_args["full_page"] is True

    @pytest.mark.asyncio
    async def test_screenshot_failure(self, manager, mock_page):
        """Test screenshot failure handling."""
        mock_page.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))

        with pytest.raises(RuntimeError, match="Screenshot failed"):
            await manager.screenshot(mock_page, "/tmp/test.png")

    @pytest.mark.asyncio
    async def test_evaluate_success(self, manager, mock_page):
        """Test JavaScript evaluation."""
        mock_page.evaluate = AsyncMock(return_value="result")

        result = await manager.evaluate(mock_page, "() => document.title")

        assert result == "result"
        mock_page.evaluate.assert_called_once_with("() => document.title")

    @pytest.mark.asyncio
    async def test_evaluate_with_arg(self, manager, mock_page):
        """Test JavaScript evaluation with argument."""
        mock_page.evaluate = AsyncMock(return_value=42)

        result = await manager.evaluate(
            mock_page, "(x) => x * 2", 21
        )

        assert result == 42
        mock_page.evaluate.assert_called_once_with("(x) => x * 2", 21)

    @pytest.mark.asyncio
    async def test_evaluate_failure(self, manager, mock_page):
        """Test evaluation failure handling."""
        mock_page.evaluate = AsyncMock(side_effect=Exception("Evaluation failed"))

        with pytest.raises(RuntimeError, match="Evaluation failed"):
            await manager.evaluate(mock_page, "() => { throw new Error(); }")


class TestCleanup:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_success(self, manager, mock_page, mock_context, mock_browser):
        """Test successful cleanup of all resources."""
        # Set up resources
        manager.pages["page1"] = mock_page
        manager.contexts["ctx1"] = mock_context
        manager.browsers["chromium"] = mock_browser
        mock_playwright = AsyncMock()
        manager.playwright = mock_playwright
        manager._initialized = True

        await manager.cleanup()

        # Verify cleanup
        mock_page.close.assert_called_once()
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()

        assert len(manager.pages) == 0
        assert len(manager.contexts) == 0
        assert len(manager.browsers) == 0
        assert manager.playwright is None
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_multiple_resources(self, manager):
        """Test cleanup with multiple resources."""
        # Create multiple mock resources
        page1 = AsyncMock(spec=Page)
        page2 = AsyncMock(spec=Page)
        context1 = AsyncMock(spec=BrowserContext)
        context2 = AsyncMock(spec=BrowserContext)
        browser1 = AsyncMock(spec=Browser)
        browser2 = AsyncMock(spec=Browser)

        manager.pages = {"page1": page1, "page2": page2}
        manager.contexts = {"ctx1": context1, "ctx2": context2}
        manager.browsers = {"chromium": browser1, "firefox": browser2}
        manager.playwright = AsyncMock()
        manager._initialized = True

        await manager.cleanup()

        # Verify all resources were closed
        page1.close.assert_called_once()
        page2.close.assert_called_once()
        context1.close.assert_called_once()
        context2.close.assert_called_once()
        browser1.close.assert_called_once()
        browser2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_errors(self, manager, mock_page, mock_context, mock_browser):
        """Test cleanup continues despite errors."""
        mock_page.close = AsyncMock(side_effect=Exception("Page close failed"))
        mock_context.close = AsyncMock(side_effect=Exception("Context close failed"))
        mock_browser.close = AsyncMock()

        manager.pages["page1"] = mock_page
        manager.contexts["ctx1"] = mock_context
        manager.browsers["chromium"] = mock_browser
        manager.playwright = AsyncMock()
        manager._initialized = True

        with pytest.raises(RuntimeError, match="Cleanup errors"):
            await manager.cleanup()

        # Verify browser still got closed despite earlier errors
        mock_browser.close.assert_called_once()
        assert len(manager.pages) == 0
        assert len(manager.contexts) == 0
        assert len(manager.browsers) == 0

    @pytest.mark.asyncio
    async def test_cleanup_empty(self, manager):
        """Test cleanup with no resources."""
        manager._initialized = True
        mock_playwright = AsyncMock()
        manager.playwright = mock_playwright

        await manager.cleanup()

        mock_playwright.stop.assert_called_once()
        assert manager.playwright is None
        assert manager._initialized is False


class TestContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager_enter_exit(self, mock_playwright, mock_browser):
        """Test using PlaywrightManager as async context manager."""
        with patch("src.browser.playwright_integration.async_playwright") as mock_async_pw:
            mock_pw_instance = mock_playwright
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)
            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)

            async with PlaywrightManager() as manager:
                assert manager._initialized is True
                assert manager.playwright is not None
                # Save reference before cleanup
                pw_ref = manager.playwright

            # Should be cleaned up after exit
            pw_ref.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_exception_handling(self, mock_playwright):
        """Test cleanup happens even when exception occurs."""
        with patch("src.browser.playwright_integration.async_playwright") as mock_async_pw:
            mock_pw_instance = mock_playwright
            mock_async_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)

            pw_ref = None
            try:
                async with PlaywrightManager() as manager:
                    pw_ref = manager.playwright
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Cleanup should still happen
            pw_ref.stop.assert_called_once()
