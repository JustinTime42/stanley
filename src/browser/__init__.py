"""Browser automation subsystem for Playwright integration and E2E testing.

This package provides comprehensive browser automation capabilities including:
- Playwright browser control and lifecycle management
- Page Object Model (POM) generation from DOM analysis
- User journey recording and playback
- Visual regression testing with screenshot comparison
- Accessibility testing with WCAG compliance
- Performance metrics collection (Core Web Vitals)
- Network request interception and mocking
"""

from src.browser.base import BaseBrowserAutomation
from src.browser.playwright_integration import PlaywrightManager
from src.browser.browser_manager import BrowserContextManager
from src.browser.page_analyzer import PageAnalyzer
from src.browser.journey_recorder import JourneyRecorder
from src.browser.visual_tester import VisualTester
from src.browser.accessibility_tester import AccessibilityTester
from src.browser.performance_monitor import PerformanceMonitor
from src.browser.network_interceptor import NetworkInterceptor

__all__ = [
    "BaseBrowserAutomation",
    "PlaywrightManager",
    "BrowserContextManager",
    "PageAnalyzer",
    "JourneyRecorder",
    "VisualTester",
    "AccessibilityTester",
    "PerformanceMonitor",
    "NetworkInterceptor",
]
