"""Test result reporters for browser automation."""

from .html_reporter import HTMLReporter
from .accessibility_reporter import AccessibilityReporter
from .performance_reporter import PerformanceReporter

__all__ = [
    "HTMLReporter",
    "AccessibilityReporter",
    "PerformanceReporter",
]
