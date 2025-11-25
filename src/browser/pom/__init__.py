"""Page Object Model (POM) generation and management.

This package provides POM auto-generation from DOM analysis,
smart element selectors, and framework-specific templates.
"""

from src.browser.pom.pom_generator import POMGenerator
from src.browser.pom.element_selector import SmartSelector

__all__ = [
    "POMGenerator",
    "SmartSelector",
]
