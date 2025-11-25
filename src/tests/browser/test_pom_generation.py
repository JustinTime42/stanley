"""Tests for Page Object Model (POM) generation.

This module contains tests for the PageAnalyzer class which generates
POMs from web pages, detects frameworks, and extracts element information.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from playwright.async_api import Page, ElementHandle
from datetime import datetime

from src.browser.page_analyzer import PageAnalyzer
from src.models.browser_models import PageElement


@pytest.fixture
def analyzer():
    """Create a PageAnalyzer instance for testing."""
    return PageAnalyzer()


@pytest.fixture
def mock_page():
    """Create a mock Page instance."""
    page = AsyncMock(spec=Page)
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example Page")
    page.evaluate = AsyncMock()
    page.query_selector_all = AsyncMock(return_value=[])
    page.query_selector = AsyncMock()
    return page


@pytest.fixture
def mock_element():
    """Create a mock ElementHandle instance."""
    element = AsyncMock(spec=ElementHandle)
    element.is_visible = AsyncMock(return_value=True)
    element.evaluate = AsyncMock()
    return element


class TestPageAnalyzerInitialization:
    """Tests for PageAnalyzer initialization."""

    def test_init(self):
        """Test PageAnalyzer initialization."""
        analyzer = PageAnalyzer()
        assert analyzer._framework_cache == {}

    def test_framework_markers(self):
        """Test framework detection markers are defined."""
        assert "react" in PageAnalyzer.FRAMEWORK_MARKERS
        assert "vue" in PageAnalyzer.FRAMEWORK_MARKERS
        assert "angular" in PageAnalyzer.FRAMEWORK_MARKERS
        assert "svelte" in PageAnalyzer.FRAMEWORK_MARKERS

    def test_interactive_selectors(self):
        """Test interactive element selectors are defined."""
        assert "button" in PageAnalyzer.INTERACTIVE_SELECTORS
        assert "input" in PageAnalyzer.INTERACTIVE_SELECTORS
        assert "link" in PageAnalyzer.INTERACTIVE_SELECTORS


class TestFrameworkDetection:
    """Tests for framework detection."""

    @pytest.mark.asyncio
    async def test_detect_react(self, analyzer, mock_page):
        """Test React framework detection."""
        mock_page.evaluate = AsyncMock(side_effect=[True, False, False, False])

        framework = await analyzer._detect_framework(mock_page)

        assert framework == "react"
        assert analyzer._framework_cache[mock_page.url] == "react"

    @pytest.mark.asyncio
    async def test_detect_vue(self, analyzer, mock_page):
        """Test Vue framework detection."""
        mock_page.evaluate = AsyncMock(side_effect=[False, True, False, False])

        framework = await analyzer._detect_framework(mock_page)

        assert framework == "vue"
        assert analyzer._framework_cache[mock_page.url] == "vue"

    @pytest.mark.asyncio
    async def test_detect_angular(self, analyzer, mock_page):
        """Test Angular framework detection."""
        mock_page.evaluate = AsyncMock(side_effect=[False, False, True, False])

        framework = await analyzer._detect_framework(mock_page)

        assert framework == "angular"
        assert analyzer._framework_cache[mock_page.url] == "angular"

    @pytest.mark.asyncio
    async def test_detect_svelte(self, analyzer, mock_page):
        """Test Svelte framework detection."""
        mock_page.evaluate = AsyncMock(side_effect=[False, False, False, True])

        framework = await analyzer._detect_framework(mock_page)

        assert framework == "svelte"
        assert analyzer._framework_cache[mock_page.url] == "svelte"

    @pytest.mark.asyncio
    async def test_detect_no_framework(self, analyzer, mock_page):
        """Test when no framework is detected."""
        mock_page.evaluate = AsyncMock(return_value=False)

        framework = await analyzer._detect_framework(mock_page)

        assert framework is None
        assert analyzer._framework_cache[mock_page.url] is None

    @pytest.mark.asyncio
    async def test_framework_cache_hit(self, analyzer, mock_page):
        """Test framework detection uses cache."""
        analyzer._framework_cache[mock_page.url] = "react"
        mock_page.evaluate = AsyncMock()

        framework = await analyzer._detect_framework(mock_page)

        assert framework == "react"
        # Should not call evaluate if cached
        mock_page.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_framework_detection_error(self, analyzer, mock_page):
        """Test framework detection handles errors gracefully."""
        mock_page.evaluate = AsyncMock(side_effect=Exception("Evaluation failed"))

        framework = await analyzer._detect_framework(mock_page)

        assert framework is None


class TestElementExtraction:
    """Tests for element extraction."""

    @pytest.mark.asyncio
    async def test_extract_elements_success(self, analyzer, mock_page):
        """Test successful element extraction."""
        elements_data = [
            {
                "name": "submit-button",
                "selector": "#submit",
                "element_type": "button",
                "attributes": {"id": "submit", "type": "submit"},
                "text_content": "Submit",
                "is_visible": True,
                "is_enabled": True,
                "aria_label": None,
                "data_testid": "submit-btn",
                "role": None,
                "placeholder": None,
            }
        ]
        mock_page.evaluate = AsyncMock(return_value=elements_data)

        elements = await analyzer._extract_elements(mock_page)

        assert len(elements) == 1
        assert elements[0]["name"] == "submit-button"
        assert elements[0]["selector"] == "#submit"

    @pytest.mark.asyncio
    async def test_extract_elements_with_invisible(self, analyzer, mock_page):
        """Test extracting elements including invisible ones."""
        elements_data = [
            {
                "name": "hidden-div",
                "selector": "#hidden",
                "element_type": "div",
                "attributes": {},
                "text_content": "",
                "is_visible": False,
                "is_enabled": True,
                "aria_label": None,
                "data_testid": None,
                "role": None,
                "placeholder": None,
            }
        ]
        mock_page.evaluate = AsyncMock(return_value=elements_data)

        elements = await analyzer._extract_elements(mock_page, include_invisible=True)

        assert len(elements) == 1
        # Verify include_invisible was passed to JavaScript
        call_args = mock_page.evaluate.call_args[0]
        assert len(call_args) == 2  # Script and argument

    @pytest.mark.asyncio
    async def test_extract_elements_error(self, analyzer, mock_page):
        """Test element extraction error handling."""
        mock_page.evaluate = AsyncMock(side_effect=Exception("Extraction failed"))

        elements = await analyzer._extract_elements(mock_page)

        assert elements == []


class TestInteractiveElementExtraction:
    """Tests for interactive element extraction."""

    @pytest.mark.asyncio
    async def test_extract_interactive_elements_buttons(self, analyzer, mock_page, mock_element):
        """Test extracting button elements."""
        mock_page.query_selector_all = AsyncMock(return_value=[mock_element])
        mock_element.evaluate = AsyncMock(
            return_value={
                "selector": "#submit",
                "tag_name": "button",
                "text": "Submit",
                "value": None,
                "placeholder": None,
                "aria_label": None,
                "data_testid": "submit-btn",
                "role": None,
                "type": "submit",
                "name": "submit",
                "disabled": False,
                "href": None,
            }
        )

        interactive = await analyzer._extract_interactive_elements(mock_page)

        assert "button" in interactive
        assert len(interactive["button"]) == 1
        assert interactive["button"][0]["selector"] == "#submit"

    @pytest.mark.asyncio
    async def test_extract_interactive_elements_inputs(self, analyzer, mock_page, mock_element):
        """Test extracting input elements."""
        mock_page.query_selector_all = AsyncMock(
            side_effect=lambda selector: [mock_element] if "input" in selector else []
        )
        mock_element.evaluate = AsyncMock(
            return_value={
                "selector": "#username",
                "tag_name": "input",
                "text": "",
                "value": "",
                "placeholder": "Enter username",
                "aria_label": "Username",
                "data_testid": "username-input",
                "role": None,
                "type": "text",
                "name": "username",
                "disabled": False,
                "href": None,
            }
        )

        interactive = await analyzer._extract_interactive_elements(mock_page)

        assert "input" in interactive
        assert len(interactive["input"]) == 1
        assert interactive["input"][0]["placeholder"] == "Enter username"

    @pytest.mark.asyncio
    async def test_extract_interactive_elements_invisible_filtered(
        self, analyzer, mock_page, mock_element
    ):
        """Test that invisible elements are filtered out."""
        mock_element.is_visible = AsyncMock(return_value=False)
        mock_page.query_selector_all = AsyncMock(return_value=[mock_element])

        interactive = await analyzer._extract_interactive_elements(mock_page)

        # All categories should be empty since element is invisible
        assert all(len(v) == 0 for v in interactive.values())

    @pytest.mark.asyncio
    async def test_extract_element_data_success(self, analyzer, mock_element):
        """Test extracting data from a single element."""
        mock_element.evaluate = AsyncMock(
            return_value={
                "selector": "[data-testid='login-btn']",
                "tag_name": "button",
                "text": "Login",
                "value": None,
                "placeholder": None,
                "aria_label": "Login button",
                "data_testid": "login-btn",
                "role": "button",
                "type": "button",
                "name": "login",
                "disabled": False,
                "href": None,
            }
        )

        element_data = await analyzer._extract_element_data(mock_element, "button")

        assert element_data is not None
        assert element_data["element_type"] == "button"
        assert element_data["data_testid"] == "login-btn"
        assert element_data["aria_label"] == "Login button"

    @pytest.mark.asyncio
    async def test_extract_element_data_error(self, analyzer, mock_element):
        """Test element data extraction error handling."""
        mock_element.evaluate = AsyncMock(side_effect=Exception("Evaluation failed"))

        element_data = await analyzer._extract_element_data(mock_element, "button")

        assert element_data is None


class TestComponentExtraction:
    """Tests for framework component extraction."""

    @pytest.mark.asyncio
    async def test_extract_react_components(self, analyzer, mock_page):
        """Test extracting React component hierarchy."""
        react_components = [
            {
                "name": "App",
                "type": "App",
                "depth": 0,
                "props": {"title": "My App"},
                "children": [],
            },
            {
                "name": "Header",
                "type": "Header",
                "depth": 1,
                "props": {"logo": "logo.png"},
                "children": [],
            },
        ]
        mock_page.evaluate = AsyncMock(return_value=react_components)

        components = await analyzer._extract_react_components(mock_page, max_depth=10)

        assert len(components) == 2
        assert components[0]["name"] == "App"
        assert components[1]["name"] == "Header"

    @pytest.mark.asyncio
    async def test_extract_vue_components(self, analyzer, mock_page):
        """Test extracting Vue component hierarchy."""
        vue_components = [
            {
                "name": "AppComponent",
                "type": "vue-component",
                "depth": 0,
                "props": {"msg": "Hello Vue"},
                "data": {},
            }
        ]
        mock_page.evaluate = AsyncMock(return_value=vue_components)

        components = await analyzer._extract_vue_components(mock_page, max_depth=10)

        assert len(components) == 1
        assert components[0]["name"] == "AppComponent"
        assert components[0]["type"] == "vue-component"

    @pytest.mark.asyncio
    async def test_extract_angular_components(self, analyzer, mock_page):
        """Test extracting Angular component hierarchy."""
        angular_components = [
            {
                "name": "app-root",
                "type": "angular-component",
                "selector": "app-root",
                "attributes": {"ng-version": "15.0.0"},
            }
        ]
        mock_page.evaluate = AsyncMock(return_value=angular_components)

        components = await analyzer._extract_angular_components(mock_page, max_depth=10)

        assert len(components) == 1
        assert components[0]["name"] == "app-root"
        assert components[0]["type"] == "angular-component"

    @pytest.mark.asyncio
    async def test_extract_components_unsupported_framework(self, analyzer, mock_page):
        """Test component extraction for unsupported framework."""
        components = await analyzer._extract_components(
            mock_page, framework="unknown", max_depth=10
        )

        assert components == []

    @pytest.mark.asyncio
    async def test_extract_components_error(self, analyzer, mock_page):
        """Test component extraction error handling."""
        mock_page.evaluate = AsyncMock(side_effect=Exception("Component extraction failed"))

        components = await analyzer._extract_react_components(mock_page, max_depth=10)

        assert components == []


class TestMetadataExtraction:
    """Tests for metadata extraction."""

    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, analyzer, mock_page):
        """Test successful metadata extraction."""
        metadata = {
            "lang": "en",
            "charset": "UTF-8",
            "viewport": {"width": 1920, "height": 1080},
            "url": "https://example.com/page",
            "origin": "https://example.com",
            "pathname": "/page",
            "has_forms": True,
            "form_count": 2,
            "link_count": 10,
            "image_count": 5,
            "meta_description": "Example page description",
            "meta_keywords": "example, test",
        }
        mock_page.evaluate = AsyncMock(return_value=metadata)

        result = await analyzer._extract_metadata(mock_page)

        assert result["lang"] == "en"
        assert result["form_count"] == 2
        assert result["viewport"]["width"] == 1920

    @pytest.mark.asyncio
    async def test_extract_metadata_error(self, analyzer, mock_page):
        """Test metadata extraction error handling."""
        mock_page.evaluate = AsyncMock(side_effect=Exception("Metadata extraction failed"))

        result = await analyzer._extract_metadata(mock_page)

        assert result == {}


class TestPageAnalysis:
    """Tests for full page analysis."""

    @pytest.mark.asyncio
    async def test_analyze_page_success(self, analyzer, mock_page):
        """Test successful page analysis."""
        mock_page.evaluate = AsyncMock(
            side_effect=[
                # Framework detection calls
                True,  # React
                # Element extraction
                [
                    {
                        "name": "button1",
                        "selector": "#btn1",
                        "element_type": "button",
                        "attributes": {},
                        "text_content": "Click me",
                        "is_visible": True,
                        "is_enabled": True,
                        "aria_label": None,
                        "data_testid": None,
                        "role": None,
                        "placeholder": None,
                    }
                ],
                # React components
                [{"name": "App", "type": "App", "depth": 0, "props": {}, "children": []}],
                # Metadata
                {
                    "lang": "en",
                    "charset": "UTF-8",
                    "viewport": {"width": 1280, "height": 720},
                    "url": "https://example.com",
                    "origin": "https://example.com",
                    "pathname": "/",
                    "has_forms": False,
                    "form_count": 0,
                    "link_count": 0,
                    "image_count": 0,
                    "meta_description": None,
                    "meta_keywords": None,
                },
            ]
        )
        mock_page.query_selector_all = AsyncMock(return_value=[])

        result = await analyzer.analyze_page(mock_page)

        assert result["url"] == "https://example.com"
        assert result["title"] == "Example Page"
        assert result["framework"] == "react"
        assert len(result["elements"]) == 1
        assert len(result["components"]) == 1
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_analyze_page_no_framework(self, analyzer, mock_page):
        """Test page analysis when no framework is detected."""
        mock_page.evaluate = AsyncMock(
            side_effect=[
                # Framework detection - all False
                False,
                False,
                False,
                False,
                # Element extraction
                [],
                # Metadata
                {},
            ]
        )
        mock_page.query_selector_all = AsyncMock(return_value=[])

        result = await analyzer.analyze_page(mock_page)

        assert result["framework"] is None
        assert result["components"] == []

    @pytest.mark.asyncio
    async def test_analyze_page_error(self, analyzer, mock_page):
        """Test page analysis error handling."""
        mock_page.title = AsyncMock(side_effect=Exception("Page analysis failed"))

        with pytest.raises(RuntimeError, match="Failed to analyze page"):
            await analyzer.analyze_page(mock_page)


class TestElementFinders:
    """Tests for element finder methods."""

    @pytest.mark.asyncio
    async def test_get_element_by_testid_success(self, analyzer, mock_page, mock_element):
        """Test finding element by test ID."""
        mock_page.query_selector = AsyncMock(return_value=mock_element)
        mock_element.evaluate = AsyncMock(
            return_value={
                "selector": "[data-testid='my-element']",
                "tag_name": "div",
                "text": "Content",
                "value": None,
                "placeholder": None,
                "aria_label": None,
                "data_testid": "my-element",
                "role": None,
                "type": None,
                "name": None,
                "disabled": False,
                "href": None,
            }
        )

        element = await analyzer.get_element_by_testid(mock_page, "my-element")

        assert element is not None
        assert element["data_testid"] == "my-element"
        mock_page.query_selector.assert_called_once_with('[data-testid="my-element"]')

    @pytest.mark.asyncio
    async def test_get_element_by_testid_not_found(self, analyzer, mock_page):
        """Test finding element by test ID when not found."""
        mock_page.query_selector = AsyncMock(return_value=None)

        element = await analyzer.get_element_by_testid(mock_page, "nonexistent")

        assert element is None

    @pytest.mark.asyncio
    async def test_get_element_by_testid_error(self, analyzer, mock_page):
        """Test finding element by test ID error handling."""
        mock_page.query_selector = AsyncMock(side_effect=Exception("Query failed"))

        element = await analyzer.get_element_by_testid(mock_page, "my-element")

        assert element is None

    @pytest.mark.asyncio
    async def test_find_elements_by_text(self, analyzer, mock_page, mock_element):
        """Test finding elements by text content."""
        mock_page.query_selector_all = AsyncMock(return_value=[mock_element])
        mock_element.evaluate = AsyncMock(
            return_value={
                "selector": "button",
                "tag_name": "button",
                "text": "Click me",
                "value": None,
                "placeholder": None,
                "aria_label": None,
                "data_testid": None,
                "role": None,
                "type": "button",
                "name": None,
                "disabled": False,
                "href": None,
            }
        )

        elements = await analyzer.find_elements_by_text(mock_page, "Click me", exact=True)

        assert len(elements) == 1
        assert elements[0]["text"] == "Click me"

    @pytest.mark.asyncio
    async def test_find_elements_by_text_case_insensitive(self, analyzer, mock_page, mock_element):
        """Test finding elements by text (case insensitive)."""
        mock_page.query_selector_all = AsyncMock(return_value=[mock_element])
        mock_element.evaluate = AsyncMock(
            return_value={
                "selector": "a",
                "tag_name": "a",
                "text": "Home",
                "value": None,
                "placeholder": None,
                "aria_label": None,
                "data_testid": None,
                "role": None,
                "type": None,
                "name": None,
                "disabled": False,
                "href": "/home",
            }
        )

        elements = await analyzer.find_elements_by_text(mock_page, "home", exact=False)

        assert len(elements) == 1

    @pytest.mark.asyncio
    async def test_find_elements_by_role(self, analyzer, mock_page, mock_element):
        """Test finding elements by ARIA role."""
        mock_page.query_selector_all = AsyncMock(return_value=[mock_element])
        mock_element.evaluate = AsyncMock(
            return_value={
                "selector": "[role='navigation']",
                "tag_name": "nav",
                "text": "",
                "value": None,
                "placeholder": None,
                "aria_label": "Main navigation",
                "data_testid": None,
                "role": "navigation",
                "type": None,
                "name": None,
                "disabled": False,
                "href": None,
            }
        )

        elements = await analyzer.find_elements_by_role(mock_page, "navigation")

        assert len(elements) == 1
        assert elements[0]["role"] == "navigation"
        mock_page.query_selector_all.assert_called_once_with('[role="navigation"]')

    @pytest.mark.asyncio
    async def test_find_elements_by_role_error(self, analyzer, mock_page):
        """Test finding elements by role error handling."""
        mock_page.query_selector_all = AsyncMock(side_effect=Exception("Query failed"))

        elements = await analyzer.find_elements_by_role(mock_page, "button")

        assert elements == []


class TestSelectorPriority:
    """Tests for selector priority in element extraction."""

    @pytest.mark.asyncio
    async def test_selector_prefers_testid(self, analyzer, mock_page):
        """Test that data-testid is preferred for selectors."""
        elements_data = [
            {
                "name": "login-button",
                "selector": "[data-testid='login-btn']",
                "element_type": "button",
                "attributes": {"id": "login", "data-testid": "login-btn"},
                "text_content": "Login",
                "is_visible": True,
                "is_enabled": True,
                "aria_label": None,
                "data_testid": "login-btn",
                "role": None,
                "placeholder": None,
            }
        ]
        mock_page.evaluate = AsyncMock(return_value=elements_data)

        elements = await analyzer._extract_elements(mock_page)

        # Selector should use data-testid even though ID is available
        assert elements[0]["selector"] == "[data-testid='login-btn']"

    @pytest.mark.asyncio
    async def test_selector_uses_id_when_no_testid(self, analyzer, mock_page):
        """Test that ID is used when data-testid is not available."""
        elements_data = [
            {
                "name": "submit",
                "selector": "#submit",
                "element_type": "button",
                "attributes": {"id": "submit"},
                "text_content": "Submit",
                "is_visible": True,
                "is_enabled": True,
                "aria_label": None,
                "data_testid": None,
                "role": None,
                "placeholder": None,
            }
        ]
        mock_page.evaluate = AsyncMock(return_value=elements_data)

        elements = await analyzer._extract_elements(mock_page)

        assert elements[0]["selector"] == "#submit"
