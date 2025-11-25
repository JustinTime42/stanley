"""Page Object Model (POM) generator from DOM analysis.

This module provides the POMGenerator class which automatically generates
PageObjectModel instances from web pages by analyzing DOM structure,
detecting frameworks, and creating smart selectors for interactive elements.

PATTERN: Analyze page structure → Generate smart selectors → Build POM with actions
CRITICAL: Generated POMs should be maintainable and framework-aware
"""

from playwright.async_api import Page
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import uuid

from src.models.browser_models import PageObjectModel, PageElement
from src.browser.page_analyzer import PageAnalyzer
from src.browser.pom.element_selector import SmartSelector

logger = logging.getLogger(__name__)


class POMGenerator:
    """Generate Page Object Models from web pages.

    This class analyzes web pages and generates structured PageObjectModel instances
    with smart selectors, framework-specific components, and action methods based on
    element types.

    PATTERN: PageAnalyzer → SmartSelector → PageObjectModel
    CRITICAL: Generated POMs should be immediately usable in test automation

    Example:
        >>> analyzer = PageAnalyzer()
        >>> generator = POMGenerator(analyzer)
        >>> pom = await generator.generate_pom(page)
        >>> print(pom.name)
        'Login Page'
        >>> print(pom.elements.keys())
        dict_keys(['username_input', 'password_input', 'submit_button'])
    """

    # Element type to action method mapping
    ACTION_TEMPLATES = {
        "button": ["click", "is_enabled", "get_text"],
        "input": ["fill", "clear", "get_value", "is_enabled"],
        "textarea": ["fill", "clear", "get_value"],
        "select": ["select_option", "get_selected", "get_options"],
        "checkbox": ["check", "uncheck", "is_checked"],
        "radio": ["check", "is_checked"],
        "link": ["click", "get_href", "get_text"],
        "form": ["submit", "reset"],
        "dialog": ["close", "is_visible"],
        "menu": ["expand", "collapse", "select_item"],
        "tab": ["click", "is_active"],
    }

    # Interactive element types to prioritize
    INTERACTIVE_TYPES = {
        "button",
        "input",
        "textarea",
        "select",
        "checkbox",
        "radio",
        "link",
        "form",
    }

    def __init__(self, page_analyzer: Optional[PageAnalyzer] = None):
        """Initialize the POM generator.

        PATTERN: Inject dependencies for testability
        CRITICAL: PageAnalyzer and SmartSelector are core dependencies

        Args:
            page_analyzer: PageAnalyzer instance (creates new if not provided)
        """
        self.page_analyzer = page_analyzer or PageAnalyzer()
        self.selector_generator = SmartSelector()

    async def generate_pom(
        self,
        page: Page,
        framework: Optional[str] = None,
        name: Optional[str] = None,
        include_invisible: bool = False,
    ) -> PageObjectModel:
        """Generate a complete Page Object Model from a page.

        PATTERN: Analyze → Extract → Generate selectors → Build POM
        CRITICAL: POM should capture all interactive elements with stable selectors

        Args:
            page: Playwright page instance to analyze
            framework: Optional framework hint (auto-detected if not provided)
            name: Optional POM name (derived from page title if not provided)
            include_invisible: Whether to include hidden elements

        Returns:
            PageObjectModel instance ready for test automation

        Raises:
            RuntimeError: If POM generation fails

        Example:
            >>> pom = await generator.generate_pom(page, framework="react")
            >>> login_btn = pom.elements["login_button"]
            >>> print(login_btn.selector)
            '[data-testid="login-submit"]'
        """
        try:
            logger.info(f"Starting POM generation for {page.url}")

            # Step 1: Analyze page structure
            analysis_result = await self.page_analyzer.analyze_page(
                page=page,
                include_invisible=include_invisible,
            )

            # Extract data from analysis
            url = analysis_result["url"]
            title = analysis_result["title"]
            detected_framework = analysis_result.get("framework")
            raw_elements = analysis_result["elements"]
            components = analysis_result.get("components", [])
            interactive_elements = analysis_result.get("interactive_elements", {})

            # Use provided framework or detected framework
            framework_to_use = framework or detected_framework

            logger.info(
                f"Analyzed page: {title}, Framework: {framework_to_use or 'None'}, "
                f"Elements: {len(raw_elements)}"
            )

            # Step 2: Generate smart selectors for elements
            elements_with_selectors = await self._generate_selectors_for_elements(
                page=page,
                elements=raw_elements,
                framework=framework_to_use,
            )

            logger.debug(
                f"Generated selectors for {len(elements_with_selectors)} elements"
            )

            # Step 3: Build PageElement models
            page_elements = self._build_page_elements(
                elements_with_selectors,
                interactive_elements,
            )

            logger.debug(f"Built {len(page_elements)} PageElement models")

            # Step 4: Generate action methods
            actions = self._generate_actions(page_elements, interactive_elements)

            logger.debug(f"Generated {len(actions)} action methods")

            # Step 5: Extract custom locators (framework-specific)
            custom_locators = self._extract_custom_locators(
                page_elements,
                framework_to_use,
            )

            # Step 6: Process components if framework detected
            processed_components = []
            if framework_to_use and components:
                processed_components = self._process_components(
                    components,
                    framework_to_use,
                )
                logger.debug(f"Processed {len(processed_components)} components")

            # Step 7: Create POM
            pom_id = str(uuid.uuid4())
            pom_name = name or self._generate_pom_name(title, url)

            pom = PageObjectModel(
                id=pom_id,
                url=url,
                name=pom_name,
                elements=page_elements,
                actions=actions,
                locator_strategy="css",
                custom_locators=custom_locators,
                components=processed_components,
                framework=framework_to_use,
                generated_at=datetime.now(),
                last_updated=datetime.now(),
            )

            logger.info(
                f"POM generation complete: {pom_name} "
                f"({len(page_elements)} elements, {len(actions)} actions)"
            )

            return pom

        except Exception as e:
            logger.error(f"POM generation failed: {e}")
            raise RuntimeError(f"Failed to generate POM: {e}")

    async def _generate_selectors_for_elements(
        self,
        page: Page,
        elements: List[Dict[str, Any]],
        framework: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate smart selectors for all elements.

        PATTERN: Process each element through SmartSelector
        CRITICAL: Validate selectors against actual page

        Args:
            page: Playwright page instance
            elements: List of raw element dictionaries
            framework: Framework name for framework-specific selectors

        Returns:
            Elements with generated smart selectors
        """
        elements_with_selectors = []

        for element in elements:
            try:
                # Generate best selector for this element
                smart_selector = await self.selector_generator.generate_selector(
                    element=element,
                    framework=framework,
                    page=page,
                )

                # Add selector to element
                element_copy = element.copy()
                element_copy["smart_selector"] = smart_selector
                elements_with_selectors.append(element_copy)

            except Exception as e:
                logger.debug(
                    f"Failed to generate selector for element "
                    f"{element.get('name', 'unknown')}: {e}"
                )
                # Use fallback selector
                element_copy = element.copy()
                element_copy["smart_selector"] = element.get(
                    "selector", element.get("element_type", "div")
                )
                elements_with_selectors.append(element_copy)

        return elements_with_selectors

    def _build_page_elements(
        self,
        elements: List[Dict[str, Any]],
        interactive_elements: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, PageElement]:
        """Build PageElement models from raw element data.

        PATTERN: Convert raw data to Pydantic models
        CRITICAL: Generate semantic element names

        Args:
            elements: Elements with smart selectors
            interactive_elements: Categorized interactive elements

        Returns:
            Dictionary mapping element names to PageElement instances
        """
        page_elements: Dict[str, PageElement] = {}
        name_counts: Dict[str, int] = {}

        # Prioritize interactive elements
        interactive_names = set()
        for category, category_elements in interactive_elements.items():
            for elem in category_elements:
                if elem.get("name"):
                    interactive_names.add(elem["name"])

        for element in elements:
            try:
                # Generate unique element name
                base_name = self._generate_element_name(element)
                element_name = self._make_unique_name(base_name, name_counts)

                # Use smart selector if available, otherwise fallback
                selector = element.get("smart_selector", element.get("selector", ""))

                # Create PageElement
                page_element = PageElement(
                    name=element_name,
                    selector=selector,
                    element_type=element.get("element_type", "div"),
                    attributes=element.get("attributes", {}),
                    text_content=element.get("text_content"),
                    is_visible=element.get("is_visible", True),
                    is_enabled=element.get("is_enabled", True),
                    aria_label=element.get("aria_label"),
                    data_testid=element.get("data_testid"),
                )

                page_elements[element_name] = page_element

            except Exception as e:
                logger.debug(f"Failed to build PageElement: {e}")
                continue

        return page_elements

    def _generate_element_name(self, element: Dict[str, Any]) -> str:
        """Generate a semantic name for an element.

        PATTERN: Derive name from attributes, text, or element type
        CRITICAL: Names should be readable and descriptive

        Args:
            element: Element data dictionary

        Returns:
            Generated element name

        Example:
            >>> element = {"data_testid": "login-submit", "element_type": "button"}
            >>> name = generator._generate_element_name(element)
            >>> name
            'login_submit_button'
        """
        # Priority: data-testid > id > name > aria-label > text > type
        if element.get("data_testid"):
            return self._sanitize_name(element["data_testid"])

        if element.get("attributes", {}).get("id"):
            return self._sanitize_name(element["attributes"]["id"])

        if element.get("name"):
            return self._sanitize_name(element["name"])

        if element.get("aria_label"):
            return self._sanitize_name(element["aria_label"])

        # Use text content for buttons and links
        element_type = element.get("element_type", "")
        if element_type in ["button", "a"] and element.get("text_content"):
            text = element["text_content"].strip()
            if text and len(text) < 50:
                return f"{self._sanitize_name(text)}_{element_type}"

        # Use placeholder for inputs
        if element_type in ["input", "textarea"] and element.get("placeholder"):
            return f"{self._sanitize_name(element['placeholder'])}_input"

        # Fallback to type with role
        role = element.get("role") or element.get("attributes", {}).get("role")
        if role:
            return f"{role}_{element_type}"

        # Ultimate fallback
        return element_type or "element"

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name to be a valid Python identifier.

        PATTERN: Convert to snake_case, remove invalid characters
        CRITICAL: Names must be valid Python identifiers

        Args:
            name: Raw name string

        Returns:
            Sanitized name suitable for Python identifier
        """
        import re

        # Convert to lowercase
        sanitized = name.lower()

        # Replace spaces and hyphens with underscores
        sanitized = re.sub(r"[\s\-]+", "_", sanitized)

        # Remove non-alphanumeric characters except underscores
        sanitized = re.sub(r"[^a-z0-9_]", "", sanitized)

        # Remove leading digits
        sanitized = re.sub(r"^[0-9]+", "", sanitized)

        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        # Ensure not empty
        if not sanitized:
            sanitized = "element"

        return sanitized

    def _make_unique_name(self, base_name: str, name_counts: Dict[str, int]) -> str:
        """Make a name unique by appending a counter if needed.

        PATTERN: Track name usage and append counter for duplicates
        CRITICAL: Ensure all element names are unique

        Args:
            base_name: Base element name
            name_counts: Dictionary tracking name usage counts

        Returns:
            Unique element name
        """
        if base_name not in name_counts:
            name_counts[base_name] = 0
            return base_name
        else:
            name_counts[base_name] += 1
            return f"{base_name}_{name_counts[base_name]}"

    def _generate_actions(
        self,
        page_elements: Dict[str, PageElement],
        interactive_elements: Dict[str, List[Dict[str, Any]]],
    ) -> List[str]:
        """Generate action method names based on element types.

        PATTERN: Map element types to appropriate action methods
        CRITICAL: Actions should match Playwright API patterns

        Args:
            page_elements: Dictionary of PageElement instances
            interactive_elements: Categorized interactive elements

        Returns:
            List of action method names

        Example:
            >>> actions = generator._generate_actions(elements, interactive)
            >>> actions
            ['click_login_button', 'fill_username_input', 'submit_login_form']
        """
        actions = set()

        # Generate actions from page elements
        for element_name, element in page_elements.items():
            element_type = element.element_type

            # Get action templates for this element type
            if element_type in self.ACTION_TEMPLATES:
                templates = self.ACTION_TEMPLATES[element_type]
                for action_template in templates:
                    action_name = f"{action_template}_{element_name}"
                    actions.add(action_name)

        # Generate actions from interactive element categories
        for category, category_elements in interactive_elements.items():
            if category in self.ACTION_TEMPLATES:
                templates = self.ACTION_TEMPLATES[category]
                for action_template in templates:
                    # Create generic category action
                    action_name = f"{action_template}_{category}"
                    actions.add(action_name)

        # Add common page-level actions
        actions.update(
            [
                "navigate_to",
                "wait_for_load",
                "take_screenshot",
                "get_title",
                "get_url",
            ]
        )

        return sorted(list(actions))

    def _extract_custom_locators(
        self,
        page_elements: Dict[str, PageElement],
        framework: Optional[str] = None,
    ) -> Dict[str, str]:
        """Extract framework-specific custom locators.

        PATTERN: Provide alternative locator strategies
        CRITICAL: Useful for framework-specific test tools

        Args:
            page_elements: Dictionary of PageElement instances
            framework: Framework name

        Returns:
            Dictionary of custom locator mappings
        """
        custom_locators: Dict[str, str] = {}

        for element_name, element in page_elements.items():
            # Add data-testid locators
            if element.data_testid:
                custom_locators[f"{element_name}_testid"] = element.data_testid

            # Add aria-label locators
            if element.aria_label:
                custom_locators[f"{element_name}_aria"] = element.aria_label

            # Add framework-specific locators
            if framework == "react" and "data-react-" in element.selector:
                custom_locators[f"{element_name}_react"] = element.selector

            elif framework == "vue" and "ref=" in element.selector:
                custom_locators[f"{element_name}_vue"] = element.selector

            elif framework == "angular" and "ng-" in element.selector:
                custom_locators[f"{element_name}_angular"] = element.selector

        return custom_locators

    def _process_components(
        self,
        components: List[Dict[str, Any]],
        framework: str,
    ) -> List[Dict[str, Any]]:
        """Process and enrich component data.

        PATTERN: Add metadata and hierarchy information
        CRITICAL: Component structure aids in test organization

        Args:
            components: Raw component data from PageAnalyzer
            framework: Framework name

        Returns:
            Processed component list with enhanced metadata
        """
        processed = []

        for component in components:
            try:
                processed_component = {
                    "name": component.get("name", "Unknown"),
                    "type": component.get("type", framework),
                    "framework": framework,
                    "depth": component.get("depth", 0),
                    "props": component.get("props", {}),
                    "children": component.get("children", []),
                }

                # Add component-specific metadata
                if framework == "react":
                    processed_component["displayName"] = component.get(
                        "type", component.get("name")
                    )

                elif framework == "vue":
                    processed_component["data"] = component.get("data", {})

                elif framework == "angular":
                    processed_component["selector"] = component.get("selector", "")
                    processed_component["attributes"] = component.get("attributes", {})

                processed.append(processed_component)

            except Exception as e:
                logger.debug(f"Failed to process component: {e}")
                continue

        return processed

    def _generate_pom_name(self, title: str, url: str) -> str:
        """Generate a POM name from page title and URL.

        PATTERN: Derive semantic name from page metadata
        CRITICAL: Names should be descriptive and unique

        Args:
            title: Page title
            url: Page URL

        Returns:
            Generated POM name

        Example:
            >>> name = generator._generate_pom_name("Login - MyApp", "https://app.com/login")
            >>> name
            'Login Page'
        """
        # Use title if available and meaningful
        if (
            title
            and title.strip()
            and title.lower()
            not in [
                "untitled",
                "blank",
                "",
            ]
        ):
            # Clean up title
            name = title.split("-")[0].split("|")[0].strip()

            # Add "Page" suffix if not present
            if not name.lower().endswith("page"):
                name = f"{name} Page"

            return name

        # Fallback to URL path
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            path = parsed.path.strip("/")

            if path:
                # Get last segment of path
                segments = path.split("/")
                last_segment = (
                    segments[-1] or segments[-2] if len(segments) > 1 else "home"
                )

                # Clean and format
                name = last_segment.replace("-", " ").replace("_", " ").title()
                return f"{name} Page"

        except Exception:
            pass

        # Ultimate fallback
        return "Unnamed Page"

    async def regenerate_pom(
        self,
        page: Page,
        existing_pom: PageObjectModel,
        update_selectors: bool = True,
    ) -> PageObjectModel:
        """Regenerate POM for an existing page to update selectors.

        PATTERN: Refresh selectors while preserving POM structure
        CRITICAL: Useful when page structure changes

        Args:
            page: Playwright page instance
            existing_pom: Existing PageObjectModel to update
            update_selectors: Whether to regenerate selectors

        Returns:
            Updated PageObjectModel

        Example:
            >>> updated_pom = await generator.regenerate_pom(page, old_pom)
        """
        try:
            logger.info(f"Regenerating POM: {existing_pom.name}")

            # Generate fresh POM
            new_pom = await self.generate_pom(
                page=page,
                framework=existing_pom.framework,
                name=existing_pom.name,
            )

            # Preserve original ID and creation timestamp
            new_pom.id = existing_pom.id
            new_pom.generated_at = existing_pom.generated_at
            new_pom.last_updated = datetime.now()

            logger.info(f"POM regenerated: {new_pom.name}")

            return new_pom

        except Exception as e:
            logger.error(f"POM regeneration failed: {e}")
            raise RuntimeError(f"Failed to regenerate POM: {e}")

    async def validate_pom(
        self,
        page: Page,
        pom: PageObjectModel,
    ) -> Dict[str, Any]:
        """Validate that POM selectors work on the current page.

        PATTERN: Check all selectors are still valid
        CRITICAL: Identify broken selectors for maintenance

        Args:
            page: Playwright page instance
            pom: PageObjectModel to validate

        Returns:
            Validation report with valid/invalid selectors

        Example:
            >>> report = await generator.validate_pom(page, pom)
            >>> report["valid_count"]
            45
            >>> report["invalid_selectors"]
            ['submit_button', 'cancel_link']
        """
        try:
            logger.info(f"Validating POM: {pom.name}")

            valid_selectors = []
            invalid_selectors = []

            # Validate each element selector
            for element_name, element in pom.elements.items():
                is_valid = await self.selector_generator.validate_selector(
                    page=page,
                    selector=element.selector,
                    expect_unique=True,
                )

                if is_valid:
                    valid_selectors.append(element_name)
                else:
                    invalid_selectors.append(element_name)

            validation_report = {
                "pom_id": pom.id,
                "pom_name": pom.name,
                "total_elements": len(pom.elements),
                "valid_count": len(valid_selectors),
                "invalid_count": len(invalid_selectors),
                "valid_selectors": valid_selectors,
                "invalid_selectors": invalid_selectors,
                "validation_passed": len(invalid_selectors) == 0,
                "timestamp": datetime.now(),
            }

            logger.info(
                f"POM validation complete: "
                f"{len(valid_selectors)}/{len(pom.elements)} selectors valid"
            )

            return validation_report

        except Exception as e:
            logger.error(f"POM validation failed: {e}")
            raise RuntimeError(f"Failed to validate POM: {e}")
