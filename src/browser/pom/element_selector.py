"""Smart element selector generator for robust test automation.

This module provides the SmartSelector class which generates robust, maintainable
selectors using multiple strategies with automatic validation. Prioritizes stable
selector strategies (data-testid, id, aria-label) over brittle ones (class names).

PATTERN: Generate multiple selector candidates and validate uniqueness
CRITICAL: Selector stability is key for maintainable test automation
"""

from playwright.async_api import Page, Error as PlaywrightError
from typing import Dict, Any, List, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class SmartSelector:
    """Generate and validate robust element selectors.

    This class generates selectors using multiple strategies with prioritization:
    1. data-testid - Most stable, explicitly for testing
    2. id - Stable if unique and semantic
    3. aria-label - Stable, accessibility-friendly
    4. Framework-specific (React key, Vue ref, Angular ng-*)
    5. Tag + attributes combo - Generated fallback
    6. XPath - Last resort

    PATTERN: Try strategies in priority order, validate uniqueness
    CRITICAL: Always validate selectors work on actual page
    """

    # Selector strategy priority order
    STRATEGY_PRIORITY = [
        "data-testid",
        "id",
        "aria-label",
        "name",
        "placeholder",
        "framework-specific",
        "role",
        "tag-attributes",
        "text",
        "xpath",
    ]

    # Framework-specific attribute patterns
    FRAMEWORK_ATTRIBUTES = {
        "react": [
            "data-react-",
            "data-reactid",
            "data-reactroot",
        ],
        "vue": [
            "data-v-",
            "v-bind:",
            ":ref",
            "ref",
        ],
        "angular": [
            "ng-",
            "[ng",
            "data-ng-",
            "_ngcontent-",
            "_nghost-",
        ],
        "svelte": [
            "data-svelte-",
        ],
    }

    # Attributes to avoid in selectors (too dynamic/unstable)
    UNSTABLE_ATTRIBUTES = {
        "class",  # Often dynamic, especially in frameworks
        "style",  # Frequently changes
        "data-reactid",  # Internal React IDs
        "data-v-",  # Internal Vue IDs (hash)
        "_nghost-",  # Internal Angular IDs
        "_ngcontent-",  # Internal Angular IDs
    }

    def __init__(self):
        """Initialize the smart selector generator."""
        self._validation_cache: Dict[str, bool] = {}

    async def generate_selector(
        self,
        element: Dict[str, Any],
        framework: Optional[str] = None,
        page: Optional[Page] = None,
    ) -> str:
        """Generate the best possible selector for an element.

        PATTERN: Try strategies in priority order, validate if page provided
        CRITICAL: Return first selector that is unique and works on page

        Args:
            element: Element data dictionary with attributes, text, etc.
            framework: Detected framework (react, vue, angular) for framework-specific selectors
            page: Optional page instance for validation

        Returns:
            Best available selector string

        Example:
            >>> element = {
            ...     "data_testid": "submit-button",
            ...     "attributes": {"id": "submit", "class": "btn btn-primary"},
            ...     "element_type": "button"
            ... }
            >>> selector = await generator.generate_selector(element)
            >>> selector
            '[data-testid="submit-button"]'
        """
        logger.debug(
            f"Generating selector for {element.get('element_type', 'unknown')} element"
        )

        # Try each strategy in priority order
        for strategy in self.STRATEGY_PRIORITY:
            try:
                selector = None

                if strategy == "data-testid":
                    selector = self._generate_testid_selector(element)
                elif strategy == "id":
                    selector = self._generate_id_selector(element)
                elif strategy == "aria-label":
                    selector = self._generate_aria_selector(element)
                elif strategy == "name":
                    selector = self._generate_name_selector(element)
                elif strategy == "placeholder":
                    selector = self._generate_placeholder_selector(element)
                elif strategy == "framework-specific" and framework:
                    selector = self._generate_framework_selector(element, framework)
                elif strategy == "role":
                    selector = self._generate_role_selector(element)
                elif strategy == "tag-attributes":
                    selector = self._generate_tag_attributes_selector(element)
                elif strategy == "text":
                    selector = self._generate_text_selector(element)
                elif strategy == "xpath":
                    selector = self._generate_xpath_selector(element)

                # If selector generated, validate it
                if selector:
                    # If page provided, validate on actual page
                    if page:
                        is_valid = await self.validate_selector(page, selector)
                        if is_valid:
                            logger.info(f"Generated {strategy} selector: {selector}")
                            return selector
                        else:
                            logger.debug(
                                f"{strategy} selector failed validation: {selector}"
                            )
                    else:
                        # No page to validate against, return first generated selector
                        logger.info(
                            f"Generated {strategy} selector (not validated): {selector}"
                        )
                        return selector

            except Exception as e:
                logger.debug(f"Failed to generate {strategy} selector: {e}")
                continue

        # Fallback to basic tag selector
        fallback = element.get("element_type", "div")
        logger.warning(f"Using fallback selector: {fallback}")
        return fallback

    async def validate_selector(
        self,
        page: Page,
        selector: str,
        expect_unique: bool = True,
    ) -> bool:
        """Validate that a selector works and optionally is unique.

        PATTERN: Check selector matches elements and is unique if required
        CRITICAL: Validation prevents flaky tests from bad selectors

        Args:
            page: Playwright page instance
            selector: CSS or XPath selector to validate
            expect_unique: Whether selector should match exactly one element

        Returns:
            True if selector is valid and meets uniqueness requirement

        Example:
            >>> is_valid = await validator.validate_selector(
            ...     page,
            ...     '[data-testid="submit"]',
            ...     expect_unique=True
            ... )
        """
        # Check cache first
        cache_key = f"{selector}:{expect_unique}"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        try:
            # Try to query the selector
            if selector.startswith("//") or selector.startswith("(//"):
                # XPath selector
                elements = await page.query_selector_all(f"xpath={selector}")
            else:
                # CSS selector
                elements = await page.query_selector_all(selector)

            # Check if any elements found
            if not elements:
                logger.debug(f"Selector matches no elements: {selector}")
                self._validation_cache[cache_key] = False
                return False

            # Check uniqueness if required
            if expect_unique and len(elements) > 1:
                logger.debug(
                    f"Selector matches {len(elements)} elements (expected 1): {selector}"
                )
                self._validation_cache[cache_key] = False
                return False

            # Selector is valid
            logger.debug(f"Selector validated successfully: {selector}")
            self._validation_cache[cache_key] = True
            return True

        except PlaywrightError as e:
            logger.debug(f"Invalid selector syntax: {selector} - {e}")
            self._validation_cache[cache_key] = False
            return False
        except Exception as e:
            logger.error(f"Selector validation error: {e}")
            self._validation_cache[cache_key] = False
            return False

    def _generate_testid_selector(self, element: Dict[str, Any]) -> Optional[str]:
        """Generate selector using data-testid attribute.

        PATTERN: data-testid is the gold standard for test selectors
        CRITICAL: This should be the primary selector strategy

        Args:
            element: Element data dictionary

        Returns:
            data-testid selector or None if not available
        """
        testid = element.get("data_testid") or element.get("attributes", {}).get(
            "data-testid"
        )

        if testid and isinstance(testid, str) and testid.strip():
            # Escape special characters in attribute value
            escaped_testid = testid.replace('"', '\\"')
            return f'[data-testid="{escaped_testid}"]'

        return None

    def _generate_id_selector(self, element: Dict[str, Any]) -> Optional[str]:
        """Generate selector using id attribute.

        PATTERN: IDs are stable if semantic and unique
        CRITICAL: Avoid auto-generated or dynamic IDs

        Args:
            element: Element data dictionary

        Returns:
            ID selector or None if not suitable
        """
        element_id = element.get("attributes", {}).get("id")

        if not element_id or not isinstance(element_id, str):
            return None

        # Skip IDs that look auto-generated or dynamic
        if self._is_dynamic_id(element_id):
            logger.debug(f"Skipping dynamic ID: {element_id}")
            return None

        # Escape special characters in ID
        escaped_id = self._escape_css_selector(element_id)
        return f"#{escaped_id}"

    def _generate_aria_selector(self, element: Dict[str, Any]) -> Optional[str]:
        """Generate selector using aria-label attribute.

        PATTERN: ARIA labels are stable and accessibility-friendly
        CRITICAL: Good for buttons, links, and interactive elements

        Args:
            element: Element data dictionary

        Returns:
            aria-label selector or None if not available
        """
        aria_label = element.get("aria_label") or element.get("attributes", {}).get(
            "aria-label"
        )

        if aria_label and isinstance(aria_label, str) and aria_label.strip():
            escaped_label = aria_label.replace('"', '\\"')
            return f'[aria-label="{escaped_label}"]'

        return None

    def _generate_name_selector(self, element: Dict[str, Any]) -> Optional[str]:
        """Generate selector using name attribute.

        PATTERN: Name attributes are stable for form elements
        CRITICAL: Good for inputs, selects, textareas

        Args:
            element: Element data dictionary

        Returns:
            name selector or None if not available
        """
        name = element.get("name") or element.get("attributes", {}).get("name")

        if name and isinstance(name, str) and name.strip():
            element_type = element.get("element_type", "")
            escaped_name = name.replace('"', '\\"')

            # Combine with element type for better specificity
            if element_type in ["input", "textarea", "select"]:
                return f'{element_type}[name="{escaped_name}"]'
            else:
                return f'[name="{escaped_name}"]'

        return None

    def _generate_placeholder_selector(self, element: Dict[str, Any]) -> Optional[str]:
        """Generate selector using placeholder attribute.

        PATTERN: Placeholders are moderately stable for inputs
        CRITICAL: Use only when better selectors unavailable

        Args:
            element: Element data dictionary

        Returns:
            placeholder selector or None if not available
        """
        placeholder = element.get("placeholder") or element.get("attributes", {}).get(
            "placeholder"
        )

        if placeholder and isinstance(placeholder, str) and placeholder.strip():
            escaped_placeholder = placeholder.replace('"', '\\"')
            return f'[placeholder="{escaped_placeholder}"]'

        return None

    def _generate_framework_selector(
        self, element: Dict[str, Any], framework: str
    ) -> Optional[str]:
        """Generate framework-specific selector.

        PATTERN: Use framework-specific attributes when available
        CRITICAL: React keys, Vue refs, Angular directives

        Args:
            element: Element data dictionary
            framework: Framework name (react, vue, angular)

        Returns:
            Framework-specific selector or None
        """
        attributes = element.get("attributes", {})

        if framework == "react":
            # Look for React-specific data attributes
            for attr_name, attr_value in attributes.items():
                if attr_name.startswith("data-react-") and not attr_name.endswith("id"):
                    escaped_value = attr_value.replace('"', '\\"')
                    return f'[{attr_name}="{escaped_value}"]'

        elif framework == "vue":
            # Look for Vue ref
            for attr_name, attr_value in attributes.items():
                if attr_name == "ref" or attr_name == "data-ref":
                    escaped_value = attr_value.replace('"', '\\"')
                    return f'[{attr_name}="{escaped_value}"]'

        elif framework == "angular":
            # Look for Angular directives (excluding internal ones)
            for attr_name, attr_value in attributes.items():
                if attr_name.startswith("ng-") and not attr_name.startswith(
                    "ng-version"
                ):
                    escaped_value = attr_value.replace('"', '\\"')
                    return f'[{attr_name}="{escaped_value}"]'

        return None

    def _generate_role_selector(self, element: Dict[str, Any]) -> Optional[str]:
        """Generate selector using ARIA role.

        PATTERN: Roles provide semantic meaning
        CRITICAL: Good for accessibility-focused tests

        Args:
            element: Element data dictionary

        Returns:
            role selector or None if not available
        """
        role = element.get("role") or element.get("attributes", {}).get("role")

        if role and isinstance(role, str) and role.strip():
            # Combine role with text content for better specificity
            text = element.get("text_content", "").strip()
            if text and len(text) < 50:  # Reasonable text length
                escaped_text = text.replace('"', '\\"')
                return f'[role="{role}"][aria-label="{escaped_text}"]'
            else:
                return f'[role="{role}"]'

        return None

    def _generate_tag_attributes_selector(
        self, element: Dict[str, Any]
    ) -> Optional[str]:
        """Generate selector combining tag with stable attributes.

        PATTERN: Build compound selector from stable attributes
        CRITICAL: Avoid classes and dynamic attributes

        Args:
            element: Element data dictionary

        Returns:
            Combined tag+attributes selector
        """
        element_type = element.get("element_type", "")
        if not element_type:
            return None

        attributes = element.get("attributes", {})
        selector_parts = [element_type]

        # Find stable attributes
        stable_attrs = []
        for attr_name, attr_value in attributes.items():
            # Skip unstable attributes
            if any(unstable in attr_name for unstable in self.UNSTABLE_ATTRIBUTES):
                continue

            # Skip empty values
            if not attr_value or not isinstance(attr_value, str):
                continue

            # Add stable attributes
            if attr_name in ["type", "href", "src", "alt", "title"]:
                escaped_value = attr_value.replace('"', '\\"')
                stable_attrs.append(f'[{attr_name}="{escaped_value}"]')

        # If we have stable attributes, combine them
        if stable_attrs:
            selector_parts.extend(stable_attrs[:2])  # Limit to 2 attributes
            return "".join(selector_parts)

        return None

    def _generate_text_selector(self, element: Dict[str, Any]) -> Optional[str]:
        """Generate selector using text content.

        PATTERN: Text selectors are human-readable but can be fragile
        CRITICAL: Use for unique, stable text content only

        Args:
            element: Element data dictionary

        Returns:
            text selector or None if text unsuitable
        """
        text = element.get("text_content", "").strip()

        if not text or len(text) > 100:  # Skip long text
            return None

        # For buttons and links, text is usually stable
        element_type = element.get("element_type", "")
        if element_type in ["button", "a"]:
            escaped_text = text.replace('"', '\\"')
            return f'{element_type}:has-text("{escaped_text}")'

        return None

    def _generate_xpath_selector(self, element: Dict[str, Any]) -> Optional[str]:
        """Generate XPath selector as last resort.

        PATTERN: XPath can express complex relationships
        CRITICAL: More fragile than CSS, use only when necessary

        Args:
            element: Element data dictionary

        Returns:
            XPath selector
        """
        element_type = element.get("element_type", "*")
        attributes = element.get("attributes", {})

        # Build XPath with available attributes
        xpath_parts = [f"//{element_type}"]
        predicates = []

        # Add attribute predicates
        for attr_name, attr_value in attributes.items():
            if attr_name in ["id", "name", "type"]:
                escaped_value = attr_value.replace("'", "\\'")
                predicates.append(f"@{attr_name}='{escaped_value}'")

        if predicates:
            xpath_parts.append(f"[{' and '.join(predicates[:2])}]")

        return "".join(xpath_parts)

    def _is_dynamic_id(self, element_id: str) -> bool:
        """Check if an ID looks auto-generated or dynamic.

        PATTERN: Detect common dynamic ID patterns
        CRITICAL: Avoid IDs that change between page loads

        Args:
            element_id: ID string to check

        Returns:
            True if ID appears dynamic/auto-generated
        """
        # Patterns that indicate dynamic IDs
        dynamic_patterns = [
            r"^[0-9]+$",  # Numeric only
            r"^[a-f0-9]{8,}$",  # Long hex strings (UUIDs)
            r".*-[0-9]{10,}$",  # Timestamp suffixes
            r"^react-",  # React auto-generated
            r"^vue-",  # Vue auto-generated
            r"^ember\d+",  # Ember auto-generated
            r"^ext-gen\d+",  # ExtJS auto-generated
            r"^\d+-\d+-\d+",  # Numeric with dashes
        ]

        for pattern in dynamic_patterns:
            if re.match(pattern, element_id):
                return True

        return False

    def _escape_css_selector(self, value: str) -> str:
        """Escape special characters in CSS selector.

        PATTERN: Ensure selector syntax is valid
        CRITICAL: Handle special characters properly

        Args:
            value: Selector value to escape

        Returns:
            Escaped selector value
        """
        # Characters that need escaping in CSS selectors
        special_chars = r'!"#$%&\'()*+,./:;<=>?@[\]^`{|}~'

        escaped = value
        for char in special_chars:
            escaped = escaped.replace(char, f"\\{char}")

        return escaped

    async def generate_multiple_selectors(
        self,
        element: Dict[str, Any],
        framework: Optional[str] = None,
        page: Optional[Page] = None,
        max_selectors: int = 3,
    ) -> List[Tuple[str, str]]:
        """Generate multiple selector options for an element.

        PATTERN: Provide fallback selectors for robustness
        CRITICAL: Return list of (strategy, selector) tuples

        Args:
            element: Element data dictionary
            framework: Detected framework
            page: Optional page for validation
            max_selectors: Maximum number of selectors to generate

        Returns:
            List of (strategy_name, selector) tuples

        Example:
            >>> selectors = await generator.generate_multiple_selectors(element)
            >>> selectors
            [
                ('data-testid', '[data-testid="submit"]'),
                ('id', '#submit-btn'),
                ('aria-label', '[aria-label="Submit Form"]')
            ]
        """
        selectors = []

        for strategy in self.STRATEGY_PRIORITY:
            if len(selectors) >= max_selectors:
                break

            try:
                selector = None

                if strategy == "data-testid":
                    selector = self._generate_testid_selector(element)
                elif strategy == "id":
                    selector = self._generate_id_selector(element)
                elif strategy == "aria-label":
                    selector = self._generate_aria_selector(element)
                elif strategy == "name":
                    selector = self._generate_name_selector(element)
                elif strategy == "placeholder":
                    selector = self._generate_placeholder_selector(element)
                elif strategy == "framework-specific" and framework:
                    selector = self._generate_framework_selector(element, framework)
                elif strategy == "role":
                    selector = self._generate_role_selector(element)
                elif strategy == "tag-attributes":
                    selector = self._generate_tag_attributes_selector(element)

                if selector:
                    # Validate if page provided
                    if page:
                        is_valid = await self.validate_selector(page, selector)
                        if is_valid:
                            selectors.append((strategy, selector))
                    else:
                        selectors.append((strategy, selector))

            except Exception as e:
                logger.debug(f"Failed to generate {strategy} selector: {e}")
                continue

        return selectors

    def clear_cache(self):
        """Clear the validation cache.

        PATTERN: Allow cache refresh for new page contexts
        CRITICAL: Call when page changes significantly
        """
        self._validation_cache.clear()
        logger.debug("Selector validation cache cleared")
