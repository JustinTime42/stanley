"""Page analyzer for DOM structure analysis and framework detection.

This module provides the PageAnalyzer class which analyzes web pages to extract
DOM structure, detect frontend frameworks, identify interactive elements, and
generate structured data suitable for Page Object Model (POM) generation.

PATTERN: Analyze DOM structure and extract semantic information for test automation
CRITICAL: Proper framework detection enables framework-specific POM generation
"""

from playwright.async_api import Page, ElementHandle
from typing import Dict, Any, List, Optional
import logging


logger = logging.getLogger(__name__)


class PageAnalyzer:
    """Analyze web page DOM structure and extract elements.

    This class analyzes web pages to:
    1. Extract interactive elements with their properties
    2. Detect frontend frameworks (React, Vue, Angular)
    3. Identify component hierarchies
    4. Generate structured data for POM creation

    PATTERN: Use Playwright Page API to analyze DOM and JavaScript context
    CRITICAL: Framework detection enables framework-specific test generation
    """

    # Framework detection markers
    FRAMEWORK_MARKERS = {
        "react": [
            "__reactInternalInstance",
            "__reactFiber",
            "_reactRootContainer",
            "__REACT_DEVTOOLS_GLOBAL_HOOK__",
        ],
        "vue": [
            "__vue__",
            "__VUE__",
            "__VUE_DEVTOOLS_GLOBAL_HOOK__",
            "_vnode",
        ],
        "angular": [
            "ng-version",
            "getAllAngularRootElements",
            "getAngularTestability",
            "__ngContext__",
        ],
        "svelte": [
            "__svelte",
            "__svelte_meta",
        ],
    }

    # Interactive element selectors
    INTERACTIVE_SELECTORS = {
        "button": "button, [role='button'], input[type='button'], input[type='submit']",
        "input": "input:not([type='button']):not([type='submit']), textarea",
        "link": "a[href], [role='link']",
        "select": "select, [role='listbox'], [role='combobox']",
        "checkbox": "input[type='checkbox'], [role='checkbox']",
        "radio": "input[type='radio'], [role='radio']",
        "form": "form",
        "dialog": "[role='dialog'], dialog",
        "menu": "[role='menu'], [role='menubar']",
        "tab": "[role='tab']",
        "slider": "input[type='range'], [role='slider']",
    }

    def __init__(self):
        """Initialize the page analyzer."""
        self._framework_cache: Dict[str, Optional[str]] = {}

    async def analyze_page(
        self,
        page: Page,
        include_invisible: bool = False,
        max_depth: int = 10,
    ) -> Dict[str, Any]:
        """Analyze page and extract structured element data.

        PATTERN: Extract all relevant information in a single page analysis pass
        CRITICAL: Results must be JSON-serializable for POM generation

        Args:
            page: Playwright page instance to analyze
            include_invisible: Whether to include hidden elements
            max_depth: Maximum depth for component hierarchy extraction

        Returns:
            Dictionary containing:
                - url: Page URL
                - title: Page title
                - framework: Detected framework (if any)
                - elements: List of extracted page elements
                - components: Component hierarchy (for frameworks)
                - interactive_elements: Categorized interactive elements
                - metadata: Additional page metadata

        Raises:
            RuntimeError: If page analysis fails
        """
        try:
            logger.info(f"Analyzing page: {page.url}")

            # Extract basic page info
            url = page.url
            title = await page.title()

            # Detect framework
            framework = await self._detect_framework(page)
            logger.info(f"Detected framework: {framework or 'None'}")

            # Extract all elements
            elements = await self._extract_elements(page, include_invisible)
            logger.debug(f"Extracted {len(elements)} elements")

            # Extract interactive elements by category
            interactive_elements = await self._extract_interactive_elements(page)
            logger.debug(
                f"Found {sum(len(v) for v in interactive_elements.values())} interactive elements"
            )

            # Extract component hierarchy if framework detected
            components = []
            if framework:
                components = await self._extract_components(page, framework, max_depth)
                logger.debug(f"Extracted {len(components)} components")

            # Extract metadata
            metadata = await self._extract_metadata(page)

            result = {
                "url": url,
                "title": title,
                "framework": framework,
                "elements": elements,
                "components": components,
                "interactive_elements": interactive_elements,
                "metadata": metadata,
            }

            logger.info(f"Page analysis complete for {url}")
            return result

        except Exception as e:
            logger.error(f"Page analysis failed: {e}")
            raise RuntimeError(f"Failed to analyze page: {e}")

    async def _detect_framework(self, page: Page) -> Optional[str]:
        """Detect frontend framework used by the page.

        PATTERN: Check for framework-specific global variables and attributes
        CRITICAL: Framework detection enables specialized POM generation

        Args:
            page: Playwright page instance

        Returns:
            Framework name (react, vue, angular, svelte) or None
        """
        # Check cache first
        cache_key = page.url
        if cache_key in self._framework_cache:
            return self._framework_cache[cache_key]

        try:
            # Check for React
            has_react = await page.evaluate(
                """() => {
                    // Check for React DevTools hook
                    if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) return true;

                    // Check for React root
                    if (document.querySelector('[data-reactroot]')) return true;

                    // Check for React fiber
                    const body = document.body;
                    if (body) {
                        const keys = Object.keys(body);
                        if (keys.some(key => key.startsWith('__reactInternalInstance') ||
                                              key.startsWith('__reactFiber') ||
                                              key.startsWith('__reactContainer'))) {
                            return true;
                        }
                    }

                    return false;
                }"""
            )

            if has_react:
                self._framework_cache[cache_key] = "react"
                return "react"

            # Check for Vue
            has_vue = await page.evaluate(
                """() => {
                    // Check for Vue DevTools
                    if (window.__VUE__ || window.__VUE_DEVTOOLS_GLOBAL_HOOK__) return true;

                    // Check for Vue app attribute
                    if (document.querySelector('[data-v-app]')) return true;

                    // Check for Vue instance on elements
                    const body = document.body;
                    if (body && body.__vue__) return true;

                    // Check for Vue 3
                    const app = document.getElementById('app');
                    if (app && app.__vue_app__) return true;

                    return false;
                }"""
            )

            if has_vue:
                self._framework_cache[cache_key] = "vue"
                return "vue"

            # Check for Angular
            has_angular = await page.evaluate(
                """() => {
                    // Check for Angular version attribute
                    if (document.querySelector('[ng-version]')) return true;

                    // Check for Angular global functions
                    if (window.getAllAngularRootElements ||
                        window.getAngularTestability ||
                        window.ng) return true;

                    // Check for Angular context
                    const body = document.body;
                    if (body && body.__ngContext__) return true;

                    return false;
                }"""
            )

            if has_angular:
                self._framework_cache[cache_key] = "angular"
                return "angular"

            # Check for Svelte
            has_svelte = await page.evaluate(
                """() => {
                    // Check for Svelte meta
                    const elements = document.querySelectorAll('*');
                    for (const el of elements) {
                        const keys = Object.keys(el);
                        if (keys.some(key => key.startsWith('__svelte'))) {
                            return true;
                        }
                    }
                    return false;
                }"""
            )

            if has_svelte:
                self._framework_cache[cache_key] = "svelte"
                return "svelte"

            # No framework detected
            self._framework_cache[cache_key] = None
            return None

        except Exception as e:
            logger.warning(f"Framework detection failed: {e}")
            return None

    async def _extract_elements(
        self, page: Page, include_invisible: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract all elements from the page with their properties.

        PATTERN: Extract comprehensive element data including selectors and attributes
        CRITICAL: Support multiple selector strategies (CSS, XPath, test IDs)

        Args:
            page: Playwright page instance
            include_invisible: Whether to include hidden elements

        Returns:
            List of element dictionaries with properties
        """
        try:
            # JavaScript to extract element data
            elements_data = await page.evaluate(
                """(includeInvisible) => {
                    const elements = [];
                    const visited = new WeakSet();

                    function isVisible(el) {
                        if (!el) return false;
                        const style = window.getComputedStyle(el);
                        return style.display !== 'none' &&
                               style.visibility !== 'hidden' &&
                               style.opacity !== '0';
                    }

                    function getSelector(el) {
                        // Prefer data-testid
                        if (el.hasAttribute('data-testid')) {
                            return `[data-testid="${el.getAttribute('data-testid')}"]`;
                        }

                        // Use ID if available
                        if (el.id) {
                            return `#${el.id}`;
                        }

                        // Use name for inputs
                        if (el.name) {
                            return `${el.tagName.toLowerCase()}[name="${el.name}"]`;
                        }

                        // Build class selector
                        if (el.className && typeof el.className === 'string') {
                            const classes = el.className.trim().split(/\\s+/).filter(c => c);
                            if (classes.length > 0) {
                                return `${el.tagName.toLowerCase()}.${classes.join('.')}`;
                            }
                        }

                        // Fallback to tag name
                        return el.tagName.toLowerCase();
                    }

                    function extractElement(el) {
                        if (visited.has(el)) return null;
                        visited.add(el);

                        // Skip if invisible and not including invisible
                        if (!includeInvisible && !isVisible(el)) return null;

                        const tagName = el.tagName.toLowerCase();

                        // Extract attributes
                        const attributes = {};
                        for (const attr of el.attributes) {
                            attributes[attr.name] = attr.value;
                        }

                        // Get text content (truncated)
                        let textContent = el.textContent?.trim() || '';
                        if (textContent.length > 200) {
                            textContent = textContent.substring(0, 200) + '...';
                        }

                        return {
                            name: el.getAttribute('name') ||
                                  el.getAttribute('id') ||
                                  el.getAttribute('data-testid') ||
                                  tagName,
                            selector: getSelector(el),
                            element_type: tagName,
                            attributes: attributes,
                            text_content: textContent,
                            is_visible: isVisible(el),
                            is_enabled: !el.disabled,
                            aria_label: el.getAttribute('aria-label') ||
                                       el.getAttribute('aria-labelledby') || null,
                            data_testid: el.getAttribute('data-testid') || null,
                            role: el.getAttribute('role') || null,
                            placeholder: el.getAttribute('placeholder') || null,
                        };
                    }

                    // Extract interactive elements
                    const interactiveSelectors = [
                        'button', 'a[href]', 'input', 'textarea', 'select',
                        '[role="button"]', '[role="link"]', '[role="checkbox"]',
                        '[role="radio"]', '[role="tab"]', '[role="menuitem"]',
                        '[onclick]', '[data-testid]'
                    ];

                    for (const selector of interactiveSelectors) {
                        const els = document.querySelectorAll(selector);
                        for (const el of els) {
                            const data = extractElement(el);
                            if (data) elements.push(data);
                        }
                    }

                    return elements;
                }""",
                include_invisible,
            )

            return elements_data

        except Exception as e:
            logger.error(f"Element extraction failed: {e}")
            return []

    async def _extract_interactive_elements(
        self, page: Page
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract interactive elements categorized by type.

        PATTERN: Categorize elements by interaction type for better test generation
        CRITICAL: Focus on elements that users can interact with

        Args:
            page: Playwright page instance

        Returns:
            Dictionary mapping element types to lists of elements
        """
        interactive_elements: Dict[str, List[Dict[str, Any]]] = {
            category: [] for category in self.INTERACTIVE_SELECTORS.keys()
        }

        try:
            for category, selector in self.INTERACTIVE_SELECTORS.items():
                try:
                    # Get all matching elements
                    elements = await page.query_selector_all(selector)

                    for element in elements:
                        try:
                            # Check visibility
                            is_visible = await element.is_visible()
                            if not is_visible:
                                continue

                            # Extract element data
                            element_data = await self._extract_element_data(
                                element, category
                            )
                            if element_data:
                                interactive_elements[category].append(element_data)

                        except Exception as e:
                            logger.debug(f"Failed to extract {category} element: {e}")
                            continue

                except Exception as e:
                    logger.debug(f"Failed to query {category} elements: {e}")
                    continue

            return interactive_elements

        except Exception as e:
            logger.error(f"Interactive element extraction failed: {e}")
            return interactive_elements

    async def _extract_element_data(
        self, element: ElementHandle, element_type: str
    ) -> Optional[Dict[str, Any]]:
        """Extract data from a single element.

        Args:
            element: Element handle
            element_type: Type of element (button, input, etc.)

        Returns:
            Element data dictionary or None if extraction fails
        """
        try:
            # Evaluate element properties in browser context
            data = await element.evaluate(
                """(el) => {
                    function getSelector(el) {
                        if (el.hasAttribute('data-testid')) {
                            return `[data-testid="${el.getAttribute('data-testid')}"]`;
                        }
                        if (el.id) return `#${el.id}`;
                        if (el.name) return `[name="${el.name}"]`;
                        if (el.className && typeof el.className === 'string') {
                            const classes = el.className.trim().split(/\\s+/).filter(c => c);
                            if (classes.length > 0) {
                                return `${el.tagName.toLowerCase()}.${classes[0]}`;
                            }
                        }
                        return el.tagName.toLowerCase();
                    }

                    const style = window.getComputedStyle(el);
                    let text = el.textContent?.trim() || '';
                    if (text.length > 100) text = text.substring(0, 100) + '...';

                    return {
                        selector: getSelector(el),
                        tag_name: el.tagName.toLowerCase(),
                        text: text,
                        value: el.value || null,
                        placeholder: el.placeholder || null,
                        aria_label: el.getAttribute('aria-label') || null,
                        data_testid: el.getAttribute('data-testid') || null,
                        role: el.getAttribute('role') || null,
                        type: el.type || null,
                        name: el.name || el.id || null,
                        disabled: el.disabled || false,
                        href: el.href || null,
                    };
                }"""
            )

            return {
                "element_type": element_type,
                **data,
            }

        except Exception as e:
            logger.debug(f"Failed to extract element data: {e}")
            return None

    async def _extract_components(
        self, page: Page, framework: str, max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract component hierarchy for framework-specific POMs.

        PATTERN: Framework-specific component extraction for better test structure
        CRITICAL: Component hierarchy enables more maintainable test code

        Args:
            page: Playwright page instance
            framework: Detected framework name
            max_depth: Maximum depth to traverse component tree

        Returns:
            List of component dictionaries with hierarchy info
        """
        try:
            if framework == "react":
                return await self._extract_react_components(page, max_depth)
            elif framework == "vue":
                return await self._extract_vue_components(page, max_depth)
            elif framework == "angular":
                return await self._extract_angular_components(page, max_depth)
            else:
                logger.debug(f"Component extraction not implemented for {framework}")
                return []

        except Exception as e:
            logger.warning(f"Component extraction failed for {framework}: {e}")
            return []

    async def _extract_react_components(
        self, page: Page, max_depth: int
    ) -> List[Dict[str, Any]]:
        """Extract React component hierarchy.

        PATTERN: Use React Fiber to extract component tree
        CRITICAL: React Fiber provides component type and props information

        Args:
            page: Playwright page instance
            max_depth: Maximum depth to traverse

        Returns:
            List of React component data
        """
        try:
            components = await page.evaluate(
                """(maxDepth) => {
                    const components = [];
                    const visited = new WeakSet();

                    function getComponentName(fiber) {
                        if (!fiber) return 'Unknown';
                        if (fiber.type && fiber.type.name) return fiber.type.name;
                        if (fiber.type && typeof fiber.type === 'string') return fiber.type;
                        if (fiber.elementType && fiber.elementType.name) return fiber.elementType.name;
                        return 'Anonymous';
                    }

                    function extractComponent(fiber, depth = 0) {
                        if (!fiber || depth > maxDepth || visited.has(fiber)) return;
                        visited.add(fiber);

                        const componentName = getComponentName(fiber);

                        // Only include user components (capitalized names)
                        if (componentName && componentName[0] === componentName[0].toUpperCase()) {
                            const component = {
                                name: componentName,
                                type: fiber.type?.displayName || componentName,
                                depth: depth,
                                props: {},
                                children: [],
                            };

                            // Extract safe props (avoid circular references)
                            if (fiber.memoizedProps) {
                                for (const key in fiber.memoizedProps) {
                                    const value = fiber.memoizedProps[key];
                                    if (typeof value === 'string' ||
                                        typeof value === 'number' ||
                                        typeof value === 'boolean') {
                                        component.props[key] = value;
                                    }
                                }
                            }

                            components.push(component);
                        }

                        // Traverse children
                        if (fiber.child) extractComponent(fiber.child, depth + 1);
                        if (fiber.sibling) extractComponent(fiber.sibling, depth);
                    }

                    // Find React root
                    const reactRoot = document.querySelector('[data-reactroot]') ||
                                     document.getElementById('root') ||
                                     document.body;

                    if (reactRoot) {
                        const fiberKey = Object.keys(reactRoot).find(
                            key => key.startsWith('__reactInternalInstance') ||
                                   key.startsWith('__reactFiber') ||
                                   key.startsWith('__reactContainer')
                        );

                        if (fiberKey) {
                            const fiber = reactRoot[fiberKey];
                            extractComponent(fiber);
                        }
                    }

                    return components;
                }""",
                max_depth,
            )

            logger.debug(f"Extracted {len(components)} React components")
            return components

        except Exception as e:
            logger.warning(f"React component extraction failed: {e}")
            return []

    async def _extract_vue_components(
        self, page: Page, max_depth: int
    ) -> List[Dict[str, Any]]:
        """Extract Vue component hierarchy.

        Args:
            page: Playwright page instance
            max_depth: Maximum depth to traverse

        Returns:
            List of Vue component data
        """
        try:
            components = await page.evaluate(
                """(maxDepth) => {
                    const components = [];
                    const visited = new WeakSet();

                    function extractComponent(vm, depth = 0) {
                        if (!vm || depth > maxDepth || visited.has(vm)) return;
                        visited.add(vm);

                        const component = {
                            name: vm.$options?.name || vm.$options?._componentTag || 'Anonymous',
                            type: 'vue-component',
                            depth: depth,
                            props: {},
                            data: {},
                        };

                        // Extract props
                        if (vm.$props) {
                            for (const key in vm.$props) {
                                const value = vm.$props[key];
                                if (typeof value === 'string' ||
                                    typeof value === 'number' ||
                                    typeof value === 'boolean') {
                                    component.props[key] = value;
                                }
                            }
                        }

                        components.push(component);

                        // Traverse children
                        if (vm.$children) {
                            for (const child of vm.$children) {
                                extractComponent(child, depth + 1);
                            }
                        }
                    }

                    // Find Vue app
                    const app = document.getElementById('app');
                    if (app && app.__vue__) {
                        extractComponent(app.__vue__);
                    } else if (app && app.__vue_app__) {
                        // Vue 3
                        const rootComponent = app.__vue_app__?._instance;
                        if (rootComponent) {
                            extractComponent(rootComponent);
                        }
                    }

                    return components;
                }""",
                max_depth,
            )

            logger.debug(f"Extracted {len(components)} Vue components")
            return components

        except Exception as e:
            logger.warning(f"Vue component extraction failed: {e}")
            return []

    async def _extract_angular_components(
        self, page: Page, max_depth: int
    ) -> List[Dict[str, Any]]:
        """Extract Angular component hierarchy.

        Args:
            page: Playwright page instance
            max_depth: Maximum depth to traverse

        Returns:
            List of Angular component data
        """
        try:
            components = await page.evaluate(
                """(maxDepth) => {
                    const components = [];

                    // Find all Angular elements
                    const elements = document.querySelectorAll('[ng-version], [_nghost-*]');

                    for (const el of elements) {
                        const context = el.__ngContext__;
                        if (context) {
                            const component = {
                                name: el.tagName.toLowerCase(),
                                type: 'angular-component',
                                selector: el.tagName.toLowerCase(),
                                attributes: {},
                            };

                            // Extract attributes
                            for (const attr of el.attributes) {
                                if (attr.name.startsWith('ng-') ||
                                    attr.name.startsWith('_ng') ||
                                    attr.name.startsWith('[') ||
                                    attr.name.startsWith('(')) {
                                    component.attributes[attr.name] = attr.value;
                                }
                            }

                            components.push(component);
                        }
                    }

                    return components;
                }""",
                max_depth,
            )

            logger.debug(f"Extracted {len(components)} Angular components")
            return components

        except Exception as e:
            logger.warning(f"Angular component extraction failed: {e}")
            return []

    async def _extract_metadata(self, page: Page) -> Dict[str, Any]:
        """Extract additional page metadata.

        PATTERN: Extract metadata useful for test context and reporting
        CRITICAL: Include language, viewport, and performance info

        Args:
            page: Playwright page instance

        Returns:
            Metadata dictionary
        """
        try:
            metadata = await page.evaluate(
                """() => {
                    return {
                        lang: document.documentElement.lang || null,
                        charset: document.characterSet || null,
                        viewport: {
                            width: window.innerWidth,
                            height: window.innerHeight,
                        },
                        url: window.location.href,
                        origin: window.location.origin,
                        pathname: window.location.pathname,
                        has_forms: document.forms.length > 0,
                        form_count: document.forms.length,
                        link_count: document.links.length,
                        image_count: document.images.length,
                        meta_description: document.querySelector('meta[name="description"]')?.content || null,
                        meta_keywords: document.querySelector('meta[name="keywords"]')?.content || null,
                    };
                }"""
            )

            return metadata

        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}

    async def get_element_by_testid(
        self, page: Page, test_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get element by data-testid attribute.

        PATTERN: Prefer test IDs for stable selectors
        CRITICAL: Test IDs are the most reliable selector strategy

        Args:
            page: Playwright page instance
            test_id: Value of data-testid attribute

        Returns:
            Element data or None if not found
        """
        try:
            element = await page.query_selector(f'[data-testid="{test_id}"]')
            if not element:
                return None

            return await self._extract_element_data(element, "testid-element")

        except Exception as e:
            logger.error(f"Failed to get element by testid '{test_id}': {e}")
            return None

    async def find_elements_by_text(
        self, page: Page, text: str, exact: bool = False
    ) -> List[Dict[str, Any]]:
        """Find elements containing specific text.

        Args:
            page: Playwright page instance
            text: Text to search for
            exact: Whether to match exact text

        Returns:
            List of matching elements
        """
        try:
            selector = f"text={text}" if exact else f"text=/{text}/i"
            elements = await page.query_selector_all(selector)

            results = []
            for element in elements:
                element_data = await self._extract_element_data(element, "text-element")
                if element_data:
                    results.append(element_data)

            return results

        except Exception as e:
            logger.error(f"Failed to find elements by text '{text}': {e}")
            return []

    async def find_elements_by_role(
        self, page: Page, role: str
    ) -> List[Dict[str, Any]]:
        """Find elements by ARIA role.

        PATTERN: Use ARIA roles for accessible element selection
        CRITICAL: ARIA roles provide semantic meaning for test automation

        Args:
            page: Playwright page instance
            role: ARIA role to search for

        Returns:
            List of matching elements
        """
        try:
            elements = await page.query_selector_all(f'[role="{role}"]')

            results = []
            for element in elements:
                element_data = await self._extract_element_data(element, f"role-{role}")
                if element_data:
                    results.append(element_data)

            return results

        except Exception as e:
            logger.error(f"Failed to find elements by role '{role}': {e}")
            return []
