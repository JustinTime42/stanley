"""Accessibility testing using axe-core for WCAG compliance audits.

This module provides the AccessibilityTester class for running automated
accessibility audits using the axe-core library, supporting different WCAG
levels and providing detailed violation reports with fix suggestions.
"""

import logging
from typing import List, Dict, Any, Literal
from playwright.async_api import Page

from src.models.browser_models import AccessibilityIssue

logger = logging.getLogger(__name__)


class AccessibilityTester:
    """Run accessibility audits using axe-core library.

    This class injects the axe-core library into web pages and runs
    WCAG compliance audits, parsing violations into structured
    AccessibilityIssue objects with fix suggestions.

    PATTERN: Use evaluate() to run JavaScript libraries in browser context.
    """

    # Axe-core CDN URL
    AXE_CORE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.7.2/axe.min.js"

    # WCAG level to axe-core tag mapping
    WCAG_TAG_MAPPING = {
        "A": ["wcag2a"],
        "AA": ["wcag2a", "wcag2aa"],
        "AAA": ["wcag2a", "wcag2aa", "wcag2aaa"],
    }

    def __init__(self):
        """Initialize the accessibility tester."""
        self._axe_injected: Dict[str, bool] = {}

    async def inject_axe(self, page: Page) -> None:
        """Inject axe-core library into the page.

        Args:
            page: Playwright page instance

        Raises:
            Exception: If axe-core injection fails
        """
        page_id = str(id(page))

        # Skip if already injected for this page
        if self._axe_injected.get(page_id, False):
            logger.debug("axe-core already injected for this page")
            return

        try:
            logger.info(f"Injecting axe-core library from {self.AXE_CORE_CDN}")
            await page.add_script_tag(url=self.AXE_CORE_CDN)

            # Verify axe is loaded
            is_loaded = await page.evaluate("() => typeof axe !== 'undefined'")
            if not is_loaded:
                raise Exception("axe-core library failed to load")

            self._axe_injected[page_id] = True
            logger.info("axe-core library injected successfully")

        except Exception as e:
            logger.error(f"Failed to inject axe-core: {e}")
            raise RuntimeError(f"Failed to inject axe-core: {e}")

    async def run_audit(
        self,
        page: Page,
        wcag_level: Literal["A", "AA", "AAA"] = "AA",
        include_best_practices: bool = True,
    ) -> List[AccessibilityIssue]:
        """Run accessibility audit on the page.

        Args:
            page: Playwright page instance
            wcag_level: WCAG conformance level (A, AA, or AAA)
            include_best_practices: Include best practice rules

        Returns:
            List of accessibility issues found

        Example:
            tester = AccessibilityTester()
            issues = await tester.run_audit(page, wcag_level="AA")
            for issue in issues:
                print(f"{issue.impact}: {issue.description}")
        """
        try:
            # Ensure axe-core is injected
            await self.inject_axe(page)

            # Build rule tags based on WCAG level
            tags = self.WCAG_TAG_MAPPING.get(wcag_level, ["wcag2a", "wcag2aa"]).copy()
            if include_best_practices:
                tags.append("best-practice")

            logger.info(
                f"Running accessibility audit with WCAG level {wcag_level} (tags: {tags})"
            )

            # Run axe.run() with the specified tags
            results = await page.evaluate(
                """
                (tags) => {
                    return new Promise((resolve, reject) => {
                        axe.run(document, {
                            runOnly: {
                                type: 'tag',
                                values: tags
                            }
                        })
                        .then(results => resolve(results))
                        .catch(error => reject(error));
                    });
                }
                """,
                tags,
            )

            # Parse violations into AccessibilityIssue objects
            issues = self._parse_violations(results.get("violations", []), wcag_level)

            logger.info(
                f"Accessibility audit completed: {len(issues)} issues found "
                f"({self._count_by_impact(issues)})"
            )

            return issues

        except Exception as e:
            logger.error(f"Accessibility audit failed: {e}")
            raise RuntimeError(f"Accessibility audit failed: {e}")

    def _parse_violations(
        self, violations: List[Dict[str, Any]], wcag_level: Literal["A", "AA", "AAA"]
    ) -> List[AccessibilityIssue]:
        """Parse axe-core violations into AccessibilityIssue objects.

        Args:
            violations: Raw violations from axe-core
            wcag_level: WCAG level used for the audit

        Returns:
            List of AccessibilityIssue objects
        """
        issues = []

        for violation in violations:
            rule_id = violation.get("id", "unknown")
            impact = violation.get("impact", "moderate")
            description = violation.get("description", "")
            help_text = violation.get("help", "")
            help_url = violation.get("helpUrl", "")
            tags = violation.get("tags", [])

            # Extract WCAG criteria from tags
            wcag_criteria = [tag for tag in tags if tag.startswith("wcag")]

            # Determine WCAG level from tags
            if "wcag2aaa" in tags or "wcag21aaa" in tags or "wcag22aaa" in tags:
                issue_level = "AAA"
            elif "wcag2aa" in tags or "wcag21aa" in tags or "wcag22aa" in tags:
                issue_level = "AA"
            else:
                issue_level = "A"

            # Process each node (element) that violates the rule
            nodes = violation.get("nodes", [])
            for idx, node in enumerate(nodes):
                target = node.get("target", [])
                selector = target[0] if target else "unknown"
                html = node.get("html", "")

                # Generate fix suggestion from node data
                fix_suggestion = self._generate_fix_suggestion(rule_id, node, help_url)

                # Create unique ID for this issue
                issue_id = f"{rule_id}_{idx}"

                issue = AccessibilityIssue(
                    id=issue_id,
                    impact=impact,
                    rule_id=rule_id,
                    description=description,
                    help_text=f"{help_text}. More info: {help_url}",
                    selector=selector,
                    html=html,
                    wcag_criteria=wcag_criteria,
                    wcag_level=issue_level,
                    fix_suggestion=fix_suggestion,
                )

                issues.append(issue)

        return issues

    def _generate_fix_suggestion(
        self, rule_id: str, node: Dict[str, Any], help_url: str
    ) -> str:
        """Generate fix suggestion based on the violation type.

        Args:
            rule_id: The axe-core rule ID
            node: The node data from axe-core
            help_url: URL to detailed help

        Returns:
            Human-readable fix suggestion
        """
        # Extract failure messages from node
        failure_summary = node.get("failureSummary", "")
        any_checks = node.get("any", [])
        all_checks = node.get("all", [])
        none_checks = node.get("none", [])

        suggestions = []

        # Rule-specific suggestions
        rule_suggestions = {
            "color-contrast": "Increase the contrast ratio between text and background colors to meet WCAG AA standards (4.5:1 for normal text, 3:1 for large text).",
            "image-alt": "Add descriptive alt text to the image using the 'alt' attribute.",
            "label": "Add a <label> element associated with this form control, or use aria-label/aria-labelledby.",
            "button-name": "Provide accessible text for the button using text content, aria-label, or aria-labelledby.",
            "link-name": "Ensure the link has descriptive text using text content, aria-label, or aria-labelledby.",
            "heading-order": "Use heading levels in sequential order (h1, h2, h3, etc.) without skipping levels.",
            "html-has-lang": "Add a 'lang' attribute to the <html> element (e.g., <html lang='en'>).",
            "valid-lang": "Use a valid language code in the 'lang' attribute (e.g., 'en', 'es', 'fr').",
            "landmark-one-main": "Ensure the page has exactly one <main> landmark or role='main'.",
            "region": "Place all page content within landmark regions (main, nav, aside, etc.).",
            "page-has-heading-one": "Add a single <h1> heading to the page to describe the main content.",
            "bypass": "Add a 'skip to main content' link at the top of the page for keyboard users.",
            "meta-viewport": "Avoid using 'user-scalable=no' or setting 'maximum-scale' below 5 in the viewport meta tag.",
            "aria-required-attr": "Add the required ARIA attributes for this role.",
            "aria-valid-attr": "Use only valid ARIA attributes as defined in the ARIA specification.",
            "aria-valid-attr-value": "Ensure ARIA attribute values are valid and properly formatted.",
            "duplicate-id": "Make sure all ID attributes on the page are unique.",
            "form-field-multiple-labels": "Ensure each form field has only one associated <label> element.",
            "frame-title": "Add a descriptive 'title' attribute to the <iframe> or <frame> element.",
        }

        # Add rule-specific suggestion if available
        if rule_id in rule_suggestions:
            suggestions.append(rule_suggestions[rule_id])

        # Add failure summary if available
        if failure_summary:
            suggestions.append(f"Issue details: {failure_summary}")

        # Add check messages
        for check in any_checks + all_checks + none_checks:
            message = check.get("message", "")
            if message:
                suggestions.append(message)

        # Build final suggestion
        if suggestions:
            fix_text = " ".join(suggestions)
        else:
            fix_text = f"Review the element and fix according to WCAG guidelines. See {help_url} for details."

        return fix_text

    def _count_by_impact(self, issues: List[AccessibilityIssue]) -> str:
        """Count issues by impact level for logging.

        Args:
            issues: List of accessibility issues

        Returns:
            Formatted string with counts by impact
        """
        counts = {"critical": 0, "serious": 0, "moderate": 0, "minor": 0}

        for issue in issues:
            if issue.impact in counts:
                counts[issue.impact] += 1

        parts = [f"{count} {level}" for level, count in counts.items() if count > 0]
        return ", ".join(parts) if parts else "no issues"

    async def run_partial_audit(
        self,
        page: Page,
        selector: str,
        wcag_level: Literal["A", "AA", "AAA"] = "AA",
        include_best_practices: bool = True,
    ) -> List[AccessibilityIssue]:
        """Run accessibility audit on a specific element and its children.

        Args:
            page: Playwright page instance
            selector: CSS selector for the element to audit
            wcag_level: WCAG conformance level (A, AA, or AAA)
            include_best_practices: Include best practice rules

        Returns:
            List of accessibility issues found in the element

        Example:
            issues = await tester.run_partial_audit(
                page,
                selector="#main-content",
                wcag_level="AA"
            )
        """
        try:
            # Ensure axe-core is injected
            await self.inject_axe(page)

            # Build rule tags
            tags = self.WCAG_TAG_MAPPING.get(wcag_level, ["wcag2a", "wcag2aa"]).copy()
            if include_best_practices:
                tags.append("best-practice")

            logger.info(
                f"Running partial accessibility audit on '{selector}' with WCAG level {wcag_level}"
            )

            # Run axe.run() on the specific element
            results = await page.evaluate(
                """
                ({selector, tags}) => {
                    return new Promise((resolve, reject) => {
                        const element = document.querySelector(selector);
                        if (!element) {
                            reject(new Error(`Element not found: ${selector}`));
                            return;
                        }

                        axe.run(element, {
                            runOnly: {
                                type: 'tag',
                                values: tags
                            }
                        })
                        .then(results => resolve(results))
                        .catch(error => reject(error));
                    });
                }
                """,
                {"selector": selector, "tags": tags},
            )

            # Parse violations
            issues = self._parse_violations(results.get("violations", []), wcag_level)

            logger.info(
                f"Partial accessibility audit completed: {len(issues)} issues found "
                f"in '{selector}' ({self._count_by_impact(issues)})"
            )

            return issues

        except Exception as e:
            logger.error(f"Partial accessibility audit failed: {e}")
            raise RuntimeError(f"Partial accessibility audit failed: {e}")

    def filter_by_impact(
        self,
        issues: List[AccessibilityIssue],
        min_impact: Literal["minor", "moderate", "serious", "critical"] = "moderate",
    ) -> List[AccessibilityIssue]:
        """Filter issues by minimum impact level.

        Args:
            issues: List of accessibility issues
            min_impact: Minimum impact level to include

        Returns:
            Filtered list of issues

        Example:
            critical_issues = tester.filter_by_impact(
                all_issues,
                min_impact="critical"
            )
        """
        impact_order = {"minor": 0, "moderate": 1, "serious": 2, "critical": 3}
        min_level = impact_order.get(min_impact, 1)

        filtered = [
            issue for issue in issues if impact_order.get(issue.impact, 0) >= min_level
        ]

        logger.debug(
            f"Filtered {len(issues)} issues to {len(filtered)} with minimum impact '{min_impact}'"
        )

        return filtered

    def group_by_rule(
        self, issues: List[AccessibilityIssue]
    ) -> Dict[str, List[AccessibilityIssue]]:
        """Group issues by rule ID.

        Args:
            issues: List of accessibility issues

        Returns:
            Dictionary mapping rule IDs to lists of issues

        Example:
            grouped = tester.group_by_rule(issues)
            for rule_id, rule_issues in grouped.items():
                print(f"{rule_id}: {len(rule_issues)} occurrences")
        """
        grouped: Dict[str, List[AccessibilityIssue]] = {}

        for issue in issues:
            if issue.rule_id not in grouped:
                grouped[issue.rule_id] = []
            grouped[issue.rule_id].append(issue)

        logger.debug(f"Grouped {len(issues)} issues into {len(grouped)} rule types")

        return grouped

    async def get_accessibility_tree(self, page: Page) -> Dict[str, Any]:
        """Get the accessibility tree snapshot from the page.

        Args:
            page: Playwright page instance

        Returns:
            Accessibility tree as a dictionary

        Example:
            tree = await tester.get_accessibility_tree(page)
            print(tree)
        """
        try:
            logger.info("Capturing accessibility tree snapshot")
            snapshot = await page.accessibility.snapshot()
            logger.debug(f"Accessibility tree captured: {snapshot}")
            return snapshot or {}
        except Exception as e:
            logger.error(f"Failed to get accessibility tree: {e}")
            raise RuntimeError(f"Failed to get accessibility tree: {e}")
