"""Smoke testing strategy for quick critical path validation.

This module provides the SmokeTestStrategy class for executing fast smoke tests
that validate critical functionality, page loads, and key elements without
running full test suites.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from playwright.async_api import Page


logger = logging.getLogger(__name__)


class SmokeTestStrategy:
    """Execute smoke tests for quick critical path validation.

    This class performs lightweight smoke testing to quickly validate that
    critical pages load, essential elements are present, and basic functionality
    works. Ideal for CI/CD pipelines and deployment validation.

    PATTERN: Focus on breadth over depth - test many critical paths quickly
    rather than exhaustive testing of individual features.
    """

    def __init__(
        self,
        timeout_ms: int = 10000,
        fail_fast: bool = True,
    ):
        """Initialize the smoke testing strategy.

        Args:
            timeout_ms: Default timeout for operations in milliseconds
            fail_fast: Stop testing on first failure
        """
        self.timeout_ms = timeout_ms
        self.fail_fast = fail_fast
        self._test_results: List[Dict[str, Any]] = []
        logger.info(
            f"Smoke test strategy initialized (timeout: {timeout_ms}ms, "
            f"fail_fast: {fail_fast})"
        )

    async def run_smoke_tests(
        self,
        page: Page,
        test_suite: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a complete smoke test suite.

        Args:
            page: Playwright page instance
            test_suite: Smoke test suite specification

        Returns:
            Dictionary with test results

        Example:
            strategy = SmokeTestStrategy()
            suite = {
                "name": "Production Smoke Tests",
                "critical_pages": [
                    {"url": "https://example.com", "title": "Home"},
                    {"url": "https://example.com/about", "title": "About"},
                ],
                "critical_elements": {
                    "https://example.com": ["header", "nav", "#main"],
                },
                "api_endpoints": [
                    "https://api.example.com/health",
                ],
            }
            results = await strategy.run_smoke_tests(page, suite)
        """
        logger.info(f"Running smoke test suite: {test_suite.get('name', 'Unnamed')}")

        start_time = datetime.now()
        results = {
            "suite_name": test_suite.get("name", "Smoke Tests"),
            "start_time": start_time.isoformat(),
            "tests": [],
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
        }

        try:
            # Test critical pages
            critical_pages = test_suite.get("critical_pages", [])
            for page_spec in critical_pages:
                if self.fail_fast and results["failed"] > 0:
                    logger.info("Fail-fast enabled, stopping tests")
                    break

                test_result = await self.test_page_load(
                    page, page_spec["url"], page_spec.get("title")
                )
                results["tests"].append(test_result)
                results["total"] += 1

                if test_result["passed"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            # Test critical elements
            critical_elements = test_suite.get("critical_elements", {})
            for url, selectors in critical_elements.items():
                if self.fail_fast and results["failed"] > 0:
                    break

                test_result = await self.test_elements_present(page, url, selectors)
                results["tests"].append(test_result)
                results["total"] += 1

                if test_result["passed"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            # Test API endpoints
            api_endpoints = test_suite.get("api_endpoints", [])
            for endpoint in api_endpoints:
                if self.fail_fast and results["failed"] > 0:
                    break

                test_result = await self.test_api_health(page, endpoint)
                results["tests"].append(test_result)
                results["total"] += 1

                if test_result["passed"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            # Test critical flows
            critical_flows = test_suite.get("critical_flows", [])
            for flow in critical_flows:
                if self.fail_fast and results["failed"] > 0:
                    break

                test_result = await self.test_critical_flow(page, flow)
                results["tests"].append(test_result)
                results["total"] += 1

                if test_result["passed"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

        except Exception as e:
            logger.error(f"Smoke test suite failed: {e}")
            results["error"] = str(e)

        finally:
            end_time = datetime.now()
            results["end_time"] = end_time.isoformat()
            results["duration_ms"] = int((end_time - start_time).total_seconds() * 1000)

        # Store results
        self._test_results.append(results)

        logger.info(
            f"Smoke tests completed: {results['passed']}/{results['total']} passed "
            f"({results['duration_ms']}ms)"
        )

        return results

    async def test_page_load(
        self, page: Page, url: str, expected_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Test that a page loads successfully.

        Args:
            page: Playwright page instance
            url: URL to test
            expected_title: Expected page title (optional)

        Returns:
            Dictionary with test result

        Example:
            result = await strategy.test_page_load(
                page,
                "https://example.com",
                expected_title="Example Domain"
            )
        """
        logger.info(f"Testing page load: {url}")

        start_time = datetime.now()
        test_result = {
            "name": f"Page Load: {url}",
            "type": "page_load",
            "url": url,
            "passed": False,
            "error": None,
            "duration_ms": 0,
            "status_code": None,
            "title": None,
        }

        try:
            # Navigate to page and capture response
            response = await page.goto(
                url, timeout=self.timeout_ms, wait_until="domcontentloaded"
            )

            # Check response status
            if response:
                test_result["status_code"] = response.status
                if response.status >= 400:
                    raise Exception(f"HTTP {response.status}")

            # Get page title
            title = await page.title()
            test_result["title"] = title

            # Validate title if expected
            if expected_title and expected_title not in title:
                raise Exception(
                    f"Title mismatch: expected '{expected_title}', got '{title}'"
                )

            # Check for critical errors in console
            console_errors = await self._check_console_errors(page)
            if console_errors:
                test_result["console_errors"] = console_errors
                logger.warning(f"Console errors detected: {len(console_errors)}")

            test_result["passed"] = True
            logger.info(f"Page load test PASSED: {url}")

        except Exception as e:
            logger.error(f"Page load test FAILED: {url} - {e}")
            test_result["error"] = str(e)

        finally:
            end_time = datetime.now()
            test_result["duration_ms"] = int(
                (end_time - start_time).total_seconds() * 1000
            )

        return test_result

    async def test_elements_present(
        self, page: Page, url: str, selectors: List[str]
    ) -> Dict[str, Any]:
        """Test that critical elements are present on a page.

        Args:
            page: Playwright page instance
            url: Page URL
            selectors: List of CSS selectors to check

        Returns:
            Dictionary with test result

        Example:
            result = await strategy.test_elements_present(
                page,
                "https://example.com",
                ["header", "nav", "#main-content", ".footer"]
            )
        """
        logger.info(f"Testing elements present on: {url}")

        start_time = datetime.now()
        test_result = {
            "name": f"Critical Elements: {url}",
            "type": "elements_present",
            "url": url,
            "passed": False,
            "error": None,
            "duration_ms": 0,
            "selectors_checked": len(selectors),
            "selectors_found": 0,
            "missing_elements": [],
        }

        try:
            # Navigate to page
            await page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")

            # Check each selector
            missing = []
            for selector in selectors:
                try:
                    is_visible = await page.is_visible(selector, timeout=2000)
                    if is_visible:
                        test_result["selectors_found"] += 1
                    else:
                        missing.append(selector)
                except Exception:
                    missing.append(selector)

            test_result["missing_elements"] = missing

            # Pass if all elements found
            if not missing:
                test_result["passed"] = True
                logger.info(
                    f"Elements test PASSED: {url} "
                    f"({test_result['selectors_found']}/{test_result['selectors_checked']})"
                )
            else:
                raise Exception(
                    f"Missing elements: {', '.join(missing)} "
                    f"({test_result['selectors_found']}/{test_result['selectors_checked']} found)"
                )

        except Exception as e:
            logger.error(f"Elements test FAILED: {url} - {e}")
            test_result["error"] = str(e)

        finally:
            end_time = datetime.now()
            test_result["duration_ms"] = int(
                (end_time - start_time).total_seconds() * 1000
            )

        return test_result

    async def test_api_health(self, page: Page, endpoint: str) -> Dict[str, Any]:
        """Test that an API endpoint is healthy.

        Args:
            page: Playwright page instance (for network access)
            endpoint: API endpoint URL

        Returns:
            Dictionary with test result

        Example:
            result = await strategy.test_api_health(
                page,
                "https://api.example.com/health"
            )
        """
        logger.info(f"Testing API health: {endpoint}")

        start_time = datetime.now()
        test_result = {
            "name": f"API Health: {endpoint}",
            "type": "api_health",
            "endpoint": endpoint,
            "passed": False,
            "error": None,
            "duration_ms": 0,
            "status_code": None,
            "response_time_ms": None,
        }

        try:
            # Make request using page.evaluate to use browser's fetch
            request_start = datetime.now()
            response_data = await page.evaluate(
                """
                async (url) => {
                    try {
                        const response = await fetch(url);
                        return {
                            status: response.status,
                            ok: response.ok,
                        };
                    } catch (error) {
                        return {
                            error: error.message
                        };
                    }
                }
                """,
                endpoint,
            )
            request_end = datetime.now()

            test_result["response_time_ms"] = int(
                (request_end - request_start).total_seconds() * 1000
            )

            # Check for errors
            if "error" in response_data:
                raise Exception(response_data["error"])

            # Check status
            test_result["status_code"] = response_data.get("status")
            if not response_data.get("ok", False):
                raise Exception(f"HTTP {test_result['status_code']}")

            test_result["passed"] = True
            logger.info(
                f"API health test PASSED: {endpoint} "
                f"(status: {test_result['status_code']}, "
                f"time: {test_result['response_time_ms']}ms)"
            )

        except Exception as e:
            logger.error(f"API health test FAILED: {endpoint} - {e}")
            test_result["error"] = str(e)

        finally:
            end_time = datetime.now()
            test_result["duration_ms"] = int(
                (end_time - start_time).total_seconds() * 1000
            )

        return test_result

    async def test_critical_flow(
        self, page: Page, flow: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test a critical user flow.

        Args:
            page: Playwright page instance
            flow: Flow specification

        Returns:
            Dictionary with test result

        Example:
            flow = {
                "name": "Homepage to Product",
                "steps": [
                    {"action": "goto", "url": "https://example.com"},
                    {"action": "click", "selector": ".product-link"},
                    {"action": "wait_for", "selector": ".product-details"},
                ]
            }
            result = await strategy.test_critical_flow(page, flow)
        """
        logger.info(f"Testing critical flow: {flow.get('name', 'Unnamed')}")

        start_time = datetime.now()
        test_result = {
            "name": f"Critical Flow: {flow.get('name', 'Unnamed')}",
            "type": "critical_flow",
            "passed": False,
            "error": None,
            "duration_ms": 0,
            "steps_completed": 0,
            "total_steps": len(flow.get("steps", [])),
        }

        try:
            steps = flow.get("steps", [])

            for idx, step in enumerate(steps):
                action = step.get("action")

                if action == "goto":
                    await page.goto(
                        step["url"],
                        timeout=self.timeout_ms,
                        wait_until="domcontentloaded",
                    )

                elif action == "click":
                    await page.click(step["selector"], timeout=self.timeout_ms)

                elif action == "fill":
                    await page.fill(
                        step["selector"], step["value"], timeout=self.timeout_ms
                    )

                elif action == "wait_for":
                    await page.wait_for_selector(
                        step["selector"], timeout=self.timeout_ms
                    )

                elif action == "check_text":
                    text = await page.text_content(step["selector"])
                    if step["expected"] not in (text or ""):
                        raise Exception(
                            f"Text mismatch: expected '{step['expected']}' in '{text}'"
                        )

                else:
                    logger.warning(f"Unknown action: {action}")

                test_result["steps_completed"] += 1

            test_result["passed"] = True
            logger.info(
                f"Critical flow test PASSED: {flow.get('name')} "
                f"({test_result['steps_completed']}/{test_result['total_steps']} steps)"
            )

        except Exception as e:
            logger.error(f"Critical flow test FAILED: {flow.get('name')} - {e}")
            test_result["error"] = str(e)

        finally:
            end_time = datetime.now()
            test_result["duration_ms"] = int(
                (end_time - start_time).total_seconds() * 1000
            )

        return test_result

    async def test_multiple_pages(
        self,
        page: Page,
        urls: List[str],
        check_title: bool = True,
        check_status: bool = True,
    ) -> Dict[str, Any]:
        """Test multiple pages for basic functionality.

        Args:
            page: Playwright page instance
            urls: List of URLs to test
            check_title: Verify page has a title
            check_status: Verify response status is OK

        Returns:
            Dictionary with aggregated test results

        Example:
            urls = [
                "https://example.com",
                "https://example.com/about",
                "https://example.com/contact",
            ]
            result = await strategy.test_multiple_pages(page, urls)
        """
        logger.info(f"Testing {len(urls)} pages")

        start_time = datetime.now()
        results = {
            "name": f"Multiple Pages Test ({len(urls)} pages)",
            "type": "multiple_pages",
            "passed": True,
            "total": len(urls),
            "passed_count": 0,
            "failed_count": 0,
            "pages": [],
        }

        for url in urls:
            page_result = await self.test_page_load(page, url)

            # Additional checks
            if check_title and not page_result.get("title"):
                page_result["passed"] = False
                page_result["error"] = "No page title found"

            if check_status and page_result.get("status_code", 0) >= 400:
                page_result["passed"] = False

            results["pages"].append(page_result)

            if page_result["passed"]:
                results["passed_count"] += 1
            else:
                results["failed_count"] += 1
                results["passed"] = False

        end_time = datetime.now()
        results["duration_ms"] = int((end_time - start_time).total_seconds() * 1000)

        logger.info(
            f"Multiple pages test completed: "
            f"{results['passed_count']}/{results['total']} passed"
        )

        return results

    async def _check_console_errors(self, page: Page) -> List[str]:
        """Check for console errors on the page.

        Args:
            page: Playwright page instance

        Returns:
            List of console error messages
        """
        try:
            errors = await page.evaluate("""
                () => {
                    // This is a simple check - in real scenarios, you'd set up
                    // console listeners before navigation
                    return [];
                }
            """)
            return errors or []
        except Exception:
            return []

    def validate_smoke_test_suite(self, test_suite: Dict[str, Any]) -> bool:
        """Validate smoke test suite specification.

        Args:
            test_suite: Test suite to validate

        Returns:
            True if valid, False otherwise

        Example:
            is_valid = strategy.validate_smoke_test_suite(suite)
            if not is_valid:
                print("Invalid test suite!")
        """
        required_fields = ["name"]

        # Check required fields
        for field in required_fields:
            if field not in test_suite:
                logger.error(f"Missing required field: {field}")
                return False

        # Check that at least one test type is defined
        test_types = [
            "critical_pages",
            "critical_elements",
            "api_endpoints",
            "critical_flows",
        ]

        has_tests = any(test_suite.get(test_type) for test_type in test_types)

        if not has_tests:
            logger.error("Test suite must define at least one test type")
            return False

        logger.debug("Test suite validation passed")
        return True

    def get_test_results(self) -> List[Dict[str, Any]]:
        """Get all test results.

        Returns:
            List of test results

        Example:
            results = strategy.get_test_results()
            for result in results:
                print(f"{result['suite_name']}: {result['passed']}/{result['total']}")
        """
        return self._test_results

    def clear_results(self) -> None:
        """Clear stored test results.

        Example:
            strategy.clear_results()
        """
        self._test_results.clear()
        logger.debug("Test results cleared")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all test executions.

        Returns:
            Dictionary with summary statistics

        Example:
            summary = strategy.get_summary()
            print(f"Overall pass rate: {summary['pass_rate']}%")
        """
        if not self._test_results:
            return {
                "total_runs": 0,
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "pass_rate": 0.0,
            }

        total_runs = len(self._test_results)
        total_tests = sum(r.get("total", 0) for r in self._test_results)
        total_passed = sum(r.get("passed", 0) for r in self._test_results)
        total_failed = sum(r.get("failed", 0) for r in self._test_results)

        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        return {
            "total_runs": total_runs,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "pass_rate": pass_rate,
        }
