"""Tester agent for test creation and execution."""

import logging
from typing import Optional, List, Dict, Any

from .base import BaseAgent
from ..models.state_models import AgentState, AgentRole, WorkflowStatus
from ..models.agent_models import AgentResponse
from ..services.memory_service import MemoryOrchestrator
from ..services.testing_service import TestingOrchestrator
from ..services.browser_service import BrowserOrchestrator
from ..services.healing_service import HealingOrchestrator
from ..models.testing_models import TestGenerationRequest, TestType
from ..models.healing_models import HealingRequest, TestFailure, FailureType
from ..models.browser_models import (
    UserJourney,
    BrowserType,
    BrowserTestSuite,
    PageObjectModel,
)

logger = logging.getLogger(__name__)


class TesterAgent(BaseAgent):
    """
    Tester agent responsible for test creation and execution.

    Responsibilities:
    - Create unit and integration tests using test generation
    - Execute E2E tests with browser automation (NEW: PRP-10)
    - Execute test suites
    - Report test results and coverage
    - Identify issues and failures
    - Automatically heal failing tests (NEW: PRP-11)
    - Generate Page Object Models for web applications (NEW: PRP-10)
    - Run visual regression, accessibility, and performance tests (NEW: PRP-10)
    """

    def __init__(
        self,
        memory_service: Optional[MemoryOrchestrator] = None,
        testing_service: Optional[TestingOrchestrator] = None,
        browser_service: Optional[BrowserOrchestrator] = None,
        healing_service: Optional[HealingOrchestrator] = None,
        auto_heal: bool = True,
    ):
        """
        Initialize tester agent.

        Args:
            memory_service: Optional memory orchestrator
            testing_service: Optional testing service (will create if not provided)
            browser_service: Optional browser orchestrator for E2E testing (PRP-10)
            healing_service: Optional healing service for self-healing tests (PRP-11)
            auto_heal: Whether to automatically heal failing tests (default: True)
        """
        super().__init__(role=AgentRole.TESTER, memory_service=memory_service)
        self.testing_service = testing_service or TestingOrchestrator()
        self.browser_service = browser_service  # Initialize on demand to avoid Playwright startup cost
        self.healing_service = healing_service  # Initialize on demand for healing
        self.auto_heal = auto_heal

    async def execute(self, state: AgentState) -> AgentResponse:
        """
        Execute testing logic with automatic test generation.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with test results
        """
        try:
            self.logger.info(f"Tester executing tests {state.get('workflow_id')}")

            # Use `or {}` to handle both missing keys and explicit None values
            implementation = state.get("implementation") or {}

            # Retrieve test examples
            context = await self.retrieve_context(
                query="test examples and patterns",
                state=state,
                k=5,
            )

            # Generate and run tests
            test_results = await self._generate_and_run_tests(implementation, state)

            # Store test results
            await self.store_result(
                content=f"Tests: {test_results['passed']}/{test_results['total']} passed, Coverage: {test_results['coverage']:.1%}",
                state=state,
                importance=0.8,
                tags=["testing", "results", "generated"],
            )

            # Create message
            test_message = self.create_message(
                content=f"Testing complete: {test_results['passed']}/{test_results['total']} passed, Coverage: {test_results['coverage']:.1%}",
                message_type="success" if test_results["all_passed"] else "warning",
                metadata={
                    "test_run_id": test_results["test_run_id"],
                    "generated_tests": test_results.get("generated_count", 0),
                },
            )

            # Determine next step based on results
            if test_results["all_passed"]:
                next_agent = AgentRole.VALIDATOR
                next_status = WorkflowStatus.VALIDATING.value
            else:
                next_agent = AgentRole.DEBUGGER
                next_status = WorkflowStatus.DEBUGGING.value

            # State updates
            state_updates = {
                "test_results": test_results,
                "status": next_status,
            }

            return self._create_success_response(
                result=test_results,
                next_agent=next_agent,
                state_updates=state_updates,
                messages=[test_message],
            )

        except Exception as e:
            self.logger.error(f"Tester execution failed: {e}")
            return self._create_error_response(
                error=f"Testing failed: {str(e)}",
            )

    async def _generate_and_run_tests(
        self, implementation: dict, state: AgentState
    ) -> dict:
        """
        Generate and run tests on implementation.

        CRITICAL: Uses TestingOrchestrator for automatic test generation

        Args:
            implementation: Implementation details
            state: Current workflow state

        Returns:
            Test results including generation metrics
        """
        import uuid

        # Handle None or empty implementation (e.g., if implementer failed)
        if not implementation or not implementation.get("files"):
            self.logger.warning("No implementation provided, returning failed test results")
            return {
                "test_run_id": str(uuid.uuid4()),
                "implementation_id": implementation.get("implementation_id") if implementation else None,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "all_passed": False,
                "coverage": 0.0,
                "error": "No implementation to test",
            }

        # Get files to test (already validated above)
        target_files = implementation.get("files", [])

        # Extract file paths if target_files contains dicts
        if target_files and isinstance(target_files[0], dict):
            target_file_paths = [f.get('path', f) for f in target_files]
        else:
            target_file_paths = target_files

        # Create test generation request
        request = TestGenerationRequest(
            target_files=target_file_paths,
            test_types=[TestType.UNIT, TestType.EDGE_CASE],
            coverage_target=0.8,
            include_property_tests=True,
            include_edge_cases=True,
            include_mocks=True,
        )

        try:
            # Generate and run tests
            results = await self.testing_service.generate_and_run_tests(
                target_files=target_file_paths, request=request
            )

            # Extract results
            execution_results = results.get("execution", {})
            total_tests = execution_results.get("total_tests", 0)
            failed_tests = execution_results.get("failed", 0)
            generated_count = results.get("generation", {}).get("count", 0)

            # Tests only pass if: some tests ran AND none failed
            # If no tests were generated/run, that's a failure
            all_passed = total_tests > 0 and failed_tests == 0

            return {
                "test_run_id": str(uuid.uuid4()),
                "implementation_id": implementation.get("implementation_id"),
                "total": total_tests,
                "passed": execution_results.get("passed", 0),
                "failed": failed_tests,
                "all_passed": all_passed,
                "coverage": execution_results.get("coverage", 0.0),
                "generated_count": generated_count,
                "generation_time_ms": results.get("total_time_ms", 0),
                "test_files": results.get("generation", {}).get("test_files", {}),
                "error": None if all_passed else f"Tests failed: {failed_tests} failed, {total_tests} total",
            }

        except Exception as e:
            self.logger.warning(
                f"Test generation failed, falling back to simple tests: {e}"
            )
            return self._run_simple_tests(implementation)

    def _run_simple_tests(self, implementation: dict) -> dict:
        """
        Run basic validation tests on implementation files.

        This performs actual validation:
        - Syntax checking for Python files
        - Basic import validation
        - Checks for required elements

        Args:
            implementation: Implementation details

        Returns:
            Test results with honest pass/fail status
        """
        import uuid
        import ast

        files = implementation.get("files", [])
        test_run_id = str(uuid.uuid4())

        if not files:
            self.logger.warning("No files to test")
            return {
                "test_run_id": test_run_id,
                "implementation_id": implementation.get("implementation_id"),
                "total": 1,
                "passed": 0,
                "failed": 1,
                "all_passed": False,
                "coverage": 0.0,
                "generated_count": 0,
                "error": "No implementation files to test",
            }

        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        test_details = []

        for file_info in files:
            file_path = file_info.get("path", "unknown")
            content = file_info.get("content", "")
            language = file_info.get("language", "python")

            if language == "python":
                # Test 1: Syntax validation
                total_tests += 1
                try:
                    ast.parse(content)
                    passed_tests += 1
                    test_details.append({
                        "test": f"syntax_check:{file_path}",
                        "passed": True,
                    })
                except SyntaxError as e:
                    failed_tests += 1
                    test_details.append({
                        "test": f"syntax_check:{file_path}",
                        "passed": False,
                        "error": str(e),
                    })

                # Test 2: Non-placeholder content check
                total_tests += 1
                if self._is_placeholder_content(content):
                    failed_tests += 1
                    test_details.append({
                        "test": f"content_check:{file_path}",
                        "passed": False,
                        "error": "File appears to contain placeholder/stub content",
                    })
                else:
                    passed_tests += 1
                    test_details.append({
                        "test": f"content_check:{file_path}",
                        "passed": True,
                    })

                # Test 3: Has meaningful implementation
                total_tests += 1
                if self._has_meaningful_code(content):
                    passed_tests += 1
                    test_details.append({
                        "test": f"implementation_check:{file_path}",
                        "passed": True,
                    })
                else:
                    failed_tests += 1
                    test_details.append({
                        "test": f"implementation_check:{file_path}",
                        "passed": False,
                        "error": "File lacks meaningful implementation",
                    })

        # Calculate coverage estimate based on content quality
        coverage = passed_tests / total_tests if total_tests > 0 else 0.0

        return {
            "test_run_id": test_run_id,
            "implementation_id": implementation.get("implementation_id"),
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "all_passed": failed_tests == 0,
            "coverage": coverage,
            "generated_count": 0,
            "test_details": test_details,
        }

    def _is_placeholder_content(self, content: str) -> bool:
        """Check if content is just placeholder/stub code."""
        placeholder_indicators = [
            "# placeholder",
            "# stub",
            "# todo",
            "# implement",
            "pass  # implement",
            "raise NotImplementedError",
            "def main():\n    pass",
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in placeholder_indicators)

    def _has_meaningful_code(self, content: str) -> bool:
        """Check if content has meaningful implementation."""
        # Count actual code lines (not comments or empty)
        lines = content.split("\n")
        code_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                code_lines += 1

        # Should have at least 5 lines of actual code
        if code_lines < 5:
            return False

        # Should have at least one function definition
        if "def " not in content and "class " not in content:
            return False

        # Should not be mostly pass statements
        pass_count = content.count("\n    pass") + content.count("\n        pass")
        if pass_count > 2:
            return False

        return True

    # ===== Self-Healing Methods (PRP-11) =====

    async def heal_failing_tests(
        self,
        test_failures: List[Dict[str, Any]],
        auto_repair: bool = True,
        detect_flaky: bool = True,
    ) -> Dict[str, Any]:
        """
        Heal failing tests using self-healing system.

        NEW: Added in PRP-11 for automatic test healing.

        Args:
            test_failures: List of test failure information
            auto_repair: Whether to automatically apply repairs
            detect_flaky: Whether to detect flaky tests

        Returns:
            Healing results with repairs and recommendations

        Raises:
            RuntimeError: If healing service not available
        """
        if not self.healing_service:
            # Lazy initialize healing service
            self.healing_service = HealingOrchestrator()

        try:
            self.logger.info(f"Healing {len(test_failures)} failing tests")

            # Convert test failures to TestFailure models
            failures = []
            for failure_data in test_failures:
                failure = TestFailure(
                    test_id=failure_data.get("test_id", "unknown"),
                    test_name=failure_data.get("test_name", "unknown"),
                    test_file=failure_data.get("test_file", ""),
                    failure_type=FailureType.RUNTIME_ERROR,  # Will be classified
                    error_message=failure_data.get("error_message", ""),
                    stack_trace=failure_data.get("stack_trace"),
                    target_file=failure_data.get("target_file", ""),
                    target_function=failure_data.get("target_function"),
                    test_framework=failure_data.get("framework", "pytest"),
                    execution_time_ms=failure_data.get("execution_time_ms"),
                )
                failures.append(failure)

            # Create healing request
            request = HealingRequest(
                test_failures=failures,
                auto_repair=auto_repair,
                detect_flaky=detect_flaky,
                optimize=False,  # Don't optimize during normal healing
                max_repair_attempts=3,
                confidence_threshold=0.7,
            )

            # Run healing
            result = await self.healing_service.heal_tests(request)

            self.logger.info(
                f"Healing complete: {result.successful_repairs}/{result.total_failures} repaired, "
                f"{result.flaky_tests_detected} flaky tests detected"
            )

            return {
                "total_failures": result.total_failures,
                "successful_repairs": result.successful_repairs,
                "failed_repairs": result.failed_repairs,
                "flaky_tests_detected": result.flaky_tests_detected,
                "repair_success_rate": result.repair_success_rate,
                "healing_time_ms": result.healing_time_ms,
                "repairs": [
                    {
                        "test_id": repair.failure_analysis.failure.test_id,
                        "strategy": repair.strategy.value,
                        "confidence": repair.confidence,
                        "test_passes": repair.test_passes,
                    }
                    for repair in result.repairs
                ],
                "flaky_tests": [
                    {
                        "test_id": flaky.test_id,
                        "flakiness_score": flaky.flakiness_score,
                        "recommended_action": flaky.recommended_action,
                    }
                    for flaky in result.flaky_tests
                ],
            }

        except Exception as e:
            self.logger.error(f"Test healing failed: {e}")
            raise RuntimeError(f"Test healing failed: {e}")

    # ===== E2E Testing Methods (PRP-10) =====

    async def run_e2e_tests(
        self,
        journeys: List[UserJourney],
        browser_type: BrowserType = BrowserType.CHROMIUM,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """
        Run E2E tests using browser automation.

        NEW: Added in PRP-10 for browser automation support.

        Args:
            journeys: List of user journeys to execute
            browser_type: Browser to use for testing
            parallel: Whether to run tests in parallel

        Returns:
            E2E test results

        Raises:
            RuntimeError: If browser service not available
        """
        if not self.browser_service:
            # Lazy initialize browser service
            self.browser_service = BrowserOrchestrator()

        try:
            self.logger.info(f"Running {len(journeys)} E2E tests with {browser_type.value}")

            results = await self.browser_service.run_e2e_tests(
                journeys=journeys,
                browser_type=browser_type,
                parallel=parallel,
            )

            passed = sum(1 for r in results if r.get("status") == "passed")
            failed = len(results) - passed

            self.logger.info(f"E2E tests complete: {passed}/{len(results)} passed")

            return {
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "all_passed": failed == 0,
                "results": results,
                "test_type": "e2e",
            }

        except Exception as e:
            self.logger.error(f"E2E test execution failed: {e}")
            raise RuntimeError(f"E2E testing failed: {e}")

    async def generate_pom(
        self,
        url: str,
        framework: Optional[str] = None,
        browser_type: BrowserType = BrowserType.CHROMIUM,
    ) -> PageObjectModel:
        """
        Generate Page Object Model from URL.

        NEW: Added in PRP-10 for automated POM generation.

        Args:
            url: URL to analyze
            framework: Frontend framework hint (react, vue, angular)
            browser_type: Browser to use for analysis

        Returns:
            Generated Page Object Model

        Raises:
            RuntimeError: If browser service not available
        """
        if not self.browser_service:
            self.browser_service = BrowserOrchestrator()

        try:
            self.logger.info(f"Generating POM for {url}")

            pom = await self.browser_service.generate_pom(
                url=url,
                framework=framework,
                browser_type=browser_type,
            )

            self.logger.info(
                f"Generated POM: {pom.name} with {len(pom.elements)} elements"
            )

            return pom

        except Exception as e:
            self.logger.error(f"POM generation failed: {e}")
            raise RuntimeError(f"POM generation failed: {e}")

    async def run_visual_tests(
        self,
        test_configs: List[Dict[str, Any]],
        browser_type: BrowserType = BrowserType.CHROMIUM,
    ) -> List[Dict[str, Any]]:
        """
        Run visual regression tests.

        NEW: Added in PRP-10 for visual testing support.

        Args:
            test_configs: List of visual test configurations
            browser_type: Browser to use for testing

        Returns:
            Visual test results

        Raises:
            RuntimeError: If browser service not available
        """
        if not self.browser_service:
            self.browser_service = BrowserOrchestrator()

        try:
            self.logger.info(f"Running {len(test_configs)} visual regression tests")

            results = await self.browser_service.run_visual_tests(
                test_configs=test_configs,
                browser_type=browser_type,
            )

            passed = sum(1 for r in results if r.get("is_match", False))
            self.logger.info(f"Visual tests complete: {passed}/{len(results)} passed")

            return results

        except Exception as e:
            self.logger.error(f"Visual testing failed: {e}")
            raise RuntimeError(f"Visual testing failed: {e}")

    async def run_accessibility_tests(
        self,
        urls: List[str],
        wcag_level: str = "AA",
        browser_type: BrowserType = BrowserType.CHROMIUM,
    ) -> List[Dict[str, Any]]:
        """
        Run accessibility tests on URLs.

        NEW: Added in PRP-10 for accessibility testing support.

        Args:
            urls: List of URLs to test
            wcag_level: WCAG compliance level (A, AA, AAA)
            browser_type: Browser to use for testing

        Returns:
            Accessibility test results with issues

        Raises:
            RuntimeError: If browser service not available
        """
        if not self.browser_service:
            self.browser_service = BrowserOrchestrator()

        try:
            self.logger.info(
                f"Running accessibility tests on {len(urls)} URLs (WCAG {wcag_level})"
            )

            results = await self.browser_service.run_accessibility_tests(
                urls=urls,
                wcag_level=wcag_level,
                browser_type=browser_type,
            )

            total_issues = sum(len(r.get("issues", [])) for r in results)
            self.logger.info(f"Accessibility tests complete: {total_issues} total issues")

            return results

        except Exception as e:
            self.logger.error(f"Accessibility testing failed: {e}")
            raise RuntimeError(f"Accessibility testing failed: {e}")

    async def run_performance_tests(
        self,
        urls: List[str],
        browser_type: BrowserType = BrowserType.CHROMIUM,
    ) -> List[Dict[str, Any]]:
        """
        Run performance tests and collect metrics.

        NEW: Added in PRP-10 for performance testing support.

        Args:
            urls: List of URLs to test
            browser_type: Browser to use for testing

        Returns:
            Performance test results with Core Web Vitals

        Raises:
            RuntimeError: If browser service not available
        """
        if not self.browser_service:
            self.browser_service = BrowserOrchestrator()

        try:
            self.logger.info(f"Running performance tests on {len(urls)} URLs")

            results = await self.browser_service.run_performance_tests(
                urls=urls,
                browser_type=browser_type,
            )

            passed = sum(1 for r in results if r.get("passes_cwv", False))
            self.logger.info(
                f"Performance tests complete: {passed}/{len(results)} pass Core Web Vitals"
            )

            return results

        except Exception as e:
            self.logger.error(f"Performance testing failed: {e}")
            raise RuntimeError(f"Performance testing failed: {e}")

    async def execute_test_suite(
        self, suite: BrowserTestSuite
    ) -> Dict[str, Any]:
        """
        Execute complete browser test suite.

        NEW: Added in PRP-10 for comprehensive test suite execution.

        Args:
            suite: Browser test suite specification

        Returns:
            Complete test suite results

        Raises:
            RuntimeError: If browser service not available
        """
        if not self.browser_service:
            self.browser_service = BrowserOrchestrator()

        try:
            self.logger.info(f"Executing test suite: {suite.name}")

            results = await self.browser_service.execute_test_suite(suite)

            self.logger.info(
                f"Test suite complete: {results['summary']['total_tests']} tests, "
                f"{results['summary']['success_rate']:.1f}% success rate"
            )

            return results

        except Exception as e:
            self.logger.error(f"Test suite execution failed: {e}")
            raise RuntimeError(f"Test suite execution failed: {e}")

    async def cleanup_browser(self) -> None:
        """
        Clean up browser resources.

        CRITICAL: Call this when done with browser testing to prevent resource leaks.

        NEW: Added in PRP-10 for browser resource management.
        """
        if self.browser_service:
            try:
                await self.browser_service.cleanup()
                self.logger.info("Browser resources cleaned up")
            except Exception as e:
                self.logger.error(f"Browser cleanup failed: {e}")
