"""User flow execution strategy for multi-step browser workflows.

This module provides the UserFlowStrategy class for executing predefined
user flows and journeys, supporting authentication, navigation, form
interactions, and complex multi-step workflows.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from playwright.async_api import Page, Browser, BrowserContext

from src.models.browser_models import (
    UserJourney,
    JourneyStep,
)

logger = logging.getLogger(__name__)


class UserFlowStrategy:
    """Execute predefined user flows and multi-step journeys.

    This class handles complex user workflows including authentication,
    navigation, form filling, and multi-page interactions with proper
    wait conditions and validation.

    PATTERN: Use explicit waits and validations at each step for reliable
    test execution across different environments and network conditions.
    """

    def __init__(self, browser: Optional[Browser] = None):
        """Initialize the user flow strategy.

        Args:
            browser: Optional browser instance to use
        """
        self.browser = browser
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._journey_results: List[Dict[str, Any]] = []
        logger.info("User flow strategy initialized")

    async def execute_journey(
        self, journey: UserJourney, browser: Optional[Browser] = None
    ) -> Dict[str, Any]:
        """Execute a complete user journey.

        Args:
            journey: User journey specification
            browser: Browser instance (overrides constructor browser)

        Returns:
            Dictionary with journey execution results

        Raises:
            Exception: If journey execution fails

        Example:
            strategy = UserFlowStrategy()
            journey = UserJourney(
                id="checkout-flow",
                name="E-commerce Checkout",
                start_url="https://shop.example.com",
                steps=[...],
            )
            result = await strategy.execute_journey(journey, browser)
        """
        logger.info(f"Executing user journey: {journey.name}")

        start_time = datetime.now()
        browser_instance = browser or self.browser

        if not browser_instance:
            raise ValueError("Browser instance required for journey execution")

        try:
            # Create new context for isolation
            self.context = await browser_instance.new_context(
                viewport={
                    "width": journey.viewport.width,
                    "height": journey.viewport.height,
                }
            )

            # Create new page
            self.page = await self.context.new_page()

            # Navigate to start URL
            logger.info(f"Navigating to start URL: {journey.start_url}")
            await self.page.goto(journey.start_url, wait_until="networkidle")

            # Execute all steps
            step_results = []
            for step in journey.steps:
                step_result = await self._execute_step(step)
                step_results.append(step_result)

                # Stop if step failed
                if not step_result["success"]:
                    logger.error(
                        f"Step {step.step_number} failed: {step_result['error']}"
                    )
                    break

            # Validate assertions
            assertion_results = await self._validate_assertions(journey.assertions)

            # Calculate duration
            end_time = datetime.now()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            # Build result
            success = all(r["success"] for r in step_results) and all(
                r["passed"] for r in assertion_results
            )

            result = {
                "journey_id": journey.id,
                "journey_name": journey.name,
                "success": success,
                "duration_ms": duration_ms,
                "steps_completed": sum(1 for r in step_results if r["success"]),
                "total_steps": len(journey.steps),
                "step_results": step_results,
                "assertions": assertion_results,
                "timestamp": start_time.isoformat(),
            }

            logger.info(
                f"Journey completed: {journey.name} - "
                f"{'SUCCESS' if success else 'FAILED'} "
                f"({result['steps_completed']}/{result['total_steps']} steps)"
            )

            self._journey_results.append(result)
            return result

        except Exception as e:
            logger.error(f"Journey execution failed: {e}")
            raise

        finally:
            # Cleanup
            if self.context:
                await self.context.close()
                self.context = None
                self.page = None

    async def _execute_step(self, step: JourneyStep) -> Dict[str, Any]:
        """Execute a single journey step.

        Args:
            step: Journey step to execute

        Returns:
            Dictionary with step execution result
        """
        logger.info(f"Executing step {step.step_number}: {step.action}")

        start_time = datetime.now()
        result = {
            "step_number": step.step_number,
            "action": step.action,
            "success": False,
            "error": None,
            "duration_ms": 0,
        }

        try:
            # Execute action based on type
            if step.action == "navigate":
                await self._action_navigate(step)

            elif step.action == "click":
                await self._action_click(step)

            elif step.action == "fill":
                await self._action_fill(step)

            elif step.action == "select":
                await self._action_select(step)

            elif step.action == "check":
                await self._action_check(step)

            elif step.action == "wait":
                await self._action_wait(step)

            elif step.action == "screenshot":
                await self._action_screenshot(step)

            elif step.action == "evaluate":
                await self._action_evaluate(step)

            else:
                raise ValueError(f"Unknown action type: {step.action}")

            # Wait for network if requested
            if step.wait_for_network:
                await self.page.wait_for_load_state("networkidle")

            # Wait for specific condition if specified
            if step.wait_for:
                await self._wait_for_condition(step.wait_for, step.wait_timeout_ms)

            # Validate step if validation provided
            if step.validate:
                validation_passed = await self._validate_step(step.validate)
                if not validation_passed:
                    raise Exception(f"Step validation failed: {step.validate}")

            # Take screenshot if requested
            if step.screenshot:
                await self._take_step_screenshot(step.step_number)

            result["success"] = True

        except Exception as e:
            logger.error(f"Step {step.step_number} failed: {e}")
            result["error"] = str(e)

        finally:
            end_time = datetime.now()
            result["duration_ms"] = int((end_time - start_time).total_seconds() * 1000)

        return result

    async def _action_navigate(self, step: JourneyStep) -> None:
        """Navigate to a URL.

        Args:
            step: Journey step with target URL
        """
        if not step.target:
            raise ValueError("Navigate action requires target URL")

        logger.debug(f"Navigating to: {step.target}")
        await self.page.goto(step.target, wait_until="networkidle")

    async def _action_click(self, step: JourneyStep) -> None:
        """Click an element.

        Args:
            step: Journey step with target selector
        """
        if not step.target:
            raise ValueError("Click action requires target selector")

        logger.debug(f"Clicking element: {step.target}")
        await self.page.click(step.target, timeout=step.wait_timeout_ms)

    async def _action_fill(self, step: JourneyStep) -> None:
        """Fill a form field.

        Args:
            step: Journey step with target selector and value
        """
        if not step.target or step.value is None:
            raise ValueError("Fill action requires target selector and value")

        logger.debug(f"Filling field: {step.target}")
        await self.page.fill(step.target, str(step.value), timeout=step.wait_timeout_ms)

    async def _action_select(self, step: JourneyStep) -> None:
        """Select an option from dropdown.

        Args:
            step: Journey step with target selector and value
        """
        if not step.target or step.value is None:
            raise ValueError("Select action requires target selector and value")

        logger.debug(f"Selecting option: {step.value} in {step.target}")
        await self.page.select_option(
            step.target, str(step.value), timeout=step.wait_timeout_ms
        )

    async def _action_check(self, step: JourneyStep) -> None:
        """Check or uncheck a checkbox.

        Args:
            step: Journey step with target selector and boolean value
        """
        if not step.target:
            raise ValueError("Check action requires target selector")

        checked = bool(step.value) if step.value is not None else True
        logger.debug(f"Setting checkbox: {step.target} to {checked}")

        if checked:
            await self.page.check(step.target, timeout=step.wait_timeout_ms)
        else:
            await self.page.uncheck(step.target, timeout=step.wait_timeout_ms)

    async def _action_wait(self, step: JourneyStep) -> None:
        """Wait for a specified duration.

        Args:
            step: Journey step with wait duration in value
        """
        wait_ms = int(step.value) if step.value else step.wait_timeout_ms
        logger.debug(f"Waiting for {wait_ms}ms")
        await self.page.wait_for_timeout(wait_ms)

    async def _action_screenshot(self, step: JourneyStep) -> None:
        """Take a screenshot.

        Args:
            step: Journey step with optional path in value
        """
        path = (
            str(step.value) if step.value else f"screenshot_step_{step.step_number}.png"
        )
        logger.debug(f"Taking screenshot: {path}")
        await self.page.screenshot(path=path)

    async def _action_evaluate(self, step: JourneyStep) -> None:
        """Execute JavaScript code.

        Args:
            step: Journey step with JavaScript code in value
        """
        if not step.value:
            raise ValueError("Evaluate action requires JavaScript code in value")

        logger.debug("Evaluating JavaScript code")
        await self.page.evaluate(str(step.value))

    async def _wait_for_condition(self, condition: str, timeout_ms: int) -> None:
        """Wait for a specific condition.

        Args:
            condition: Condition to wait for (selector, url, etc.)
            timeout_ms: Timeout in milliseconds
        """
        logger.debug(f"Waiting for condition: {condition}")

        # Try different wait strategies based on condition format
        if condition.startswith("url:"):
            # Wait for URL
            expected_url = condition[4:]
            await self.page.wait_for_url(expected_url, timeout=timeout_ms)

        elif condition.startswith("selector:"):
            # Wait for selector
            selector = condition[9:]
            await self.page.wait_for_selector(selector, timeout=timeout_ms)

        elif condition.startswith("function:"):
            # Wait for function to return true
            func = condition[9:]
            await self.page.wait_for_function(func, timeout=timeout_ms)

        else:
            # Default: treat as selector
            await self.page.wait_for_selector(condition, timeout=timeout_ms)

    async def _validate_step(self, validation: str) -> bool:
        """Validate a step's outcome.

        Args:
            validation: Validation expression

        Returns:
            True if validation passed, False otherwise
        """
        try:
            logger.debug(f"Validating: {validation}")

            # Support different validation formats
            if validation.startswith("visible:"):
                # Check element visibility
                selector = validation[8:]
                return await self.page.is_visible(selector)

            elif validation.startswith("text:"):
                # Check text content
                parts = validation[5:].split("|")
                selector = parts[0]
                expected_text = parts[1] if len(parts) > 1 else ""
                actual_text = await self.page.text_content(selector)
                return expected_text in (actual_text or "")

            elif validation.startswith("url:"):
                # Check current URL
                expected_url = validation[4:]
                current_url = self.page.url
                return expected_url in current_url

            else:
                # Evaluate as JavaScript boolean expression
                result = await self.page.evaluate(validation)
                return bool(result)

        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return False

    async def _validate_assertions(self, assertions: List[str]) -> List[Dict[str, Any]]:
        """Validate journey-level assertions.

        Args:
            assertions: List of assertion expressions

        Returns:
            List of assertion results
        """
        results = []

        for assertion in assertions:
            result = {
                "assertion": assertion,
                "passed": False,
                "error": None,
            }

            try:
                result["passed"] = await self._validate_step(assertion)
            except Exception as e:
                result["error"] = str(e)

            results.append(result)
            logger.debug(
                f"Assertion {'PASSED' if result['passed'] else 'FAILED'}: {assertion}"
            )

        return results

    async def _take_step_screenshot(self, step_number: int) -> None:
        """Take a screenshot for a specific step.

        Args:
            step_number: Step number for filename
        """
        try:
            filename = (
                f"step_{step_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            await self.page.screenshot(path=filename)
            logger.debug(f"Screenshot saved: {filename}")
        except Exception as e:
            logger.warning(f"Failed to take screenshot: {e}")

    async def execute_authentication_flow(
        self,
        login_url: str,
        username: str,
        password: str,
        username_selector: str = 'input[name="username"]',
        password_selector: str = 'input[name="password"]',
        submit_selector: str = 'button[type="submit"]',
        success_url_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a standard authentication flow.

        Args:
            login_url: Login page URL
            username: Username credential
            password: Password credential
            username_selector: Username field selector
            password_selector: Password field selector
            submit_selector: Submit button selector
            success_url_pattern: Expected URL pattern after successful login

        Returns:
            Dictionary with authentication result

        Example:
            strategy = UserFlowStrategy(browser)
            result = await strategy.execute_authentication_flow(
                login_url="https://example.com/login",
                username="user@example.com",
                password="secret123",
                success_url_pattern="/dashboard"
            )
        """
        logger.info(f"Executing authentication flow for: {login_url}")

        # Create journey for authentication
        auth_journey = UserJourney(
            id="auth-flow",
            name="Authentication Flow",
            description="Standard login workflow",
            start_url=login_url,
            steps=[
                JourneyStep(
                    step_number=1,
                    action="fill",
                    target=username_selector,
                    value=username,
                ),
                JourneyStep(
                    step_number=2,
                    action="fill",
                    target=password_selector,
                    value=password,
                ),
                JourneyStep(
                    step_number=3,
                    action="click",
                    target=submit_selector,
                    wait_for_network=True,
                ),
            ],
            assertions=([f"url:{success_url_pattern}"] if success_url_pattern else []),
        )

        return await self.execute_journey(auth_journey)

    async def execute_form_submission(
        self,
        form_url: str,
        form_data: Dict[str, Any],
        submit_selector: str = 'button[type="submit"]',
    ) -> Dict[str, Any]:
        """Execute a form submission flow.

        Args:
            form_url: Form page URL
            form_data: Dictionary mapping selectors to values
            submit_selector: Submit button selector

        Returns:
            Dictionary with form submission result

        Example:
            result = await strategy.execute_form_submission(
                form_url="https://example.com/contact",
                form_data={
                    'input[name="name"]': "John Doe",
                    'input[name="email"]': "john@example.com",
                    'textarea[name="message"]': "Hello world",
                },
            )
        """
        logger.info(f"Executing form submission for: {form_url}")

        # Build steps from form data
        steps = []
        step_num = 1

        for selector, value in form_data.items():
            steps.append(
                JourneyStep(
                    step_number=step_num,
                    action="fill",
                    target=selector,
                    value=value,
                )
            )
            step_num += 1

        # Add submit step
        steps.append(
            JourneyStep(
                step_number=step_num,
                action="click",
                target=submit_selector,
                wait_for_network=True,
            )
        )

        # Create journey
        form_journey = UserJourney(
            id="form-submission",
            name="Form Submission Flow",
            description="Form filling and submission",
            start_url=form_url,
            steps=steps,
        )

        return await self.execute_journey(form_journey)

    def get_journey_results(self) -> List[Dict[str, Any]]:
        """Get all journey execution results.

        Returns:
            List of journey results

        Example:
            results = strategy.get_journey_results()
            for result in results:
                print(f"{result['journey_name']}: {result['success']}")
        """
        return self._journey_results

    def clear_results(self) -> None:
        """Clear stored journey results.

        Example:
            strategy.clear_results()
        """
        self._journey_results.clear()
        logger.debug("Journey results cleared")
