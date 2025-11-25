"""Journey recorder for capturing and replaying user interactions.

This module provides the JourneyRecorder class for recording browser interactions,
generating smart selectors, and creating executable Playwright test code from
recorded user journeys.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from playwright.async_api import Page

from src.models.browser_models import UserJourney, JourneyStep, Viewport, BrowserType

logger = logging.getLogger(__name__)


class JourneyRecorder:
    """Records user interactions and generates Playwright test code.

    This class captures user interactions (clicks, fills, navigation, etc.),
    generates smart selectors for elements, and creates executable Playwright
    test code from recorded journeys.
    """

    def __init__(self):
        """Initialize the journey recorder."""
        self.recording = False
        self.journey: Optional[UserJourney] = None
        self.steps: List[JourneyStep] = []
        self.step_counter = 0
        self.start_time: Optional[datetime] = None
        self.page: Optional[Page] = None
        self._action_handlers: Dict[str, Any] = {}

        logger.info("JourneyRecorder initialized")

    async def start_recording(
        self, page: Page, journey_name: str, description: str = "", start_url: str = ""
    ) -> None:
        """Start recording a user journey.

        Args:
            page: Playwright page to record
            journey_name: Name for the journey
            description: Journey description
            start_url: Starting URL for the journey
        """
        if self.recording:
            logger.warning("Recording already in progress")
            return

        self.recording = True
        self.page = page
        self.steps = []
        self.step_counter = 0
        self.start_time = datetime.now()

        # Get current URL if not provided
        if not start_url:
            start_url = page.url

        # Get viewport info
        viewport_size = page.viewport_size
        viewport = Viewport(
            width=viewport_size.get("width", 1280),
            height=viewport_size.get("height", 720),
        )

        # Initialize journey
        self.journey = UserJourney(
            id=f"journey_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=journey_name,
            description=description,
            start_url=start_url,
            viewport=viewport,
            browser=BrowserType.CHROMIUM,
            recorded_at=self.start_time,
        )

        # Expose recording function to page
        await page.expose_function("recordAction", self._record_action)

        # Inject recording script into page
        await page.add_init_script(self._get_recording_script())

        logger.info(f"Started recording journey: {journey_name}")

    async def stop_recording(self) -> Optional[UserJourney]:
        """Stop recording and finalize the journey.

        Returns:
            The completed UserJourney or None if not recording
        """
        if not self.recording:
            logger.warning("No recording in progress")
            return None

        self.recording = False

        # Calculate duration
        if self.start_time and self.journey:
            duration = datetime.now() - self.start_time
            self.journey.duration_ms = int(duration.total_seconds() * 1000)
            self.journey.steps = self.steps

        logger.info(f"Stopped recording. Captured {len(self.steps)} steps")

        return self.journey

    async def _record_action(self, action_data: Dict[str, Any]) -> None:
        """Record an action from the browser.

        Args:
            action_data: Action data from browser including type, target, value, etc.
        """
        if not self.recording:
            return

        try:
            action_type = action_data.get("action", "unknown")

            # Generate selector for the target element
            selector = None
            if "selector" in action_data:
                selector = action_data["selector"]

            # Create journey step
            step = JourneyStep(
                step_number=self.step_counter,
                action=action_type,
                target=selector,
                value=action_data.get("value"),
                wait_for=action_data.get("waitFor"),
                wait_timeout_ms=action_data.get("waitTimeout", 5000),
                wait_for_network=action_data.get("waitForNetwork", False),
            )

            self.steps.append(step)
            self.step_counter += 1

            logger.debug(
                f"Recorded step {self.step_counter}: {action_type} on {selector}"
            )

        except Exception as e:
            logger.error(f"Error recording action: {e}", exc_info=True)

    def _get_recording_script(self) -> str:
        """Get the JavaScript recording script to inject.

        Returns:
            JavaScript code for recording browser interactions
        """
        return """
(() => {
    // Helper to generate smart selector for an element
    function generateSelector(element) {
        // Priority 1: data-testid
        if (element.hasAttribute('data-testid')) {
            return `[data-testid="${element.getAttribute('data-testid')}"]`;
        }

        // Priority 2: id
        if (element.id) {
            return `#${element.id}`;
        }

        // Priority 3: name attribute
        if (element.name) {
            return `[name="${element.name}"]`;
        }

        // Priority 4: aria-label
        if (element.hasAttribute('aria-label')) {
            return `[aria-label="${element.getAttribute('aria-label')}"]`;
        }

        // Priority 5: placeholder
        if (element.hasAttribute('placeholder')) {
            return `[placeholder="${element.getAttribute('placeholder')}"]`;
        }

        // Priority 6: role + accessible name
        if (element.hasAttribute('role')) {
            const role = element.getAttribute('role');
            const text = element.textContent?.trim().substring(0, 30);
            if (text) {
                return `[role="${role}"]:has-text("${text}")`;
            }
            return `[role="${role}"]`;
        }

        // Priority 7: class + tag
        if (element.className && typeof element.className === 'string') {
            const classes = element.className.split(' ').filter(c => c.trim());
            if (classes.length > 0) {
                return `${element.tagName.toLowerCase()}.${classes[0]}`;
            }
        }

        // Priority 8: text content for buttons and links
        if (['BUTTON', 'A'].includes(element.tagName)) {
            const text = element.textContent?.trim().substring(0, 30);
            if (text) {
                return `${element.tagName.toLowerCase()}:has-text("${text}")`;
            }
        }

        // Priority 9: CSS path
        let path = [];
        let current = element;
        while (current && current !== document.body) {
            let selector = current.tagName.toLowerCase();
            if (current.id) {
                path.unshift(`#${current.id}`);
                break;
            }

            // Add nth-child if needed
            if (current.parentElement) {
                const siblings = Array.from(current.parentElement.children);
                const index = siblings.indexOf(current);
                if (siblings.length > 1) {
                    selector += `:nth-child(${index + 1})`;
                }
            }

            path.unshift(selector);
            current = current.parentElement;
        }

        return path.join(' > ');
    }

    // Record click events
    window.addEventListener('click', (e) => {
        const target = e.target;
        const selector = generateSelector(target);

        window.recordAction({
            action: 'click',
            selector: selector,
            tagName: target.tagName,
            text: target.textContent?.trim().substring(0, 50),
            timestamp: Date.now()
        });
    }, true);

    // Record input events
    const inputHandler = (e) => {
        const target = e.target;
        if (!['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) {
            return;
        }

        const selector = generateSelector(target);

        // Debounce input events
        clearTimeout(target._recordTimeout);
        target._recordTimeout = setTimeout(() => {
            window.recordAction({
                action: 'fill',
                selector: selector,
                value: target.value,
                inputType: target.type,
                timestamp: Date.now()
            });
        }, 500);
    };

    window.addEventListener('input', inputHandler, true);
    window.addEventListener('change', inputHandler, true);

    // Record navigation
    let lastUrl = window.location.href;
    const checkNavigation = () => {
        const currentUrl = window.location.href;
        if (currentUrl !== lastUrl) {
            window.recordAction({
                action: 'navigate',
                value: currentUrl,
                timestamp: Date.now()
            });
            lastUrl = currentUrl;
        }
    };

    // Check for URL changes (for SPAs)
    setInterval(checkNavigation, 500);

    // Record select changes
    window.addEventListener('change', (e) => {
        const target = e.target;
        if (target.tagName === 'SELECT') {
            const selector = generateSelector(target);
            const selectedOption = target.options[target.selectedIndex];

            window.recordAction({
                action: 'select',
                selector: selector,
                value: target.value,
                label: selectedOption?.text,
                timestamp: Date.now()
            });
        }
    }, true);

    // Record keyboard events (for special keys)
    window.addEventListener('keydown', (e) => {
        // Only record special keys
        const specialKeys = ['Enter', 'Escape', 'Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
        if (specialKeys.includes(e.key)) {
            const target = e.target;
            const selector = generateSelector(target);

            window.recordAction({
                action: 'press',
                selector: selector,
                value: e.key,
                timestamp: Date.now()
            });
        }
    }, true);

    // Record file uploads
    window.addEventListener('change', (e) => {
        const target = e.target;
        if (target.tagName === 'INPUT' && target.type === 'file') {
            const selector = generateSelector(target);

            window.recordAction({
                action: 'upload',
                selector: selector,
                value: Array.from(target.files || []).map(f => f.name).join(', '),
                timestamp: Date.now()
            });
        }
    }, true);

    // Record hover events (throttled)
    let lastHoverTarget = null;
    window.addEventListener('mouseover', (e) => {
        const target = e.target;
        if (target === lastHoverTarget) return;

        lastHoverTarget = target;
        const selector = generateSelector(target);

        // Only record meaningful hover events (buttons, links, etc.)
        if (['BUTTON', 'A', 'INPUT'].includes(target.tagName) ||
            target.hasAttribute('role')) {
            window.recordAction({
                action: 'hover',
                selector: selector,
                timestamp: Date.now()
            });
        }
    }, true);

    console.log('Journey recording script initialized');
})();
        """

    async def generate_test(self, test_name: Optional[str] = None) -> str:
        """Generate Playwright test code from recorded journey.

        Args:
            test_name: Optional test name (defaults to journey name)

        Returns:
            Generated Playwright test code as a string
        """
        if not self.journey:
            logger.error("No journey to generate test from")
            return ""

        if not test_name:
            test_name = self.journey.name.replace(" ", "_").lower()

        # Generate test code
        test_code = self._generate_test_code(test_name)

        logger.info(f"Generated test code for journey: {self.journey.name}")

        return test_code

    def _generate_test_code(self, test_name: str) -> str:
        """Generate the actual test code.

        Args:
            test_name: Name for the test

        Returns:
            Generated test code
        """
        if not self.journey:
            return ""

        # Start building the test
        lines = [
            "import { test, expect } from '@playwright/test';",
            "",
            f"test('{test_name}', async ({{ page }}) => {{",
            f"  // {self.journey.description}" if self.journey.description else "",
            f"  await page.goto('{self.journey.start_url}');",
            "",
        ]

        # Generate code for each step
        for step in self.journey.steps:
            step_code = self._generate_step_code(step)
            if step_code:
                lines.append(f"  {step_code}")

        lines.append("});")

        return "\n".join(lines)

    def _generate_step_code(self, step: JourneyStep) -> str:
        """Generate code for a single step.

        Args:
            step: The journey step

        Returns:
            Generated code for the step
        """
        action = step.action
        target = step.target or ""
        value = step.value

        # Generate code based on action type
        if action == "click":
            return f"await page.locator('{target}').click();"

        elif action == "fill":
            # Escape quotes in value
            safe_value = str(value).replace("'", "\\'") if value else ""
            return f"await page.locator('{target}').fill('{safe_value}');"

        elif action == "navigate":
            return f"await page.goto('{value}');"

        elif action == "select":
            safe_value = str(value).replace("'", "\\'") if value else ""
            return f"await page.locator('{target}').selectOption('{safe_value}');"

        elif action == "press":
            return f"await page.locator('{target}').press('{value}');"

        elif action == "upload":
            return f"await page.locator('{target}').setInputFiles('{value}');"

        elif action == "hover":
            return f"await page.locator('{target}').hover();"

        elif action == "check":
            return f"await page.locator('{target}').check();"

        elif action == "uncheck":
            return f"await page.locator('{target}').uncheck();"

        elif action == "wait":
            if step.wait_for:
                return f"await page.waitForSelector('{step.wait_for}', {{ timeout: {step.wait_timeout_ms} }});"
            return f"await page.waitForTimeout({step.wait_timeout_ms});"

        else:
            logger.warning(f"Unknown action type: {action}")
            return f"// Unknown action: {action}"

    async def playback_journey(
        self, page: Page, journey: Optional[UserJourney] = None
    ) -> bool:
        """Playback a recorded journey.

        Args:
            page: Playwright page to playback on
            journey: Journey to playback (defaults to current journey)

        Returns:
            True if playback succeeded, False otherwise
        """
        if not journey:
            journey = self.journey

        if not journey:
            logger.error("No journey to playback")
            return False

        try:
            logger.info(f"Starting playback of journey: {journey.name}")

            # Navigate to start URL
            await page.goto(journey.start_url)

            # Execute each step
            for i, step in enumerate(journey.steps):
                try:
                    await self._execute_step(page, step)

                    # Add delay between steps if configured
                    if journey.step_delay_ms > 0:
                        await page.wait_for_timeout(journey.step_delay_ms)

                    logger.debug(f"Executed step {i + 1}/{len(journey.steps)}")

                except Exception as e:
                    logger.error(f"Error executing step {i + 1}: {e}", exc_info=True)
                    return False

            logger.info(
                f"Successfully completed playback of {len(journey.steps)} steps"
            )
            return True

        except Exception as e:
            logger.error(f"Error during journey playback: {e}", exc_info=True)
            return False

    async def _execute_step(self, page: Page, step: JourneyStep) -> None:
        """Execute a single journey step.

        Args:
            page: Playwright page
            step: Step to execute
        """
        action = step.action
        target = step.target
        value = step.value

        # Wait for network idle if needed
        if step.wait_for_network:
            await page.wait_for_load_state("networkidle")

        # Execute action
        if action == "click" and target:
            await page.locator(target).click(timeout=step.wait_timeout_ms)

        elif action == "fill" and target and value is not None:
            await page.locator(target).fill(str(value), timeout=step.wait_timeout_ms)

        elif action == "navigate" and value:
            await page.goto(str(value))

        elif action == "select" and target and value is not None:
            await page.locator(target).select_option(
                str(value), timeout=step.wait_timeout_ms
            )

        elif action == "press" and target and value:
            await page.locator(target).press(str(value), timeout=step.wait_timeout_ms)

        elif action == "upload" and target and value:
            await page.locator(target).set_input_files(
                str(value), timeout=step.wait_timeout_ms
            )

        elif action == "hover" and target:
            await page.locator(target).hover(timeout=step.wait_timeout_ms)

        elif action == "check" and target:
            await page.locator(target).check(timeout=step.wait_timeout_ms)

        elif action == "uncheck" and target:
            await page.locator(target).uncheck(timeout=step.wait_timeout_ms)

        elif action == "wait":
            if step.wait_for:
                await page.wait_for_selector(
                    step.wait_for, timeout=step.wait_timeout_ms
                )
            else:
                await page.wait_for_timeout(step.wait_timeout_ms)

        # Take screenshot if needed
        if step.screenshot:
            screenshot_path = f"screenshots/step_{step.step_number}.png"
            await page.screenshot(path=screenshot_path)
            logger.debug(f"Screenshot saved: {screenshot_path}")

        # Execute validation if present
        if step.validate:
            # Validation could be a selector check or custom assertion
            await page.wait_for_selector(step.validate, timeout=step.wait_timeout_ms)

    def export_journey(self, file_path: str) -> None:
        """Export journey to JSON file.

        Args:
            file_path: Path to save the journey JSON
        """
        if not self.journey:
            logger.error("No journey to export")
            return

        try:
            with open(file_path, "w") as f:
                json.dump(self.journey.model_dump(), f, indent=2, default=str)

            logger.info(f"Journey exported to: {file_path}")

        except Exception as e:
            logger.error(f"Error exporting journey: {e}", exc_info=True)

    def import_journey(self, file_path: str) -> Optional[UserJourney]:
        """Import journey from JSON file.

        Args:
            file_path: Path to the journey JSON file

        Returns:
            Imported UserJourney or None if import failed
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            journey = UserJourney(**data)
            self.journey = journey
            self.steps = journey.steps

            logger.info(f"Journey imported from: {file_path}")
            return journey

        except Exception as e:
            logger.error(f"Error importing journey: {e}", exc_info=True)
            return None

    def get_journey_summary(self) -> Dict[str, Any]:
        """Get a summary of the current journey.

        Returns:
            Dictionary with journey summary information
        """
        if not self.journey:
            return {"status": "no_journey", "message": "No journey recorded"}

        action_counts = {}
        for step in self.journey.steps:
            action = step.action
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "status": "active" if self.recording else "completed",
            "name": self.journey.name,
            "description": self.journey.description,
            "start_url": self.journey.start_url,
            "total_steps": len(self.journey.steps),
            "action_breakdown": action_counts,
            "duration_ms": self.journey.duration_ms,
            "recorded_at": self.journey.recorded_at.isoformat()
            if self.journey.recorded_at
            else None,
        }
