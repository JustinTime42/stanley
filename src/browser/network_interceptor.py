"""Network request interception and mocking for browser automation.

This module provides the NetworkInterceptor class for intercepting and mocking
network requests using Playwright's route API. Supports request abortion,
response mocking with delays, and request tracking.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime

from playwright.async_api import Route, Request, BrowserContext, Page

from src.models.browser_models import NetworkMock

logger = logging.getLogger(__name__)


class NetworkInterceptor:
    """Intercept and mock network requests in browser automation.

    This class provides functionality to:
    - Intercept network requests using Playwright route API
    - Mock API responses with configurable delays
    - Abort requests based on patterns
    - Track intercepted requests for debugging
    - Support multiple mocks simultaneously

    Example:
        interceptor = NetworkInterceptor()
        mock = NetworkMock(
            url_pattern="**/api/users",
            method="GET",
            status=200,
            body={"users": []},
            delay_ms=100
        )
        await interceptor.setup_interception(page, [mock])
    """

    def __init__(self):
        """Initialize the network interceptor."""
        self.mocks: List[NetworkMock] = []
        self.intercepted_requests: List[Dict[str, Any]] = []
        self._route_handlers: Dict[str, Callable] = {}
        self._active_patterns: List[str] = []
        self._mock_call_counts: Dict[str, int] = {}

    async def setup_interception(
        self,
        context_or_page: Union[BrowserContext, Page],
        mocks: List[NetworkMock],
    ) -> None:
        """Setup route handlers for network interception.

        Args:
            context_or_page: Browser context or page to intercept requests on
            mocks: List of network mocks to apply

        Example:
            mocks = [
                NetworkMock(url_pattern="**/api/*", method="GET", status=200),
                NetworkMock(url_pattern="**/slow-api", delay_ms=2000)
            ]
            await interceptor.setup_interception(page, mocks)
        """
        self.mocks = mocks
        logger.info(f"Setting up network interception with {len(mocks)} mocks")

        # Setup route handlers for each mock
        for mock in mocks:
            mock_id = f"{mock.url_pattern}:{mock.method}"
            self._mock_call_counts[mock_id] = 0

            # Create and register handler
            handler = self._create_handler(mock)
            self._route_handlers[mock_id] = handler
            self._active_patterns.append(mock.url_pattern)

            # Register route with Playwright
            await context_or_page.route(mock.url_pattern, handler)
            logger.debug(
                f"Registered route handler for {mock.url_pattern} ({mock.method})"
            )

        logger.info(
            f"Network interception setup complete. "
            f"Intercepting {len(self._active_patterns)} patterns"
        )

    def _create_handler(self, mock: NetworkMock) -> Callable:
        """Create a route handler function for the given mock.

        Args:
            mock: Network mock configuration

        Returns:
            Async handler function for Playwright route

        The handler:
        - Checks HTTP method matches
        - Evaluates predicate if exists
        - Tracks request
        - Aborts if configured
        - Delays response if configured
        - Fulfills with mock response
        """

        async def handler(route: Route, request: Request) -> None:
            """Handle intercepted network request.

            Args:
                route: Playwright route object
                request: Playwright request object
            """
            mock_id = f"{mock.url_pattern}:{mock.method}"

            # Track the request
            request_data = {
                "url": request.url,
                "method": request.method,
                "headers": dict(request.headers),
                "timestamp": datetime.now().isoformat(),
                "mock_id": mock_id,
            }

            # Add post data if available
            try:
                if request.post_data:
                    request_data["post_data"] = request.post_data
            except Exception:
                # post_data may not be available for all requests
                pass

            self.intercepted_requests.append(request_data)

            # Check HTTP method
            if request.method.upper() != mock.method.upper():
                logger.debug(
                    f"Method mismatch for {request.url}: "
                    f"expected {mock.method}, got {request.method}. "
                    f"Continuing without mock."
                )
                await route.continue_()
                return

            # Check predicate if exists
            if mock.predicate:
                try:
                    # The predicate is a string expression that can be evaluated
                    # For safety, we only support basic checks
                    # In production, this could use a safer evaluation method
                    if not self._evaluate_predicate(mock.predicate, request):
                        logger.debug(
                            f"Predicate failed for {request.url}. "
                            f"Continuing without mock."
                        )
                        await route.continue_()
                        return
                except Exception as e:
                    logger.warning(
                        f"Error evaluating predicate for {request.url}: {e}. "
                        f"Continuing without mock."
                    )
                    await route.continue_()
                    return

            # Check if we've exceeded the times limit
            if mock.times is not None:
                call_count = self._mock_call_counts.get(mock_id, 0)
                if call_count >= mock.times:
                    logger.debug(
                        f"Mock {mock_id} has reached its limit ({mock.times} times). "
                        f"Continuing without mock."
                    )
                    await route.continue_()
                    return

            # Increment call count
            self._mock_call_counts[mock_id] = self._mock_call_counts.get(mock_id, 0) + 1

            logger.debug(
                f"Intercepted {request.method} {request.url} "
                f"(mock: {mock_id}, call {self._mock_call_counts[mock_id]})"
            )

            # Handle request abortion
            if mock.abort:
                logger.debug(f"Aborting request to {request.url}")
                await route.abort()
                return

            # Apply delay if configured
            if mock.delay_ms > 0:
                logger.debug(
                    f"Delaying response for {request.url} by {mock.delay_ms}ms"
                )
                await asyncio.sleep(mock.delay_ms / 1000.0)

            # Prepare response body
            body = self._prepare_response_body(mock.body)

            # Prepare headers with defaults
            headers = {
                "Content-Type": "application/json",
                **mock.headers,
            }

            # Fulfill the request with mock response
            try:
                await route.fulfill(
                    status=mock.status,
                    headers=headers,
                    body=body,
                )
                logger.debug(f"Fulfilled {request.url} with status {mock.status}")
            except Exception as e:
                logger.error(
                    f"Error fulfilling request {request.url}: {e}. "
                    f"Falling back to continue."
                )
                await route.continue_()

        return handler

    def _evaluate_predicate(self, predicate: str, request: Request) -> bool:
        """Evaluate a predicate expression against a request.

        Args:
            predicate: Predicate expression string
            request: Playwright request object

        Returns:
            True if predicate passes, False otherwise

        Note:
            This is a simplified implementation. In production,
            consider using a safer evaluation method or a DSL.
        """
        # Create a safe context with request properties
        context = {
            "url": request.url,
            "method": request.method,
            "headers": dict(request.headers),
        }

        try:
            # Simple contains/equals checks
            if "contains" in predicate.lower():
                # e.g., "url contains 'api'"
                parts = predicate.split("contains")
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip().strip("'\"")
                    return value in str(context.get(field, ""))

            if "==" in predicate:
                # e.g., "method == 'POST'"
                parts = predicate.split("==")
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip().strip("'\"")
                    return str(context.get(field, "")) == value

            # Default to True if predicate format is not recognized
            logger.warning(
                f"Unrecognized predicate format: {predicate}. Defaulting to True."
            )
            return True
        except Exception as e:
            logger.error(f"Error evaluating predicate '{predicate}': {e}")
            return False

    def _prepare_response_body(self, body: Any) -> str:
        """Prepare response body for fulfillment.

        Args:
            body: Response body (dict, list, str, or None)

        Returns:
            String representation of the body

        If body is a dict or list, it will be JSON-encoded.
        If body is None, returns empty string.
        """
        if body is None:
            return ""

        if isinstance(body, (dict, list)):
            try:
                return json.dumps(body)
            except Exception as e:
                logger.error(f"Error JSON encoding body: {e}")
                return str(body)

        return str(body)

    async def add_mock(
        self,
        context_or_page: Union[BrowserContext, Page],
        mock: NetworkMock,
    ) -> None:
        """Add a single mock to the active interception.

        Args:
            context_or_page: Browser context or page
            mock: Network mock to add

        Example:
            mock = NetworkMock(url_pattern="**/api/new", status=201)
            await interceptor.add_mock(page, mock)
        """
        mock_id = f"{mock.url_pattern}:{mock.method}"
        self._mock_call_counts[mock_id] = 0

        # Create and register handler
        handler = self._create_handler(mock)
        self._route_handlers[mock_id] = handler
        self._active_patterns.append(mock.url_pattern)

        # Register route with Playwright
        await context_or_page.route(mock.url_pattern, handler)

        self.mocks.append(mock)
        logger.info(f"Added mock for {mock.url_pattern} ({mock.method})")

    async def clear_mocks(
        self,
        context_or_page: Union[BrowserContext, Page],
    ) -> None:
        """Clear all active mocks and unroute patterns.

        Args:
            context_or_page: Browser context or page to clear routes from

        Example:
            await interceptor.clear_mocks(page)
        """
        logger.info(f"Clearing {len(self._active_patterns)} network mocks")

        # Unroute all patterns
        for pattern in self._active_patterns:
            try:
                await context_or_page.unroute(pattern)
                logger.debug(f"Unrouted pattern: {pattern}")
            except Exception as e:
                logger.warning(f"Error unrouting pattern {pattern}: {e}")

        # Clear all tracking
        self.mocks.clear()
        self._route_handlers.clear()
        self._active_patterns.clear()
        self._mock_call_counts.clear()

        logger.info("Network mocks cleared")

    def get_intercepted_requests(
        self,
        url_pattern: Optional[str] = None,
        method: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get intercepted requests with optional filtering.

        Args:
            url_pattern: Optional URL pattern to filter by (glob)
            method: Optional HTTP method to filter by

        Returns:
            List of intercepted request data

        Example:
            # Get all requests
            all_requests = interceptor.get_intercepted_requests()

            # Get POST requests only
            posts = interceptor.get_intercepted_requests(method="POST")
        """
        requests = self.intercepted_requests

        if method:
            requests = [r for r in requests if r["method"].upper() == method.upper()]

        if url_pattern:
            # Simple pattern matching (can be enhanced)
            import fnmatch

            requests = [r for r in requests if fnmatch.fnmatch(r["url"], url_pattern)]

        return requests

    def clear_intercepted_requests(self) -> None:
        """Clear the intercepted requests history.

        Example:
            interceptor.clear_intercepted_requests()
        """
        count = len(self.intercepted_requests)
        self.intercepted_requests.clear()
        logger.debug(f"Cleared {count} intercepted requests")

    def get_mock_stats(self) -> Dict[str, int]:
        """Get statistics about mock usage.

        Returns:
            Dictionary mapping mock_id to call count

        Example:
            stats = interceptor.get_mock_stats()
            # {"**/api/users:GET": 5, "**/api/posts:POST": 2}
        """
        return dict(self._mock_call_counts)

    async def wait_for_request(
        self,
        context_or_page: Union[BrowserContext, Page],
        url_pattern: str,
        timeout: float = 30000,
    ) -> Optional[Dict[str, Any]]:
        """Wait for a specific request to be intercepted.

        Args:
            context_or_page: Browser context or page
            url_pattern: URL pattern to wait for (glob)
            timeout: Timeout in milliseconds

        Returns:
            Request data if found, None if timeout

        Example:
            request = await interceptor.wait_for_request(
                page,
                "**/api/users",
                timeout=5000
            )
        """
        import fnmatch

        start_time = asyncio.get_event_loop().time()
        timeout_seconds = timeout / 1000.0

        logger.debug(f"Waiting for request matching {url_pattern}")

        while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            # Check if matching request exists
            for request in self.intercepted_requests:
                if fnmatch.fnmatch(request["url"], url_pattern):
                    logger.debug(f"Found matching request: {request['url']}")
                    return request

            # Wait a bit before checking again
            await asyncio.sleep(0.1)

        logger.warning(
            f"Timeout waiting for request matching {url_pattern} after {timeout}ms"
        )
        return None
