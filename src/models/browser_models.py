"""Browser automation data models for Playwright integration and E2E testing.

This module defines all the Pydantic models needed for browser automation,
including browser configurations, page object models, user journeys,
visual testing, accessibility, and performance metrics.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from enum import Enum
from datetime import datetime


class BrowserType(str, Enum):
    """Supported browser engines."""

    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"
    CHROME = "chrome"
    EDGE = "edge"
    SAFARI = "safari"


class DeviceType(str, Enum):
    """Device emulation types."""

    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"


class TestType(str, Enum):
    """Browser test types."""

    E2E = "e2e"
    VISUAL = "visual"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    SMOKE = "smoke"
    REGRESSION = "regression"


class Viewport(BaseModel):
    """Browser viewport configuration."""

    width: int = Field(default=1280, description="Viewport width")
    height: int = Field(default=720, description="Viewport height")
    device_scale_factor: float = Field(default=1.0, description="Device pixel ratio")
    is_mobile: bool = Field(default=False, description="Mobile viewport")
    has_touch: bool = Field(default=False, description="Touch support")


class PageElement(BaseModel):
    """Represents a page element for POM."""

    name: str = Field(description="Element name")
    selector: str = Field(description="CSS/XPath selector")
    element_type: str = Field(description="Element type (button, input, etc.)")
    attributes: Dict[str, str] = Field(
        default_factory=dict, description="Element attributes"
    )
    text_content: Optional[str] = Field(default=None, description="Element text")
    is_visible: bool = Field(default=True, description="Visibility state")
    is_enabled: bool = Field(default=True, description="Enabled state")
    aria_label: Optional[str] = Field(default=None, description="Accessibility label")
    data_testid: Optional[str] = Field(default=None, description="Test ID attribute")


class PageObjectModel(BaseModel):
    """Page Object Model representation."""

    id: str = Field(description="POM identifier")
    url: str = Field(description="Page URL")
    name: str = Field(description="Page name")
    elements: Dict[str, PageElement] = Field(
        default_factory=dict, description="Page elements"
    )

    # Actions
    actions: List[str] = Field(default_factory=list, description="Available actions")

    # Locators
    locator_strategy: str = Field(default="css", description="Primary locator strategy")
    custom_locators: Dict[str, str] = Field(
        default_factory=dict, description="Custom locators"
    )

    # Component structure (for React/Vue/Angular)
    components: List[Dict[str, Any]] = Field(
        default_factory=list, description="Component tree"
    )

    # Metadata
    framework: Optional[str] = Field(default=None, description="Frontend framework")
    generated_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)


class JourneyStep(BaseModel):
    """Individual step in a user journey."""

    step_number: int = Field(description="Step sequence number")
    action: str = Field(description="Action type (click, fill, navigate, etc.)")
    target: Optional[str] = Field(default=None, description="Target selector")
    value: Optional[Any] = Field(default=None, description="Input value if applicable")

    # Wait conditions
    wait_for: Optional[str] = Field(default=None, description="Wait condition")
    wait_timeout_ms: int = Field(default=5000, description="Wait timeout")

    # Validation
    validation_expr: Optional[str] = Field(default=None, description="Validation expression")
    screenshot: bool = Field(default=False, description="Take screenshot after step")

    # Network
    wait_for_network: bool = Field(default=False, description="Wait for network idle")
    expected_requests: List[str] = Field(
        default_factory=list, description="Expected API calls"
    )


class UserJourney(BaseModel):
    """User journey test specification."""

    id: str = Field(description="Journey identifier")
    name: str = Field(description="Journey name")
    description: str = Field(description="Journey description")

    # Steps
    steps: List[JourneyStep] = Field(default_factory=list, description="Journey steps")

    # Assertions
    assertions: List[str] = Field(
        default_factory=list, description="Journey assertions"
    )
    expected_outcomes: Dict[str, Any] = Field(
        default_factory=dict, description="Expected results"
    )

    # Configuration
    start_url: str = Field(description="Starting URL")
    viewport: Viewport = Field(
        default_factory=Viewport, description="Viewport settings"
    )
    browser: BrowserType = Field(
        default=BrowserType.CHROMIUM, description="Browser to use"
    )

    # Timing
    timeout_ms: int = Field(default=30000, description="Journey timeout")
    step_delay_ms: int = Field(default=0, description="Delay between steps")

    # Recording metadata
    recorded_at: Optional[datetime] = Field(
        default=None, description="Recording timestamp"
    )
    duration_ms: Optional[int] = Field(default=None, description="Journey duration")


class VisualTestResult(BaseModel):
    """Visual regression test result."""

    test_id: str = Field(description="Test identifier")
    page_url: str = Field(description="Page URL tested")

    # Screenshots
    baseline_path: str = Field(description="Baseline screenshot path")
    actual_path: str = Field(description="Actual screenshot path")
    diff_path: Optional[str] = Field(default=None, description="Diff image path")

    # Comparison results
    match_percentage: float = Field(description="Similarity percentage")
    pixel_difference: int = Field(description="Number of different pixels")
    is_match: bool = Field(description="Whether images match within threshold")

    # Diff regions
    diff_regions: List[Dict[str, int]] = Field(
        default_factory=list,
        description="Regions with differences (x, y, width, height)",
    )

    # Configuration
    threshold: float = Field(default=0.01, description="Difference threshold")
    ignore_regions: List[Dict[str, int]] = Field(
        default_factory=list, description="Ignored regions"
    )

    # Metadata
    browser: BrowserType = Field(description="Browser used")
    viewport: Viewport = Field(description="Viewport settings")
    timestamp: datetime = Field(default_factory=datetime.now)


class AccessibilityIssue(BaseModel):
    """Accessibility violation details."""

    id: str = Field(description="Issue identifier")
    impact: Literal["minor", "moderate", "serious", "critical"] = Field(
        description="Impact level"
    )
    rule_id: str = Field(description="WCAG rule ID")
    description: str = Field(description="Issue description")
    help_text: str = Field(description="How to fix")

    # Location
    selector: str = Field(description="Element selector")
    html: str = Field(description="Element HTML")

    # WCAG info
    wcag_criteria: List[str] = Field(default_factory=list, description="WCAG criteria")
    wcag_level: Literal["A", "AA", "AAA"] = Field(description="WCAG level")

    # Fix suggestion
    fix_suggestion: Optional[str] = Field(default=None, description="Suggested fix")


class PerformanceMetrics(BaseModel):
    """Performance metrics from browser."""

    url: str = Field(description="Page URL")

    # Core Web Vitals
    lcp: float = Field(description="Largest Contentful Paint (ms)")
    fid: float = Field(description="First Input Delay (ms)")
    cls: float = Field(description="Cumulative Layout Shift")

    # Additional metrics
    ttfb: float = Field(description="Time to First Byte (ms)")
    fcp: float = Field(description="First Contentful Paint (ms)")
    tti: float = Field(description="Time to Interactive (ms)")
    speed_index: float = Field(description="Speed Index")

    # Resource timing
    dom_content_loaded: float = Field(description="DOM Content Loaded (ms)")
    load_complete: float = Field(description="Load Complete (ms)")

    # Network
    total_requests: int = Field(description="Total network requests")
    total_size_kb: float = Field(description="Total transfer size (KB)")

    # Memory
    js_heap_size_mb: float = Field(description="JS heap size (MB)")

    # Thresholds
    passes_cwv: bool = Field(description="Passes Core Web Vitals")

    # Metadata
    browser: BrowserType = Field(description="Browser used")
    timestamp: datetime = Field(default_factory=datetime.now)


class NetworkMock(BaseModel):
    """Network request mock configuration."""

    url_pattern: str = Field(description="URL pattern to match")
    method: str = Field(default="GET", description="HTTP method")

    # Response
    status: int = Field(default=200, description="Response status code")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Response headers"
    )
    body: Optional[Any] = Field(default=None, description="Response body")

    # Behavior
    delay_ms: int = Field(default=0, description="Response delay")
    abort: bool = Field(default=False, description="Abort request")

    # Conditions
    times: Optional[int] = Field(default=None, description="Number of times to mock")
    predicate: Optional[str] = Field(default=None, description="Condition function")


class BrowserTestSuite(BaseModel):
    """Browser test suite specification."""

    id: str = Field(description="Suite identifier")
    name: str = Field(description="Suite name")

    # Tests
    journeys: List[UserJourney] = Field(
        default_factory=list, description="User journeys"
    )
    visual_tests: List[Dict[str, Any]] = Field(
        default_factory=list, description="Visual tests"
    )
    accessibility_tests: List[Dict[str, Any]] = Field(
        default_factory=list, description="A11y tests"
    )
    performance_tests: List[Dict[str, Any]] = Field(
        default_factory=list, description="Perf tests"
    )

    # Configuration
    browsers: List[BrowserType] = Field(
        default_factory=lambda: [BrowserType.CHROMIUM], description="Browsers to test"
    )
    viewports: List[Viewport] = Field(
        default_factory=list, description="Viewports to test"
    )

    # Page Objects
    page_objects: Dict[str, PageObjectModel] = Field(
        default_factory=dict, description="Page object models"
    )

    # Network
    network_mocks: List[NetworkMock] = Field(
        default_factory=list, description="Network mocks"
    )

    # Parallel execution
    parallel: bool = Field(default=True, description="Run tests in parallel")
    workers: int = Field(default=4, description="Number of parallel workers")

    # Reporting
    generate_html_report: bool = Field(default=True, description="Generate HTML report")
    screenshot_on_failure: bool = Field(
        default=True, description="Screenshot on failure"
    )
    video_on_failure: bool = Field(default=False, description="Record video on failure")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_run: Optional[datetime] = Field(default=None)
