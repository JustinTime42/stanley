"""Tests for Accessibility Testing.

This module contains tests for the AccessibilityTester class which performs
WCAG compliance audits using axe-core.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from playwright.async_api import Page

from src.browser.accessibility_tester import AccessibilityTester
from src.models.browser_models import AccessibilityIssue


@pytest.fixture
def tester():
    """Create an AccessibilityTester instance."""
    return AccessibilityTester()


@pytest.fixture
def mock_page():
    """Create a mock Page instance."""
    page = AsyncMock(spec=Page)
    page.add_script_tag = AsyncMock()
    page.evaluate = AsyncMock()
    return page


@pytest.fixture
def sample_axe_violation():
    """Create a sample axe-core violation."""
    return {
        "id": "color-contrast",
        "impact": "serious",
        "description": "Elements must have sufficient color contrast",
        "help": "Ensure text has sufficient contrast",
        "helpUrl": "https://dequeuniversity.com/rules/axe/4.7/color-contrast",
        "tags": ["wcag2aa", "wcag143"],
        "nodes": [
            {
                "target": ["#main-heading"],
                "html": '<h1 id="main-heading">Welcome</h1>',
                "failureSummary": "Fix the following: Element has insufficient color contrast",
                "any": [
                    {
                        "id": "color-contrast",
                        "message": "Element has insufficient color contrast of 2.5:1",
                    }
                ],
                "all": [],
                "none": [],
            }
        ],
    }


@pytest.fixture
def sample_axe_results(sample_axe_violation):
    """Create sample axe-core results."""
    return {
        "violations": [sample_axe_violation],
        "passes": [],
        "incomplete": [],
        "inapplicable": [],
    }


class TestAccessibilityTesterInitialization:
    """Tests for AccessibilityTester initialization."""

    def test_init(self):
        """Test AccessibilityTester initialization."""
        tester = AccessibilityTester()
        assert tester._axe_injected == {}

    def test_axe_core_cdn_url(self):
        """Test that AXE_CORE_CDN is defined."""
        assert hasattr(AccessibilityTester, "AXE_CORE_CDN")
        assert "axe" in AccessibilityTester.AXE_CORE_CDN

    def test_wcag_tag_mapping(self):
        """Test WCAG level to tag mapping."""
        assert "A" in AccessibilityTester.WCAG_TAG_MAPPING
        assert "AA" in AccessibilityTester.WCAG_TAG_MAPPING
        assert "AAA" in AccessibilityTester.WCAG_TAG_MAPPING
        assert "wcag2aa" in AccessibilityTester.WCAG_TAG_MAPPING["AA"]


class TestAxeInjection:
    """Tests for axe-core library injection."""

    @pytest.mark.asyncio
    async def test_inject_axe_success(self, tester, mock_page):
        """Test successful axe-core injection."""
        mock_page.evaluate = AsyncMock(return_value=True)

        await tester.inject_axe(mock_page)

        mock_page.add_script_tag.assert_called_once_with(url=tester.AXE_CORE_CDN)
        mock_page.evaluate.assert_called_once()
        assert str(id(mock_page)) in tester._axe_injected
        assert tester._axe_injected[str(id(mock_page))] is True

    @pytest.mark.asyncio
    async def test_inject_axe_already_injected(self, tester, mock_page):
        """Test that axe-core is not injected twice."""
        page_id = str(id(mock_page))
        tester._axe_injected[page_id] = True

        await tester.inject_axe(mock_page)

        # Should not call add_script_tag if already injected
        mock_page.add_script_tag.assert_not_called()

    @pytest.mark.asyncio
    async def test_inject_axe_load_failure(self, tester, mock_page):
        """Test axe-core injection when library fails to load."""
        mock_page.evaluate = AsyncMock(return_value=False)

        with pytest.raises(Exception, match="axe-core library failed to load"):
            await tester.inject_axe(mock_page)

    @pytest.mark.asyncio
    async def test_inject_axe_script_tag_failure(self, tester, mock_page):
        """Test axe-core injection when script tag fails."""
        mock_page.add_script_tag = AsyncMock(side_effect=Exception("Script load failed"))

        with pytest.raises(Exception, match="Failed to inject axe-core"):
            await tester.inject_axe(mock_page)


class TestAccessibilityAudit:
    """Tests for running accessibility audits."""

    @pytest.mark.asyncio
    async def test_run_audit_success(self, tester, mock_page, sample_axe_results):
        """Test successful accessibility audit."""
        mock_page.evaluate = AsyncMock(side_effect=[True, sample_axe_results])

        issues = await tester.run_audit(mock_page, wcag_level="AA")

        assert len(issues) == 1
        assert isinstance(issues[0], AccessibilityIssue)
        assert issues[0].rule_id == "color-contrast"
        assert issues[0].impact == "serious"

    @pytest.mark.asyncio
    async def test_run_audit_wcag_a(self, tester, mock_page):
        """Test audit with WCAG Level A."""
        mock_page.evaluate = AsyncMock(side_effect=[True, {"violations": []}])

        await tester.run_audit(mock_page, wcag_level="A")

        # Verify the correct tags were passed
        call_args = mock_page.evaluate.call_args_list[1]
        tags = call_args[0][1]
        assert "wcag2a" in tags

    @pytest.mark.asyncio
    async def test_run_audit_wcag_aaa(self, tester, mock_page):
        """Test audit with WCAG Level AAA."""
        mock_page.evaluate = AsyncMock(side_effect=[True, {"violations": []}])

        await tester.run_audit(mock_page, wcag_level="AAA")

        # Verify AAA tags were passed
        call_args = mock_page.evaluate.call_args_list[1]
        tags = call_args[0][1]
        assert "wcag2a" in tags
        assert "wcag2aa" in tags
        assert "wcag2aaa" in tags

    @pytest.mark.asyncio
    async def test_run_audit_with_best_practices(self, tester, mock_page):
        """Test audit including best practices."""
        mock_page.evaluate = AsyncMock(side_effect=[True, {"violations": []}])

        await tester.run_audit(mock_page, wcag_level="AA", include_best_practices=True)

        call_args = mock_page.evaluate.call_args_list[1]
        tags = call_args[0][1]
        assert "best-practice" in tags

    @pytest.mark.asyncio
    async def test_run_audit_without_best_practices(self, tester, mock_page):
        """Test audit excluding best practices."""
        mock_page.evaluate = AsyncMock(side_effect=[True, {"violations": []}])

        await tester.run_audit(mock_page, wcag_level="AA", include_best_practices=False)

        call_args = mock_page.evaluate.call_args_list[1]
        tags = call_args[0][1]
        assert "best-practice" not in tags

    @pytest.mark.asyncio
    async def test_run_audit_no_violations(self, tester, mock_page):
        """Test audit with no violations."""
        mock_page.evaluate = AsyncMock(side_effect=[True, {"violations": []}])

        issues = await tester.run_audit(mock_page)

        assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_run_audit_multiple_violations(self, tester, mock_page):
        """Test audit with multiple violations."""
        violations = [
            {
                "id": "color-contrast",
                "impact": "serious",
                "description": "Color contrast issue",
                "help": "Fix contrast",
                "helpUrl": "https://example.com",
                "tags": ["wcag2aa"],
                "nodes": [{"target": ["#elem1"], "html": "<div>", "any": [], "all": [], "none": []}],
            },
            {
                "id": "image-alt",
                "impact": "critical",
                "description": "Images must have alt text",
                "help": "Add alt text",
                "helpUrl": "https://example.com",
                "tags": ["wcag2a"],
                "nodes": [{"target": ["#img1"], "html": "<img>", "any": [], "all": [], "none": []}],
            },
        ]
        mock_page.evaluate = AsyncMock(side_effect=[True, {"violations": violations}])

        issues = await tester.run_audit(mock_page)

        assert len(issues) == 2
        assert issues[0].rule_id == "color-contrast"
        assert issues[1].rule_id == "image-alt"

    @pytest.mark.asyncio
    async def test_run_audit_failure(self, tester, mock_page):
        """Test audit failure handling."""
        mock_page.evaluate = AsyncMock(side_effect=Exception("Audit failed"))

        with pytest.raises(Exception, match="Accessibility audit failed"):
            await tester.run_audit(mock_page)


class TestViolationParsing:
    """Tests for parsing axe-core violations."""

    def test_parse_violations_basic(self, tester, sample_axe_violation):
        """Test parsing a basic violation."""
        issues = tester._parse_violations([sample_axe_violation], "AA")

        assert len(issues) == 1
        issue = issues[0]
        assert issue.rule_id == "color-contrast"
        assert issue.impact == "serious"
        assert issue.description == "Elements must have sufficient color contrast"
        assert issue.selector == "#main-heading"

    def test_parse_violations_wcag_level_detection(self, tester):
        """Test WCAG level detection from tags."""
        # AAA violation
        violation_aaa = {
            "id": "test-rule",
            "impact": "minor",
            "description": "Test",
            "help": "Help",
            "helpUrl": "http://example.com",
            "tags": ["wcag2aaa"],
            "nodes": [{"target": ["#test"], "html": "<div>", "any": [], "all": [], "none": []}],
        }

        issues = tester._parse_violations([violation_aaa], "AAA")
        assert issues[0].wcag_level == "AAA"

        # AA violation
        violation_aa = {
            "id": "test-rule",
            "impact": "minor",
            "description": "Test",
            "help": "Help",
            "helpUrl": "http://example.com",
            "tags": ["wcag2aa"],
            "nodes": [{"target": ["#test"], "html": "<div>", "any": [], "all": [], "none": []}],
        }

        issues = tester._parse_violations([violation_aa], "AA")
        assert issues[0].wcag_level == "AA"

        # A violation
        violation_a = {
            "id": "test-rule",
            "impact": "minor",
            "description": "Test",
            "help": "Help",
            "helpUrl": "http://example.com",
            "tags": ["wcag2a"],
            "nodes": [{"target": ["#test"], "html": "<div>", "any": [], "all": [], "none": []}],
        }

        issues = tester._parse_violations([violation_a], "A")
        assert issues[0].wcag_level == "A"

    def test_parse_violations_multiple_nodes(self, tester):
        """Test parsing violation with multiple nodes."""
        violation = {
            "id": "label",
            "impact": "critical",
            "description": "Form elements must have labels",
            "help": "Add labels",
            "helpUrl": "http://example.com",
            "tags": ["wcag2a"],
            "nodes": [
                {"target": ["#input1"], "html": "<input>", "any": [], "all": [], "none": []},
                {"target": ["#input2"], "html": "<input>", "any": [], "all": [], "none": []},
            ],
        }

        issues = tester._parse_violations([violation], "A")

        assert len(issues) == 2
        assert issues[0].selector == "#input1"
        assert issues[1].selector == "#input2"

    def test_parse_violations_wcag_criteria(self, tester):
        """Test extracting WCAG criteria from tags."""
        violation = {
            "id": "test-rule",
            "impact": "moderate",
            "description": "Test",
            "help": "Help",
            "helpUrl": "http://example.com",
            "tags": ["wcag2a", "wcag2aa", "wcag111", "wcag143", "section508"],
            "nodes": [{"target": ["#test"], "html": "<div>", "any": [], "all": [], "none": []}],
        }

        issues = tester._parse_violations([violation], "AA")

        # Should extract only wcag tags
        wcag_criteria = issues[0].wcag_criteria
        assert "wcag2a" in wcag_criteria
        assert "wcag2aa" in wcag_criteria
        assert "wcag111" in wcag_criteria
        assert "wcag143" in wcag_criteria
        assert "section508" not in wcag_criteria


class TestFixSuggestions:
    """Tests for generating fix suggestions."""

    def test_generate_fix_suggestion_color_contrast(self, tester):
        """Test fix suggestion for color contrast issue."""
        node = {
            "target": ["#text"],
            "html": "<p>",
            "failureSummary": "Element has insufficient color contrast",
            "any": [],
            "all": [],
            "none": [],
        }

        suggestion = tester._generate_fix_suggestion(
            "color-contrast", node, "http://example.com"
        )

        assert "contrast ratio" in suggestion.lower()
        assert "4.5:1" in suggestion

    def test_generate_fix_suggestion_image_alt(self, tester):
        """Test fix suggestion for missing alt text."""
        node = {"target": ["#img"], "html": "<img>", "any": [], "all": [], "none": []}

        suggestion = tester._generate_fix_suggestion(
            "image-alt", node, "http://example.com"
        )

        assert "alt" in suggestion.lower()

    def test_generate_fix_suggestion_with_failure_summary(self, tester):
        """Test fix suggestion includes failure summary."""
        node = {
            "target": ["#elem"],
            "html": "<div>",
            "failureSummary": "This is the specific issue",
            "any": [],
            "all": [],
            "none": [],
        }

        suggestion = tester._generate_fix_suggestion(
            "unknown-rule", node, "http://example.com"
        )

        assert "This is the specific issue" in suggestion

    def test_generate_fix_suggestion_with_check_messages(self, tester):
        """Test fix suggestion includes check messages."""
        node = {
            "target": ["#elem"],
            "html": "<div>",
            "any": [{"message": "Fix this specific thing"}],
            "all": [{"message": "And this too"}],
            "none": [],
        }

        suggestion = tester._generate_fix_suggestion(
            "test-rule", node, "http://example.com"
        )

        assert "Fix this specific thing" in suggestion
        assert "And this too" in suggestion

    def test_generate_fix_suggestion_fallback(self, tester):
        """Test fix suggestion fallback for unknown rules."""
        node = {"target": ["#elem"], "html": "<div>", "any": [], "all": [], "none": []}

        suggestion = tester._generate_fix_suggestion(
            "unknown-rule", node, "http://example.com"
        )

        assert "http://example.com" in suggestion
        assert "WCAG" in suggestion


class TestPartialAudit:
    """Tests for partial accessibility audit on specific elements."""

    @pytest.mark.asyncio
    async def test_run_partial_audit_success(self, tester, mock_page):
        """Test successful partial audit on element."""
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,  # axe injection check
                {"violations": []},  # audit results
            ]
        )

        issues = await tester.run_partial_audit(mock_page, selector="#main-content")

        assert len(issues) == 0
        # Verify selector was passed to evaluate
        call_args = mock_page.evaluate.call_args_list[1]
        assert call_args[0][1]["selector"] == "#main-content"

    @pytest.mark.asyncio
    async def test_run_partial_audit_with_violations(self, tester, mock_page):
        """Test partial audit finding violations."""
        violation = {
            "id": "heading-order",
            "impact": "moderate",
            "description": "Heading levels should increase by one",
            "help": "Fix heading order",
            "helpUrl": "http://example.com",
            "tags": ["wcag2a"],
            "nodes": [{"target": ["#content h3"], "html": "<h3>", "any": [], "all": [], "none": []}],
        }

        mock_page.evaluate = AsyncMock(side_effect=[True, {"violations": [violation]}])

        issues = await tester.run_partial_audit(
            mock_page, selector="#main-content", wcag_level="A"
        )

        assert len(issues) == 1
        assert issues[0].rule_id == "heading-order"

    @pytest.mark.asyncio
    async def test_run_partial_audit_element_not_found(self, tester, mock_page):
        """Test partial audit when element is not found."""
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,
                Exception("Element not found: #nonexistent"),
            ]
        )

        with pytest.raises(Exception, match="Partial accessibility audit failed"):
            await tester.run_partial_audit(mock_page, selector="#nonexistent")


class TestImpactFiltering:
    """Tests for filtering issues by impact level."""

    def test_filter_by_impact_critical(self, tester):
        """Test filtering for critical issues only."""
        issues = [
            AccessibilityIssue(
                id="1",
                impact="critical",
                rule_id="test1",
                description="Critical issue",
                help_text="Fix it",
                selector="#e1",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
            AccessibilityIssue(
                id="2",
                impact="serious",
                rule_id="test2",
                description="Serious issue",
                help_text="Fix it",
                selector="#e2",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
            AccessibilityIssue(
                id="3",
                impact="moderate",
                rule_id="test3",
                description="Moderate issue",
                help_text="Fix it",
                selector="#e3",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
        ]

        filtered = tester.filter_by_impact(issues, min_impact="critical")

        assert len(filtered) == 1
        assert filtered[0].impact == "critical"

    def test_filter_by_impact_serious(self, tester):
        """Test filtering for serious and above."""
        issues = [
            AccessibilityIssue(
                id="1",
                impact="critical",
                rule_id="test1",
                description="Test",
                help_text="Fix",
                selector="#e1",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
            AccessibilityIssue(
                id="2",
                impact="serious",
                rule_id="test2",
                description="Test",
                help_text="Fix",
                selector="#e2",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
            AccessibilityIssue(
                id="3",
                impact="moderate",
                rule_id="test3",
                description="Test",
                help_text="Fix",
                selector="#e3",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
        ]

        filtered = tester.filter_by_impact(issues, min_impact="serious")

        assert len(filtered) == 2
        assert all(i.impact in ["critical", "serious"] for i in filtered)

    def test_filter_by_impact_moderate(self, tester):
        """Test filtering for moderate and above."""
        issues = [
            AccessibilityIssue(
                id="1",
                impact="serious",
                rule_id="test1",
                description="Test",
                help_text="Fix",
                selector="#e1",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
            AccessibilityIssue(
                id="2",
                impact="moderate",
                rule_id="test2",
                description="Test",
                help_text="Fix",
                selector="#e2",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
            AccessibilityIssue(
                id="3",
                impact="minor",
                rule_id="test3",
                description="Test",
                help_text="Fix",
                selector="#e3",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
        ]

        filtered = tester.filter_by_impact(issues, min_impact="moderate")

        assert len(filtered) == 2


class TestIssueGrouping:
    """Tests for grouping issues by rule."""

    def test_group_by_rule_single_rule(self, tester):
        """Test grouping issues with single rule."""
        issues = [
            AccessibilityIssue(
                id="1",
                impact="serious",
                rule_id="color-contrast",
                description="Test",
                help_text="Fix",
                selector="#e1",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
            AccessibilityIssue(
                id="2",
                impact="serious",
                rule_id="color-contrast",
                description="Test",
                help_text="Fix",
                selector="#e2",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
        ]

        grouped = tester.group_by_rule(issues)

        assert len(grouped) == 1
        assert "color-contrast" in grouped
        assert len(grouped["color-contrast"]) == 2

    def test_group_by_rule_multiple_rules(self, tester):
        """Test grouping issues with multiple rules."""
        issues = [
            AccessibilityIssue(
                id="1",
                impact="serious",
                rule_id="color-contrast",
                description="Test",
                help_text="Fix",
                selector="#e1",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
            AccessibilityIssue(
                id="2",
                impact="critical",
                rule_id="image-alt",
                description="Test",
                help_text="Fix",
                selector="#e2",
                html="<img>",
                wcag_criteria=[],
                wcag_level="A",
            ),
        ]

        grouped = tester.group_by_rule(issues)

        assert len(grouped) == 2
        assert "color-contrast" in grouped
        assert "image-alt" in grouped
        assert len(grouped["color-contrast"]) == 1
        assert len(grouped["image-alt"]) == 1


class TestAccessibilityTree:
    """Tests for accessibility tree snapshot."""

    @pytest.mark.asyncio
    async def test_get_accessibility_tree_success(self, tester, mock_page):
        """Test getting accessibility tree snapshot."""
        tree_snapshot = {
            "role": "WebArea",
            "name": "Test Page",
            "children": [
                {"role": "heading", "name": "Main Heading", "level": 1},
                {"role": "button", "name": "Click Me"},
            ],
        }

        mock_page.accessibility = AsyncMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=tree_snapshot)

        tree = await tester.get_accessibility_tree(mock_page)

        assert tree["role"] == "WebArea"
        assert len(tree["children"]) == 2

    @pytest.mark.asyncio
    async def test_get_accessibility_tree_empty(self, tester, mock_page):
        """Test getting accessibility tree when empty."""
        mock_page.accessibility = AsyncMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value=None)

        tree = await tester.get_accessibility_tree(mock_page)

        assert tree == {}

    @pytest.mark.asyncio
    async def test_get_accessibility_tree_failure(self, tester, mock_page):
        """Test accessibility tree failure handling."""
        mock_page.accessibility = AsyncMock()
        mock_page.accessibility.snapshot = AsyncMock(
            side_effect=Exception("Snapshot failed")
        )

        with pytest.raises(Exception, match="Failed to get accessibility tree"):
            await tester.get_accessibility_tree(mock_page)


class TestCountByImpact:
    """Tests for counting issues by impact level."""

    def test_count_by_impact_single_level(self, tester):
        """Test counting with single impact level."""
        issues = [
            AccessibilityIssue(
                id=str(i),
                impact="serious",
                rule_id="test",
                description="Test",
                help_text="Fix",
                selector=f"#e{i}",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            )
            for i in range(3)
        ]

        count_str = tester._count_by_impact(issues)

        assert "3 serious" in count_str

    def test_count_by_impact_multiple_levels(self, tester):
        """Test counting with multiple impact levels."""
        issues = [
            AccessibilityIssue(
                id="1",
                impact="critical",
                rule_id="test",
                description="Test",
                help_text="Fix",
                selector="#e1",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
            AccessibilityIssue(
                id="2",
                impact="serious",
                rule_id="test",
                description="Test",
                help_text="Fix",
                selector="#e2",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
            AccessibilityIssue(
                id="3",
                impact="moderate",
                rule_id="test",
                description="Test",
                help_text="Fix",
                selector="#e3",
                html="<div>",
                wcag_criteria=[],
                wcag_level="AA",
            ),
        ]

        count_str = tester._count_by_impact(issues)

        assert "1 critical" in count_str
        assert "1 serious" in count_str
        assert "1 moderate" in count_str

    def test_count_by_impact_no_issues(self, tester):
        """Test counting with no issues."""
        issues = []

        count_str = tester._count_by_impact(issues)

        assert count_str == "no issues"
