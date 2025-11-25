"""Tests for Visual Regression Testing.

This module contains tests for the VisualTester class which performs
visual regression testing using screenshot comparison.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.async_api import Page

from src.browser.visual_tester import VisualTester
from src.models.browser_models import BrowserType, Viewport, VisualTestResult


@pytest.fixture
def tester(tmp_path):
    """Create a VisualTester instance with temp directory."""
    return VisualTester(threshold=0.01, screenshots_dir=tmp_path)


@pytest.fixture
def mock_page():
    """Create a mock Page instance."""
    page = AsyncMock(spec=Page)
    page.url = "https://example.com"
    page.screenshot = AsyncMock(return_value=b"screenshot_data")
    return page


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    # Create a simple 100x100 red image
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    return img


@pytest.fixture
def baseline_image(tmp_path):
    """Create a baseline image file."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    baseline_path = tmp_path / "baseline.png"
    img.save(baseline_path)
    return str(baseline_path)


@pytest.fixture
def actual_image(tmp_path):
    """Create an actual image file."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    actual_path = tmp_path / "actual.png"
    img.save(actual_path)
    return str(actual_path)


class TestVisualTesterInitialization:
    """Tests for VisualTester initialization."""

    def test_init_default(self):
        """Test VisualTester initialization with defaults."""
        tester = VisualTester()
        assert tester.threshold == 0.01
        assert tester.screenshots_dir == Path("screenshots")

    def test_init_custom_threshold(self, tmp_path):
        """Test initialization with custom threshold."""
        tester = VisualTester(threshold=0.05, screenshots_dir=tmp_path)
        assert tester.threshold == 0.05
        assert tester.screenshots_dir == tmp_path

    def test_init_invalid_threshold(self):
        """Test initialization with invalid threshold."""
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            VisualTester(threshold=1.5)

        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            VisualTester(threshold=-0.1)

    def test_screenshots_dir_created(self, tmp_path):
        """Test that screenshots directory is created."""
        screenshots_dir = tmp_path / "screenshots"
        tester = VisualTester(screenshots_dir=screenshots_dir)
        assert screenshots_dir.exists()
        assert screenshots_dir.is_dir()


class TestScreenshotCapture:
    """Tests for screenshot capture functionality."""

    @pytest.mark.asyncio
    async def test_capture_screenshot_success(self, tester, mock_page, tmp_path):
        """Test successful screenshot capture."""
        mock_page.screenshot = AsyncMock(return_value=b"screenshot_bytes")

        path = await tester.capture_screenshot(mock_page, "test_screenshot")

        assert path == str((tmp_path / "test_screenshot.png").absolute())
        mock_page.screenshot.assert_called_once_with(
            path=str(tmp_path / "test_screenshot.png"), full_page=False, clip=None
        )

    @pytest.mark.asyncio
    async def test_capture_screenshot_full_page(self, tester, mock_page):
        """Test capturing full page screenshot."""
        await tester.capture_screenshot(mock_page, "fullpage", full_page=True)

        call_args = mock_page.screenshot.call_args
        assert call_args[1]["full_page"] is True

    @pytest.mark.asyncio
    async def test_capture_screenshot_with_clip(self, tester, mock_page):
        """Test capturing screenshot with clip region."""
        clip_region = {"x": 10, "y": 20, "width": 100, "height": 200}

        await tester.capture_screenshot(mock_page, "clipped", clip=clip_region)

        call_args = mock_page.screenshot.call_args
        assert call_args[1]["clip"] == clip_region

    @pytest.mark.asyncio
    async def test_capture_screenshot_failure(self, tester, mock_page):
        """Test screenshot capture failure handling."""
        mock_page.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))

        with pytest.raises(Exception, match="Screenshot failed"):
            await tester.capture_screenshot(mock_page, "test")


class TestImageComparison:
    """Tests for image comparison functionality."""

    def test_compare_identical_images(self, tester, baseline_image, actual_image):
        """Test comparing two identical images."""
        result = tester.compare_images(
            baseline_path=baseline_image,
            actual_path=actual_image,
            test_id="test_identical",
        )

        assert isinstance(result, VisualTestResult)
        assert result.is_match is True
        assert result.match_percentage == 100.0
        assert result.pixel_difference == 0
        assert result.diff_path is None

    def test_compare_different_images(self, tester, tmp_path):
        """Test comparing two different images."""
        # Create baseline (red)
        baseline_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        baseline_path = tmp_path / "baseline.png"
        baseline_img.save(baseline_path)

        # Create actual (blue)
        actual_img = Image.new("RGB", (100, 100), color=(0, 0, 255))
        actual_path = tmp_path / "actual.png"
        actual_img.save(actual_path)

        result = tester.compare_images(
            baseline_path=str(baseline_path),
            actual_path=str(actual_path),
            test_id="test_different",
        )

        assert result.is_match is False
        assert result.match_percentage < 100.0
        assert result.pixel_difference > 0
        assert result.diff_path is not None

    def test_compare_partially_different_images(self, tester, tmp_path):
        """Test comparing images with partial differences."""
        # Create baseline (all red)
        baseline_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        baseline_path = tmp_path / "baseline.png"
        baseline_img.save(baseline_path)

        # Create actual (red with small blue square)
        actual_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        # Draw a 10x10 blue square in the corner
        for x in range(10):
            for y in range(10):
                actual_img.putpixel((x, y), (0, 0, 255))
        actual_path = tmp_path / "actual.png"
        actual_img.save(actual_path)

        result = tester.compare_images(
            baseline_path=str(baseline_path),
            actual_path=str(actual_path),
            test_id="test_partial",
        )

        # 100 pixels different out of 10000 = 1%
        assert result.pixel_difference == 100
        assert 98.0 < result.match_percentage < 100.0

    def test_compare_with_ignore_regions(self, tester, tmp_path):
        """Test image comparison with ignore regions."""
        # Create baseline (red)
        baseline_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        baseline_path = tmp_path / "baseline.png"
        baseline_img.save(baseline_path)

        # Create actual (blue)
        actual_img = Image.new("RGB", (100, 100), color=(0, 0, 255))
        actual_path = tmp_path / "actual.png"
        actual_img.save(actual_path)

        # Ignore the entire image
        ignore_regions = [{"x": 0, "y": 0, "width": 100, "height": 100}]

        result = tester.compare_images(
            baseline_path=str(baseline_path),
            actual_path=str(actual_path),
            ignore_regions=ignore_regions,
            test_id="test_ignore",
        )

        # After masking, both images should be identical (gray)
        assert result.is_match is True
        assert result.pixel_difference == 0

    def test_compare_different_sizes(self, tester, tmp_path):
        """Test comparing images with different sizes."""
        # Create baseline (100x100)
        baseline_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        baseline_path = tmp_path / "baseline.png"
        baseline_img.save(baseline_path)

        # Create actual (200x200)
        actual_img = Image.new("RGB", (200, 200), color=(255, 0, 0))
        actual_path = tmp_path / "actual.png"
        actual_img.save(actual_path)

        # Should resize actual to match baseline
        result = tester.compare_images(
            baseline_path=str(baseline_path),
            actual_path=str(actual_path),
            test_id="test_resize",
        )

        # After resizing, should still match
        assert result.is_match is True

    def test_compare_baseline_not_found(self, tester, actual_image):
        """Test comparison when baseline doesn't exist."""
        with pytest.raises(FileNotFoundError):
            tester.compare_images(
                baseline_path="/nonexistent/baseline.png",
                actual_path=actual_image,
                test_id="test_missing",
            )

    def test_compare_actual_not_found(self, tester, baseline_image):
        """Test comparison when actual doesn't exist."""
        with pytest.raises(FileNotFoundError):
            tester.compare_images(
                baseline_path=baseline_image,
                actual_path="/nonexistent/actual.png",
                test_id="test_missing",
            )


class TestIgnoreRegions:
    """Tests for ignore regions functionality."""

    def test_apply_ignore_regions_single(self, tester, sample_image):
        """Test applying a single ignore region."""
        regions = [{"x": 10, "y": 10, "width": 20, "height": 20}]

        result_img = tester._apply_ignore_regions(sample_image, regions)

        # Check that the region was masked (should be gray)
        pixel = result_img.getpixel((15, 15))
        assert pixel == (128, 128, 128)

        # Check that outside region is unchanged
        pixel_outside = result_img.getpixel((50, 50))
        assert pixel_outside == (255, 0, 0)

    def test_apply_ignore_regions_multiple(self, tester, sample_image):
        """Test applying multiple ignore regions."""
        regions = [
            {"x": 10, "y": 10, "width": 20, "height": 20},
            {"x": 50, "y": 50, "width": 30, "height": 30},
        ]

        result_img = tester._apply_ignore_regions(sample_image, regions)

        # Both regions should be masked
        assert result_img.getpixel((15, 15)) == (128, 128, 128)
        assert result_img.getpixel((60, 60)) == (128, 128, 128)

    def test_apply_ignore_regions_invalid(self, tester, sample_image):
        """Test that invalid ignore regions are handled gracefully."""
        regions = [{"x": "invalid", "y": 10, "width": 20, "height": 20}]

        # Should not raise exception, just skip invalid region
        result_img = tester._apply_ignore_regions(sample_image, regions)

        assert result_img is not None


class TestDiffGeneration:
    """Tests for diff image generation."""

    def test_generate_diff_image(self, tester, tmp_path):
        """Test generating a diff image."""
        # Create baseline and actual arrays
        baseline = np.zeros((100, 100, 3), dtype=np.uint8)
        baseline[:, :] = [255, 0, 0]  # Red

        actual = np.zeros((100, 100, 3), dtype=np.uint8)
        actual[:, :] = [255, 0, 0]  # Red
        actual[10:20, 10:20] = [0, 0, 255]  # Blue square

        # Create diff mask
        diff_mask = np.any(baseline != actual, axis=2)

        diff_path = tester._generate_diff_image(
            baseline, actual, diff_mask, "test_diff"
        )

        assert diff_path != ""
        assert Path(diff_path).exists()
        assert Path(diff_path).name == "test_diff_diff.png"

    def test_generate_diff_image_error(self, tester):
        """Test diff image generation error handling."""
        # Invalid arrays
        baseline = np.array([])
        actual = np.array([])
        diff_mask = np.array([])

        diff_path = tester._generate_diff_image(
            baseline, actual, diff_mask, "test_error"
        )

        # Should return empty string on error
        assert diff_path == ""


class TestDiffRegions:
    """Tests for finding diff regions."""

    def test_find_diff_regions_single(self, tester):
        """Test finding a single diff region."""
        # Create a diff mask with one region
        diff_mask = np.zeros((100, 100), dtype=bool)
        diff_mask[10:30, 20:50] = True

        regions = tester._find_diff_regions(diff_mask)

        assert len(regions) == 1
        assert regions[0]["x"] == 20
        assert regions[0]["y"] == 10
        assert regions[0]["width"] == 30
        assert regions[0]["height"] == 20

    def test_find_diff_regions_no_diff(self, tester):
        """Test finding regions when there's no difference."""
        diff_mask = np.zeros((100, 100), dtype=bool)

        regions = tester._find_diff_regions(diff_mask)

        assert len(regions) == 0

    def test_find_diff_regions_small(self, tester):
        """Test that small regions are filtered out."""
        diff_mask = np.zeros((100, 100), dtype=bool)
        # Create a 2x2 region (4 pixels, below min_region_size=10)
        diff_mask[10:12, 10:12] = True

        regions = tester._find_diff_regions(diff_mask, min_region_size=10)

        assert len(regions) == 0

    def test_find_diff_regions_error(self, tester):
        """Test diff region finding error handling."""
        # Invalid mask
        diff_mask = np.array([])

        regions = tester._find_diff_regions(diff_mask)

        assert regions == []


class TestCaptureAndCompare:
    """Tests for the convenience capture_and_compare method."""

    @pytest.mark.asyncio
    async def test_capture_and_compare_create_baseline(self, tester, mock_page, tmp_path):
        """Test capture_and_compare creates baseline if it doesn't exist."""
        mock_page.screenshot = AsyncMock()

        # Mock the file operations
        with patch.object(Path, "exists", return_value=False):
            with patch.object(Image, "open") as mock_open:
                mock_img = Mock()
                mock_img.size = (1280, 720)
                mock_open.return_value = mock_img

                result = await tester.capture_and_compare(
                    page=mock_page,
                    test_id="new_test",
                )

                # Should create both actual and baseline
                assert mock_page.screenshot.call_count == 2
                assert result.is_match is True
                assert result.pixel_difference == 0

    @pytest.mark.asyncio
    async def test_capture_and_compare_with_existing_baseline(
        self, tester, mock_page, tmp_path
    ):
        """Test capture_and_compare with existing baseline."""
        # Create a baseline file
        baseline_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        baseline_path = tmp_path / "test_baseline.png"
        baseline_img.save(baseline_path)

        actual_path = tmp_path / "test_actual.png"

        mock_page.screenshot = AsyncMock()

        with patch.object(tester, "compare_images") as mock_compare:
            mock_compare.return_value = VisualTestResult(
                test_id="test",
                page_url="https://example.com",
                baseline_path=str(baseline_path),
                actual_path=str(actual_path),
                diff_path=None,
                match_percentage=100.0,
                pixel_difference=0,
                is_match=True,
                diff_regions=[],
                threshold=0.01,
                ignore_regions=[],
                browser=BrowserType.CHROMIUM,
                viewport=Viewport(),
            )

            await tester.capture_and_compare(
                page=mock_page,
                test_id="test",
            )

            # Should capture actual and compare
            mock_page.screenshot.assert_called_once()
            mock_compare.assert_called_once()

    @pytest.mark.asyncio
    async def test_capture_and_compare_with_ignore_regions(self, tester, mock_page):
        """Test capture_and_compare with ignore regions."""
        ignore_regions = [{"x": 0, "y": 0, "width": 100, "height": 50}]

        with patch.object(tester, "compare_images") as mock_compare:
            mock_compare.return_value = Mock(spec=VisualTestResult)

            # Create a fake baseline
            with patch.object(Path, "exists", return_value=True):
                await tester.capture_and_compare(
                    page=mock_page,
                    test_id="test",
                    ignore_regions=ignore_regions,
                )

                # Verify ignore_regions passed to compare_images
                call_args = mock_compare.call_args
                assert call_args[1]["ignore_regions"] == ignore_regions


class TestBaselineUpdate:
    """Tests for baseline update functionality."""

    def test_update_baseline_success(self, tester, tmp_path):
        """Test successful baseline update."""
        # Create an actual image
        actual_img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        actual_path = tmp_path / "test_actual.png"
        actual_img.save(actual_path)

        baseline_path = tmp_path / "test_baseline.png"

        result = tester.update_baseline("test")

        assert result is True
        assert baseline_path.exists()

        # Verify baseline matches actual
        baseline_img = Image.open(baseline_path)
        assert baseline_img.getpixel((50, 50)) == (0, 255, 0)

    def test_update_baseline_actual_not_found(self, tester):
        """Test update baseline when actual doesn't exist."""
        result = tester.update_baseline("nonexistent")

        assert result is False

    def test_update_baseline_with_custom_name(self, tester, tmp_path):
        """Test updating baseline with custom name."""
        # Create an actual image
        actual_img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        actual_path = tmp_path / "test_actual.png"
        actual_img.save(actual_path)

        result = tester.update_baseline("test", baseline_name="custom")

        assert result is True
        assert (tmp_path / "custom_baseline.png").exists()


class TestVisualTestResult:
    """Tests for VisualTestResult data structure."""

    def test_visual_test_result_creation(self):
        """Test creating a VisualTestResult."""
        result = VisualTestResult(
            test_id="test123",
            page_url="https://example.com",
            baseline_path="/path/to/baseline.png",
            actual_path="/path/to/actual.png",
            diff_path="/path/to/diff.png",
            match_percentage=98.5,
            pixel_difference=150,
            is_match=False,
            diff_regions=[{"x": 10, "y": 20, "width": 30, "height": 40}],
            threshold=0.01,
            ignore_regions=[],
            browser=BrowserType.CHROMIUM,
            viewport=Viewport(width=1920, height=1080),
        )

        assert result.test_id == "test123"
        assert result.match_percentage == 98.5
        assert result.is_match is False
        assert len(result.diff_regions) == 1

    def test_visual_test_result_with_ignore_regions(self):
        """Test VisualTestResult with ignore regions."""
        ignore_regions = [
            {"x": 0, "y": 0, "width": 100, "height": 50},
            {"x": 200, "y": 200, "width": 50, "height": 50},
        ]

        result = VisualTestResult(
            test_id="test",
            page_url="https://example.com",
            baseline_path="/baseline.png",
            actual_path="/actual.png",
            diff_path=None,
            match_percentage=100.0,
            pixel_difference=0,
            is_match=True,
            diff_regions=[],
            threshold=0.01,
            ignore_regions=ignore_regions,
            browser=BrowserType.FIREFOX,
            viewport=Viewport(),
        )

        assert len(result.ignore_regions) == 2
        assert result.ignore_regions[0]["width"] == 100
