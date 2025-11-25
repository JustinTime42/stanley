"""Visual testing module for browser automation.

This module provides visual regression testing capabilities using Playwright
for screenshot capture and PIL/Pillow with numpy for image comparison.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image, ImageDraw
from playwright.async_api import Page

from src.models.browser_models import VisualTestResult, BrowserType, Viewport

logger = logging.getLogger(__name__)


class VisualTester:
    """Visual regression testing for browser automation.

    Captures screenshots using Playwright and compares them using PIL/Pillow
    and numpy for pixel-by-pixel analysis. Supports ignore regions for dynamic
    content and generates visual diff images highlighting differences.

    Attributes:
        threshold: Maximum allowed difference percentage (default 0.01 = 1%)
        screenshots_dir: Directory for storing screenshots
    """

    def __init__(self, threshold: float = 0.01, screenshots_dir: Optional[Path] = None):
        """Initialize the visual tester.

        Args:
            threshold: Maximum allowed difference percentage (0.0-1.0)
            screenshots_dir: Directory to save screenshots (defaults to ./screenshots)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self.threshold = threshold
        self.screenshots_dir = screenshots_dir or Path("screenshots")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VisualTester initialized with threshold={threshold}")

    async def capture_screenshot(
        self,
        page: Page,
        name: str,
        full_page: bool = False,
        clip: Optional[Dict[str, int]] = None,
    ) -> str:
        """Capture a screenshot of the current page.

        Args:
            page: Playwright page instance
            name: Screenshot name (without extension)
            full_page: Whether to capture the full scrollable page
            clip: Optional clip region {x, y, width, height}

        Returns:
            Absolute path to the saved screenshot

        Raises:
            Exception: If screenshot capture fails
        """
        try:
            screenshot_path = self.screenshots_dir / f"{name}.png"

            logger.info(f"Capturing screenshot: {name} (full_page={full_page})")

            await page.screenshot(
                path=str(screenshot_path), full_page=full_page, clip=clip
            )

            logger.info(f"Screenshot saved: {screenshot_path}")
            return str(screenshot_path.absolute())

        except Exception as e:
            logger.error(f"Failed to capture screenshot '{name}': {e}")
            raise

    def compare_images(
        self,
        baseline_path: str,
        actual_path: str,
        ignore_regions: Optional[List[Dict[str, int]]] = None,
        test_id: str = "visual_test",
        page_url: str = "",
        browser: BrowserType = BrowserType.CHROMIUM,
        viewport: Optional[Viewport] = None,
    ) -> VisualTestResult:
        """Compare two images and generate a visual test result.

        Args:
            baseline_path: Path to baseline image
            actual_path: Path to actual/current image
            ignore_regions: List of regions to ignore [{x, y, width, height}, ...]
            test_id: Unique test identifier
            page_url: URL of the page being tested
            browser: Browser type used for the test
            viewport: Viewport configuration used

        Returns:
            VisualTestResult with comparison metrics and paths

        Raises:
            FileNotFoundError: If baseline or actual image doesn't exist
            ValueError: If images have different dimensions
        """
        try:
            logger.info(
                f"Comparing images: baseline={baseline_path}, actual={actual_path}"
            )

            # Load images
            baseline_img = Image.open(baseline_path).convert("RGB")
            actual_img = Image.open(actual_path).convert("RGB")

            # Validate dimensions
            if baseline_img.size != actual_img.size:
                logger.warning(
                    f"Image size mismatch: baseline={baseline_img.size}, "
                    f"actual={actual_img.size}. Resizing actual to match baseline."
                )
                actual_img = actual_img.resize(baseline_img.size, Image.LANCZOS)

            # Apply ignore regions if provided
            if ignore_regions:
                logger.info(f"Applying {len(ignore_regions)} ignore regions")
                baseline_img = self._apply_ignore_regions(baseline_img, ignore_regions)
                actual_img = self._apply_ignore_regions(actual_img, ignore_regions)

            # Convert to numpy arrays for comparison
            baseline_array = np.array(baseline_img)
            actual_array = np.array(actual_img)

            # Calculate pixel differences
            diff_mask = np.any(baseline_array != actual_array, axis=2)
            pixel_difference = np.sum(diff_mask)
            total_pixels = baseline_array.shape[0] * baseline_array.shape[1]

            # Calculate match percentage
            match_percentage = ((total_pixels - pixel_difference) / total_pixels) * 100
            difference_percentage = pixel_difference / total_pixels

            # Determine if images match within threshold
            is_match = difference_percentage <= self.threshold

            logger.info(
                f"Comparison result: match={is_match}, "
                f"match_percentage={match_percentage:.2f}%, "
                f"pixel_difference={pixel_difference}, "
                f"threshold={self.threshold * 100}%"
            )

            # Generate diff image and find diff regions
            diff_path = None
            diff_regions = []
            if pixel_difference > 0:
                diff_path = self._generate_diff_image(
                    baseline_array, actual_array, diff_mask, test_id
                )
                diff_regions = self._find_diff_regions(diff_mask)
                logger.info(f"Found {len(diff_regions)} diff regions")

            # Create viewport if not provided
            if viewport is None:
                width, height = baseline_img.size
                viewport = Viewport(width=width, height=height)

            return VisualTestResult(
                test_id=test_id,
                page_url=page_url,
                baseline_path=baseline_path,
                actual_path=actual_path,
                diff_path=diff_path,
                match_percentage=match_percentage,
                pixel_difference=pixel_difference,
                is_match=is_match,
                diff_regions=diff_regions,
                threshold=self.threshold,
                ignore_regions=ignore_regions or [],
                browser=browser,
                viewport=viewport,
            )

        except FileNotFoundError as e:
            logger.error(f"Image file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            raise

    def _apply_ignore_regions(
        self, img: Image.Image, regions: List[Dict[str, int]]
    ) -> Image.Image:
        """Apply ignore regions by masking them with a neutral color.

        Args:
            img: PIL Image to modify
            regions: List of regions to ignore [{x, y, width, height}, ...]

        Returns:
            Modified image with ignore regions masked
        """
        # Create a copy to avoid modifying the original
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)

        # Mask each ignore region with gray color
        for region in regions:
            try:
                x = region.get("x", 0)
                y = region.get("y", 0)
                width = region.get("width", 0)
                height = region.get("height", 0)

                # Draw filled rectangle over the ignore region
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=(128, 128, 128),  # Neutral gray
                )

                logger.debug(
                    f"Masked ignore region: x={x}, y={y}, w={width}, h={height}"
                )

            except Exception as e:
                logger.warning(f"Failed to apply ignore region {region}: {e}")
                continue

        return img_copy

    def _generate_diff_image(
        self,
        baseline: np.ndarray,
        actual: np.ndarray,
        diff_mask: np.ndarray,
        test_id: str,
    ) -> str:
        """Generate a visual diff image highlighting differences.

        Creates a side-by-side comparison with the diff highlighted in red.

        Args:
            baseline: Baseline image as numpy array
            actual: Actual image as numpy array
            diff_mask: Boolean mask of differences
            test_id: Test identifier for naming the diff image

        Returns:
            Path to the generated diff image
        """
        try:
            height, width = baseline.shape[:2]

            # Create a diff visualization image
            # Start with the actual image
            diff_visual = actual.copy()

            # Highlight differences in red
            diff_visual[diff_mask] = [255, 0, 0]  # Red color for differences

            # Create side-by-side comparison: baseline | actual | diff
            combined_width = width * 3
            combined = np.zeros((height, combined_width, 3), dtype=np.uint8)

            # Place images side by side
            combined[:, :width] = baseline
            combined[:, width : width * 2] = actual
            combined[:, width * 2 :] = diff_visual

            # Convert to PIL Image and save
            diff_img = Image.fromarray(combined)
            diff_path = self.screenshots_dir / f"{test_id}_diff.png"
            diff_img.save(diff_path)

            logger.info(f"Diff image saved: {diff_path}")
            return str(diff_path.absolute())

        except Exception as e:
            logger.error(f"Failed to generate diff image: {e}")
            # Return empty string on failure to avoid breaking the flow
            return ""

    def _find_diff_regions(
        self, diff_mask: np.ndarray, min_region_size: int = 10
    ) -> List[Dict[str, int]]:
        """Find bounding boxes of difference regions.

        Args:
            diff_mask: Boolean mask of differences
            min_region_size: Minimum region size to include (in pixels)

        Returns:
            List of difference regions [{x, y, width, height}, ...]
        """
        regions = []

        try:
            # Find connected components (groups of different pixels)
            # Simple approach: find rows and columns with differences
            rows_with_diff = np.any(diff_mask, axis=1)
            cols_with_diff = np.any(diff_mask, axis=0)

            if not np.any(rows_with_diff) or not np.any(cols_with_diff):
                return regions

            # Find contiguous regions
            # This is a simplified approach - could use scipy.ndimage.label for more accuracy
            row_indices = np.where(rows_with_diff)[0]
            col_indices = np.where(cols_with_diff)[0]

            if len(row_indices) > 0 and len(col_indices) > 0:
                # Create a bounding box for the entire diff region
                # For more granular regions, we'd need connected component analysis
                y_min, y_max = row_indices[0], row_indices[-1]
                x_min, x_max = col_indices[0], col_indices[-1]

                width = x_max - x_min + 1
                height = y_max - y_min + 1

                # Only add if region is large enough
                if width * height >= min_region_size:
                    regions.append(
                        {
                            "x": int(x_min),
                            "y": int(y_min),
                            "width": int(width),
                            "height": int(height),
                        }
                    )

        except Exception as e:
            logger.warning(f"Failed to find diff regions: {e}")

        return regions

    async def capture_and_compare(
        self,
        page: Page,
        test_id: str,
        baseline_name: Optional[str] = None,
        full_page: bool = False,
        ignore_regions: Optional[List[Dict[str, int]]] = None,
        browser: BrowserType = BrowserType.CHROMIUM,
        viewport: Optional[Viewport] = None,
    ) -> VisualTestResult:
        """Convenience method to capture and compare in one call.

        If baseline doesn't exist, creates it. Otherwise compares with baseline.

        Args:
            page: Playwright page instance
            test_id: Unique test identifier
            baseline_name: Name for baseline image (defaults to test_id)
            full_page: Whether to capture full scrollable page
            ignore_regions: Regions to ignore in comparison
            browser: Browser type used
            viewport: Viewport configuration

        Returns:
            VisualTestResult with comparison results
        """
        baseline_name = baseline_name or test_id
        baseline_path = self.screenshots_dir / f"{baseline_name}_baseline.png"
        actual_path = self.screenshots_dir / f"{test_id}_actual.png"

        # Capture current screenshot
        await self.capture_screenshot(page, f"{test_id}_actual", full_page=full_page)

        # If baseline doesn't exist, create it
        if not baseline_path.exists():
            logger.info(f"Baseline not found, creating: {baseline_path}")
            await self.capture_screenshot(
                page, f"{baseline_name}_baseline", full_page=full_page
            )

            # Return a perfect match result for new baseline
            if viewport is None:
                img = Image.open(str(actual_path))
                width, height = img.size
                viewport = Viewport(width=width, height=height)

            return VisualTestResult(
                test_id=test_id,
                page_url=page.url,
                baseline_path=str(baseline_path.absolute()),
                actual_path=str(actual_path.absolute()),
                diff_path=None,
                match_percentage=100.0,
                pixel_difference=0,
                is_match=True,
                diff_regions=[],
                threshold=self.threshold,
                ignore_regions=ignore_regions or [],
                browser=browser,
                viewport=viewport,
            )

        # Compare with existing baseline
        return self.compare_images(
            baseline_path=str(baseline_path.absolute()),
            actual_path=str(actual_path.absolute()),
            ignore_regions=ignore_regions,
            test_id=test_id,
            page_url=page.url,
            browser=browser,
            viewport=viewport,
        )

    def update_baseline(
        self, test_id: str, baseline_name: Optional[str] = None
    ) -> bool:
        """Update the baseline image with the current actual image.

        Useful when visual changes are intentional and should become the new baseline.

        Args:
            test_id: Test identifier
            baseline_name: Baseline name (defaults to test_id)

        Returns:
            True if baseline was updated successfully
        """
        try:
            baseline_name = baseline_name or test_id
            actual_path = self.screenshots_dir / f"{test_id}_actual.png"
            baseline_path = self.screenshots_dir / f"{baseline_name}_baseline.png"

            if not actual_path.exists():
                logger.error(f"Actual image not found: {actual_path}")
                return False

            # Copy actual to baseline
            actual_img = Image.open(actual_path)
            actual_img.save(baseline_path)

            logger.info(f"Baseline updated: {baseline_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to update baseline: {e}")
            return False
