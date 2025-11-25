"""Report generator with factory pattern for quality analysis.

This module coordinates all report generators and provides a unified interface
for generating quality reports in multiple formats.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pathlib import Path

from ..models.quality_models import QualityReport
from .reporters.html_reporter import HtmlReporter
from .reporters.json_reporter import JsonReporter
from .reporters.markdown_reporter import MarkdownReporter

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Supported report formats."""

    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    MD = "md"  # Alias for markdown
    ALL = "all"  # Generate all formats


class ReportGenerator:
    """
    Report generator coordinating all reporters.

    PATTERN: Factory pattern for reporter selection
    CRITICAL: Provides unified interface for all report formats
    GOTCHA: Handles file I/O and path management
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        include_badges: bool = True,
        pretty_json: bool = True,
    ):
        """
        Initialize report generator.

        Args:
            output_dir: Default output directory for saved reports
            include_badges: Include visual badges in reports
            pretty_json: Pretty-print JSON output
        """
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.include_badges = include_badges
        self.pretty_json = pretty_json
        self.logger = logger

        # Initialize reporters
        self.reporters: Dict[ReportFormat, Any] = {
            ReportFormat.HTML: HtmlReporter(),
            ReportFormat.JSON: JsonReporter(pretty=pretty_json),
            ReportFormat.MARKDOWN: MarkdownReporter(include_badges=include_badges),
        }

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"ReportGenerator initialized with output_dir: {self.output_dir}")

    def generate(
        self,
        report: QualityReport,
        format: Union[str, ReportFormat] = ReportFormat.HTML,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate report in specified format.

        Args:
            report: Quality report to generate
            format: Output format (html, json, markdown, all)
            output_path: Optional custom output path
            **kwargs: Additional format-specific options

        Returns:
            Report content as string (or path if saved)
        """
        # Convert string to enum
        if isinstance(format, str):
            format = ReportFormat(format.lower())

        self.logger.info(f"Generating {format.value} report for {report.report_id}")

        # Handle "all" format
        if format == ReportFormat.ALL:
            return self._generate_all_formats(report, output_path, **kwargs)

        # Normalize markdown aliases
        if format == ReportFormat.MD:
            format = ReportFormat.MARKDOWN

        # Get appropriate reporter
        reporter = self.get_reporter(format)

        # Generate report
        content = reporter.generate_report(report)

        # Save if output path provided
        if output_path:
            output_path = self._resolve_output_path(output_path, format)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Report saved to: {output_path}")
            return output_path

        return content

    def generate_and_save(
        self,
        report: QualityReport,
        format: Union[str, ReportFormat] = ReportFormat.HTML,
        filename: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate and save report to output directory.

        Args:
            report: Quality report to generate
            format: Output format
            filename: Custom filename (without extension)
            **kwargs: Additional format-specific options

        Returns:
            Path to saved file
        """
        # Convert string to enum
        if isinstance(format, str):
            format = ReportFormat(format.lower())

        # Generate default filename
        if not filename:
            filename = f"quality_report_{report.report_id}"

        # Handle "all" format
        if format == ReportFormat.ALL:
            return self._generate_and_save_all_formats(report, filename, **kwargs)

        # Normalize markdown aliases
        if format == ReportFormat.MD:
            format = ReportFormat.MARKDOWN

        # Determine file extension
        extension = self._get_file_extension(format)
        output_path = self.output_dir / f"{filename}.{extension}"

        # Generate and save
        content = self.generate(report, format, **kwargs)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        self.logger.info(f"{format.value.upper()} report saved to: {output_path}")
        return str(output_path)

    def generate_pr_comment(self, report: QualityReport) -> str:
        """
        Generate concise PR comment.

        Args:
            report: Quality report

        Returns:
            Markdown string for PR comment
        """
        self.logger.info(f"Generating PR comment for {report.report_id}")
        reporter = self.get_reporter(ReportFormat.MARKDOWN)
        return reporter.generate_pr_comment(report)

    def generate_ci_output(self, report: QualityReport) -> str:
        """
        Generate CI/CD-friendly JSON output.

        Args:
            report: Quality report

        Returns:
            JSON string for CI systems
        """
        self.logger.info(f"Generating CI output for {report.report_id}")
        reporter = self.get_reporter(ReportFormat.JSON)
        return reporter.generate_ci_output(report)

    def generate_metrics_summary(self, report: QualityReport) -> str:
        """
        Generate simplified metrics summary (JSON).

        Args:
            report: Quality report

        Returns:
            JSON string with summary metrics
        """
        self.logger.info(f"Generating metrics summary for {report.report_id}")
        reporter = self.get_reporter(ReportFormat.JSON)
        return reporter.generate_metrics_only(report)

    def generate_api_response(
        self, report: QualityReport, include_details: bool = True
    ) -> Dict[str, Any]:
        """
        Generate API-friendly response.

        Args:
            report: Quality report
            include_details: Include full details

        Returns:
            Dictionary suitable for API response
        """
        self.logger.info(f"Generating API response for {report.report_id}")
        reporter = self.get_reporter(ReportFormat.JSON)
        return reporter.generate_api_response(report, include_details)

    def export_for_integration(
        self, report: QualityReport, system: str = "generic"
    ) -> str:
        """
        Export report for external system integration.

        Args:
            report: Quality report
            system: Target system (generic, sonarqube, codecov)

        Returns:
            System-specific formatted output
        """
        self.logger.info(f"Exporting report for {system} integration")
        reporter = self.get_reporter(ReportFormat.JSON)
        return reporter.export_for_integration(report, system)

    def generate_comparison_table(self, reports: List[QualityReport]) -> str:
        """
        Generate comparison table for multiple reports.

        Args:
            reports: List of quality reports to compare

        Returns:
            Markdown table comparing reports
        """
        self.logger.info(f"Generating comparison table for {len(reports)} reports")
        reporter = self.get_reporter(ReportFormat.MARKDOWN)
        return reporter.generate_summary_table(reports)

    def get_reporter(self, format: ReportFormat) -> Any:
        """
        Get reporter for specified format.

        Args:
            format: Report format

        Returns:
            Reporter instance

        Raises:
            ValueError: If format not supported
        """
        # Normalize markdown aliases
        if format == ReportFormat.MD:
            format = ReportFormat.MARKDOWN

        if format not in self.reporters:
            raise ValueError(f"Unsupported report format: {format}")

        return self.reporters[format]

    def _generate_all_formats(
        self, report: QualityReport, base_path: Optional[str] = None, **kwargs
    ) -> Dict[str, str]:
        """Generate reports in all formats."""
        results = {}

        for format in [ReportFormat.HTML, ReportFormat.JSON, ReportFormat.MARKDOWN]:
            if base_path:
                extension = self._get_file_extension(format)
                output_path = f"{base_path}.{extension}"
            else:
                output_path = None

            content = self.generate(report, format, output_path, **kwargs)
            results[format.value] = content

        self.logger.info("Generated reports in all formats")
        return results

    def _generate_and_save_all_formats(
        self, report: QualityReport, filename: str, **kwargs
    ) -> Dict[str, str]:
        """Generate and save reports in all formats."""
        results = {}

        for format in [ReportFormat.HTML, ReportFormat.JSON, ReportFormat.MARKDOWN]:
            path = self.generate_and_save(report, format, filename, **kwargs)
            results[format.value] = path

        self.logger.info(f"Saved reports in all formats to {self.output_dir}")
        return results

    def _get_file_extension(self, format: ReportFormat) -> str:
        """Get file extension for format."""
        extensions = {
            ReportFormat.HTML: "html",
            ReportFormat.JSON: "json",
            ReportFormat.MARKDOWN: "md",
            ReportFormat.MD: "md",
        }
        return extensions.get(format, "txt")

    def _resolve_output_path(self, path: str, format: ReportFormat) -> str:
        """Resolve output path with correct extension."""
        path_obj = Path(path)

        # Add extension if missing
        expected_ext = self._get_file_extension(format)
        if path_obj.suffix != f".{expected_ext}":
            path_obj = path_obj.with_suffix(f".{expected_ext}")

        return str(path_obj)

    def set_output_dir(self, output_dir: str) -> None:
        """
        Set output directory for saved reports.

        Args:
            output_dir: New output directory path
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory set to: {self.output_dir}")

    def cleanup_old_reports(self, keep_count: int = 10) -> int:
        """
        Clean up old reports, keeping only the most recent.

        Args:
            keep_count: Number of recent reports to keep

        Returns:
            Number of reports deleted
        """
        self.logger.info(f"Cleaning up old reports (keeping {keep_count})")

        # Get all report files
        report_files = []
        for ext in ["html", "json", "md"]:
            report_files.extend(self.output_dir.glob(f"*.{ext}"))

        # Sort by modification time
        report_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Delete old reports
        deleted_count = 0
        for old_file in report_files[keep_count:]:
            try:
                old_file.unlink()
                deleted_count += 1
                self.logger.debug(f"Deleted old report: {old_file}")
            except Exception as e:
                self.logger.warning(f"Failed to delete {old_file}: {e}")

        self.logger.info(f"Cleaned up {deleted_count} old reports")
        return deleted_count

    def get_available_formats(self) -> List[str]:
        """
        Get list of available report formats.

        Returns:
            List of format names
        """
        return [fmt.value for fmt in ReportFormat if fmt != ReportFormat.ALL]

    def __repr__(self) -> str:
        """String representation."""
        return f"ReportGenerator(output_dir={self.output_dir}, formats={len(self.reporters)})"
