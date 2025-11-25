"""Base classes for quality analysis components."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..models.quality_models import (
    QualityReport,
    QualityDimension,
)

logger = logging.getLogger(__name__)


class BaseQualityAnalyzer(ABC):
    """
    Abstract base class for quality analysis components.

    PATTERN: Abstract interface for quality analysis operations
    CRITICAL: All analyzers must implement async methods
    GOTCHA: Analysis operations should be isolated and testable
    """

    def __init__(self):
        """Initialize base quality analyzer."""
        self.logger = logger

    @abstractmethod
    async def analyze(
        self, target_path: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform quality analysis on target code.

        Args:
            target_path: Path to code to analyze
            context: Optional additional context (config, baselines, etc.)

        Returns:
            Dictionary containing analysis results
        """
        pass

    @abstractmethod
    async def report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate report from analysis results.

        Args:
            analysis_results: Results from analyze() method

        Returns:
            Formatted report dictionary
        """
        pass

    def _validate_path(self, path: str) -> bool:
        """
        Validate that target path exists and is accessible.

        Args:
            path: Path to validate

        Returns:
            True if path is valid
        """
        import os
        return os.path.exists(path)

    def _get_quality_dimension(self) -> QualityDimension:
        """
        Get the quality dimension this analyzer handles.

        Returns:
            QualityDimension enum value

        Note:
            Subclasses should override to specify their dimension
        """
        return QualityDimension.COVERAGE  # Default, subclasses override
