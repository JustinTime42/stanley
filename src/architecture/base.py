"""Base architect class defining architecture interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..models.architecture_models import ArchitectureDesign, PatternMatch, TechnologyEvaluation


class BaseArchitect(ABC):
    """
    Abstract base class for architecture components.

    PATTERN: Abstract base class with async methods
    CRITICAL: All architecture operations should be async for LLM/analysis integration
    """

    @abstractmethod
    async def design_architecture(
        self,
        requirements: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> ArchitectureDesign:
        """
        Design system architecture based on requirements.

        Args:
            requirements: System requirements
            constraints: Design constraints

        Returns:
            Architecture design
        """
        pass

    @abstractmethod
    async def validate_consistency(
        self,
        design: ArchitectureDesign,
    ) -> Dict[str, Any]:
        """
        Validate architecture consistency.

        Args:
            design: Architecture design to validate

        Returns:
            Validation results
        """
        pass

    @abstractmethod
    async def recognize_patterns(
        self,
        codebase_path: str,
    ) -> List[PatternMatch]:
        """
        Recognize architecture patterns in codebase.

        Args:
            codebase_path: Path to codebase

        Returns:
            List of detected patterns
        """
        pass

    @abstractmethod
    async def recommend_technologies(
        self,
        requirements: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> List[TechnologyEvaluation]:
        """
        Recommend technologies based on requirements.

        Args:
            requirements: System requirements
            constraints: Technology constraints

        Returns:
            List of technology evaluations
        """
        pass
