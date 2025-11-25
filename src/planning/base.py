"""Base planner class defining planning interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models.planning_models import Solution, PlanningContext


class BasePlanner(ABC):
    """
    Abstract base class for planning components.

    PATTERN: Abstract base class with async methods
    CRITICAL: All planning operations should be async for LLM integration
    """

    @abstractmethod
    async def generate_solutions(
        self,
        problem: str,
        context: PlanningContext,
    ) -> List[Solution]:
        """
        Generate solution alternatives for a problem.

        Args:
            problem: Problem description
            context: Planning context with constraints

        Returns:
            List of solution alternatives
        """
        pass

    @abstractmethod
    async def analyze_trade_offs(
        self,
        solutions: List[Solution],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze trade-offs between solutions.

        Args:
            solutions: List of solutions to analyze
            weights: Optional dimension weights

        Returns:
            Trade-off analysis results
        """
        pass
