"""Solution space exploration engine for generating alternatives."""

import logging
import json
from typing import List, Dict
from ..models.planning_models import (
    Solution,
    SolutionApproach,
    PlanningContext,
    TradeOffDimension,
)

logger = logging.getLogger(__name__)


class SolutionExplorer:
    """
    Generate multiple solution alternatives for a problem.

    PATTERN: Generate base solution -> create variations -> score and rank
    CRITICAL: Bound exploration to prevent explosion (max 5 solutions)
    GOTCHA: Use LLM creatively but with constraints
    """

    def __init__(
        self,
        llm_service=None,
        decomposition_service=None,
        max_solutions: int = 5,
    ):
        """
        Initialize solution explorer.

        Args:
            llm_service: LLM service for creative generation
            decomposition_service: Decomposition service for task analysis
            max_solutions: Maximum number of solutions to generate
        """
        self.llm_service = llm_service
        self.decomposition_service = decomposition_service
        self.max_solutions = max_solutions
        self.logger = logging.getLogger(__name__)

    async def explore_solutions(
        self,
        problem: str,
        context: PlanningContext,
        strategy: str = "breadth_first",
    ) -> List[Solution]:
        """
        Generate solution alternatives.

        PATTERN: Guided variation with constraints
        CRITICAL: Ensure diversity in solutions

        Args:
            problem: Problem description
            context: Planning context with constraints
            strategy: Exploration strategy (breadth_first, depth_first, hybrid)

        Returns:
            List of solution alternatives
        """
        solutions = []

        # Generate initial solution
        base_solution = await self._generate_base_solution(problem, context)
        solutions.append(base_solution)

        # Generate variations using different approaches
        approaches = [
            SolutionApproach.INCREMENTAL,
            SolutionApproach.BIG_BANG,
            SolutionApproach.STRANGLER,
            SolutionApproach.PARALLEL,
        ]

        for approach in approaches[: self.max_solutions - 1]:
            # Generate variation with specific approach
            variation = await self._generate_variation(
                problem,
                base_solution,
                approach,
                context,
            )

            # Check if variation is significantly different
            if self._is_sufficiently_different(variation, solutions):
                solutions.append(variation)

            if len(solutions) >= self.max_solutions:
                break

        # Score and rank solutions
        for solution in solutions:
            solution.trade_offs = await self._estimate_trade_offs(solution, context)
            solution.confidence_score = self._calculate_confidence(solution)

        self.logger.info(f"Generated {len(solutions)} solution alternatives")

        return solutions

    async def _generate_base_solution(
        self,
        problem: str,
        context: PlanningContext,
    ) -> Solution:
        """
        Generate base solution using LLM or heuristics.

        Args:
            problem: Problem description
            context: Planning context

        Returns:
            Base solution
        """
        if not self.llm_service:
            # Fallback: Create simple solution
            return Solution(
                name="Basic Solution",
                description=f"Standard approach to: {problem}",
                approach=SolutionApproach.INCREMENTAL,
                components=["Core Component", "Data Layer", "API Layer"],
                technologies=["Python", "FastAPI", "PostgreSQL"],
                estimated_effort_hours=160.0,
                estimated_cost=20000.0,
            )

        # Use LLM to generate base solution
        prompt = f"""Generate a solution for the following problem:

Problem: {problem}

Context:
- Budget: ${context.budget_limit or 'Not specified'}
- Timeline: {context.timeline_weeks or 'Not specified'} weeks
- Team capabilities: {', '.join(context.team_capabilities) if context.team_capabilities else 'Not specified'}

Generate a detailed solution including:
1. Solution name (concise)
2. Description (2-3 sentences)
3. Key components (3-5 major components)
4. Technologies required
5. Estimated effort in hours
6. Estimated cost in USD
7. Key advantages (3-5 points)
8. Key disadvantages (2-3 points)
9. Assumptions made

Format as JSON with keys: name, description, components (array), technologies (array),
estimated_effort_hours (number), estimated_cost (number), advantages (array),
disadvantages (array), assumptions (array)"""

        try:
            from ..models.llm_models import LLMRequest

            request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                agent_role="planner",
                task_description=f"Generate base solution for: {problem[:50]}",
                temperature=0.7,
                use_cache=True,
            )

            response = await self.llm_service.generate_response(request)

            # Parse JSON response (handle markdown code blocks)
            solution_data = self._extract_json_from_response(response.content)

            return Solution(
                name=solution_data.get("name", "Generated Solution"),
                description=solution_data.get("description", ""),
                approach=SolutionApproach.INCREMENTAL,
                components=solution_data.get("components", []),
                technologies=solution_data.get("technologies", []),
                estimated_effort_hours=float(solution_data.get("estimated_effort_hours", 100)),
                estimated_cost=float(solution_data.get("estimated_cost", 10000)),
                advantages=solution_data.get("advantages", []),
                disadvantages=solution_data.get("disadvantages", []),
                assumptions=solution_data.get("assumptions", []),
            )

        except Exception as e:
            self.logger.error(f"Failed to generate base solution with LLM: {e}")
            # Return fallback solution
            return Solution(
                name="Basic Solution",
                description=f"Standard approach to: {problem}",
                approach=SolutionApproach.INCREMENTAL,
                components=["Core Component"],
                estimated_effort_hours=100.0,
                estimated_cost=10000.0,
            )

    async def _generate_variation(
        self,
        problem: str,
        base_solution: Solution,
        approach: SolutionApproach,
        context: PlanningContext,
    ) -> Solution:
        """
        Generate solution variation with specific approach.

        CRITICAL: Use LLM creatively but with constraints

        Args:
            problem: Problem description
            base_solution: Base solution to vary from
            approach: Solution approach to use
            context: Planning context

        Returns:
            Solution variation
        """
        if not self.llm_service:
            # Fallback: Create simple variation
            return Solution(
                name=f"{approach.value.title()} Approach",
                description=f"Using {approach.value} strategy for: {problem}",
                approach=approach,
                components=[f"{approach.value} Component {i}" for i in range(1, 4)],
                estimated_effort_hours=base_solution.estimated_effort_hours * 1.2,
                estimated_cost=base_solution.estimated_cost * 1.1,
            )

        prompt = f"""Generate an alternative solution for: {problem}

Use approach: {approach.value}
Base solution components: {base_solution.components}

Constraints:
- Budget: ${context.budget_limit or 'Not specified'}
- Timeline: {context.timeline_weeks or 'Not specified'} weeks
- Team capabilities: {', '.join(context.team_capabilities) if context.team_capabilities else 'Not specified'}

Generate a distinctly different approach that:
1. Uses {approach.value} methodology
2. Has different technologies or patterns
3. Has different trade-offs than base solution
4. Addresses the same requirements

Format as JSON with keys: name, description, components (array), technologies (array),
estimated_effort_hours (number), estimated_cost (number), advantages (array),
disadvantages (array), assumptions (array)"""

        try:
            from ..models.llm_models import LLMRequest

            request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                agent_role="planner",
                task_description=f"Generate {approach.value} variation",
                temperature=0.8,  # Higher temperature for creativity
                use_cache=True,
            )

            response = await self.llm_service.generate_response(request)

            # Parse JSON response (handle markdown code blocks)
            solution_data = self._extract_json_from_response(response.content)

            return Solution(
                name=solution_data.get("name", f"{approach.value} Solution"),
                description=solution_data.get("description", ""),
                approach=approach,
                components=solution_data.get("components", []),
                technologies=solution_data.get("technologies", []),
                estimated_effort_hours=float(solution_data.get("estimated_effort_hours", 100)),
                estimated_cost=float(solution_data.get("estimated_cost", 10000)),
                advantages=solution_data.get("advantages", []),
                disadvantages=solution_data.get("disadvantages", []),
                assumptions=solution_data.get("assumptions", []),
            )

        except Exception as e:
            self.logger.error(f"Failed to generate variation with LLM: {e}")
            # Return fallback variation
            return Solution(
                name=f"{approach.value.title()} Approach",
                description=f"Using {approach.value} strategy for: {problem}",
                approach=approach,
                components=[f"Component {i}" for i in range(1, 4)],
                estimated_effort_hours=base_solution.estimated_effort_hours * 1.1,
                estimated_cost=base_solution.estimated_cost * 1.1,
            )

    def _extract_json_from_response(self, response: str) -> dict:
        """
        Extract JSON from LLM response, handling markdown and extra text.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be extracted
        """
        import re

        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text (without markdown)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: raise error with helpful message
        self.logger.error(f"Could not extract valid JSON from LLM response: {response[:500]}")
        raise ValueError(f"Could not extract valid JSON from LLM response. Response preview: {response[:200]}...")

    def _is_sufficiently_different(
        self,
        solution: Solution,
        existing_solutions: List[Solution],
        similarity_threshold: float = 0.7,
    ) -> bool:
        """
        Check if solution is sufficiently different from existing ones.

        Args:
            solution: Solution to check
            existing_solutions: Existing solutions
            similarity_threshold: Maximum similarity allowed

        Returns:
            True if sufficiently different
        """
        if not existing_solutions:
            return True

        # Simple similarity check based on component overlap
        solution_components = set(solution.components)

        for existing in existing_solutions:
            existing_components = set(existing.components)

            # Calculate Jaccard similarity
            if solution_components and existing_components:
                intersection = len(solution_components & existing_components)
                union = len(solution_components | existing_components)
                similarity = intersection / union if union > 0 else 0

                if similarity >= similarity_threshold:
                    return False

        return True

    async def _estimate_trade_offs(
        self,
        solution: Solution,
        context: PlanningContext,
    ) -> Dict[str, float]:
        """
        Estimate trade-off scores for a solution.

        Args:
            solution: Solution to estimate
            context: Planning context

        Returns:
            Dictionary of trade-off scores (0-1 normalized)
        """
        trade_offs = {}

        # Cost score (lower cost = higher score)
        if context.budget_limit and context.budget_limit > 0:
            cost_ratio = solution.estimated_cost / context.budget_limit
            trade_offs[TradeOffDimension.COST.value] = max(0, min(1, 1 - cost_ratio))
        else:
            trade_offs[TradeOffDimension.COST.value] = 0.5

        # Time score (based on approach and effort)
        effort_weeks = solution.estimated_effort_hours / 40  # Assume 40 hours/week
        if context.timeline_weeks and context.timeline_weeks > 0:
            time_ratio = effort_weeks / context.timeline_weeks
            trade_offs[TradeOffDimension.TIME.value] = max(0, min(1, 1 - time_ratio))
        else:
            trade_offs[TradeOffDimension.TIME.value] = 0.5

        # Complexity score (fewer components = lower complexity)
        component_count = len(solution.components)
        complexity_normalized = max(0, min(1, 1 - (component_count / 10)))
        trade_offs[TradeOffDimension.COMPLEXITY.value] = complexity_normalized

        # Approach-based scores
        risk_score = 0.5  # Default risk score
        if solution.approach == SolutionApproach.INCREMENTAL:
            trade_offs[TradeOffDimension.SCALABILITY.value] = 0.7
            trade_offs[TradeOffDimension.MAINTAINABILITY.value] = 0.8
            trade_offs[TradeOffDimension.RELIABILITY.value] = 0.8
            risk_score = 0.8
        elif solution.approach == SolutionApproach.BIG_BANG:
            trade_offs[TradeOffDimension.SCALABILITY.value] = 0.9
            trade_offs[TradeOffDimension.MAINTAINABILITY.value] = 0.6
            trade_offs[TradeOffDimension.RELIABILITY.value] = 0.3
            risk_score = 0.3
        elif solution.approach == SolutionApproach.STRANGLER:
            trade_offs[TradeOffDimension.SCALABILITY.value] = 0.8
            trade_offs[TradeOffDimension.MAINTAINABILITY.value] = 0.7
            trade_offs[TradeOffDimension.RELIABILITY.value] = 0.7
            risk_score = 0.7
        else:
            trade_offs[TradeOffDimension.SCALABILITY.value] = 0.7
            trade_offs[TradeOffDimension.MAINTAINABILITY.value] = 0.7
            trade_offs[TradeOffDimension.RELIABILITY.value] = 0.6
            risk_score = 0.6

        # Store risk level from trade-offs (inverse of reliability)
        solution.risk_level = 1 - risk_score

        return trade_offs

    def _calculate_confidence(self, solution: Solution) -> float:
        """
        Calculate confidence score for a solution.

        Args:
            solution: Solution to score

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence

        # Increase confidence if solution has detailed components
        if len(solution.components) >= 3:
            confidence += 0.1

        # Increase confidence if solution has advantages listed
        if len(solution.advantages) >= 3:
            confidence += 0.1

        # Increase confidence if solution has realistic estimates
        if solution.estimated_effort_hours > 0 and solution.estimated_cost > 0:
            confidence += 0.1

        # Increase confidence if solution has assumptions documented
        if len(solution.assumptions) > 0:
            confidence += 0.1

        # Decrease confidence if solution has many disadvantages
        if len(solution.disadvantages) > len(solution.advantages):
            confidence -= 0.1

        return max(0, min(1, confidence))
