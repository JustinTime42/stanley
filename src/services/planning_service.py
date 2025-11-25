"""Planning orchestrator service integrating solution exploration and trade-off analysis."""

import logging
from typing import List, Dict, Any, Optional
from ..planning import SolutionExplorer, TradeOffAnalyzer, DecisionDocumenter
from ..models.planning_models import Solution, PlanningContext, DecisionRecord, TradeOffAnalysisResult

logger = logging.getLogger(__name__)


class PlanningOrchestrator:
    """
    High-level planning service facade.

    PATTERN: Service facade pattern integrating all planning components
    CRITICAL: Single entry point for planning operations
    GOTCHA: Initialize components lazily to avoid circular dependencies
    """

    def __init__(
        self,
        llm_service=None,
        decomposition_service=None,
        max_solutions: int = 5,
    ):
        """
        Initialize planning orchestrator.

        Args:
            llm_service: LLM service for solution generation
            decomposition_service: Decomposition service for task analysis
            max_solutions: Maximum number of solutions to explore
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.solution_explorer = SolutionExplorer(
            llm_service=llm_service,
            decomposition_service=decomposition_service,
            max_solutions=max_solutions,
        )

        self.trade_off_analyzer = TradeOffAnalyzer()
        self.decision_documenter = DecisionDocumenter()

        self.logger.info("Planning orchestrator initialized")

    async def plan_solution(
        self,
        problem: str,
        context: PlanningContext,
        explore_alternatives: bool = True,
    ) -> Dict[str, Any]:
        """
        Plan solution with exploration and trade-off analysis.

        PATTERN: Explore -> Analyze -> Document -> Return
        CRITICAL: Generate multiple alternatives and select best

        Args:
            problem: Problem description
            context: Planning context
            explore_alternatives: Whether to explore multiple solutions

        Returns:
            Planning result with solutions, analysis, and decision
        """
        self.logger.info(f"Planning solution for: {problem[:50]}...")

        # Step 1: Generate solution alternatives
        if explore_alternatives:
            solutions = await self.solution_explorer.explore_solutions(
                problem=problem,
                context=context,
            )
        else:
            # Generate single solution
            base_solution = await self.solution_explorer._generate_base_solution(
                problem=problem,
                context=context,
            )
            solutions = [base_solution]

        self.logger.info(f"Generated {len(solutions)} solution alternatives")

        # Step 2: Analyze trade-offs
        if len(solutions) > 1:
            trade_off_analysis = self.trade_off_analyzer.analyze_trade_offs(solutions)

            # Get best solution
            best_idx = trade_off_analysis.best_solution_index
            selected_solution = solutions[best_idx]

            self.logger.info(
                f"Selected solution: {selected_solution.name} "
                f"(score: {trade_off_analysis.scores[best_idx]:.3f})"
            )
        else:
            # Single solution, create trivial analysis
            trade_off_analysis = TradeOffAnalysisResult(
                scores=[1.0],
                ranking=[0],
                best_solution_index=0,
                trade_off_matrix=[[0.5] * 10],
                weights_used={},
            )
            selected_solution = solutions[0]

        # Step 3: Document decision
        decision_record = self.decision_documenter.create_adr(
            title=f"Solution for: {problem[:50]}",
            solutions=solutions,
            selected_solution=selected_solution,
            trade_off_analysis=trade_off_analysis,
            context=context,
        )

        # Step 4: Generate explanation
        explanation = self.trade_off_analyzer.explain_decision(
            solutions=solutions,
            analysis_result=trade_off_analysis,
            top_k=min(3, len(solutions)),
        )

        return {
            "solutions": solutions,
            "selected_solution": selected_solution,
            "trade_off_analysis": trade_off_analysis,
            "decision_record": decision_record,
            "explanation": explanation,
            "alternatives_count": len(solutions),
        }

    async def generate_alternatives(
        self,
        problem: str,
        context: PlanningContext,
        num_alternatives: Optional[int] = None,
    ) -> List[Solution]:
        """
        Generate solution alternatives.

        Args:
            problem: Problem description
            context: Planning context
            num_alternatives: Number of alternatives (uses max_solutions if None)

        Returns:
            List of solution alternatives
        """
        # Override max_solutions if specified
        original_max = self.solution_explorer.max_solutions

        if num_alternatives:
            self.solution_explorer.max_solutions = num_alternatives

        solutions = await self.solution_explorer.explore_solutions(
            problem=problem,
            context=context,
        )

        # Restore original max
        self.solution_explorer.max_solutions = original_max

        return solutions

    def analyze_trade_offs(
        self,
        solutions: List[Solution],
        weights: Optional[Dict[str, float]] = None,
    ) -> TradeOffAnalysisResult:
        """
        Analyze trade-offs between solutions.

        Args:
            solutions: List of solutions
            weights: Optional dimension weights

        Returns:
            Trade-off analysis result
        """
        return self.trade_off_analyzer.analyze_trade_offs(solutions, weights)

    def create_decision_record(
        self,
        title: str,
        solutions: List[Solution],
        selected_solution: Solution,
        trade_off_analysis: TradeOffAnalysisResult,
        context: PlanningContext,
    ) -> DecisionRecord:
        """
        Create architecture decision record.

        Args:
            title: Decision title
            solutions: All solutions considered
            selected_solution: Selected solution
            trade_off_analysis: Trade-off analysis results
            context: Planning context

        Returns:
            Decision record
        """
        return self.decision_documenter.create_adr(
            title=title,
            solutions=solutions,
            selected_solution=selected_solution,
            trade_off_analysis=trade_off_analysis,
            context=context,
        )

    def get_decision_history(self, decision_id: str) -> List[DecisionRecord]:
        """
        Get decision history.

        Args:
            decision_id: Decision ID

        Returns:
            List of decisions in history
        """
        return self.decision_documenter.get_decision_history(decision_id)

    def get_all_decisions(self) -> List[DecisionRecord]:
        """
        Get all decisions.

        Returns:
            List of all decision records
        """
        return self.decision_documenter.get_all_decisions()

    async def explain_solution(
        self,
        solution: Solution,
        alternatives: List[Solution],
    ) -> str:
        """
        Generate explanation for a solution.

        Args:
            solution: Solution to explain
            alternatives: Alternative solutions for comparison

        Returns:
            Explanation text
        """
        all_solutions = [solution] + [a for a in alternatives if a.id != solution.id]

        # Create analysis
        analysis = self.trade_off_analyzer.analyze_trade_offs(all_solutions)

        # Generate explanation
        explanation = self.trade_off_analyzer.explain_decision(
            solutions=all_solutions,
            analysis_result=analysis,
        )

        return explanation
