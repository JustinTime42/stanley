"""Trade-off analysis using TOPSIS (Technique for Order of Preference by Similarity)."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from ..models.planning_models import Solution, TradeOffDimension, TradeOffAnalysisResult

logger = logging.getLogger(__name__)


class TradeOffAnalyzer:
    """
    Multi-criteria decision analysis using TOPSIS algorithm.

    PATTERN: TOPSIS - find solution closest to ideal and farthest from negative-ideal
    CRITICAL: Normalize scores across different dimensions
    GOTCHA: Handle cases where all solutions have same score on a dimension
    """

    def __init__(self):
        """Initialize trade-off analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_trade_offs(
        self,
        solutions: List[Solution],
        weights: Optional[Dict[str, float]] = None,
    ) -> TradeOffAnalysisResult:
        """
        Analyze trade-offs using TOPSIS algorithm.

        PATTERN: Build matrix -> Normalize -> Weight -> Find ideal -> Calculate distances -> Score
        CRITICAL: Handle edge cases (single solution, equal scores, etc.)

        Args:
            solutions: List of solutions to analyze
            weights: Optional dimension weights (defaults to equal)

        Returns:
            Trade-off analysis result with rankings
        """
        if not solutions:
            raise ValueError("Cannot analyze trade-offs for empty solution list")

        if len(solutions) == 1:
            # Single solution - trivial case
            return TradeOffAnalysisResult(
                scores=[1.0],
                ranking=[0],
                best_solution_index=0,
                trade_off_matrix=[[0.5] * len(TradeOffDimension)],
                weights_used=weights or {},
            )

        # Get all dimensions
        dimensions = [dim.value for dim in TradeOffDimension]

        # Default to equal weights if not provided
        if not weights:
            weights = {dim: 1.0 / len(dimensions) for dim in dimensions}

        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v / weight_sum for k, v in weights.items()}

        # Build decision matrix
        matrix = []
        for solution in solutions:
            row = []
            for dim in dimensions:
                # Get trade-off score, default to 0.5 if not present
                score = solution.trade_offs.get(dim, 0.5)
                row.append(score)
            matrix.append(row)

        matrix_np = np.array(matrix)

        # Step 1: Normalize matrix using vector normalization
        # Calculate column norms (avoiding division by zero)
        col_norms = np.sqrt((matrix_np ** 2).sum(axis=0))
        col_norms = np.where(col_norms == 0, 1e-10, col_norms)  # Prevent division by zero

        normalized_matrix = matrix_np / col_norms

        # Step 2: Apply weights
        weight_array = np.array([weights.get(dim, 1.0 / len(dimensions)) for dim in dimensions])
        weighted_matrix = normalized_matrix * weight_array

        # Step 3: Determine ideal and negative-ideal solutions
        # For all dimensions, higher is better (already normalized to 0-1)
        ideal = weighted_matrix.max(axis=0)
        negative_ideal = weighted_matrix.min(axis=0)

        # Step 4: Calculate distances to ideal and negative-ideal
        dist_to_ideal = np.sqrt(((weighted_matrix - ideal) ** 2).sum(axis=1))
        dist_to_negative = np.sqrt(((weighted_matrix - negative_ideal) ** 2).sum(axis=1))

        # Step 5: Calculate TOPSIS scores
        # Avoid division by zero
        denominator = dist_to_ideal + dist_to_negative
        denominator = np.where(denominator == 0, 1e-10, denominator)

        scores = dist_to_negative / denominator

        # Step 6: Rank solutions (higher score = better)
        ranking = np.argsort(scores)[::-1].tolist()  # Descending order
        best_solution_index = int(np.argmax(scores))

        self.logger.info(
            f"Trade-off analysis complete: Best solution index {best_solution_index} "
            f"with score {scores[best_solution_index]:.3f}"
        )

        return TradeOffAnalysisResult(
            scores=scores.tolist(),
            ranking=ranking,
            best_solution_index=best_solution_index,
            trade_off_matrix=matrix,
            weights_used=weights,
            normalized_matrix=normalized_matrix.tolist(),
        )

    def analyze_sensitivity(
        self,
        solutions: List[Solution],
        dimension: str,
        weight_range: tuple = (0.0, 1.0),
        steps: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform sensitivity analysis by varying weight of a dimension.

        Args:
            solutions: List of solutions
            dimension: Dimension to vary
            weight_range: Range of weights to test (min, max)
            steps: Number of steps in the range

        Returns:
            List of analysis results for each weight value
        """
        results = []

        # Get all dimensions
        dimensions = [dim.value for dim in TradeOffDimension]

        # Create weight range
        weights_to_test = np.linspace(weight_range[0], weight_range[1], steps)

        for weight in weights_to_test:
            # Distribute remaining weight equally among other dimensions
            remaining_weight = 1.0 - weight
            other_dims = [d for d in dimensions if d != dimension]
            other_weight = remaining_weight / len(other_dims) if other_dims else 0

            # Build weights dict
            weights = {dim: other_weight if dim != dimension else weight for dim in dimensions}

            # Analyze with these weights
            analysis = self.analyze_trade_offs(solutions, weights)

            results.append({
                "weight": float(weight),
                "dimension": dimension,
                "best_solution_index": analysis.best_solution_index,
                "scores": analysis.scores,
                "ranking": analysis.ranking,
            })

        return results

    def compare_solutions(
        self,
        solution1: Solution,
        solution2: Solution,
    ) -> Dict[str, Any]:
        """
        Compare two solutions across all dimensions.

        Args:
            solution1: First solution
            solution2: Second solution

        Returns:
            Comparison results
        """
        dimensions = [dim.value for dim in TradeOffDimension]

        comparison = {
            "solution1_name": solution1.name,
            "solution2_name": solution2.name,
            "dimension_comparison": {},
            "winner_by_dimension": {},
            "overall_winner": None,
        }

        solution1_wins = 0
        solution2_wins = 0

        for dim in dimensions:
            score1 = solution1.trade_offs.get(dim, 0.5)
            score2 = solution2.trade_offs.get(dim, 0.5)

            comparison["dimension_comparison"][dim] = {
                "solution1_score": score1,
                "solution2_score": score2,
                "difference": score1 - score2,
            }

            if score1 > score2:
                comparison["winner_by_dimension"][dim] = solution1.name
                solution1_wins += 1
            elif score2 > score1:
                comparison["winner_by_dimension"][dim] = solution2.name
                solution2_wins += 1
            else:
                comparison["winner_by_dimension"][dim] = "tie"

        # Determine overall winner by counting wins
        if solution1_wins > solution2_wins:
            comparison["overall_winner"] = solution1.name
        elif solution2_wins > solution1_wins:
            comparison["overall_winner"] = solution2.name
        else:
            comparison["overall_winner"] = "tie"

        comparison["solution1_total_wins"] = solution1_wins
        comparison["solution2_total_wins"] = solution2_wins

        return comparison

    def identify_pareto_optimal(
        self,
        solutions: List[Solution],
    ) -> List[int]:
        """
        Identify Pareto-optimal solutions (not dominated by any other solution).

        A solution is Pareto-optimal if no other solution is better in all dimensions.

        Args:
            solutions: List of solutions

        Returns:
            Indices of Pareto-optimal solutions
        """
        dimensions = [dim.value for dim in TradeOffDimension]
        pareto_optimal = []

        for i, solution in enumerate(solutions):
            is_dominated = False

            # Check if this solution is dominated by any other
            for j, other in enumerate(solutions):
                if i == j:
                    continue

                # Check if other dominates solution
                # (better or equal in all dimensions, better in at least one)
                better_count = 0
                worse_count = 0

                for dim in dimensions:
                    score = solution.trade_offs.get(dim, 0.5)
                    other_score = other.trade_offs.get(dim, 0.5)

                    if other_score > score:
                        better_count += 1
                    elif other_score < score:
                        worse_count += 1

                # If other is better in all dimensions, solution is dominated
                if better_count > 0 and worse_count == 0:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_optimal.append(i)

        self.logger.info(f"Found {len(pareto_optimal)} Pareto-optimal solutions")

        return pareto_optimal

    def explain_decision(
        self,
        solutions: List[Solution],
        analysis_result: TradeOffAnalysisResult,
        top_k: int = 3,
    ) -> str:
        """
        Generate human-readable explanation of the decision.

        Args:
            solutions: List of solutions
            analysis_result: Analysis result
            top_k: Number of top solutions to explain

        Returns:
            Explanation text
        """
        explanation = "# Trade-off Analysis Results\n\n"

        # Top solutions
        explanation += f"## Top {top_k} Solutions\n\n"

        for rank, idx in enumerate(analysis_result.ranking[:top_k]):
            solution = solutions[idx]
            score = analysis_result.scores[idx]

            explanation += f"### {rank + 1}. {solution.name} (Score: {score:.3f})\n\n"
            explanation += f"{solution.description}\n\n"

            # Show trade-offs
            explanation += "**Trade-offs:**\n"
            dimensions = [dim.value for dim in TradeOffDimension]
            for dim in dimensions:
                dim_score = solution.trade_offs.get(dim, 0.5)
                bar_length = int(dim_score * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                explanation += f"- {dim.title()}: {bar} ({dim_score:.2f})\n"

            explanation += "\n"

        # Weights used
        explanation += "## Weights Used\n\n"
        for dim, weight in analysis_result.weights_used.items():
            explanation += f"- {dim.title()}: {weight:.2f}\n"

        explanation += "\n## Decision Rationale\n\n"

        best_idx = analysis_result.best_solution_index
        best_solution = solutions[best_idx]

        explanation += f"The recommended solution is **{best_solution.name}** because:\n\n"

        # Identify strongest dimensions
        dimensions = [dim.value for dim in TradeOffDimension]
        strong_dims = []
        for dim in dimensions:
            score = best_solution.trade_offs.get(dim, 0.5)
            if score >= 0.7:
                strong_dims.append((dim, score))

        if strong_dims:
            explanation += "**Strengths:**\n"
            for dim, score in sorted(strong_dims, key=lambda x: x[1], reverse=True):
                explanation += f"- High {dim} score ({score:.2f})\n"
            explanation += "\n"

        # Show advantages
        if best_solution.advantages:
            explanation += "**Key Advantages:**\n"
            for adv in best_solution.advantages:
                explanation += f"- {adv}\n"
            explanation += "\n"

        return explanation
