"""Decision documentation using Architecture Decision Records (ADR) format."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..models.planning_models import Solution, DecisionRecord, PlanningContext, TradeOffAnalysisResult

logger = logging.getLogger(__name__)


class DecisionDocumenter:
    """
    Generate and manage Architecture Decision Records.

    PATTERN: ADR standard format with full context and rationale
    CRITICAL: Maintain traceability and version history
    GOTCHA: Ensure decisions are immutable once accepted
    """

    def __init__(self):
        """Initialize decision documenter."""
        self.logger = logging.getLogger(__name__)
        self.decisions: Dict[str, DecisionRecord] = {}

    def create_adr(
        self,
        title: str,
        solutions: List[Solution],
        selected_solution: Solution,
        trade_off_analysis: TradeOffAnalysisResult,
        context: PlanningContext,
        decision_context: Optional[str] = None,
    ) -> DecisionRecord:
        """
        Create Architecture Decision Record.

        PATTERN: Standard ADR format with full context
        CRITICAL: Capture all alternatives and rationale

        Args:
            title: Decision title
            solutions: All solutions considered
            selected_solution: Selected solution
            trade_off_analysis: Trade-off analysis results
            context: Planning context
            decision_context: Optional additional context

        Returns:
            Decision record
        """
        # Build context description
        context_text = self._build_context_section(context, decision_context)

        # Build decision text
        decision_text = self._build_decision_section(selected_solution)

        # Build consequences text
        consequences_text = self._build_consequences_section(selected_solution)

        # Build selection rationale
        rationale = self._build_rationale_section(
            selected_solution,
            solutions,
            trade_off_analysis,
        )

        # Convert solutions to dict for JSON serialization
        alternatives_dict = [self._solution_to_dict(s) for s in solutions]

        # Create decision record
        decision = DecisionRecord(
            title=title,
            status="accepted",
            context=context_text,
            decision=decision_text,
            consequences=consequences_text,
            alternatives_considered=alternatives_dict,
            selection_rationale=rationale,
            trade_off_analysis={
                "scores": trade_off_analysis.scores,
                "ranking": trade_off_analysis.ranking,
                "best_solution_index": trade_off_analysis.best_solution_index,
                "weights_used": trade_off_analysis.weights_used,
            },
        )

        # Store decision
        self.decisions[decision.id] = decision

        self.logger.info(f"Created ADR: {title} (ID: {decision.id})")

        return decision

    def _build_context_section(
        self,
        context: PlanningContext,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Build context section of ADR.

        Args:
            context: Planning context
            additional_context: Optional additional context

        Returns:
            Context section text
        """
        sections = []

        if additional_context:
            sections.append(f"{additional_context}\n")

        # Project constraints
        if any([context.budget_limit, context.timeline_weeks, context.team_capabilities]):
            sections.append("## Project Constraints\n")

            if context.budget_limit:
                sections.append(f"- **Budget**: ${context.budget_limit:,.2f}")

            if context.timeline_weeks:
                sections.append(f"- **Timeline**: {context.timeline_weeks} weeks")

            if context.team_capabilities:
                sections.append(
                    f"- **Team Capabilities**: {', '.join(context.team_capabilities)}"
                )

            sections.append("")

        # Quality requirements
        if context.quality_requirements:
            sections.append("## Quality Requirements\n")
            for requirement, threshold in context.quality_requirements.items():
                sections.append(f"- **{requirement.title()}**: {threshold}")
            sections.append("")

        # Regulatory requirements
        if context.regulatory_requirements:
            sections.append("## Regulatory Requirements\n")
            for req in context.regulatory_requirements:
                sections.append(f"- {req}")
            sections.append("")

        # Existing architecture
        if context.existing_architecture:
            sections.append("## Current Architecture\n")
            sections.append("Existing system that must be considered:")
            for key, value in context.existing_architecture.items():
                sections.append(f"- **{key}**: {value}")
            sections.append("")

        return "\n".join(sections)

    def _build_decision_section(self, solution: Solution) -> str:
        """
        Build decision section of ADR.

        Args:
            solution: Selected solution

        Returns:
            Decision section text
        """
        sections = [
            f"We will implement: **{solution.name}**\n",
            f"{solution.description}\n",
            f"**Approach**: {solution.approach.value.title()}\n",
        ]

        # Components
        if solution.components:
            sections.append("## Key Components\n")
            for component in solution.components:
                sections.append(f"- {component}")
            sections.append("")

        # Technologies
        if solution.technologies:
            sections.append("## Technologies\n")
            sections.append(", ".join(solution.technologies))
            sections.append("")

        # Dependencies
        if solution.dependencies:
            sections.append("## Dependencies\n")
            for dep in solution.dependencies:
                sections.append(f"- {dep}")
            sections.append("")

        # Estimates
        sections.append("## Estimates\n")
        sections.append(f"- **Effort**: {solution.estimated_effort_hours:.1f} hours")
        sections.append(f"- **Cost**: ${solution.estimated_cost:,.2f}")
        sections.append(f"- **Risk Level**: {solution.risk_level:.2f}")
        sections.append("")

        return "\n".join(sections)

    def _build_consequences_section(self, solution: Solution) -> str:
        """
        Build consequences section of ADR.

        Args:
            solution: Selected solution

        Returns:
            Consequences section text
        """
        sections = []

        # Positive consequences
        if solution.advantages:
            sections.append("## Positive Consequences\n")
            for advantage in solution.advantages:
                sections.append(f"- {advantage}")
            sections.append("")

        # Negative consequences
        if solution.disadvantages:
            sections.append("## Negative Consequences\n")
            for disadvantage in solution.disadvantages:
                sections.append(f"- {disadvantage}")
            sections.append("")

        # Risks and assumptions
        if solution.assumptions:
            sections.append("## Assumptions and Risks\n")
            sections.append(f"**Risk Level**: {solution.risk_level:.2f}\n")
            sections.append("**Key Assumptions**:")
            for assumption in solution.assumptions:
                sections.append(f"- {assumption}")
            sections.append("")

        return "\n".join(sections)

    def _build_rationale_section(
        self,
        selected_solution: Solution,
        all_solutions: List[Solution],
        analysis: TradeOffAnalysisResult,
    ) -> str:
        """
        Build rationale section explaining why this solution was chosen.

        Args:
            selected_solution: Selected solution
            all_solutions: All solutions considered
            analysis: Trade-off analysis results

        Returns:
            Rationale text
        """
        sections = []

        # TOPSIS score
        selected_idx = next(
            (i for i, s in enumerate(all_solutions) if s.id == selected_solution.id),
            analysis.best_solution_index,
        )

        topsis_score = analysis.scores[selected_idx]

        sections.append("This solution was selected based on comprehensive trade-off analysis.\n")
        sections.append(f"**TOPSIS Score**: {topsis_score:.3f}\n")

        # Key factors
        sections.append("## Key Selection Factors\n")

        # Identify strongest dimensions
        strong_dimensions = []
        for dim, score in selected_solution.trade_offs.items():
            if score >= 0.7:
                strong_dimensions.append((dim, score))

        if strong_dimensions:
            sections.append("**Strengths in Critical Dimensions**:")
            for dim, score in sorted(strong_dimensions, key=lambda x: x[1], reverse=True):
                sections.append(f"- {dim.title()}: {score:.2f}")
            sections.append("")

        # Weights consideration
        sections.append("**Decision Weights Applied**:")
        for dim, weight in sorted(
            analysis.weights_used.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]:  # Top 5 weights
            sections.append(f"- {dim.title()}: {weight:.2f}")
        sections.append("")

        # Comparison with other solutions
        if len(all_solutions) > 1:
            sections.append("## Comparison with Alternatives\n")

            for rank, idx in enumerate(analysis.ranking[:3]):  # Top 3
                if idx == selected_idx:
                    continue  # Skip selected solution

                other_solution = all_solutions[idx]
                other_score = analysis.scores[idx]

                sections.append(
                    f"**{other_solution.name}** (Score: {other_score:.3f}): "
                    f"{other_solution.description[:100]}..."
                )

            sections.append("")

        # Alignment with constraints
        sections.append("## Alignment with Project Constraints\n")
        sections.append("This solution best balances:")
        sections.append("- Project timeline and budget constraints")
        sections.append("- Team capabilities and learning curve")
        sections.append("- Quality and performance requirements")
        sections.append("- Risk tolerance and mitigation strategies")

        return "\n".join(sections)

    def _solution_to_dict(self, solution: Solution) -> Dict[str, Any]:
        """
        Convert solution to dict for JSON serialization.

        Args:
            solution: Solution to convert

        Returns:
            Dictionary representation
        """
        return {
            "id": solution.id,
            "name": solution.name,
            "description": solution.description,
            "approach": solution.approach.value,
            "components": solution.components,
            "technologies": solution.technologies,
            "estimated_effort_hours": solution.estimated_effort_hours,
            "estimated_cost": solution.estimated_cost,
            "advantages": solution.advantages,
            "disadvantages": solution.disadvantages,
            "confidence_score": solution.confidence_score,
        }

    def update_decision_status(
        self,
        decision_id: str,
        new_status: str,
        reason: Optional[str] = None,
    ) -> DecisionRecord:
        """
        Update decision status.

        Args:
            decision_id: Decision ID
            new_status: New status (proposed|accepted|deprecated|superseded)
            reason: Optional reason for status change

        Returns:
            Updated decision record

        Raises:
            ValueError: If decision not found
        """
        if decision_id not in self.decisions:
            raise ValueError(f"Decision {decision_id} not found")

        decision = self.decisions[decision_id]
        old_status = decision.status

        decision.status = new_status
        decision.updated_at = datetime.now()

        if reason:
            decision.context += f"\n\n**Status Update ({datetime.now().isoformat()})**\n"
            decision.context += f"Status changed from {old_status} to {new_status}\n"
            decision.context += f"Reason: {reason}\n"

        self.logger.info(
            f"Updated decision {decision_id} status: {old_status} -> {new_status}"
        )

        return decision

    def supersede_decision(
        self,
        old_decision_id: str,
        new_decision: DecisionRecord,
    ) -> DecisionRecord:
        """
        Supersede an old decision with a new one.

        Args:
            old_decision_id: ID of decision to supersede
            new_decision: New decision

        Returns:
            New decision record

        Raises:
            ValueError: If old decision not found
        """
        if old_decision_id not in self.decisions:
            raise ValueError(f"Decision {old_decision_id} not found")

        # Mark old decision as superseded
        self.update_decision_status(
            old_decision_id,
            "superseded",
            f"Superseded by decision {new_decision.id}",
        )

        # Link new decision to old
        new_decision.supersedes = old_decision_id
        new_decision.related_decisions.append(old_decision_id)

        # Store new decision
        self.decisions[new_decision.id] = new_decision

        self.logger.info(f"Decision {old_decision_id} superseded by {new_decision.id}")

        return new_decision

    def get_decision_history(self, decision_id: str) -> List[DecisionRecord]:
        """
        Get full history of a decision including superseded decisions.

        Args:
            decision_id: Decision ID

        Returns:
            List of decisions in chronological order
        """
        if decision_id not in self.decisions:
            raise ValueError(f"Decision {decision_id} not found")

        history = []
        current_id = decision_id

        # Follow supersedes chain backwards
        while current_id:
            decision = self.decisions.get(current_id)
            if not decision:
                break

            history.append(decision)
            current_id = decision.supersedes

        # Reverse to get chronological order
        history.reverse()

        return history

    def export_to_markdown(self, decision: DecisionRecord) -> str:
        """
        Export decision to Markdown format.

        Args:
            decision: Decision record

        Returns:
            Markdown text
        """
        sections = [
            f"# {decision.title}\n",
            f"**Status**: {decision.status.upper()}",
            f"**Date**: {decision.created_at.strftime('%Y-%m-%d')}",
            f"**Author**: {decision.author}\n",
        ]

        if decision.supersedes:
            sections.append(f"**Supersedes**: {decision.supersedes}\n")

        if decision.related_decisions:
            sections.append(
                f"**Related Decisions**: {', '.join(decision.related_decisions)}\n"
            )

        sections.extend([
            "## Context\n",
            decision.context,
            "\n## Decision\n",
            decision.decision,
            "\n## Consequences\n",
            decision.consequences,
            "\n## Rationale\n",
            decision.selection_rationale,
        ])

        return "\n".join(sections)

    def get_all_decisions(self) -> List[DecisionRecord]:
        """
        Get all decisions.

        Returns:
            List of all decision records
        """
        return list(self.decisions.values())

    def get_active_decisions(self) -> List[DecisionRecord]:
        """
        Get all active (accepted) decisions.

        Returns:
            List of active decision records
        """
        return [d for d in self.decisions.values() if d.status == "accepted"]
