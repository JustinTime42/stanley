"""Technology evaluation and selection based on requirements and constraints."""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..models.architecture_models import Technology, TechnologyEvaluation
from ..models.planning_models import PlanningContext

logger = logging.getLogger(__name__)


class TechnologySelector:
    """
    Evaluate and select technologies based on requirements.

    PATTERN: Multi-criteria evaluation with filtering
    CRITICAL: Consider constraints and compatibility
    GOTCHA: Technology database may be incomplete, handle gracefully
    """

    def __init__(
        self,
        technology_db_file: Optional[str] = None,
        trade_off_analyzer=None,
    ):
        """
        Initialize technology selector.

        Args:
            technology_db_file: Path to technology database JSON
            trade_off_analyzer: Trade-off analyzer for scoring
        """
        self.logger = logging.getLogger(__name__)
        self.trade_off_analyzer = trade_off_analyzer
        self.technologies: Dict[str, Technology] = {}

        # Default technology database location
        if technology_db_file is None:
            technology_db_file = str(
                Path(__file__).parent.parent / "data" / "technology_database.json"
            )

        self.technology_db_file = technology_db_file
        self._load_technologies()

    def _load_technologies(self) -> None:
        """Load technologies from JSON file."""
        try:
            with open(self.technology_db_file, "r") as f:
                data = json.load(f)

            technologies_data = data.get("technologies", [])

            for tech_data in technologies_data:
                tech = Technology(**tech_data)
                self.technologies[tech.name] = tech

            self.logger.info(
                f"Loaded {len(self.technologies)} technologies from {self.technology_db_file}"
            )

        except FileNotFoundError:
            self.logger.warning(
                f"Technology database not found: {self.technology_db_file}, using defaults"
            )
            self._load_default_technologies()

        except Exception as e:
            self.logger.error(f"Failed to load technologies: {e}, using defaults")
            self._load_default_technologies()

    def _load_default_technologies(self) -> None:
        """Load default technology catalog."""
        default_techs = [
            Technology(
                name="Python",
                category="language",
                license="PSF",
                maturity="adopt",
                learning_curve=0.3,
                community_size="large",
                documentation_quality=0.9,
                suitability_scores={
                    "web_development": 0.9,
                    "data_science": 0.95,
                    "automation": 0.9,
                    "performance": 0.6,
                },
                compatible_with=["FastAPI", "Django", "Flask", "PostgreSQL"],
            ),
            Technology(
                name="FastAPI",
                category="web_framework",
                license="MIT",
                maturity="adopt",
                learning_curve=0.4,
                community_size="large",
                documentation_quality=0.9,
                suitability_scores={
                    "api_development": 0.95,
                    "async_processing": 0.9,
                    "microservices": 0.85,
                },
                compatible_with=["Python", "PostgreSQL", "Redis"],
            ),
            Technology(
                name="PostgreSQL",
                category="database",
                license="PostgreSQL",
                maturity="adopt",
                learning_curve=0.4,
                community_size="large",
                documentation_quality=0.85,
                suitability_scores={
                    "relational_data": 0.95,
                    "scalability": 0.8,
                    "reliability": 0.9,
                },
                compatible_with=["Python", "FastAPI", "Django"],
            ),
            Technology(
                name="Redis",
                category="cache",
                license="BSD",
                maturity="adopt",
                learning_curve=0.3,
                community_size="large",
                documentation_quality=0.8,
                suitability_scores={
                    "caching": 0.95,
                    "real_time": 0.9,
                    "scalability": 0.85,
                },
                compatible_with=["Python", "FastAPI"],
            ),
            Technology(
                name="Docker",
                category="containerization",
                license="Apache 2.0",
                maturity="adopt",
                learning_curve=0.5,
                community_size="large",
                documentation_quality=0.85,
                suitability_scores={
                    "deployment": 0.9,
                    "microservices": 0.95,
                    "devops": 0.9,
                },
                compatible_with=["Kubernetes", "Python", "FastAPI"],
            ),
        ]

        for tech in default_techs:
            self.technologies[tech.name] = tech

        self.logger.info(f"Loaded {len(default_techs)} default technologies")

    async def recommend_stack(
        self,
        requirements: Dict[str, Any],
        context: PlanningContext,
        max_recommendations: int = 10,
    ) -> List[TechnologyEvaluation]:
        """
        Recommend technology stack based on requirements.

        PATTERN: Multi-criteria evaluation with filtering
        CRITICAL: Check compatibility of selections

        Args:
            requirements: System requirements
            context: Planning context with constraints
            max_recommendations: Maximum number of recommendations

        Returns:
            List of technology evaluations, sorted by score
        """
        evaluations = []

        # Get candidate technologies
        candidates = self._filter_candidates(requirements, context)

        # Evaluate each technology
        for tech in candidates:
            evaluation = await self._evaluate_technology(
                tech,
                requirements,
                context,
            )
            evaluations.append(evaluation)

        # Sort by score
        evaluations.sort(key=lambda e: e.score, reverse=True)

        # Take top N
        top_evaluations = evaluations[:max_recommendations]

        # Check compatibility of top choices
        compatible_stack = self._ensure_compatibility(top_evaluations)

        self.logger.info(
            f"Recommended {len(compatible_stack)} compatible technologies"
        )

        return compatible_stack

    def _filter_candidates(
        self,
        requirements: Dict[str, Any],
        context: PlanningContext,
    ) -> List[Technology]:
        """
        Filter technology candidates based on requirements.

        Args:
            requirements: System requirements
            context: Planning context

        Returns:
            List of candidate technologies
        """
        candidates = []

        for tech in self.technologies.values():
            # Filter by maturity
            if tech.maturity == "hold":
                continue  # Skip technologies on hold

            # Filter by team capabilities if specified
            if context.team_capabilities:
                # If team doesn't have experience, prefer lower learning curve
                if tech.name not in context.team_capabilities:
                    if tech.learning_curve > 0.7:
                        continue  # Skip high learning curve technologies

            # Filter by licensing constraints
            if context.regulatory_requirements:
                if "commercial" in str(context.regulatory_requirements).lower():
                    if "GPL" in tech.license:
                        continue  # Skip GPL for commercial projects

            candidates.append(tech)

        return candidates

    async def _evaluate_technology(
        self,
        technology: Technology,
        requirements: Dict[str, Any],
        context: PlanningContext,
    ) -> TechnologyEvaluation:
        """
        Evaluate single technology.

        PATTERN: Weighted scoring across multiple factors
        CRITICAL: Consider all constraints

        Args:
            technology: Technology to evaluate
            requirements: System requirements
            context: Planning context

        Returns:
            Technology evaluation
        """
        score = 0.0
        pros = []
        cons = []
        risks = []

        # Check team capability match
        if context.team_capabilities:
            if technology.name in context.team_capabilities:
                score += 0.2
                pros.append("Team has experience with this technology")
            elif technology.learning_curve < 0.3:
                score += 0.15
                pros.append("Low learning curve")
            else:
                cons.append(f"High learning curve: {technology.learning_curve:.2f}")
                risks.append("Team may need training")

        # Check maturity
        if technology.maturity == "adopt":
            score += 0.3
            pros.append("Mature and production-ready")
        elif technology.maturity == "trial":
            score += 0.2
            pros.append("Worth trying for new projects")
        elif technology.maturity == "assess":
            score += 0.1
            cons.append("Still being assessed, may have limited support")
            risks.append("Limited production track record")
        else:  # hold
            score = 0.0
            cons.append("Technology on hold, not recommended")

        # Check cost constraints
        if context.budget_limit:
            total_cost = technology.licensing_cost + technology.operational_cost

            # Assume 10% of budget can be allocated to technology costs
            tech_budget = context.budget_limit * 0.1

            if total_cost == 0:
                score += 0.2
                pros.append("Open source / free")
            elif total_cost <= tech_budget:
                score += 0.15
                pros.append(f"Within budget (${total_cost:,.2f})")
            else:
                cons.append(f"High cost: ${total_cost:,.2f}")
                risks.append("May exceed technology budget")

        # Check suitability for requirements
        requirements_text = str(requirements).lower()

        # Match suitability scores to requirements
        max_suitability_boost = 0.3
        suitability_matches = []

        for req_type, suitability_score in technology.suitability_scores.items():
            if req_type.replace("_", " ") in requirements_text:
                suitability_matches.append((req_type, suitability_score))

        if suitability_matches:
            # Calculate average suitability for matched requirements
            avg_suitability = sum(s for _, s in suitability_matches) / len(suitability_matches)
            score += avg_suitability * max_suitability_boost

            for req_type, suitability_score in sorted(
                suitability_matches, key=lambda x: x[1], reverse=True
            )[:3]:  # Top 3
                pros.append(f"Well-suited for {req_type.replace('_', ' ')} ({suitability_score:.2f})")

        # Check community and documentation
        if technology.community_size == "large":
            score += 0.1
            pros.append("Large community support")
        elif technology.community_size == "small":
            cons.append("Small community, limited support")
            risks.append("May be harder to find solutions to problems")

        if technology.documentation_quality >= 0.8:
            pros.append("Excellent documentation")
        elif technology.documentation_quality < 0.5:
            cons.append("Poor documentation quality")
            risks.append("May require significant research time")

        # Determine migration effort
        migration_effort = self._estimate_migration_effort(technology, context)

        # Determine recommendation
        recommendation = self._get_recommendation(score, technology.maturity)

        # Normalize score to 0-1 range
        score = min(max(score, 0.0), 1.0)

        return TechnologyEvaluation(
            technology=technology.model_dump(),
            score=score,
            pros=pros,
            cons=cons,
            risks=risks,
            migration_effort=migration_effort,
            recommendation=recommendation,
        )

    def _estimate_migration_effort(
        self,
        technology: Technology,
        context: PlanningContext,
    ) -> str:
        """
        Estimate migration effort.

        Args:
            technology: Technology to migrate to
            context: Planning context

        Returns:
            Migration effort level (low|medium|high)
        """
        # Check if existing architecture uses similar technologies
        if context.existing_architecture:
            existing_techs = str(context.existing_architecture).lower()

            # If technology already in use
            if technology.name.lower() in existing_techs:
                return "low"

            # If compatible technologies in use
            if any(compat.lower() in existing_techs for compat in technology.compatible_with):
                return "medium"

            # New technology with no compatibility
            return "high"

        # No existing architecture, new project
        return "low"

    def _get_recommendation(self, score: float, maturity: str) -> str:
        """
        Get recommendation based on score and maturity.

        Args:
            score: Evaluation score
            maturity: Technology maturity

        Returns:
            Recommendation (adopt|trial|assess|hold)
        """
        if score >= 0.7 and maturity == "adopt":
            return "adopt"
        elif score >= 0.5 and maturity in ["adopt", "trial"]:
            return "trial"
        elif score >= 0.3:
            return "assess"
        else:
            return "hold"

    def _ensure_compatibility(
        self,
        evaluations: List[TechnologyEvaluation],
    ) -> List[TechnologyEvaluation]:
        """
        Ensure selected technologies are compatible.

        PATTERN: Filter incompatible pairs, prefer compatible clusters
        CRITICAL: Don't recommend incompatible technologies

        Args:
            evaluations: List of technology evaluations

        Returns:
            Filtered list ensuring compatibility
        """
        compatible = []

        for evaluation in evaluations:
            tech_name = evaluation.technology["name"]
            incompatible_with = evaluation.technology.get("incompatible_with", [])

            # Check if compatible with already selected technologies
            is_compatible = True

            for already_selected in compatible:
                selected_name = already_selected.technology["name"]

                # Check if current tech is incompatible with selected
                if selected_name in incompatible_with:
                    is_compatible = False
                    break

                # Check if selected is incompatible with current
                selected_incompatible = already_selected.technology.get("incompatible_with", [])
                if tech_name in selected_incompatible:
                    is_compatible = False
                    break

            if is_compatible:
                compatible.append(evaluation)

        return compatible

    def get_technology(self, name: str) -> Optional[Technology]:
        """
        Get technology by name.

        Args:
            name: Technology name

        Returns:
            Technology or None
        """
        return self.technologies.get(name)

    def add_technology(self, technology: Technology) -> None:
        """
        Add technology to database.

        Args:
            technology: Technology to add
        """
        self.technologies[technology.name] = technology
        self.logger.info(f"Added technology: {technology.name}")

    def save_technologies(self) -> None:
        """Save technologies to file."""
        try:
            data = {
                "technologies": [t.model_dump() for t in self.technologies.values()]
            }

            # Ensure directory exists
            Path(self.technology_db_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self.technology_db_file, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.info(
                f"Saved {len(self.technologies)} technologies to {self.technology_db_file}"
            )

        except Exception as e:
            self.logger.error(f"Failed to save technologies: {e}")
