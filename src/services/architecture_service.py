"""Architecture orchestrator service integrating pattern recognition and technology selection."""

import logging
from typing import List, Dict, Any, Optional
from ..architecture import (
    PatternLibrary,
    PatternRecognizer,
    TechnologySelector,
    ConsistencyChecker,
)
from ..models.architecture_models import (
    ArchitectureDesign,
    PatternMatch,
    TechnologyEvaluation,
)
from ..models.planning_models import PlanningContext

logger = logging.getLogger(__name__)


class ArchitectureOrchestrator:
    """
    High-level architecture service facade.

    PATTERN: Service facade pattern integrating all architecture components
    CRITICAL: Single entry point for architecture operations
    GOTCHA: Components may not all be available, handle gracefully
    """

    def __init__(
        self,
        ast_parser=None,
        trade_off_analyzer=None,
    ):
        """
        Initialize architecture orchestrator.

        Args:
            ast_parser: AST parser for code analysis
            trade_off_analyzer: Trade-off analyzer for technology scoring
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.pattern_library = PatternLibrary()
        self.pattern_recognizer = PatternRecognizer(
            ast_parser=ast_parser,
            pattern_library=self.pattern_library,
        )
        self.technology_selector = TechnologySelector(
            trade_off_analyzer=trade_off_analyzer,
        )
        self.consistency_checker = ConsistencyChecker(
            pattern_library=self.pattern_library,
        )

        self.logger.info("Architecture orchestrator initialized")

    async def design_system(
        self,
        requirements: Dict[str, Any],
        context: PlanningContext,
        codebase_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Design system architecture.

        PATTERN: Match patterns -> Select technologies -> Create design -> Validate
        CRITICAL: Comprehensive architecture design

        Args:
            requirements: System requirements
            context: Planning context
            codebase_path: Optional existing codebase to analyze

        Returns:
            Architecture design result
        """
        self.logger.info("Designing system architecture...")

        # Step 1: Match suitable patterns
        constraints = {
            "budget": context.budget_limit,
            "timeline": context.timeline_weeks,
            "team_size": len(context.team_capabilities) if context.team_capabilities else 5,
        }

        suitable_patterns = self.pattern_library.match_context_to_patterns(
            requirements=requirements,
            constraints=constraints,
        )

        selected_patterns = [p[0].pattern_type for p in suitable_patterns[:3]]  # Top 3

        self.logger.info(f"Selected patterns: {selected_patterns}")

        # Step 2: Detect existing patterns if codebase provided
        detected_patterns = []
        if codebase_path:
            detected_patterns = await self.pattern_recognizer.recognize_patterns(
                codebase_path=codebase_path,
            )
            self.logger.info(f"Detected {len(detected_patterns)} existing patterns")

        # Step 3: Recommend technologies
        tech_evaluations = await self.technology_selector.recommend_stack(
            requirements=requirements,
            context=context,
        )

        selected_technologies = {
            eval.technology["name"]: eval.technology
            for eval in tech_evaluations[:10]  # Top 10
        }

        self.logger.info(f"Selected {len(selected_technologies)} technologies")

        # Step 4: Create architecture design
        design = ArchitectureDesign(
            name=requirements.get("name", "System Architecture"),
            description=requirements.get("description", "Architecture design"),
            patterns=selected_patterns,
            technologies=selected_technologies,
        )

        # Step 5: Validate consistency
        consistency_result = await self.consistency_checker.check_consistency(
            design=design,
            detected_patterns=detected_patterns,
        )

        design.consistency_score = consistency_result["consistency_score"]
        design.completeness_score = consistency_result["completeness_score"]

        self.logger.info(
            f"Architecture design complete: "
            f"consistency={design.consistency_score:.2f}, "
            f"completeness={design.completeness_score:.2f}"
        )

        return {
            "design": design,
            "suitable_patterns": suitable_patterns,
            "detected_patterns": detected_patterns,
            "technology_evaluations": tech_evaluations,
            "consistency_result": consistency_result,
        }

    async def validate_architecture(
        self,
        design: ArchitectureDesign,
        codebase_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate architecture design.

        Args:
            design: Architecture design
            codebase_path: Optional codebase to validate against

        Returns:
            Validation results
        """
        # Detect patterns if codebase provided
        detected_patterns = None
        if codebase_path:
            detected_patterns = await self.pattern_recognizer.recognize_patterns(
                codebase_path=codebase_path,
            )

        # Check consistency
        consistency_result = await self.consistency_checker.check_consistency(
            design=design,
            detected_patterns=detected_patterns,
        )

        return {
            "consistency_result": consistency_result,
            "detected_patterns": detected_patterns,
            "is_valid": (
                consistency_result["consistency_score"] >= 0.7
                and consistency_result["completeness_score"] >= 0.8
            ),
        }

    async def recognize_patterns(
        self,
        codebase_path: str,
        confidence_threshold: float = 0.7,
    ) -> List[PatternMatch]:
        """
        Recognize architecture patterns in codebase.

        Args:
            codebase_path: Path to codebase
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of detected patterns
        """
        return await self.pattern_recognizer.recognize_patterns(
            codebase_path=codebase_path,
            confidence_threshold=confidence_threshold,
        )

    async def recommend_technologies(
        self,
        requirements: Dict[str, Any],
        context: PlanningContext,
        max_recommendations: int = 10,
    ) -> List[TechnologyEvaluation]:
        """
        Recommend technologies.

        Args:
            requirements: System requirements
            context: Planning context
            max_recommendations: Maximum recommendations

        Returns:
            List of technology evaluations
        """
        return await self.technology_selector.recommend_stack(
            requirements=requirements,
            context=context,
            max_recommendations=max_recommendations,
        )

    def get_pattern_library(self) -> PatternLibrary:
        """Get pattern library."""
        return self.pattern_library

    def get_technology_selector(self) -> TechnologySelector:
        """Get technology selector."""
        return self.technology_selector
