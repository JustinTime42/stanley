"""Pattern library for managing architecture pattern catalog."""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..models.architecture_models import PatternDefinition, ArchitecturePattern

logger = logging.getLogger(__name__)


class PatternLibrary:
    """
    Repository for architecture pattern definitions.

    PATTERN: Repository pattern for pattern storage and retrieval
    CRITICAL: Load patterns from JSON, support pattern matching
    GOTCHA: Handle missing pattern files gracefully
    """

    def __init__(self, pattern_file: Optional[str] = None):
        """
        Initialize pattern library.

        Args:
            pattern_file: Path to pattern definitions JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.patterns: Dict[str, PatternDefinition] = {}

        # Default pattern file location
        if pattern_file is None:
            pattern_file = str(
                Path(__file__).parent.parent / "data" / "architecture_patterns.json"
            )

        self.pattern_file = pattern_file
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load patterns from JSON file."""
        try:
            with open(self.pattern_file, "r") as f:
                data = json.load(f)

            patterns_data = data.get("patterns", [])

            for pattern_data in patterns_data:
                pattern = PatternDefinition(**pattern_data)
                self.patterns[pattern.pattern_type] = pattern

            self.logger.info(f"Loaded {len(self.patterns)} patterns from {self.pattern_file}")

        except FileNotFoundError:
            self.logger.warning(f"Pattern file not found: {self.pattern_file}, using default patterns")
            self._load_default_patterns()

        except Exception as e:
            self.logger.error(f"Failed to load patterns: {e}, using default patterns")
            self._load_default_patterns()

    def _load_default_patterns(self) -> None:
        """Load default patterns if file not available."""
        # Define core patterns
        default_patterns = [
            PatternDefinition(
                name="Layered Architecture",
                pattern_type=ArchitecturePattern.LAYERED.value,
                description="Organize system into horizontal layers with clear separation of concerns",
                when_to_use=[
                    "Need clear separation between UI, business logic, and data",
                    "Team organized by technical expertise",
                    "Standard enterprise application",
                ],
                when_not_to_use=[
                    "Need high scalability",
                    "Microservices architecture required",
                ],
                structure={
                    "layers": ["presentation", "business", "data"],
                    "rules": ["Dependencies flow downward only", "Each layer has specific responsibilities"],
                },
                benefits=[
                    "Clear separation of concerns",
                    "Easy to understand and maintain",
                    "Team can specialize by layer",
                ],
                drawbacks=[
                    "Can lead to monolithic architecture",
                    "Performance overhead from layer crossings",
                ],
                examples=["Traditional web applications", "Enterprise systems"],
                related_patterns=[ArchitecturePattern.CLEAN.value, ArchitecturePattern.ONION.value],
            ),
            PatternDefinition(
                name="Microservices",
                pattern_type=ArchitecturePattern.MICROSERVICES.value,
                description="Decompose application into small, independent services",
                when_to_use=[
                    "Need independent deployment and scaling",
                    "Large team that can be organized into smaller units",
                    "Different parts require different technologies",
                ],
                when_not_to_use=[
                    "Small application or team",
                    "No need for independent scaling",
                    "Distributed systems expertise lacking",
                ],
                structure={
                    "components": ["independent services", "API gateway", "service discovery"],
                    "communication": ["REST APIs", "message queues", "events"],
                },
                benefits=[
                    "Independent deployment and scaling",
                    "Technology diversity",
                    "Team autonomy",
                ],
                drawbacks=[
                    "Operational complexity",
                    "Distributed system challenges",
                    "Testing complexity",
                ],
                examples=["Netflix", "Amazon", "Uber"],
                related_patterns=[ArchitecturePattern.EVENT_DRIVEN.value, ArchitecturePattern.SOA.value],
            ),
            PatternDefinition(
                name="Event-Driven Architecture",
                pattern_type=ArchitecturePattern.EVENT_DRIVEN.value,
                description="Components communicate through events and event handlers",
                when_to_use=[
                    "Loose coupling between components needed",
                    "Asynchronous processing required",
                    "Real-time data processing",
                ],
                when_not_to_use=[
                    "Simple CRUD operations",
                    "Strict consistency required",
                ],
                structure={
                    "components": ["event producers", "event bus", "event consumers"],
                    "patterns": ["pub-sub", "event sourcing"],
                },
                benefits=[
                    "Loose coupling",
                    "Scalability",
                    "Real-time processing",
                ],
                drawbacks=[
                    "Eventual consistency",
                    "Debugging difficulty",
                    "Event schema management",
                ],
                examples=["Real-time analytics", "IoT systems"],
                related_patterns=[ArchitecturePattern.EVENT_SOURCING.value, ArchitecturePattern.CQRS.value],
            ),
        ]

        for pattern in default_patterns:
            self.patterns[pattern.pattern_type] = pattern

        self.logger.info(f"Loaded {len(default_patterns)} default patterns")

    def get_pattern(self, pattern_type: str) -> Optional[PatternDefinition]:
        """
        Get pattern by type.

        Args:
            pattern_type: Pattern type

        Returns:
            Pattern definition or None
        """
        return self.patterns.get(pattern_type)

    def get_all_patterns(self) -> List[PatternDefinition]:
        """
        Get all patterns.

        Returns:
            List of all pattern definitions
        """
        return list(self.patterns.values())

    def match_context_to_patterns(
        self,
        requirements: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> List[tuple[PatternDefinition, float]]:
        """
        Match requirements and constraints to suitable patterns.

        PATTERN: Rule-based matching with scoring
        CRITICAL: Return patterns sorted by suitability

        Args:
            requirements: System requirements
            constraints: Design constraints

        Returns:
            List of (pattern, suitability_score) tuples, sorted by score
        """
        scored_patterns = []

        for pattern in self.patterns.values():
            score = self._calculate_pattern_suitability(
                pattern,
                requirements,
                constraints,
            )

            if score > 0.3:  # Only include patterns with reasonable suitability
                scored_patterns.append((pattern, score))

        # Sort by score descending
        scored_patterns.sort(key=lambda x: x[1], reverse=True)

        self.logger.info(f"Matched {len(scored_patterns)} suitable patterns")

        return scored_patterns

    def _calculate_pattern_suitability(
        self,
        pattern: PatternDefinition,
        requirements: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> float:
        """
        Calculate suitability score for a pattern.

        Args:
            pattern: Pattern definition
            requirements: System requirements
            constraints: Design constraints

        Returns:
            Suitability score (0-1)
        """
        score = 0.5  # Base score

        # Check if requirements mention pattern characteristics
        requirements_text = str(requirements).lower()

        # Microservices indicators
        if pattern.pattern_type == ArchitecturePattern.MICROSERVICES.value:
            if any(kw in requirements_text for kw in ["scalability", "independent", "distributed"]):
                score += 0.3
            if any(kw in requirements_text for kw in ["monolith", "simple", "small"]):
                score -= 0.3

        # Layered indicators
        elif pattern.pattern_type == ArchitecturePattern.LAYERED.value:
            if any(kw in requirements_text for kw in ["separation", "layers", "traditional"]):
                score += 0.3
            if any(kw in requirements_text for kw in ["microservice", "distributed"]):
                score -= 0.2

        # Event-driven indicators
        elif pattern.pattern_type == ArchitecturePattern.EVENT_DRIVEN.value:
            if any(kw in requirements_text for kw in ["event", "real-time", "async"]):
                score += 0.3
            if any(kw in requirements_text for kw in ["sync", "crud", "simple"]):
                score -= 0.2

        # Check constraints
        if constraints.get("team_size"):
            team_size = constraints["team_size"]

            # Microservices better for larger teams
            if pattern.pattern_type == ArchitecturePattern.MICROSERVICES.value:
                if team_size > 20:
                    score += 0.2
                elif team_size < 5:
                    score -= 0.2

            # Layered architecture good for medium teams
            elif pattern.pattern_type == ArchitecturePattern.LAYERED.value:
                if 5 <= team_size <= 20:
                    score += 0.2

        # Check complexity constraints
        if constraints.get("acceptable_complexity"):
            complexity = constraints["acceptable_complexity"]

            # Microservices high complexity
            if pattern.pattern_type == ArchitecturePattern.MICROSERVICES.value:
                if complexity == "low":
                    score -= 0.3
                elif complexity == "high":
                    score += 0.2

        return max(0, min(1, score))

    def add_pattern(self, pattern: PatternDefinition) -> None:
        """
        Add new pattern to library.

        Args:
            pattern: Pattern definition
        """
        self.patterns[pattern.pattern_type] = pattern
        self.logger.info(f"Added pattern: {pattern.name}")

    def save_patterns(self) -> None:
        """Save patterns to file."""
        try:
            data = {
                "patterns": [p.model_dump() for p in self.patterns.values()]
            }

            # Ensure directory exists
            Path(self.pattern_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self.pattern_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.info(f"Saved {len(self.patterns)} patterns to {self.pattern_file}")

        except Exception as e:
            self.logger.error(f"Failed to save patterns: {e}")
