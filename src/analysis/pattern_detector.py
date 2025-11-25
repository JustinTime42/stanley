"""Code pattern detection."""

import logging
from typing import List, Optional, Callable, Dict

from ..models.analysis_models import (
    ASTNode,
    CodeEntity,
    Pattern,
    PatternType,
    NodeType,
)

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Detector for code patterns and anti-patterns.

    PATTERN: Registered pattern matchers
    CRITICAL: Balance between accuracy and performance
    """

    def __init__(self):
        """Initialize pattern detector."""
        self.logger = logger
        self.patterns: Dict[str, Callable] = {}
        self._register_patterns()

    def _register_patterns(self):
        """Register pattern detection functions."""
        # Design patterns
        self.patterns["singleton"] = self._detect_singleton
        self.patterns["factory"] = self._detect_factory
        self.patterns["observer"] = self._detect_observer

        # Anti-patterns
        self.patterns["god_class"] = self._detect_god_class
        self.patterns["long_method"] = self._detect_long_method
        self.patterns["magic_numbers"] = self._detect_magic_numbers

        # Code smells
        self.patterns["duplicate_code"] = self._detect_duplicate_code
        self.patterns["deep_nesting"] = self._detect_deep_nesting

    async def detect_patterns(
        self, entities: List[CodeEntity], ast: ASTNode
    ) -> List[Pattern]:
        """
        Detect all patterns in code.

        Args:
            entities: List of code entities
            ast: Parsed AST

        Returns:
            List of detected patterns
        """
        detected = []

        for entity in entities:
            for pattern_name, detector in self.patterns.items():
                try:
                    result = await detector(entity, ast)
                    if result and result.confidence > 0.5:
                        detected.append(result)
                except Exception as e:
                    self.logger.warning(f"Error detecting pattern {pattern_name}: {e}")

        return detected

    async def _detect_singleton(
        self, entity: CodeEntity, ast: ASTNode
    ) -> Optional[Pattern]:
        """
        Detect singleton pattern.

        PATTERN: Private constructor + static instance
        """
        if entity.type != NodeType.CLASS:
            return None

        # Simplified singleton detection based on naming convention
        # Full implementation would analyze AST for singleton characteristics
        confidence = 0.6 if entity.name.endswith("Singleton") else 0.5

        if confidence > 0.5:
            return Pattern(
                name="Singleton",
                type=PatternType.DESIGN_PATTERN,
                confidence=confidence,
                location=entity,
                description="Singleton pattern detected",
                recommendation="Ensure singleton is necessary; consider dependency injection",
            )

        return None

    async def _detect_factory(
        self, entity: CodeEntity, ast: ASTNode
    ) -> Optional[Pattern]:
        """Detect factory pattern."""
        if entity.type != NodeType.CLASS and entity.type != NodeType.FUNCTION:
            return None

        # Look for factory indicators
        if "create" in entity.name.lower() or "factory" in entity.name.lower():
            return Pattern(
                name="Factory",
                type=PatternType.DESIGN_PATTERN,
                confidence=0.7,
                location=entity,
                description="Factory pattern detected",
                recommendation="Good use of creational pattern",
            )

        return None

    async def _detect_observer(
        self, entity: CodeEntity, ast: ASTNode
    ) -> Optional[Pattern]:
        """Detect observer pattern."""
        if entity.type != NodeType.CLASS:
            return None

        # Look for observer indicators
        name_lower = entity.name.lower()
        if "observer" in name_lower or "listener" in name_lower:
            return Pattern(
                name="Observer",
                type=PatternType.DESIGN_PATTERN,
                confidence=0.75,
                location=entity,
                description="Observer pattern detected",
                recommendation="Good use of behavioral pattern",
            )

        return None

    async def _detect_god_class(
        self, entity: CodeEntity, ast: ASTNode
    ) -> Optional[Pattern]:
        """
        Detect god class anti-pattern.

        GOTCHA: Class doing too many things
        """
        if entity.type != NodeType.CLASS:
            return None

        # Heuristic: Very large classes are likely god classes
        lines = entity.line_end - entity.line_start

        if lines > 500:
            confidence = min(0.9, 0.5 + (lines - 500) / 1000)

            return Pattern(
                name="God Class",
                type=PatternType.ANTI_PATTERN,
                confidence=confidence,
                location=entity,
                description=f"Very large class with {lines} lines",
                recommendation="Consider breaking into smaller, focused classes",
            )

        return None

    async def _detect_long_method(
        self, entity: CodeEntity, ast: ASTNode
    ) -> Optional[Pattern]:
        """Detect long method anti-pattern."""
        if entity.type not in {NodeType.FUNCTION, NodeType.METHOD}:
            return None

        lines = entity.line_end - entity.line_start

        if lines > 100:
            confidence = min(0.9, 0.5 + (lines - 100) / 200)

            return Pattern(
                name="Long Method",
                type=PatternType.ANTI_PATTERN,
                confidence=confidence,
                location=entity,
                description=f"Method is {lines} lines long",
                recommendation="Consider extracting smaller methods",
            )

        return None

    async def _detect_magic_numbers(
        self, entity: CodeEntity, ast: ASTNode
    ) -> Optional[Pattern]:
        """Detect magic numbers code smell."""
        # Count number literals in entity
        number_count = self._count_number_literals(ast)

        if number_count > 5:
            return Pattern(
                name="Magic Numbers",
                type=PatternType.CODE_SMELL,
                confidence=0.7,
                location=entity,
                description=f"Found {number_count} magic numbers",
                recommendation="Extract numbers as named constants",
            )

        return None

    async def _detect_duplicate_code(
        self, entity: CodeEntity, ast: ASTNode
    ) -> Optional[Pattern]:
        """Detect duplicate code (simplified)."""
        # This would require comparing with other entities
        # Simplified implementation for now
        return None

    async def _detect_deep_nesting(
        self, entity: CodeEntity, ast: ASTNode
    ) -> Optional[Pattern]:
        """Detect deep nesting code smell."""
        max_depth = self._calculate_max_depth(ast)

        if max_depth > 5:
            return Pattern(
                name="Deep Nesting",
                type=PatternType.CODE_SMELL,
                confidence=0.8,
                location=entity,
                description=f"Maximum nesting depth of {max_depth}",
                recommendation="Reduce nesting with early returns or extraction",
            )

        return None

    def _count_number_literals(self, ast: ASTNode) -> int:
        """Count number literals in AST."""
        count = 0

        def visit(node: ASTNode):
            nonlocal count
            if node.node_type in {"number", "integer", "float"}:
                count += 1

            for child in node.children:
                visit(child)

        visit(ast)
        return count

    def _calculate_max_depth(self, ast: ASTNode, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = current_depth

        nesting_types = {
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
        }

        for child in ast.children:
            if child.node_type in nesting_types:
                child_depth = self._calculate_max_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_max_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)

        return max_depth
