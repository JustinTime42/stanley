"""Architecture consistency validation and checking."""

import logging
from typing import List, Dict, Any, Optional
from ..models.architecture_models import ArchitectureDesign, PatternMatch

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """
    Validate architecture consistency and adherence to patterns.

    PATTERN: Rule-based validation with scoring
    CRITICAL: Non-blocking warnings for inconsistencies
    GOTCHA: Don't fail builds on minor inconsistencies
    """

    def __init__(self, pattern_library=None):
        """
        Initialize consistency checker.

        Args:
            pattern_library: Pattern library for pattern rules
        """
        self.pattern_library = pattern_library
        self.logger = logging.getLogger(__name__)

    async def check_consistency(
        self,
        design: ArchitectureDesign,
        detected_patterns: Optional[List[PatternMatch]] = None,
    ) -> Dict[str, Any]:
        """
        Check architecture consistency.

        PATTERN: Run multiple validation rules, aggregate results
        CRITICAL: Provide actionable recommendations

        Args:
            design: Architecture design to check
            detected_patterns: Optional detected patterns

        Returns:
            Consistency check results
        """
        results = {
            "consistency_score": 0.0,
            "completeness_score": 0.0,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "checks_passed": [],
        }

        # Run consistency checks
        consistency_checks = await self._run_consistency_checks(design, detected_patterns)
        results["consistency_score"] = consistency_checks["score"]
        results["issues"].extend(consistency_checks["issues"])
        results["warnings"].extend(consistency_checks["warnings"])
        results["checks_passed"].extend(consistency_checks["passed"])

        # Run completeness checks
        completeness_checks = await self._run_completeness_checks(design)
        results["completeness_score"] = completeness_checks["score"]
        results["issues"].extend(completeness_checks["issues"])
        results["warnings"].extend(completeness_checks["warnings"])
        results["checks_passed"].extend(completeness_checks["passed"])

        # Run pattern adherence checks
        if detected_patterns:
            pattern_checks = await self._check_pattern_adherence(design, detected_patterns)
            results["recommendations"].extend(pattern_checks["recommendations"])
            results["warnings"].extend(pattern_checks["warnings"])

        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results["recommendations"].extend(recommendations)

        self.logger.info(
            f"Consistency check complete: consistency={results['consistency_score']:.2f}, "
            f"completeness={results['completeness_score']:.2f}"
        )

        return results

    async def _run_consistency_checks(
        self,
        design: ArchitectureDesign,
        detected_patterns: Optional[List[PatternMatch]] = None,
    ) -> Dict[str, Any]:
        """
        Run architecture consistency checks.

        Args:
            design: Architecture design
            detected_patterns: Detected patterns

        Returns:
            Consistency check results
        """
        score = 1.0  # Start with perfect score
        issues = []
        warnings = []
        passed = []

        # Check 1: Components have clear responsibilities
        if design.components:
            components_with_responsibilities = sum(
                1
                for comp in design.components.values()
                if comp.get("responsibilities") and len(comp["responsibilities"]) > 0
            )

            if components_with_responsibilities == len(design.components):
                passed.append("All components have defined responsibilities")
            else:
                missing = len(design.components) - components_with_responsibilities
                warnings.append(f"{missing} components lack clear responsibilities")
                score -= 0.1

        # Check 2: Technology choices are compatible
        if design.technologies:
            incompatible_pairs = self._check_technology_compatibility(design.technologies)

            if not incompatible_pairs:
                passed.append("All technologies are compatible")
            else:
                for tech1, tech2 in incompatible_pairs:
                    issues.append(f"Incompatible technologies: {tech1} and {tech2}")
                score -= 0.2 * len(incompatible_pairs)

        # Check 3: Pattern consistency
        if detected_patterns and len(design.patterns) > 0:
            # Check if declared patterns match detected patterns
            detected_pattern_names = {p.pattern for p in detected_patterns}

            for declared_pattern in design.patterns:
                if declared_pattern not in detected_pattern_names:
                    warnings.append(
                        f"Declared pattern '{declared_pattern}' not detected in codebase"
                    )
                    score -= 0.05

        # Check 4: Layer violations (for layered architectures)
        if any("layered" in p.lower() for p in design.patterns):
            layer_violations = self._check_layer_violations(design)

            if not layer_violations:
                passed.append("No layer violations detected")
            else:
                for violation in layer_violations:
                    issues.append(violation)
                score -= 0.1 * len(layer_violations)

        # Check 5: Component coupling
        coupling_issues = self._check_component_coupling(design)

        if not coupling_issues:
            passed.append("Component coupling is acceptable")
        else:
            for issue in coupling_issues:
                warnings.append(issue)
            score -= 0.05 * len(coupling_issues)

        return {
            "score": max(0, min(1, score)),
            "issues": issues,
            "warnings": warnings,
            "passed": passed,
        }

    async def _run_completeness_checks(
        self,
        design: ArchitectureDesign,
    ) -> Dict[str, Any]:
        """
        Run architecture completeness checks.

        Args:
            design: Architecture design

        Returns:
            Completeness check results
        """
        score = 0.0
        issues = []
        warnings = []
        passed = []

        # Check 1: Has name and description
        if design.name and design.description:
            score += 0.2
            passed.append("Architecture has name and description")
        else:
            issues.append("Architecture missing name or description")

        # Check 2: Has components defined
        if design.components and len(design.components) > 0:
            score += 0.2
            passed.append(f"Architecture has {len(design.components)} components defined")
        else:
            issues.append("Architecture has no components defined")

        # Check 3: Has technology stack
        if design.technologies and len(design.technologies) > 0:
            score += 0.2
            passed.append(f"Architecture has {len(design.technologies)} technologies defined")
        else:
            warnings.append("Architecture has no technology stack defined")

        # Check 4: Has patterns identified
        if design.patterns and len(design.patterns) > 0:
            score += 0.2
            passed.append(f"Architecture uses {len(design.patterns)} patterns")
        else:
            warnings.append("No architecture patterns identified")

        # Check 5: Has quality attributes
        if design.quality_attributes and len(design.quality_attributes) > 0:
            score += 0.2
            passed.append("Quality attributes defined")
        else:
            warnings.append("No quality attributes defined")

        return {
            "score": max(0, min(1, score)),
            "issues": issues,
            "warnings": warnings,
            "passed": passed,
        }

    async def _check_pattern_adherence(
        self,
        design: ArchitectureDesign,
        detected_patterns: List[PatternMatch],
    ) -> Dict[str, Any]:
        """
        Check adherence to detected patterns.

        Args:
            design: Architecture design
            detected_patterns: Detected patterns

        Returns:
            Pattern adherence results
        """
        warnings = []
        recommendations = []

        for pattern_match in detected_patterns:
            # Check confidence
            if pattern_match.confidence < 0.8:
                warnings.append(
                    f"Pattern '{pattern_match.pattern}' detected with low confidence "
                    f"({pattern_match.confidence:.2f})"
                )

            # Add recommendations from pattern
            if pattern_match.recommendations:
                recommendations.extend(pattern_match.recommendations)

            # Check if pattern follows best practices
            if not pattern_match.matches_best_practice:
                warnings.append(
                    f"Pattern '{pattern_match.pattern}' doesn't follow best practices"
                )

        return {
            "warnings": warnings,
            "recommendations": recommendations,
        }

    def _check_technology_compatibility(
        self,
        technologies: Dict[str, Any],
    ) -> List[tuple[str, str]]:
        """
        Check technology compatibility.

        Args:
            technologies: Technology choices

        Returns:
            List of incompatible technology pairs
        """
        incompatible_pairs = []

        tech_names = list(technologies.keys())

        for i, tech1_name in enumerate(tech_names):
            tech1 = technologies[tech1_name]

            for tech2_name in tech_names[i + 1 :]:
                # Check if incompatible
                incompatible_with = tech1.get("incompatible_with", [])

                if tech2_name in incompatible_with:
                    incompatible_pairs.append((tech1_name, tech2_name))

        return incompatible_pairs

    def _check_layer_violations(self, design: ArchitectureDesign) -> List[str]:
        """
        Check for layered architecture violations.

        Args:
            design: Architecture design

        Returns:
            List of violation descriptions
        """
        violations = []

        # Define expected layer order (lower layers cannot depend on higher layers)
        layer_hierarchy = [
            "presentation",
            "api",
            "business",
            "service",
            "domain",
            "data",
            "persistence",
        ]

        # Check component dependencies respect layer order
        for comp_name, comp_data in design.components.items():
            comp_layer = self._identify_layer(comp_name, comp_data)

            if comp_layer is None:
                continue  # Skip components not in a clear layer

            dependencies = comp_data.get("dependencies", [])

            for dep in dependencies:
                dep_layer = self._identify_layer(dep, design.components.get(dep, {}))

                if dep_layer is None:
                    continue

                # Check if dependency violates layer hierarchy
                try:
                    comp_layer_idx = layer_hierarchy.index(comp_layer)
                    dep_layer_idx = layer_hierarchy.index(dep_layer)

                    # Violation if depending on higher layer
                    if dep_layer_idx < comp_layer_idx:
                        violations.append(
                            f"Layer violation: {comp_name} ({comp_layer}) "
                            f"depends on {dep} ({dep_layer})"
                        )

                except ValueError:
                    # Layer not in hierarchy, skip
                    pass

        return violations

    def _identify_layer(
        self,
        component_name: str,
        component_data: Dict[str, Any],
    ) -> Optional[str]:
        """
        Identify which layer a component belongs to.

        Args:
            component_name: Component name
            component_data: Component data

        Returns:
            Layer name or None
        """
        layer_keywords = {
            "presentation": ["presentation", "ui", "view", "controller"],
            "api": ["api", "endpoint", "route"],
            "business": ["business", "logic", "service"],
            "service": ["service"],
            "domain": ["domain", "model"],
            "data": ["data", "dao", "repository"],
            "persistence": ["persistence", "database"],
        }

        component_name_lower = component_name.lower()
        component_type = component_data.get("type", "").lower()

        for layer, keywords in layer_keywords.items():
            if any(kw in component_name_lower or kw in component_type for kw in keywords):
                return layer

        return None

    def _check_component_coupling(self, design: ArchitectureDesign) -> List[str]:
        """
        Check component coupling levels.

        Args:
            design: Architecture design

        Returns:
            List of coupling issues
        """
        issues = []

        # Calculate coupling for each component
        for comp_name, comp_data in design.components.items():
            dependencies = comp_data.get("dependencies", [])
            dependents = comp_data.get("dependents", [])

            total_coupling = len(dependencies) + len(dependents)

            # High coupling threshold
            if total_coupling > 5:
                issues.append(
                    f"High coupling detected for {comp_name}: "
                    f"{len(dependencies)} dependencies, {len(dependents)} dependents"
                )

        return issues

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on check results.

        Args:
            results: Check results

        Returns:
            List of recommendations
        """
        recommendations = []

        # Low consistency score
        if results["consistency_score"] < 0.7:
            recommendations.append(
                "Architecture consistency is below recommended threshold (0.7). "
                "Review and address identified issues."
            )

        # Low completeness score
        if results["completeness_score"] < 0.8:
            recommendations.append(
                "Architecture definition is incomplete. "
                "Ensure all components, technologies, and patterns are documented."
            )

        # Many warnings
        if len(results["warnings"]) > 5:
            recommendations.append(
                f"Found {len(results['warnings'])} warnings. "
                "Consider addressing these to improve architecture quality."
            )

        # Critical issues
        if len(results["issues"]) > 0:
            recommendations.append(
                f"Found {len(results['issues'])} critical issues that should be addressed immediately."
            )

        return recommendations

    async def validate_patterns(
        self,
        design: ArchitectureDesign,
    ) -> List[str]:
        """
        Validate that declared patterns are correctly implemented.

        Args:
            design: Architecture design

        Returns:
            List of validation messages
        """
        validations = []

        for pattern_name in design.patterns:
            # Get pattern definition
            if self.pattern_library:
                pattern_def = self.pattern_library.get_pattern(pattern_name.lower())

                if pattern_def:
                    # Check if pattern structure requirements are met
                    # This would check specific requirements for each pattern
                    # For now, just log that we're checking
                    validations.append(
                        f"Validating pattern: {pattern_name}"
                    )

        return validations
