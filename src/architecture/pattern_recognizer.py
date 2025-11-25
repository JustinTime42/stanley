"""Pattern recognition for detecting architecture patterns in code."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..models.architecture_models import PatternMatch, ArchitecturePattern

logger = logging.getLogger(__name__)


class PatternRecognizer:
    """
    Detect architecture patterns in codebase.

    PATTERN: AST analysis combined with structural matching
    CRITICAL: Calculate confidence scores for each pattern
    GOTCHA: Different patterns have different detection difficulties
    """

    def __init__(self, ast_parser=None, pattern_library=None):
        """
        Initialize pattern recognizer.

        Args:
            ast_parser: AST parser for code analysis
            pattern_library: Pattern library for pattern definitions
        """
        self.ast_parser = ast_parser
        self.pattern_library = pattern_library
        self.logger = logging.getLogger(__name__)

    async def recognize_patterns(
        self,
        codebase_path: str,
        confidence_threshold: float = 0.7,
    ) -> List[PatternMatch]:
        """
        Recognize architecture patterns in codebase.

        PATTERN: Structural matching with confidence scoring
        CRITICAL: Use pattern-specific detection rules

        Args:
            codebase_path: Path to codebase
            confidence_threshold: Minimum confidence to report (0-1)

        Returns:
            List of detected patterns
        """
        patterns_found = []

        # Analyze codebase structure
        structure = await self._analyze_structure(codebase_path)

        # Check each pattern type
        for pattern_type in ArchitecturePattern:
            match = await self._check_pattern(pattern_type, structure)

            if match and match.confidence >= confidence_threshold:
                patterns_found.append(match)

        self.logger.info(
            f"Recognized {len(patterns_found)} patterns in {codebase_path}"
        )

        return patterns_found

    async def _analyze_structure(self, codebase_path: str) -> Dict[str, Any]:
        """
        Analyze codebase structure.

        Args:
            codebase_path: Path to codebase

        Returns:
            Structure analysis results
        """
        structure = {
            "root_path": codebase_path,
            "directories": [],
            "files": [],
            "modules": [],
            "classes": [],
            "functions": [],
            "imports": [],
        }

        try:
            path = Path(codebase_path)

            # Analyze directory structure
            if path.exists() and path.is_dir():
                for item in path.rglob("*.py"):
                    structure["files"].append(str(item.relative_to(path)))

                # Get directory structure
                directories = set()
                for file in structure["files"]:
                    parts = Path(file).parts
                    for i in range(1, len(parts)):
                        directories.add("/".join(parts[:i]))

                structure["directories"] = sorted(list(directories))

                # Analyze file contents if AST parser available
                if self.ast_parser:
                    # This would use the AST parser to analyze Python files
                    # For now, use basic directory analysis
                    pass

        except Exception as e:
            self.logger.error(f"Failed to analyze structure: {e}")

        return structure

    async def _check_pattern(
        self,
        pattern_type: ArchitecturePattern,
        structure: Dict[str, Any],
    ) -> Optional[PatternMatch]:
        """
        Check if pattern exists in codebase structure.

        PATTERN: Pattern-specific detection rules
        CRITICAL: Calculate evidence-based confidence scores

        Args:
            pattern_type: Pattern to check
            structure: Codebase structure

        Returns:
            Pattern match or None
        """
        confidence = 0.0
        evidence = []
        recommendations = []

        if pattern_type == ArchitecturePattern.LAYERED:
            # Check for layer separation
            has_layers, layer_evidence = self._check_layer_separation(structure)
            if has_layers:
                confidence += 0.4
                evidence.extend(layer_evidence)

            # Check for dependency rules
            follows_rules, rule_evidence = self._check_dependency_rules(structure)
            if follows_rules:
                confidence += 0.3
                evidence.extend(rule_evidence)
            else:
                recommendations.append("Ensure dependencies only flow downward through layers")

            # Check for layer interfaces
            has_interfaces, interface_evidence = self._check_layer_interfaces(structure)
            if has_interfaces:
                confidence += 0.3
                evidence.extend(interface_evidence)

        elif pattern_type == ArchitecturePattern.MICROSERVICES:
            # Check for service boundaries
            has_services, service_evidence = self._check_service_boundaries(structure)
            if has_services:
                confidence += 0.35
                evidence.extend(service_evidence)

            # Check for API definitions
            has_apis, api_evidence = self._check_api_definitions(structure)
            if has_apis:
                confidence += 0.35
                evidence.extend(api_evidence)

            # Check for independent deployability
            is_independent, independence_evidence = self._check_independent_deployability(structure)
            if is_independent:
                confidence += 0.3
                evidence.extend(independence_evidence)
            else:
                recommendations.append("Ensure services can be deployed independently")

        elif pattern_type == ArchitecturePattern.EVENT_DRIVEN:
            # Check for event handlers
            has_events, event_evidence = self._check_event_patterns(structure)
            if has_events:
                confidence += 0.4
                evidence.extend(event_evidence)

            # Check for message queues
            has_queues, queue_evidence = self._check_message_queues(structure)
            if has_queues:
                confidence += 0.3
                evidence.extend(queue_evidence)

            # Check for async processing
            is_async, async_evidence = self._check_async_processing(structure)
            if is_async:
                confidence += 0.3
                evidence.extend(async_evidence)

        elif pattern_type == ArchitecturePattern.REPOSITORY:
            # Check for repository pattern
            has_repos, repo_evidence = self._check_repository_pattern(structure)
            if has_repos:
                confidence += 0.5
                evidence.extend(repo_evidence)

        elif pattern_type == ArchitecturePattern.MVC:
            # Check for MVC components
            has_mvc, mvc_evidence = self._check_mvc_pattern(structure)
            if has_mvc:
                confidence += 0.5
                evidence.extend(mvc_evidence)

        # Only return match if confidence is above minimum threshold
        if confidence > 0.3:
            return PatternMatch(
                pattern=pattern_type.value,
                confidence=min(confidence, 1.0),
                location=structure.get("root_path", ""),
                evidence=evidence,
                matches_best_practice=confidence > 0.8,
                recommendations=recommendations,
            )

        return None

    def _check_layer_separation(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for layered architecture separation."""
        evidence = []
        layer_keywords = ["presentation", "business", "data", "service", "repository", "controller"]

        directories = structure.get("directories", [])

        # Check if directories contain layer indicators
        found_layers = []
        for dir_name in directories:
            dir_lower = dir_name.lower()
            for keyword in layer_keywords:
                if keyword in dir_lower:
                    found_layers.append(keyword)
                    evidence.append(f"Layer detected: {dir_name}")
                    break

        has_layers = len(found_layers) >= 2

        return has_layers, evidence

    def _check_dependency_rules(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check if dependencies follow layered rules."""
        # Simplified check - would need full import analysis
        # For now, assume follows rules if layered structure exists
        evidence = []

        directories = structure.get("directories", [])
        if any("controller" in d.lower() or "presentation" in d.lower() for d in directories):
            if any("service" in d.lower() or "business" in d.lower() for d in directories):
                evidence.append("Controller and service layers properly separated")
                return True, evidence

        return False, evidence

    def _check_layer_interfaces(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for layer interfaces."""
        evidence = []

        files = structure.get("files", [])

        # Look for interface or base files
        interface_files = [f for f in files if "interface" in f.lower() or "base" in f.lower()]

        if interface_files:
            evidence.append(f"Found {len(interface_files)} interface/base files")
            return True, evidence

        return False, evidence

    def _check_service_boundaries(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for microservices boundaries."""
        evidence = []

        directories = structure.get("directories", [])

        # Look for service-oriented directory structure
        service_dirs = [d for d in directories if "service" in d.lower()]

        if len(service_dirs) >= 2:
            evidence.append(f"Found {len(service_dirs)} service directories")
            return True, evidence

        return False, evidence

    def _check_api_definitions(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for API definitions."""
        evidence = []

        files = structure.get("files", [])

        # Look for API-related files
        api_files = [f for f in files if any(kw in f.lower() for kw in ["api", "endpoint", "route"])]

        if api_files:
            evidence.append(f"Found {len(api_files)} API-related files")
            return True, evidence

        return False, evidence

    def _check_independent_deployability(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for independent deployability indicators."""
        evidence = []

        files = structure.get("files", [])

        # Look for deployment configuration files
        deployment_files = [
            f
            for f in files
            if any(kw in f.lower() for kw in ["dockerfile", "docker-compose", "k8s", "kubernetes"])
        ]

        if deployment_files:
            evidence.append("Found deployment configurations")
            return True, evidence

        return False, evidence

    def _check_event_patterns(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for event-driven patterns."""
        evidence = []

        files = structure.get("files", [])

        # Look for event-related files
        event_files = [
            f
            for f in files
            if any(kw in f.lower() for kw in ["event", "handler", "listener", "subscriber"])
        ]

        if event_files:
            evidence.append(f"Found {len(event_files)} event-related files")
            return True, evidence

        return False, evidence

    def _check_message_queues(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for message queue usage."""
        evidence = []

        files = structure.get("files", [])

        # Look for queue-related files
        queue_files = [
            f
            for f in files
            if any(kw in f.lower() for kw in ["queue", "broker", "pubsub", "kafka", "rabbitmq"])
        ]

        if queue_files:
            evidence.append("Found message queue components")
            return True, evidence

        return False, evidence

    def _check_async_processing(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for async processing patterns."""
        evidence = []

        files = structure.get("files", [])

        # Look for async-related files
        async_files = [f for f in files if any(kw in f.lower() for kw in ["async", "worker", "task"])]

        if async_files:
            evidence.append("Found async processing components")
            return True, evidence

        return False, evidence

    def _check_repository_pattern(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for repository pattern."""
        evidence = []

        files = structure.get("files", [])
        directories = structure.get("directories", [])

        # Look for repository files or directories
        repo_indicators = [
            item
            for item in files + directories
            if "repository" in item.lower() or "repo" in item.lower()
        ]

        if repo_indicators:
            evidence.append("Found repository pattern indicators")
            return True, evidence

        return False, evidence

    def _check_mvc_pattern(self, structure: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for MVC pattern."""
        evidence = []

        directories = structure.get("directories", [])

        # Look for MVC directories
        has_model = any("model" in d.lower() for d in directories)
        has_view = any("view" in d.lower() or "template" in d.lower() for d in directories)
        has_controller = any("controller" in d.lower() for d in directories)

        if has_model:
            evidence.append("Model layer detected")
        if has_view:
            evidence.append("View layer detected")
        if has_controller:
            evidence.append("Controller layer detected")

        has_mvc = has_model and has_view and has_controller

        return has_mvc, evidence
