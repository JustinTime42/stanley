"""Threshold manager for configurable quality gates.

PATTERN: Configuration management with YAML/JSON support
CRITICAL: Support per-project/module overrides with precedence rules
GOTCHA: Thresholds must be validated before use
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from ..models.quality_models import (
    QualityThreshold,
    QualityDimension,
    CoverageType,
)

logger = logging.getLogger(__name__)


class ThresholdManager:
    """
    Manages quality thresholds with hierarchical overrides.

    PATTERN: Configuration loading with project/module override support
    CRITICAL: Precedence: module > project > global defaults
    GOTCHA: Invalid thresholds should fail fast with clear errors

    The threshold manager handles loading, validation, and resolution of
    quality thresholds from configuration files with support for:
    - Global default thresholds
    - Project-specific overrides
    - Module-level overrides
    - Runtime threshold adjustments
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        defaults: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize threshold manager.

        Args:
            config_path: Path to configuration file (YAML or JSON)
            defaults: Optional default thresholds to use if no config
        """
        self.logger = logger
        self.config_path = Path(config_path) if config_path else None
        self.defaults = defaults or self._get_default_thresholds()

        # Storage for loaded thresholds
        self.global_thresholds: Dict[str, QualityThreshold] = {}
        self.project_thresholds: Dict[str, Dict[str, QualityThreshold]] = {}
        self.module_thresholds: Dict[str, Dict[str, QualityThreshold]] = {}

        # Load configuration if provided
        if self.config_path and self.config_path.exists():
            self.load_configuration(self.config_path)
        else:
            self._load_defaults()

        self.logger.info("ThresholdManager initialized")

    def _get_default_thresholds(self) -> Dict[str, Any]:
        """
        Get default threshold configurations.

        PATTERN: Sensible defaults for common quality metrics
        CRITICAL: These should be production-ready baselines

        Returns:
            Default threshold configuration
        """
        return {
            "coverage": {
                "line_coverage": {
                    "dimension": "coverage",
                    "metric": "line_coverage",
                    "min_value": 80.0,
                    "target_value": 90.0,
                    "enforcement": "error",
                    "allow_override": False,
                },
                "branch_coverage": {
                    "dimension": "coverage",
                    "metric": "branch_coverage",
                    "min_value": 75.0,
                    "target_value": 85.0,
                    "enforcement": "error",
                    "allow_override": False,
                },
                "mutation_score": {
                    "dimension": "coverage",
                    "metric": "mutation_score",
                    "min_value": 75.0,
                    "target_value": 85.0,
                    "enforcement": "warning",
                    "allow_override": True,
                },
            },
            "static": {
                "code_quality_score": {
                    "dimension": "static",
                    "metric": "code_quality_score",
                    "min_value": 8.0,
                    "max_value": 10.0,
                    "target_value": 9.0,
                    "enforcement": "error",
                    "allow_override": False,
                },
                "critical_issues": {
                    "dimension": "static",
                    "metric": "critical_issues",
                    "max_value": 0,
                    "enforcement": "error",
                    "allow_override": False,
                },
                "high_issues": {
                    "dimension": "static",
                    "metric": "high_issues",
                    "max_value": 5,
                    "enforcement": "warning",
                    "allow_override": True,
                },
            },
            "security": {
                "critical_vulnerabilities": {
                    "dimension": "security",
                    "metric": "critical_vulnerabilities",
                    "max_value": 0,
                    "enforcement": "error",
                    "allow_override": False,
                },
                "high_vulnerabilities": {
                    "dimension": "security",
                    "metric": "high_vulnerabilities",
                    "max_value": 0,
                    "enforcement": "error",
                    "allow_override": True,
                },
                "medium_vulnerabilities": {
                    "dimension": "security",
                    "metric": "medium_vulnerabilities",
                    "max_value": 5,
                    "enforcement": "warning",
                    "allow_override": True,
                },
            },
            "complexity": {
                "cyclomatic_complexity": {
                    "dimension": "complexity",
                    "metric": "cyclomatic_complexity",
                    "max_value": 15.0,
                    "enforcement": "warning",
                    "allow_override": True,
                },
                "cognitive_complexity": {
                    "dimension": "complexity",
                    "metric": "cognitive_complexity",
                    "max_value": 20.0,
                    "enforcement": "warning",
                    "allow_override": True,
                },
                "maintainability_index": {
                    "dimension": "complexity",
                    "metric": "maintainability_index",
                    "min_value": 65.0,
                    "target_value": 85.0,
                    "enforcement": "warning",
                    "allow_override": True,
                },
            },
            "performance": {
                "regression_threshold": {
                    "dimension": "performance",
                    "metric": "regression_threshold",
                    "max_value": 10.0,  # Max 10% regression
                    "enforcement": "warning",
                    "allow_override": True,
                },
            },
        }

    def _load_defaults(self) -> None:
        """
        Load default thresholds into global configuration.

        PATTERN: Convert dict config to Pydantic models
        """
        self.logger.info("Loading default thresholds")

        for dimension, metrics in self.defaults.items():
            for metric_name, config in metrics.items():
                try:
                    threshold = QualityThreshold(**config)
                    key = f"{dimension}.{metric_name}"
                    self.global_thresholds[key] = threshold
                except Exception as e:
                    self.logger.error(f"Failed to load default threshold {dimension}.{metric_name}: {e}")

    def load_configuration(self, config_path: Union[str, Path]) -> None:
        """
        Load thresholds from configuration file.

        PATTERN: Support both YAML and JSON formats
        CRITICAL: Validate all loaded thresholds

        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)

        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}")
            self._load_defaults()
            return

        try:
            # Load file based on extension
            if config_path.suffix in [".yaml", ".yml"]:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

            # Parse configuration structure
            self._parse_configuration(config)

            self.logger.info(f"Loaded thresholds from {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            self._load_defaults()

    def _parse_configuration(self, config: Dict[str, Any]) -> None:
        """
        Parse configuration structure into threshold storage.

        PATTERN: Support global, project, and module levels
        CRITICAL: Validate threshold values during parsing

        Args:
            config: Configuration dictionary
        """
        # Load global thresholds
        if "global" in config:
            self._load_threshold_level(config["global"], self.global_thresholds)
        elif "thresholds" in config:
            # Backward compatibility: top-level thresholds are global
            self._load_threshold_level(config["thresholds"], self.global_thresholds)

        # Load project overrides
        if "projects" in config:
            for project_name, project_config in config["projects"].items():
                if project_name not in self.project_thresholds:
                    self.project_thresholds[project_name] = {}
                self._load_threshold_level(
                    project_config.get("thresholds", {}),
                    self.project_thresholds[project_name]
                )

        # Load module overrides
        if "modules" in config:
            for module_name, module_config in config["modules"].items():
                if module_name not in self.module_thresholds:
                    self.module_thresholds[module_name] = {}
                self._load_threshold_level(
                    module_config.get("thresholds", {}),
                    self.module_thresholds[module_name]
                )

    def _load_threshold_level(
        self,
        config: Dict[str, Any],
        target_dict: Dict[str, QualityThreshold]
    ) -> None:
        """
        Load thresholds at a specific level (global/project/module).

        Args:
            config: Threshold configuration
            target_dict: Dictionary to store parsed thresholds
        """
        for dimension, metrics in config.items():
            if isinstance(metrics, dict):
                for metric_name, threshold_config in metrics.items():
                    try:
                        # Ensure dimension is included
                        if "dimension" not in threshold_config:
                            threshold_config["dimension"] = dimension
                        if "metric" not in threshold_config:
                            threshold_config["metric"] = metric_name

                        threshold = QualityThreshold(**threshold_config)
                        key = f"{dimension}.{metric_name}"
                        target_dict[key] = threshold

                    except Exception as e:
                        self.logger.error(
                            f"Failed to load threshold {dimension}.{metric_name}: {e}"
                        )

    def get_thresholds(
        self,
        project: Optional[str] = None,
        module: Optional[str] = None,
        dimension: Optional[QualityDimension] = None,
    ) -> List[QualityThreshold]:
        """
        Get applicable thresholds with override precedence.

        PATTERN: Module > Project > Global precedence
        CRITICAL: Later overrides take precedence

        Args:
            project: Optional project name for project-specific thresholds
            module: Optional module name for module-specific thresholds
            dimension: Optional filter by quality dimension

        Returns:
            List of applicable thresholds with overrides applied
        """
        # Start with global thresholds
        thresholds = dict(self.global_thresholds)

        # Apply project overrides
        if project and project in self.project_thresholds:
            thresholds.update(self.project_thresholds[project])

        # Apply module overrides (highest precedence)
        if module and module in self.module_thresholds:
            thresholds.update(self.module_thresholds[module])

        # Filter by dimension if specified
        result = list(thresholds.values())
        if dimension:
            result = [t for t in result if t.dimension == dimension]

        return result

    def get_threshold(
        self,
        dimension: QualityDimension,
        metric: str,
        project: Optional[str] = None,
        module: Optional[str] = None,
    ) -> Optional[QualityThreshold]:
        """
        Get a specific threshold with override precedence.

        Args:
            dimension: Quality dimension
            metric: Specific metric name
            project: Optional project name
            module: Optional module name

        Returns:
            Threshold if found, None otherwise
        """
        # Convert enum to string value
        dimension_str = dimension.value if isinstance(dimension, QualityDimension) else dimension
        key = f"{dimension_str}.{metric}"

        # Check module level first (highest precedence)
        if module and module in self.module_thresholds:
            if key in self.module_thresholds[module]:
                return self.module_thresholds[module][key]

        # Check project level
        if project and project in self.project_thresholds:
            if key in self.project_thresholds[project]:
                return self.project_thresholds[project][key]

        # Fall back to global
        return self.global_thresholds.get(key)

    def check_threshold(
        self,
        threshold: QualityThreshold,
        actual_value: float,
    ) -> bool:
        """
        Check if a value meets threshold requirements.

        PATTERN: Support min, max, and target thresholds
        CRITICAL: Handle missing values gracefully

        Args:
            threshold: Threshold configuration
            actual_value: Actual metric value to check

        Returns:
            True if threshold is met, False otherwise
        """
        # Check minimum threshold
        if threshold.min_value is not None:
            if actual_value < threshold.min_value:
                self.logger.debug(
                    f"{threshold.metric} below minimum: {actual_value} < {threshold.min_value}"
                )
                return False

        # Check maximum threshold
        if threshold.max_value is not None:
            if actual_value > threshold.max_value:
                self.logger.debug(
                    f"{threshold.metric} above maximum: {actual_value} > {threshold.max_value}"
                )
                return False

        return True

    def check_violations(
        self,
        metrics: Dict[str, float],
        project: Optional[str] = None,
        module: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check multiple metrics against thresholds.

        PATTERN: Batch threshold checking with violation details
        CRITICAL: Return actionable violation information

        Args:
            metrics: Dictionary of metric values
            project: Optional project name
            module: Optional module name

        Returns:
            List of threshold violations with details
        """
        violations = []
        thresholds = self.get_thresholds(project=project, module=module)

        for threshold in thresholds:
            metric_key = threshold.metric

            # Find matching metric value
            actual_value = None
            for key, value in metrics.items():
                if key == metric_key or key.endswith(f".{metric_key}"):
                    actual_value = value
                    break

            if actual_value is None:
                # Metric not provided, skip check
                continue

            # Check threshold
            if not self.check_threshold(threshold, actual_value):
                violation = {
                    "dimension": threshold.dimension,
                    "metric": threshold.metric,
                    "threshold": threshold,
                    "actual_value": actual_value,
                    "min_value": threshold.min_value,
                    "max_value": threshold.max_value,
                    "target_value": threshold.target_value,
                    "enforcement": threshold.enforcement,
                    "allow_override": threshold.allow_override,
                }
                violations.append(violation)

        return violations

    def add_threshold(
        self,
        threshold: QualityThreshold,
        project: Optional[str] = None,
        module: Optional[str] = None,
    ) -> None:
        """
        Add or update a threshold at runtime.

        PATTERN: Runtime threshold adjustment
        GOTCHA: Runtime changes are not persisted to config

        Args:
            threshold: Threshold to add or update
            project: Optional project name
            module: Optional module name
        """
        key = f"{threshold.dimension}.{threshold.metric}"

        if module:
            if module not in self.module_thresholds:
                self.module_thresholds[module] = {}
            self.module_thresholds[module][key] = threshold
        elif project:
            if project not in self.project_thresholds:
                self.project_thresholds[project] = {}
            self.project_thresholds[project][key] = threshold
        else:
            self.global_thresholds[key] = threshold

        self.logger.info(f"Added threshold: {key} (level: {'module' if module else 'project' if project else 'global'})")

    def remove_threshold(
        self,
        dimension: QualityDimension,
        metric: str,
        project: Optional[str] = None,
        module: Optional[str] = None,
    ) -> bool:
        """
        Remove a threshold at runtime.

        Args:
            dimension: Quality dimension
            metric: Specific metric name
            project: Optional project name
            module: Optional module name

        Returns:
            True if threshold was removed, False if not found
        """
        key = f"{dimension}.{metric}"

        if module and module in self.module_thresholds:
            if key in self.module_thresholds[module]:
                del self.module_thresholds[module][key]
                self.logger.info(f"Removed module threshold: {key}")
                return True

        if project and project in self.project_thresholds:
            if key in self.project_thresholds[project]:
                del self.project_thresholds[project][key]
                self.logger.info(f"Removed project threshold: {key}")
                return True

        if key in self.global_thresholds:
            del self.global_thresholds[key]
            self.logger.info(f"Removed global threshold: {key}")
            return True

        return False

    def export_configuration(self, format: str = "yaml") -> str:
        """
        Export current threshold configuration.

        PATTERN: Serialize current state for persistence
        CRITICAL: Include all levels (global, project, module)

        Args:
            format: Output format ('yaml' or 'json')

        Returns:
            Serialized configuration string
        """
        config = {
            "global": self._serialize_thresholds(self.global_thresholds),
        }

        if self.project_thresholds:
            config["projects"] = {
                project: {"thresholds": self._serialize_thresholds(thresholds)}
                for project, thresholds in self.project_thresholds.items()
            }

        if self.module_thresholds:
            config["modules"] = {
                module: {"thresholds": self._serialize_thresholds(thresholds)}
                for module, thresholds in self.module_thresholds.items()
            }

        if format == "json":
            return json.dumps(config, indent=2)
        else:
            return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def _serialize_thresholds(
        self, thresholds: Dict[str, QualityThreshold]
    ) -> Dict[str, Any]:
        """
        Serialize thresholds to dictionary format.

        Args:
            thresholds: Dictionary of thresholds

        Returns:
            Serialized threshold configuration
        """
        result = {}

        for key, threshold in thresholds.items():
            dimension = threshold.dimension
            metric = threshold.metric

            if dimension not in result:
                result[dimension] = {}

            result[dimension][metric] = {
                "dimension": threshold.dimension,
                "metric": threshold.metric,
                "min_value": threshold.min_value,
                "max_value": threshold.max_value,
                "target_value": threshold.target_value,
                "enforcement": threshold.enforcement,
                "allow_override": threshold.allow_override,
            }

            # Remove None values
            result[dimension][metric] = {
                k: v for k, v in result[dimension][metric].items() if v is not None
            }

        return result
