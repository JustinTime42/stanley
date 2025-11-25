"""Fractal task decomposition subsystem.

This module provides recursive task decomposition with dependency management,
complexity estimation, model assignment, and progress tracking.
"""

from .base import BaseDecomposer
from .fractal_decomposer import FractalDecomposer
from .dependency_manager import DependencyManager
from .complexity_estimator import ComplexityEstimator
from .task_assigner import TaskAssigner
from .progress_tracker import ProgressTracker

__all__ = [
    "BaseDecomposer",
    "FractalDecomposer",
    "DependencyManager",
    "ComplexityEstimator",
    "TaskAssigner",
    "ProgressTracker",
]
