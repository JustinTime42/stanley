"""Quality gate external tool integrations.

This module provides integrations with external quality analysis tools:
- MutmutIntegration: Mutation testing with mutmut
- BanditIntegration: Security vulnerability scanning
- SonarIntegration: SonarPython/SonarQube integration
- ProspectorIntegration: Meta-linter aggregating multiple tools

Each integration handles:
- Subprocess execution with proper error handling
- Resource limits and timeout handling
- Result parsing and normalization
- Installation checking and fallback behavior
"""

from .mutmut_integration import MutmutIntegration
from .bandit_integration import BanditIntegration
from .sonar_integration import SonarIntegration
from .prospector_integration import ProspectorIntegration

__all__ = [
    "MutmutIntegration",
    "BanditIntegration",
    "SonarIntegration",
    "ProspectorIntegration",
]
