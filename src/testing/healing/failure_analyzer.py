"""Main failure analyzer orchestrator."""

import logging
from typing import Dict, Any, Optional

from src.models.healing_models import (
    TestFailure,
    FailureAnalysis,
    FailureType,
    RepairStrategy,
)
from src.testing.healing.analyzers.syntax_analyzer import SyntaxErrorAnalyzer
from src.testing.healing.analyzers.assertion_analyzer import AssertionAnalyzer
from src.testing.healing.analyzers.runtime_analyzer import RuntimeAnalyzer
from src.testing.healing.analyzers.timeout_analyzer import TimeoutAnalyzer

logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """Orchestrate test failure root cause analysis.

    PATTERN: Multi-strategy analysis with analyzer selection
    CRITICAL: Must handle framework-specific error formats
    """

    def __init__(self, ast_parser=None, dependency_analyzer=None):
        """Initialize failure analyzer."""
        self.ast_parser = ast_parser
        self.dependency_analyzer = dependency_analyzer

        # Register specialized analyzers
        self.analyzers = {
            FailureType.SYNTAX_ERROR: SyntaxErrorAnalyzer(),
            FailureType.ASSERTION_FAILED: AssertionAnalyzer(),
            FailureType.ATTRIBUTE_ERROR: RuntimeAnalyzer(),
            FailureType.TYPE_ERROR: RuntimeAnalyzer(),
            FailureType.TIMEOUT: TimeoutAnalyzer(),
            "default": RuntimeAnalyzer(),
        }

    async def analyze_failure(
        self, failure: TestFailure, code_diff: Optional[Dict] = None
    ) -> FailureAnalysis:
        """Analyze test failure to identify root cause."""
        try:
            # Parse error message
            error_info = self._parse_error(failure.error_message, failure.stack_trace)

            # Classify failure type
            failure_type = self._classify_failure(error_info)
            failure.failure_type = failure_type

            # Get appropriate analyzer
            analyzer = self.analyzers.get(failure_type, self.analyzers["default"])

            # Perform analysis
            root_cause_info = await analyzer.analyze(failure, error_info)

            # Check code changes if available
            if code_diff:
                related_changes = self._find_related_changes(
                    failure.target_function, code_diff
                )
                root_cause_info["code_changes"] = related_changes

            # Suggest repair strategies
            strategies = self._suggest_strategies(failure_type, root_cause_info)

            return FailureAnalysis(
                failure=failure,
                root_cause=root_cause_info["description"],
                confidence=root_cause_info["confidence"],
                code_changes=root_cause_info.get("code_changes"),
                suggested_strategies=strategies,
                evidence=root_cause_info.get("evidence", []),
            )

        except Exception as e:
            logger.error(f"Failure analysis error: {e}")
            # Return low-confidence analysis
            return FailureAnalysis(
                failure=failure,
                root_cause="Unable to determine root cause",
                confidence=0.3,
                suggested_strategies=[RepairStrategy.REGENERATE],
                evidence=[f"Analysis error: {str(e)}"],
            )

    def _parse_error(
        self, error_message: str, stack_trace: Optional[str]
    ) -> Dict[str, Any]:
        """Parse error message and stack trace."""
        return {
            "message": error_message,
            "stack_trace": stack_trace,
            "lines": error_message.split("\n") if error_message else [],
        }

    def _classify_failure(self, error_info: Dict) -> FailureType:
        """Classify failure based on error patterns."""
        error_msg = error_info.get("message", "").lower()

        if "syntaxerror" in error_msg:
            return FailureType.SYNTAX_ERROR
        elif "importerror" in error_msg or "modulenotfound" in error_msg:
            return FailureType.IMPORT_ERROR
        elif "attributeerror" in error_msg:
            return FailureType.ATTRIBUTE_ERROR
        elif "assertion" in error_msg or "assert" in error_msg:
            return FailureType.ASSERTION_FAILED
        elif "timeout" in error_msg or "timed out" in error_msg:
            return FailureType.TIMEOUT
        elif "mock" in error_msg:
            return FailureType.MOCK_ERROR
        elif "typeerror" in error_msg:
            return FailureType.TYPE_ERROR
        else:
            return FailureType.RUNTIME_ERROR

    def _find_related_changes(
        self, target_function: Optional[str], code_diff: Dict
    ) -> Dict[str, Any]:
        """Find code changes related to the failing test."""
        # Simplified - in real implementation would use AST comparison
        return {
            "changed_functions": code_diff.get("functions", []),
            "changed_files": code_diff.get("files", []),
        }

    def _suggest_strategies(
        self, failure_type: FailureType, root_cause_info: Dict
    ) -> list:
        """Suggest repair strategies based on failure type."""
        strategies = []

        if failure_type == FailureType.SYNTAX_ERROR:
            strategies = [RepairStrategy.REGENERATE]
        elif failure_type == FailureType.ASSERTION_FAILED:
            strategies = [RepairStrategy.UPDATE_ASSERTION, RepairStrategy.REGENERATE]
        elif failure_type == FailureType.ATTRIBUTE_ERROR:
            strategies = [RepairStrategy.UPDATE_SIGNATURE, RepairStrategy.REGENERATE]
        elif failure_type == FailureType.IMPORT_ERROR:
            strategies = [RepairStrategy.UPDATE_IMPORT]
        elif failure_type == FailureType.TIMEOUT:
            strategies = [RepairStrategy.ADD_WAIT, RepairStrategy.REGENERATE]
        elif failure_type == FailureType.MOCK_ERROR:
            strategies = [RepairStrategy.UPDATE_MOCK, RepairStrategy.REGENERATE]
        else:
            strategies = [RepairStrategy.REGENERATE]

        return strategies
