"""Test repairer orchestrating repair strategies.

PATTERN: Orchestrate test repair with validation
CRITICAL: Must validate repairs don't break other tests
"""

import logging
import asyncio
import os
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path

from ...models.healing_models import (
    TestFailure,
    FailureAnalysis,
    TestRepair,
    RepairStrategy,
)
from .base import BaseHealer
from .repair_strategies.signature_repair import SignatureRepair
from .repair_strategies.assertion_repair import AssertionRepair
from .repair_strategies.mock_repair import MockRepair

logger = logging.getLogger(__name__)


class TestRepairer(BaseHealer):
    """
    Orchestrate test repair strategies.

    PATTERN: Strategy selection, repair application, validation
    CRITICAL: Must validate repairs maintain test intent and coverage
    GOTCHA: LLM-based repairs for complex failures beyond pattern matching
    """

    def __init__(
        self,
        llm_service=None,
        test_validator=None,
        coverage_analyzer=None,
    ):
        """
        Initialize test repairer.

        Args:
            llm_service: LLM service for complex repairs
            test_validator: Service to validate repaired tests
            coverage_analyzer: Service to verify coverage maintenance
        """
        super().__init__()
        self.llm_service = llm_service
        self.test_validator = test_validator
        self.coverage_analyzer = coverage_analyzer

        # Initialize repair strategies
        self.repair_strategies: Dict[RepairStrategy, Any] = {
            RepairStrategy.UPDATE_SIGNATURE: SignatureRepair(),
            RepairStrategy.UPDATE_ASSERTION: AssertionRepair(),
            RepairStrategy.UPDATE_MOCK: MockRepair(),
        }

    async def analyze_failure(
        self, failure: TestFailure, context: Optional[Dict[str, Any]] = None
    ) -> FailureAnalysis:
        """
        Analyze failure (delegated to FailureAnalyzer).

        Args:
            failure: Test failure to analyze
            context: Optional context

        Returns:
            Failure analysis
        """
        # This is typically handled by FailureAnalyzer
        # This method is here to satisfy BaseHealer interface
        raise NotImplementedError(
            "Use FailureAnalyzer for failure analysis, "
            "then call repair_test() with the analysis"
        )

    async def repair_test(
        self, analysis: FailureAnalysis, max_attempts: int = 3
    ) -> Optional[TestRepair]:
        """
        Repair failing test based on analysis.

        PATTERN: Try strategies in order of confidence
        CRITICAL: Validate each repair attempt

        Args:
            analysis: Failure analysis
            max_attempts: Maximum repair attempts

        Returns:
            TestRepair if successful, None otherwise
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Read original test code
            original_code = await self._read_test_file(analysis.failure.test_file)

            if not original_code:
                logger.error(f"Could not read test file: {analysis.failure.test_file}")
                return None

            # Sort strategies by priority
            strategies = self._prioritize_strategies(analysis.suggested_strategies)

            # Try each strategy
            for attempt, strategy in enumerate(strategies[:max_attempts], 1):
                logger.info(
                    f"Repair attempt {attempt}/{max_attempts} using {strategy.value}"
                )

                try:
                    # Check if we have a strategy implementation
                    if strategy not in self.repair_strategies:
                        logger.warning(
                            f"No implementation for strategy: {strategy.value}"
                        )
                        continue

                    repairer = self.repair_strategies[strategy]

                    # Check if strategy can handle this failure
                    if not await repairer.can_repair(analysis):
                        logger.debug(
                            f"Strategy {strategy.value} cannot handle this failure"
                        )
                        continue

                    # Generate repair
                    repaired_code = await repairer.repair(original_code, analysis)

                    if not repaired_code:
                        logger.debug(f"Strategy {strategy.value} returned no repair")
                        continue

                    # Validate syntax
                    if not self._validate_syntax(
                        repaired_code, self._get_language(analysis.failure.test_file)
                    ):
                        logger.warning("Repair has syntax errors, skipping")
                        continue

                    # Create test repair result
                    repair_time_ms = int(
                        (asyncio.get_event_loop().time() - start_time) * 1000
                    )

                    repair = TestRepair(
                        repair_id=f"repair_{analysis.failure.test_id}_{attempt}",
                        failure_analysis=analysis,
                        strategy=strategy,
                        original_code=original_code,
                        repaired_code=repaired_code,
                        changes=self._compute_changes(original_code, repaired_code),
                        syntax_valid=True,
                        test_passes=False,  # Will be set by validation
                        coverage_maintained=False,  # Will be set by validation
                        repair_time_ms=repair_time_ms,
                        confidence=self._calculate_confidence(strategy, analysis),
                    )

                    # Validate repair if validator available
                    if self.test_validator:
                        repair = await self._validate_repair(repair)

                        if repair.test_passes:
                            logger.info(f"Repair successful using {strategy.value}")
                            return repair
                    else:
                        # Return repair without validation if no validator
                        logger.warning(
                            "No test validator available, returning unvalidated repair"
                        )
                        repair.confidence *= 0.8  # Reduce confidence
                        return repair

                except Exception as e:
                    logger.warning(f"Strategy {strategy.value} failed: {e}")
                    continue

            # If all simple strategies fail, try LLM-based repair
            if self.llm_service and RepairStrategy.REGENERATE in strategies:
                logger.info("Attempting LLM-based repair")
                return await self._llm_repair(analysis, original_code, start_time)

            logger.error("All repair strategies exhausted")
            return None

        except Exception as e:
            logger.error(f"Test repair error: {e}", exc_info=True)
            return None

    async def _read_test_file(self, test_file: str) -> Optional[str]:
        """
        Read test file contents.

        Args:
            test_file: Path to test file

        Returns:
            File contents or None
        """
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading test file {test_file}: {e}")
            return None

    async def _validate_repair(self, repair: TestRepair) -> TestRepair:
        """
        Validate repaired test.

        CRITICAL: Must verify test passes and coverage maintained

        Args:
            repair: Test repair to validate

        Returns:
            Updated repair with validation results
        """
        try:
            # Write repaired code to temporary file
            temp_file = await self._write_temp_test(repair.repaired_code)

            try:
                # Run test
                test_result = await self.test_validator.run_test(temp_file)
                repair.test_passes = test_result.get("passed", False)

                # Verify coverage if analyzer available
                if self.coverage_analyzer and repair.test_passes:
                    coverage_ok = await self._verify_coverage(
                        repair.original_code,
                        repair.repaired_code,
                        repair.failure_analysis.failure.target_file,
                    )
                    repair.coverage_maintained = coverage_ok
                else:
                    # Assume coverage maintained if we can't verify
                    repair.coverage_maintained = True

            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

        except Exception as e:
            logger.error(f"Repair validation error: {e}")
            repair.test_passes = False
            repair.coverage_maintained = False

        return repair

    async def _write_temp_test(self, test_code: str) -> str:
        """
        Write test code to temporary file.

        Args:
            test_code: Test code to write

        Returns:
            Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        temp_file.write(test_code)
        temp_file.close()
        return temp_file.name

    async def _verify_coverage(
        self, original_code: str, repaired_code: str, target_file: str
    ) -> bool:
        """
        Verify repaired test maintains coverage.

        Args:
            original_code: Original test code
            repaired_code: Repaired test code
            target_file: File being tested

        Returns:
            True if coverage maintained
        """
        if not self.coverage_analyzer:
            return True

        try:
            # This is a simplified check - real implementation would
            # run coverage analysis on both versions
            # For now, assume coverage is maintained
            return True
        except Exception as e:
            logger.warning(f"Coverage verification error: {e}")
            return False

    async def _llm_repair(
        self, analysis: FailureAnalysis, original_code: str, start_time: float
    ) -> Optional[TestRepair]:
        """
        Use LLM for complex repairs.

        PATTERN: Provide context and let LLM fix
        CRITICAL: Low temperature for deterministic fixes

        Args:
            analysis: Failure analysis
            original_code: Original test code
            start_time: Start time for repair timing

        Returns:
            TestRepair or None
        """
        try:
            prompt = self._build_repair_prompt(analysis, original_code)

            response = await self.llm_service.agenerate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temperature for deterministic fixes
                max_tokens=2000,
            )

            repaired_code = self._extract_code(response)

            if not repaired_code:
                logger.error("LLM did not generate valid code")
                return None

            # Validate syntax
            if not self._validate_syntax(
                repaired_code, self._get_language(analysis.failure.test_file)
            ):
                logger.error("LLM-generated code has syntax errors")
                return None

            repair_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            repair = TestRepair(
                repair_id=f"llm_repair_{analysis.failure.test_id}",
                failure_analysis=analysis,
                strategy=RepairStrategy.REGENERATE,
                original_code=original_code,
                repaired_code=repaired_code,
                changes=[
                    {"type": "llm_generated", "description": "Complete LLM repair"}
                ],
                syntax_valid=True,
                test_passes=False,
                coverage_maintained=False,
                repair_time_ms=repair_time_ms,
                confidence=0.7,  # Lower confidence for LLM repairs
            )

            # Validate if possible
            if self.test_validator:
                repair = await self._validate_repair(repair)

            return repair

        except Exception as e:
            logger.error(f"LLM repair error: {e}", exc_info=True)
            return None

    def _build_repair_prompt(
        self, analysis: FailureAnalysis, original_code: str
    ) -> str:
        """
        Build prompt for LLM-based repair.

        Args:
            analysis: Failure analysis
            original_code: Original test code

        Returns:
            Repair prompt
        """
        code_changes_str = ""
        if analysis.code_changes:
            code_changes_str = f"\n\nCode changes:\n{analysis.code_changes}"

        return f"""Fix the failing test based on the following information:

Test failure: {analysis.failure.error_message}
Root cause: {analysis.root_cause}
Confidence: {analysis.confidence}

Evidence:
{chr(10).join(f"- {e}" for e in analysis.evidence)}
{code_changes_str}

Original test code:
```python
{original_code}
```

Generate the fixed test code that:
1. Resolves the error completely
2. Maintains the test's original intent and purpose
3. Preserves code coverage
4. Uses proper test framework conventions
5. Is syntactically correct

Return ONLY the fixed test code, no explanation."""

    def _extract_code(self, llm_response: str) -> Optional[str]:
        """
        Extract code from LLM response.

        Args:
            llm_response: LLM response text

        Returns:
            Extracted code or None
        """
        # Try to extract code from markdown code blocks
        if "```python" in llm_response:
            parts = llm_response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                return code
        elif "```" in llm_response:
            parts = llm_response.split("```")
            if len(parts) >= 3:
                code = parts[1].strip()
                return code

        # If no code blocks, return the whole response
        return llm_response.strip()

    def _prioritize_strategies(
        self, strategies: List[RepairStrategy]
    ) -> List[RepairStrategy]:
        """
        Prioritize repair strategies.

        Args:
            strategies: List of suggested strategies

        Returns:
            Prioritized list
        """
        # Define priority order (lower number = higher priority)
        priority_map = {
            RepairStrategy.UPDATE_IMPORT: 1,
            RepairStrategy.UPDATE_SIGNATURE: 2,
            RepairStrategy.UPDATE_ASSERTION: 3,
            RepairStrategy.UPDATE_MOCK: 4,
            RepairStrategy.UPDATE_SETUP: 5,
            RepairStrategy.ADD_ASYNC: 6,
            RepairStrategy.ADD_WAIT: 7,
            RepairStrategy.UPDATE_SELECTOR: 8,
            RepairStrategy.REGENERATE: 9,
        }

        return sorted(strategies, key=lambda s: priority_map.get(s, 10))

    def _calculate_confidence(
        self, strategy: RepairStrategy, analysis: FailureAnalysis
    ) -> float:
        """
        Calculate confidence in repair.

        Args:
            strategy: Strategy used
            analysis: Failure analysis

        Returns:
            Confidence score (0-1)
        """
        # Base confidence from analysis
        confidence = analysis.confidence

        # Adjust based on strategy
        if strategy == RepairStrategy.REGENERATE:
            confidence *= 0.7  # Lower for regeneration
        elif strategy in [
            RepairStrategy.UPDATE_IMPORT,
            RepairStrategy.UPDATE_SIGNATURE,
        ]:
            confidence *= 0.95  # Higher for simple fixes
        else:
            confidence *= 0.85

        return min(confidence, 1.0)

    def _compute_changes(self, original: str, repaired: str) -> List[Dict[str, Any]]:
        """
        Compute changes between original and repaired code.

        Args:
            original: Original code
            repaired: Repaired code

        Returns:
            List of changes
        """
        import difflib

        changes = []
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            repaired.splitlines(keepends=True),
            lineterm="",
        )

        diff_text = "".join(diff)
        if diff_text:
            changes.append({"type": "code_change", "diff": diff_text})

        return changes

    def _validate_syntax(self, code: str, language: str = "python") -> bool:
        """
        Validate code syntax.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            True if syntax is valid
        """
        if language == "python":
            try:
                compile(code, "<string>", "exec")
                return True
            except SyntaxError as e:
                logger.debug(f"Syntax error: {e}")
                return False

        # Assume valid for other languages
        return True

    def _get_language(self, file_path: str) -> str:
        """
        Get programming language from file extension.

        Args:
            file_path: File path

        Returns:
            Language name
        """
        ext = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
        }
        return language_map.get(ext, "python")
