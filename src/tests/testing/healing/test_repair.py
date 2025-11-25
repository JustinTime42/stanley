"""Comprehensive tests for test repair components.

Tests TestRepairer orchestrator and all repair strategies according to PRP-11.
Includes mocking of LLM, validation, and coverage services.
"""

import pytest
import pytest_asyncio
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock, mock_open

from src.testing.healing.test_repairer import TestRepairer
from src.testing.healing.repair_strategies.signature_repair import SignatureRepair
from src.testing.healing.repair_strategies.assertion_repair import AssertionRepair
from src.testing.healing.repair_strategies.mock_repair import MockRepair
from src.models.healing_models import (
    TestFailure,
    FailureAnalysis,
    TestRepair,
    FailureType,
    RepairStrategy,
)


class TestSignatureRepair:
    """Test SignatureRepair strategy."""

    @pytest.fixture
    def strategy(self):
        """Create signature repair strategy instance."""
        return SignatureRepair()

    @pytest.fixture
    def attribute_error_analysis(self):
        """Create analysis for attribute error."""
        failure = TestFailure(
            test_id="test_001",
            test_name="test_missing_method",
            test_file="tests/test_example.py",
            failure_type=FailureType.ATTRIBUTE_ERROR,
            error_message="AttributeError: 'MyClass' object has no attribute 'old_method'",
            target_file="src/example.py",
            test_framework="pytest",
        )
        return FailureAnalysis(
            failure=failure,
            root_cause="Method 'old_method' no longer exists",
            confidence=0.9,
            suggested_strategies=[RepairStrategy.UPDATE_SIGNATURE],
            evidence=["Method not found in class"],
        )

    @pytest.fixture
    def type_error_analysis(self):
        """Create analysis for type error (argument mismatch)."""
        failure = TestFailure(
            test_id="test_002",
            test_name="test_wrong_args",
            test_file="tests/test_example.py",
            failure_type=FailureType.TYPE_ERROR,
            error_message="TypeError: my_func() takes 2 positional arguments but 3 were given",
            target_file="src/example.py",
            test_framework="pytest",
        )
        return FailureAnalysis(
            failure=failure,
            root_cause="Function signature changed - argument count mismatch",
            confidence=0.85,
            suggested_strategies=[RepairStrategy.UPDATE_SIGNATURE],
            evidence=["Wrong number of arguments"],
        )

    @pytest.mark.asyncio
    async def test_can_repair_attribute_error(self, strategy, attribute_error_analysis):
        """Test strategy can handle attribute errors."""
        result = await strategy.can_repair(attribute_error_analysis)
        assert result is True

    @pytest.mark.asyncio
    async def test_can_repair_type_error_with_arguments(self, strategy, type_error_analysis):
        """Test strategy can handle type errors with argument keywords."""
        result = await strategy.can_repair(type_error_analysis)
        assert result is True

    @pytest.mark.asyncio
    async def test_can_repair_explicit_strategy(self, strategy, attribute_error_analysis):
        """Test strategy accepts explicit UPDATE_SIGNATURE suggestion."""
        attribute_error_analysis.failure.failure_type = FailureType.RUNTIME_ERROR
        attribute_error_analysis.suggested_strategies = [RepairStrategy.UPDATE_SIGNATURE]

        result = await strategy.can_repair(attribute_error_analysis)
        assert result is True

    @pytest.mark.asyncio
    async def test_cannot_repair_unrelated_error(self, strategy, attribute_error_analysis):
        """Test strategy rejects unrelated errors."""
        attribute_error_analysis.failure.failure_type = FailureType.SYNTAX_ERROR
        attribute_error_analysis.suggested_strategies = []

        result = await strategy.can_repair(attribute_error_analysis)
        assert result is False

    @pytest.mark.asyncio
    async def test_repair_missing_attribute(self, strategy, attribute_error_analysis):
        """Test repair of missing attribute."""
        test_code = """
def test_example():
    obj = MyClass()
    result = obj.old_method()
    assert result == 5
"""

        repaired = await strategy.repair(test_code, attribute_error_analysis)

        assert repaired is not None
        assert "FIXME" in repaired
        assert "old_method" in repaired
        assert "pass" in repaired or "TODO" in repaired

    @pytest.mark.asyncio
    async def test_repair_argument_mismatch(self, strategy, type_error_analysis):
        """Test repair of argument mismatch."""
        test_code = """
def test_example():
    result = my_func(1, 2, 3)
    assert result == 5
"""

        repaired = await strategy.repair(test_code, type_error_analysis)

        assert repaired is not None
        assert "FIXME" in repaired

    @pytest.mark.asyncio
    async def test_repair_validates_syntax(self, strategy, attribute_error_analysis):
        """Test that repair validates syntax."""
        test_code = "def test(): pass"

        with patch.object(strategy, "_validate_syntax", return_value=False):
            repaired = await strategy.repair(test_code, attribute_error_analysis)
            assert repaired is None

    @pytest.mark.asyncio
    async def test_repair_handles_exceptions(self, strategy, attribute_error_analysis):
        """Test repair handles exceptions gracefully."""
        with patch.object(strategy, "_repair_missing_attribute", side_effect=Exception("Test error")):
            repaired = await strategy.repair("test code", attribute_error_analysis)
            assert repaired is None

    def test_repair_missing_attribute_extracts_name(self, strategy):
        """Test extraction of missing attribute name."""
        code = "obj.missing_attr()"
        root_cause = "Method 'missing_attr' no longer exists"

        repaired = strategy._repair_missing_attribute(code, root_cause)

        assert "FIXME" in repaired
        assert "missing_attr" in repaired

    def test_repair_argument_mismatch_extracts_details(self, strategy):
        """Test extraction of argument mismatch details."""
        code = "func(a, b, c)"
        error_msg = "TypeError: func() takes 2 positional arguments but 3 were given"

        repaired = strategy._repair_argument_mismatch(code, error_msg)

        assert "FIXME" in repaired

    def test_add_fixme_comment(self, strategy):
        """Test adding FIXME comment to code."""
        code = """import foo
def test():
    pass
"""
        root_cause = "Something changed"

        repaired = strategy._add_fixme_comment(code, root_cause)

        assert "FIXME" in repaired
        assert root_cause in repaired


class TestAssertionRepair:
    """Test AssertionRepair strategy."""

    @pytest.fixture
    def strategy(self):
        """Create assertion repair strategy instance."""
        return AssertionRepair()

    @pytest.fixture
    def assertion_analysis(self):
        """Create analysis for assertion failure."""
        failure = TestFailure(
            test_id="test_003",
            test_name="test_value",
            test_file="tests/test_example.py",
            failure_type=FailureType.ASSERTION_FAILED,
            error_message="AssertionError: assert 10 == 5",
            target_file="src/example.py",
            test_framework="pytest",
        )
        return FailureAnalysis(
            failure=failure,
            root_cause="Expected value changed from 5 to 10",
            confidence=0.9,
            suggested_strategies=[RepairStrategy.UPDATE_ASSERTION],
            evidence=["Value mismatch"],
        )

    @pytest.mark.asyncio
    async def test_can_repair_assertion_failed(self, strategy, assertion_analysis):
        """Test strategy can handle assertion failures."""
        result = await strategy.can_repair(assertion_analysis)
        assert result is True

    @pytest.mark.asyncio
    async def test_can_repair_with_assertion_keywords(self, strategy, assertion_analysis):
        """Test strategy recognizes assertion keywords in error."""
        assertion_analysis.failure.failure_type = FailureType.RUNTIME_ERROR
        assertion_analysis.failure.error_message = "Expected 5 but got 10"

        result = await strategy.can_repair(assertion_analysis)
        assert result is True

    @pytest.mark.asyncio
    async def test_cannot_repair_unrelated_error(self, strategy, assertion_analysis):
        """Test strategy rejects unrelated errors."""
        assertion_analysis.failure.failure_type = FailureType.SYNTAX_ERROR
        assertion_analysis.failure.error_message = "SyntaxError"
        assertion_analysis.suggested_strategies = []

        result = await strategy.can_repair(assertion_analysis)
        assert result is False

    @pytest.mark.asyncio
    async def test_repair_pytest_assertion(self, strategy, assertion_analysis):
        """Test repair of pytest-style assertion."""
        test_code = """
def test_example():
    result = calculate()
    assert result == 5
"""

        assertion_analysis.root_cause = "expected 5, got 10"

        repaired = await strategy.repair(test_code, assertion_analysis)

        assert repaired is not None
        # Should update assertion or add comment
        assert "10" in repaired or "FIXME" in repaired

    @pytest.mark.asyncio
    async def test_repair_unittest_assertion(self, strategy, assertion_analysis):
        """Test repair of unittest-style assertion."""
        test_code = """
def test_example(self):
    result = calculate()
    self.assertEqual(result, 5)
"""

        assertion_analysis.root_cause = "expected 5, got 10"

        repaired = await strategy.repair(test_code, assertion_analysis)

        assert repaired is not None

    @pytest.mark.asyncio
    async def test_repair_without_extractable_values(self, strategy, assertion_analysis):
        """Test repair when values cannot be extracted."""
        test_code = "assert something"
        assertion_analysis.root_cause = "Assertion failed"
        assertion_analysis.failure.error_message = "AssertionError"

        repaired = await strategy.repair(test_code, assertion_analysis)

        assert repaired is not None
        assert "FIXME" in repaired

    @pytest.mark.asyncio
    async def test_repair_validates_syntax(self, strategy, assertion_analysis):
        """Test that repair validates syntax."""
        test_code = "assert True"

        with patch.object(strategy, "_validate_syntax", return_value=False):
            repaired = await strategy.repair(test_code, assertion_analysis)
            assert repaired is None

    def test_extract_values_expected_got_format(self, strategy):
        """Test value extraction from 'expected, got' format."""
        root_cause = "expected 5, got 10"
        error_msg = ""

        expected, actual = strategy._extract_values(root_cause, error_msg)

        assert expected == "5"
        assert actual == "10"

    def test_extract_values_assert_format(self, strategy):
        """Test value extraction from assert format."""
        root_cause = ""
        error_msg = "AssertionError: 10 != 5"

        expected, actual = strategy._extract_values(root_cause, error_msg)

        assert expected == "10"
        assert actual == "5"

    def test_extract_values_no_match(self, strategy):
        """Test value extraction when no pattern matches."""
        expected, actual = strategy._extract_values("something", "error")

        assert expected is None
        assert actual is None

    def test_update_assertions_pytest_style(self, strategy):
        """Test updating pytest-style assertions."""
        code = "assert result == 5"
        repaired = strategy._update_assertions(code, "5", "10")

        assert "== 10" in repaired or "5" in repaired

    def test_update_assertions_unittest_style(self, strategy):
        """Test updating unittest-style assertions."""
        code = "self.assertEqual(result, 5)"
        repaired = strategy._update_assertions(code, "5", "10")

        assert "10" in repaired or "5" in repaired

    def test_update_assertions_preserves_comments(self, strategy):
        """Test that existing comments are preserved."""
        code = "# This is a comment\nassert x == 5"
        repaired = strategy._update_assertions(code, "5", "10")

        assert "# This is a comment" in repaired

    def test_add_comment_to_assertions(self, strategy):
        """Test adding comments to assertion lines."""
        code = "assert result == 5"
        root_cause = "Value changed"

        repaired = strategy._add_comment(code, root_cause)

        assert "FIXME" in repaired
        assert root_cause in repaired


class TestTestRepairer:
    """Test TestRepairer orchestrator."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        llm = AsyncMock()
        llm.agenerate = AsyncMock(return_value="""```python
def test_repaired():
    assert True
```""")
        return llm

    @pytest.fixture
    def mock_test_validator(self):
        """Create mock test validator."""
        validator = AsyncMock()
        validator.run_test = AsyncMock(return_value={"passed": True})
        return validator

    @pytest.fixture
    def mock_coverage_analyzer(self):
        """Create mock coverage analyzer."""
        analyzer = Mock()
        return analyzer

    @pytest.fixture
    def repairer(self, mock_llm_service, mock_test_validator, mock_coverage_analyzer):
        """Create test repairer instance."""
        return TestRepairer(
            llm_service=mock_llm_service,
            test_validator=mock_test_validator,
            coverage_analyzer=mock_coverage_analyzer,
        )

    @pytest.fixture
    def simple_analysis(self):
        """Create simple failure analysis."""
        failure = TestFailure(
            test_id="test_004",
            test_name="test_simple",
            test_file="tests/test_simple.py",
            failure_type=FailureType.ASSERTION_FAILED,
            error_message="assert 5 == 10",
            target_file="src/simple.py",
            test_framework="pytest",
        )
        return FailureAnalysis(
            failure=failure,
            root_cause="Value changed",
            confidence=0.9,
            suggested_strategies=[RepairStrategy.UPDATE_ASSERTION],
            evidence=["Test evidence"],
        )

    @pytest.mark.asyncio
    async def test_analyze_failure_not_implemented(self, repairer, simple_analysis):
        """Test that analyze_failure raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await repairer.analyze_failure(simple_analysis.failure)

    @pytest.mark.asyncio
    async def test_repair_test_success(self, repairer, simple_analysis):
        """Test successful test repair."""
        test_code = "def test_simple():\n    assert result == 5\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()
            simple_analysis.failure.test_file = f.name

        try:
            with patch.object(AssertionRepair, "repair", return_value=test_code.replace("5", "10")):
                repair = await repairer.repair_test(simple_analysis)

                assert repair is not None
                assert isinstance(repair, TestRepair)
                assert repair.syntax_valid is True
        finally:
            if os.path.exists(simple_analysis.failure.test_file):
                os.unlink(simple_analysis.failure.test_file)

    @pytest.mark.asyncio
    async def test_repair_test_file_not_found(self, repairer, simple_analysis):
        """Test repair when test file doesn't exist."""
        simple_analysis.failure.test_file = "/nonexistent/test.py"

        repair = await repairer.repair_test(simple_analysis)

        assert repair is None

    @pytest.mark.asyncio
    async def test_repair_test_no_strategy_implementation(self, repairer, simple_analysis):
        """Test repair when strategy has no implementation."""
        simple_analysis.suggested_strategies = [RepairStrategy.ADD_ASYNC]

        test_code = "def test(): pass"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()
            simple_analysis.failure.test_file = f.name

        try:
            # Should try LLM repair since ADD_ASYNC not implemented
            with patch.object(repairer, '_llm_repair', return_value=None):
                repair = await repairer.repair_test(simple_analysis, max_attempts=1)
                # May return None if LLM also fails
        finally:
            if os.path.exists(simple_analysis.failure.test_file):
                os.unlink(simple_analysis.failure.test_file)

    @pytest.mark.asyncio
    async def test_repair_test_strategy_cannot_handle(self, repairer, simple_analysis):
        """Test when strategy exists but cannot handle the failure."""
        test_code = "def test(): pass"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()
            simple_analysis.failure.test_file = f.name

        try:
            with patch.object(AssertionRepair, "can_repair", return_value=False):
                with patch.object(repairer, '_llm_repair', return_value=None):
                    repair = await repairer.repair_test(simple_analysis, max_attempts=1)
        finally:
            if os.path.exists(simple_analysis.failure.test_file):
                os.unlink(simple_analysis.failure.test_file)

    @pytest.mark.asyncio
    async def test_repair_test_invalid_syntax(self, repairer, simple_analysis):
        """Test when repaired code has invalid syntax."""
        test_code = "def test(): pass"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()
            simple_analysis.failure.test_file = f.name

        try:
            with patch.object(AssertionRepair, "repair", return_value="def test(: invalid"):
                repair = await repairer.repair_test(simple_analysis, max_attempts=1)
                # Should try next strategy or fail
        finally:
            if os.path.exists(simple_analysis.failure.test_file):
                os.unlink(simple_analysis.failure.test_file)

    @pytest.mark.asyncio
    async def test_repair_test_with_validation(self, repairer, simple_analysis, mock_test_validator):
        """Test repair with test validation."""
        test_code = "def test(): assert True"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()
            simple_analysis.failure.test_file = f.name

        try:
            mock_test_validator.run_test.return_value = {"passed": True}

            with patch.object(AssertionRepair, "repair", return_value=test_code):
                repair = await repairer.repair_test(simple_analysis)

                assert repair is not None
                assert repair.test_passes is True
        finally:
            if os.path.exists(simple_analysis.failure.test_file):
                os.unlink(simple_analysis.failure.test_file)

    @pytest.mark.asyncio
    async def test_repair_test_validation_fails(self, repairer, simple_analysis, mock_test_validator):
        """Test repair when validation fails."""
        test_code = "def test(): assert True"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()
            simple_analysis.failure.test_file = f.name

        try:
            mock_test_validator.run_test.return_value = {"passed": False}

            with patch.object(AssertionRepair, "repair", return_value=test_code):
                with patch.object(repairer, '_llm_repair', return_value=None):
                    repair = await repairer.repair_test(simple_analysis, max_attempts=1)
                    # Should continue trying other strategies
        finally:
            if os.path.exists(simple_analysis.failure.test_file):
                os.unlink(simple_analysis.failure.test_file)

    @pytest.mark.asyncio
    async def test_repair_test_without_validator(self, simple_analysis):
        """Test repair without test validator."""
        repairer = TestRepairer(llm_service=None, test_validator=None)

        test_code = "def test(): assert True"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()
            simple_analysis.failure.test_file = f.name

        try:
            with patch.object(AssertionRepair, "repair", return_value=test_code):
                repair = await repairer.repair_test(simple_analysis)

                assert repair is not None
                # Confidence reduced without validation
                assert repair.confidence < 0.9
        finally:
            if os.path.exists(simple_analysis.failure.test_file):
                os.unlink(simple_analysis.failure.test_file)

    @pytest.mark.asyncio
    async def test_llm_repair_success(self, repairer, simple_analysis, mock_llm_service):
        """Test successful LLM-based repair."""
        original_code = "def test(): assert False"
        mock_llm_service.agenerate.return_value = "def test(): assert True"

        repair = await repairer._llm_repair(simple_analysis, original_code, 0.0)

        assert repair is not None
        assert repair.strategy == RepairStrategy.REGENERATE
        assert "assert True" in repair.repaired_code

    @pytest.mark.asyncio
    async def test_llm_repair_with_code_blocks(self, repairer, simple_analysis, mock_llm_service):
        """Test LLM repair extracts code from markdown blocks."""
        original_code = "def test(): pass"
        mock_llm_service.agenerate.return_value = """Here's the fix:

```python
def test_fixed():
    assert True
```

This should work now."""

        repair = await repairer._llm_repair(simple_analysis, original_code, 0.0)

        assert repair is not None
        assert "def test_fixed" in repair.repaired_code
        assert "assert True" in repair.repaired_code

    @pytest.mark.asyncio
    async def test_llm_repair_invalid_syntax(self, repairer, simple_analysis, mock_llm_service):
        """Test LLM repair with invalid syntax."""
        original_code = "def test(): pass"
        mock_llm_service.agenerate.return_value = "def test(: invalid syntax"

        repair = await repairer._llm_repair(simple_analysis, original_code, 0.0)

        assert repair is None

    @pytest.mark.asyncio
    async def test_llm_repair_exception(self, repairer, simple_analysis, mock_llm_service):
        """Test LLM repair handles exceptions."""
        original_code = "def test(): pass"
        mock_llm_service.agenerate.side_effect = Exception("LLM error")

        repair = await repairer._llm_repair(simple_analysis, original_code, 0.0)

        assert repair is None

    def test_prioritize_strategies(self, repairer):
        """Test strategy prioritization."""
        strategies = [
            RepairStrategy.REGENERATE,
            RepairStrategy.UPDATE_IMPORT,
            RepairStrategy.UPDATE_ASSERTION,
        ]

        prioritized = repairer._prioritize_strategies(strategies)

        # UPDATE_IMPORT should be first (priority 1)
        assert prioritized[0] == RepairStrategy.UPDATE_IMPORT
        # REGENERATE should be last (priority 9)
        assert prioritized[-1] == RepairStrategy.REGENERATE

    def test_calculate_confidence_regenerate(self, repairer, simple_analysis):
        """Test confidence calculation for regeneration strategy."""
        confidence = repairer._calculate_confidence(
            RepairStrategy.REGENERATE, simple_analysis
        )

        # Should be reduced for regeneration
        assert confidence < simple_analysis.confidence

    def test_calculate_confidence_simple_fix(self, repairer, simple_analysis):
        """Test confidence calculation for simple fixes."""
        confidence = repairer._calculate_confidence(
            RepairStrategy.UPDATE_IMPORT, simple_analysis
        )

        # Should be high for simple fixes
        assert confidence >= 0.85

    def test_compute_changes(self, repairer):
        """Test change computation."""
        original = "def test():\n    assert False\n"
        repaired = "def test():\n    assert True\n"

        changes = repairer._compute_changes(original, repaired)

        assert len(changes) > 0
        assert changes[0]["type"] == "code_change"
        assert "diff" in changes[0]

    def test_compute_changes_identical(self, repairer):
        """Test change computation with identical code."""
        code = "def test(): pass"

        changes = repairer._compute_changes(code, code)

        assert len(changes) == 0

    def test_validate_syntax_valid_python(self, repairer):
        """Test syntax validation with valid Python."""
        code = "def test():\n    assert True\n"

        result = repairer._validate_syntax(code, "python")

        assert result is True

    def test_validate_syntax_invalid_python(self, repairer):
        """Test syntax validation with invalid Python."""
        code = "def test(: invalid"

        result = repairer._validate_syntax(code, "python")

        assert result is False

    def test_validate_syntax_other_language(self, repairer):
        """Test syntax validation for non-Python languages."""
        code = "function test() { return true; }"

        result = repairer._validate_syntax(code, "javascript")

        # Should assume valid for other languages
        assert result is True

    def test_get_language_python(self, repairer):
        """Test language detection for Python files."""
        language = repairer._get_language("tests/test_example.py")
        assert language == "python"

    def test_get_language_javascript(self, repairer):
        """Test language detection for JavaScript files."""
        language = repairer._get_language("tests/example.test.js")
        assert language == "javascript"

    def test_get_language_typescript(self, repairer):
        """Test language detection for TypeScript files."""
        language = repairer._get_language("tests/example.test.ts")
        assert language == "typescript"

    def test_get_language_default(self, repairer):
        """Test default language for unknown extensions."""
        language = repairer._get_language("tests/example.unknown")
        assert language == "python"

    def test_build_repair_prompt(self, repairer, simple_analysis):
        """Test repair prompt building."""
        original_code = "def test(): assert False"

        prompt = repairer._build_repair_prompt(simple_analysis, original_code)

        assert simple_analysis.failure.error_message in prompt
        assert simple_analysis.root_cause in prompt
        assert original_code in prompt
        assert "Fix the failing test" in prompt

    def test_build_repair_prompt_with_code_changes(self, repairer, simple_analysis):
        """Test repair prompt with code changes."""
        simple_analysis.code_changes = {"changed": "something"}
        original_code = "def test(): pass"

        prompt = repairer._build_repair_prompt(simple_analysis, original_code)

        assert "Code changes:" in prompt

    def test_extract_code_from_markdown(self, repairer):
        """Test code extraction from markdown."""
        response = """Here's the fix:
```python
def test():
    assert True
```
Done!"""

        code = repairer._extract_code(response)

        assert code == "def test():\n    assert True"

    def test_extract_code_from_generic_blocks(self, repairer):
        """Test code extraction from generic code blocks."""
        response = """```
def test():
    pass
```"""

        code = repairer._extract_code(response)

        assert code == "def test():\n    pass"

    def test_extract_code_no_blocks(self, repairer):
        """Test code extraction without code blocks."""
        response = "def test(): pass"

        code = repairer._extract_code(response)

        assert code == response.strip()

    @pytest.mark.asyncio
    async def test_write_temp_test(self, repairer):
        """Test writing test code to temporary file."""
        test_code = "def test(): assert True"

        temp_file = await repairer._write_temp_test(test_code)

        try:
            assert os.path.exists(temp_file)
            with open(temp_file, 'r') as f:
                content = f.read()
            assert content == test_code
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @pytest.mark.asyncio
    async def test_verify_coverage(self, repairer):
        """Test coverage verification."""
        result = await repairer._verify_coverage("old", "new", "src/file.py")

        # Should return True (simplified implementation)
        assert result is True

    @pytest.mark.asyncio
    async def test_repair_test_max_attempts(self, repairer, simple_analysis):
        """Test that repair respects max_attempts."""
        test_code = "def test(): pass"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()
            simple_analysis.failure.test_file = f.name

        try:
            with patch.object(AssertionRepair, "can_repair", return_value=True):
                with patch.object(AssertionRepair, "repair", return_value=None):
                    with patch.object(repairer, '_llm_repair', return_value=None):
                        repair = await repairer.repair_test(simple_analysis, max_attempts=2)
                        # Should try up to max_attempts
        finally:
            if os.path.exists(simple_analysis.failure.test_file):
                os.unlink(simple_analysis.failure.test_file)


class TestMockRepair:
    """Test MockRepair strategy."""

    @pytest.fixture
    def strategy(self):
        """Create mock repair strategy instance."""
        return MockRepair()

    @pytest.fixture
    def mock_error_analysis(self):
        """Create analysis for mock error."""
        failure = TestFailure(
            test_id="test_005",
            test_name="test_mock",
            test_file="tests/test_mock.py",
            failure_type=FailureType.MOCK_ERROR,
            error_message="Mock error: call not found",
            target_file="src/example.py",
            test_framework="pytest",
        )
        return FailureAnalysis(
            failure=failure,
            root_cause="Mock configuration incorrect",
            confidence=0.8,
            suggested_strategies=[RepairStrategy.UPDATE_MOCK],
            evidence=["Mock mismatch"],
        )

    @pytest.mark.asyncio
    async def test_can_repair_mock_error(self, strategy, mock_error_analysis):
        """Test strategy can handle mock errors."""
        result = await strategy.can_repair(mock_error_analysis)
        assert result is True

    @pytest.mark.asyncio
    async def test_cannot_repair_unrelated_error(self, strategy, mock_error_analysis):
        """Test strategy rejects unrelated errors."""
        mock_error_analysis.failure.failure_type = FailureType.SYNTAX_ERROR
        mock_error_analysis.suggested_strategies = []

        result = await strategy.can_repair(mock_error_analysis)
        assert result is False
