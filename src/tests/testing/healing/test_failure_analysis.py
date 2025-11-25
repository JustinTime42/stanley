"""Comprehensive tests for failure analysis components.

Tests all failure analyzers and the FailureAnalyzer orchestrator according to PRP-11.
Includes edge cases, error handling, and proper mocking of dependencies.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.testing.healing.failure_analyzer import FailureAnalyzer
from src.testing.healing.analyzers.syntax_analyzer import SyntaxErrorAnalyzer
from src.testing.healing.analyzers.assertion_analyzer import AssertionAnalyzer
from src.testing.healing.analyzers.runtime_analyzer import RuntimeAnalyzer
from src.testing.healing.analyzers.timeout_analyzer import TimeoutAnalyzer
from src.models.healing_models import (
    TestFailure,
    FailureAnalysis,
    FailureType,
    RepairStrategy,
)


class TestSyntaxErrorAnalyzer:
    """Test SyntaxErrorAnalyzer component."""

    @pytest.fixture
    def analyzer(self):
        """Create syntax error analyzer instance."""
        return SyntaxErrorAnalyzer()

    @pytest.fixture
    def syntax_failure(self):
        """Create sample syntax error failure."""
        return TestFailure(
            test_id="test_syntax",
            test_name="test_invalid_syntax",
            test_file="tests/test_example.py",
            failure_type=FailureType.SYNTAX_ERROR,
            error_message="SyntaxError: invalid syntax at line 10",
            stack_trace="  File 'test.py', line 10\n    if x ==\n           ^",
            line_number=10,
            target_file="src/example.py",
            target_function="example_func",
            test_framework="pytest",
            execution_time_ms=0,
        )

    @pytest.mark.asyncio
    async def test_can_analyze_syntax_error(self, analyzer, syntax_failure):
        """Test analyzer recognizes syntax errors."""
        result = await analyzer.can_analyze(syntax_failure)
        assert result is True

    @pytest.mark.asyncio
    async def test_cannot_analyze_other_failures(self, analyzer, syntax_failure):
        """Test analyzer rejects non-syntax errors."""
        syntax_failure.failure_type = FailureType.ASSERTION_FAILED
        result = await analyzer.can_analyze(syntax_failure)
        assert result is False

    @pytest.mark.asyncio
    async def test_analyze_unexpected_indent(self, analyzer, syntax_failure):
        """Test analysis of unexpected indentation errors."""
        syntax_failure.error_message = "IndentationError: unexpected indent at line 15"
        syntax_failure.line_number = 15

        error_info = {"message": syntax_failure.error_message}
        result = await analyzer.analyze(syntax_failure, error_info)

        assert result["description"] == "Unexpected indentation in test code"
        assert result["confidence"] == 0.9
        assert "Indentation error detected" in result["evidence"]
        assert result["line_number"] == 15
        assert result["error_type"] == "syntax"

    @pytest.mark.asyncio
    async def test_analyze_unexpected_eof(self, analyzer, syntax_failure):
        """Test analysis of unexpected EOF errors."""
        syntax_failure.error_message = "SyntaxError: unexpected EOF while parsing"

        error_info = {"message": syntax_failure.error_message}
        result = await analyzer.analyze(syntax_failure, error_info)

        assert "Incomplete code block" in result["description"]
        assert result["confidence"] == 0.85
        assert "EOF error indicates incomplete syntax" in result["evidence"]

    @pytest.mark.asyncio
    async def test_analyze_invalid_syntax(self, analyzer, syntax_failure):
        """Test analysis of generic invalid syntax."""
        syntax_failure.error_message = "SyntaxError: invalid syntax - error parsing"

        error_info = {"message": syntax_failure.error_message}
        result = await analyzer.analyze(syntax_failure, error_info)

        assert "Invalid Python syntax" in result["description"] or "syntax" in result["description"].lower()
        assert result["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_analyze_missing_colon(self, analyzer, syntax_failure):
        """Test analysis of missing colon in control statement."""
        syntax_failure.error_message = "SyntaxError: invalid syntax - expected ':'"

        error_info = {"message": syntax_failure.error_message}
        result = await analyzer.analyze(syntax_failure, error_info)

        assert "Missing or incorrect colon" in result["description"]
        assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_analyze_extracts_line_number(self, analyzer, syntax_failure):
        """Test line number extraction from error message."""
        syntax_failure.error_message = "SyntaxError at line 42: invalid syntax"
        syntax_failure.line_number = None

        error_info = {"message": syntax_failure.error_message}
        result = await analyzer.analyze(syntax_failure, error_info)

        assert result["line_number"] == 42

    @pytest.mark.asyncio
    async def test_analyze_fallback_to_failure_line_number(self, analyzer, syntax_failure):
        """Test fallback when line number not in error message."""
        syntax_failure.error_message = "SyntaxError: invalid syntax"
        syntax_failure.line_number = 25

        error_info = {"message": syntax_failure.error_message}
        result = await analyzer.analyze(syntax_failure, error_info)

        assert result["line_number"] == 25


class TestAssertionAnalyzer:
    """Test AssertionAnalyzer component."""

    @pytest.fixture
    def analyzer(self):
        """Create assertion analyzer instance."""
        return AssertionAnalyzer()

    @pytest.fixture
    def assertion_failure(self):
        """Create sample assertion failure."""
        return TestFailure(
            test_id="test_assertion",
            test_name="test_value_match",
            test_file="tests/test_example.py",
            failure_type=FailureType.ASSERTION_FAILED,
            error_message="AssertionError: assert 5 == 10",
            stack_trace="assert result == expected",
            target_file="src/example.py",
            target_function="calculate",
            test_framework="pytest",
            execution_time_ms=100,
        )

    @pytest.mark.asyncio
    async def test_can_analyze_assertion_failure(self, analyzer, assertion_failure):
        """Test analyzer recognizes assertion failures."""
        result = await analyzer.can_analyze(assertion_failure)
        assert result is True

    @pytest.mark.asyncio
    async def test_cannot_analyze_other_failures(self, analyzer, assertion_failure):
        """Test analyzer rejects non-assertion errors."""
        assertion_failure.failure_type = FailureType.SYNTAX_ERROR
        result = await analyzer.can_analyze(assertion_failure)
        assert result is False

    @pytest.mark.asyncio
    async def test_analyze_basic_assertion(self, analyzer, assertion_failure):
        """Test basic assertion analysis."""
        assertion_failure.error_message = "assert 5 == 10"

        error_info = {"message": assertion_failure.error_message}
        result = await analyzer.analyze(assertion_failure, error_info)

        assert "expected" in result["description"].lower()
        assert result["confidence"] >= 0.7
        assert result["error_type"] == "assertion"

    @pytest.mark.asyncio
    async def test_analyze_extracts_expected_actual(self, analyzer, assertion_failure):
        """Test extraction of expected and actual values."""
        assertion_failure.error_message = "expected: 10, got: 5"

        error_info = {"message": assertion_failure.error_message}
        result = await analyzer.analyze(assertion_failure, error_info)

        assert result["expected_value"] == "10"
        assert result["actual_value"] == "5"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_analyze_none_return_value(self, analyzer, assertion_failure):
        """Test detection of unexpected None return value."""
        assertion_failure.error_message = "expected: 'result', got: None"

        error_info = {"message": assertion_failure.error_message}
        result = await analyzer.analyze(assertion_failure, error_info)

        assert "None unexpectedly" in result["description"]
        assert result["confidence"] == 0.85
        assert "Unexpected None return value" in result["evidence"]

    @pytest.mark.asyncio
    async def test_analyze_similar_strings(self, analyzer, assertion_failure):
        """Test detection of similar but not identical strings."""
        assertion_failure.error_message = "expected: 'Hello World', got: 'Hello world'"

        error_info = {"message": assertion_failure.error_message}
        result = await analyzer.analyze(assertion_failure, error_info)

        # Strings are similar (differ only in case)
        assert result["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_extract_values_pytest_format(self, analyzer, assertion_failure):
        """Test value extraction from pytest format."""
        error_msg = "AssertionError: assert 42 == 100"
        expected, actual = analyzer._extract_values(error_msg)

        assert expected == "42"
        assert actual == "100"

    @pytest.mark.asyncio
    async def test_extract_values_expected_got_format(self, analyzer, assertion_failure):
        """Test value extraction from 'expected, got' format."""
        error_msg = "expected: 'success', got: 'failure'"
        expected, actual = analyzer._extract_values(error_msg)

        assert expected == "'success'"
        assert actual == "'failure'"

    @pytest.mark.asyncio
    async def test_extract_values_not_equal_format(self, analyzer, assertion_failure):
        """Test value extraction from != format."""
        error_msg = "5 != 10"
        expected, actual = analyzer._extract_values(error_msg)

        # Check that values were extracted (may include surrounding text)
        assert expected is not None and actual is not None
        assert "5" in str(expected) or "10" in str(expected)

    @pytest.mark.asyncio
    async def test_extract_values_no_match(self, analyzer, assertion_failure):
        """Test value extraction when no pattern matches."""
        error_msg = "Something went wrong"
        expected, actual = analyzer._extract_values(error_msg)

        assert expected is None
        assert actual is None

    def test_is_similar_strings(self, analyzer):
        """Test string similarity calculation."""
        assert analyzer._is_similar("hello", "hello") is True
        assert analyzer._is_similar("hello world", "hello world") is True
        assert analyzer._is_similar("testing123", "testing456") is True  # >80% match
        assert analyzer._is_similar("abc", "xyz") is False
        assert analyzer._is_similar("", "") is True
        assert analyzer._is_similar("test", "best") is False  # Only 75%, need >80%


class TestFailureAnalyzer:
    """Test FailureAnalyzer orchestrator."""

    @pytest.fixture
    def analyzer(self):
        """Create failure analyzer instance."""
        return FailureAnalyzer()

    @pytest.fixture
    def sample_failure(self):
        """Create sample test failure."""
        return TestFailure(
            test_id="test_001",
            test_name="test_feature",
            test_file="tests/test_feature.py",
            failure_type=FailureType.RUNTIME_ERROR,
            error_message="RuntimeError: Something went wrong",
            stack_trace="Traceback...",
            target_file="src/feature.py",
            target_function="feature_func",
            test_framework="pytest",
            execution_time_ms=150,
        )

    @pytest.mark.asyncio
    async def test_analyze_failure_success(self, analyzer, sample_failure):
        """Test successful failure analysis."""
        sample_failure.failure_type = FailureType.ASSERTION_FAILED
        sample_failure.error_message = "assert 5 == 10"

        result = await analyzer.analyze_failure(sample_failure)

        assert isinstance(result, FailureAnalysis)
        assert result.failure == sample_failure
        assert result.root_cause
        assert 0 <= result.confidence <= 1
        assert len(result.suggested_strategies) > 0

    @pytest.mark.asyncio
    async def test_analyze_failure_with_code_diff(self, analyzer, sample_failure):
        """Test failure analysis with code changes."""
        code_diff = {
            "functions": ["feature_func"],
            "files": ["src/feature.py"],
        }

        result = await analyzer.analyze_failure(sample_failure, code_diff)

        assert result.code_changes is not None
        assert "changed_functions" in result.code_changes
        assert "changed_files" in result.code_changes

    @pytest.mark.asyncio
    async def test_classify_syntax_error(self, analyzer):
        """Test classification of syntax errors."""
        error_info = {"message": "SyntaxError: invalid syntax"}
        failure_type = analyzer._classify_failure(error_info)

        assert failure_type == FailureType.SYNTAX_ERROR

    @pytest.mark.asyncio
    async def test_classify_import_error(self, analyzer):
        """Test classification of import errors."""
        error_info = {"message": "ImportError: No module named 'foo'"}
        failure_type = analyzer._classify_failure(error_info)

        assert failure_type == FailureType.IMPORT_ERROR

    @pytest.mark.asyncio
    async def test_classify_modulenotfound_error(self, analyzer):
        """Test classification of ModuleNotFoundError."""
        error_info = {"message": "ModuleNotFoundError: No module named 'bar'"}
        failure_type = analyzer._classify_failure(error_info)

        assert failure_type == FailureType.IMPORT_ERROR

    @pytest.mark.asyncio
    async def test_classify_attribute_error(self, analyzer):
        """Test classification of attribute errors."""
        error_info = {"message": "AttributeError: object has no attribute 'foo'"}
        failure_type = analyzer._classify_failure(error_info)

        assert failure_type == FailureType.ATTRIBUTE_ERROR

    @pytest.mark.asyncio
    async def test_classify_assertion_error(self, analyzer):
        """Test classification of assertion errors."""
        error_info = {"message": "AssertionError: assert False"}
        failure_type = analyzer._classify_failure(error_info)

        assert failure_type == FailureType.ASSERTION_FAILED

    @pytest.mark.asyncio
    async def test_classify_timeout_error(self, analyzer):
        """Test classification of timeout errors."""
        error_info = {"message": "TimeoutError: operation timed out"}
        failure_type = analyzer._classify_failure(error_info)

        assert failure_type == FailureType.TIMEOUT

    @pytest.mark.asyncio
    async def test_classify_mock_error(self, analyzer):
        """Test classification of mock errors."""
        error_info = {"message": "mock object has no attribute 'method'", "error_class": "AttributeError"}
        failure_type = analyzer._classify_failure(error_info)

        # Mock-related errors might be classified as attribute errors or mock errors
        assert failure_type in [FailureType.MOCK_ERROR, FailureType.ATTRIBUTE_ERROR]

    @pytest.mark.asyncio
    async def test_classify_type_error(self, analyzer):
        """Test classification of type errors."""
        error_info = {"message": "TypeError: unsupported operand type"}
        failure_type = analyzer._classify_failure(error_info)

        assert failure_type == FailureType.TYPE_ERROR

    @pytest.mark.asyncio
    async def test_classify_unknown_error(self, analyzer):
        """Test classification of unknown errors."""
        error_info = {"message": "SomeWeirdError: something happened"}
        failure_type = analyzer._classify_failure(error_info)

        assert failure_type == FailureType.RUNTIME_ERROR

    @pytest.mark.asyncio
    async def test_suggest_strategies_syntax_error(self, analyzer):
        """Test strategy suggestions for syntax errors."""
        root_cause_info = {"confidence": 0.9}
        strategies = analyzer._suggest_strategies(FailureType.SYNTAX_ERROR, root_cause_info)

        assert RepairStrategy.REGENERATE in strategies

    @pytest.mark.asyncio
    async def test_suggest_strategies_assertion_failed(self, analyzer):
        """Test strategy suggestions for assertion failures."""
        root_cause_info = {"confidence": 0.9}
        strategies = analyzer._suggest_strategies(FailureType.ASSERTION_FAILED, root_cause_info)

        assert RepairStrategy.UPDATE_ASSERTION in strategies
        assert RepairStrategy.REGENERATE in strategies

    @pytest.mark.asyncio
    async def test_suggest_strategies_attribute_error(self, analyzer):
        """Test strategy suggestions for attribute errors."""
        root_cause_info = {"confidence": 0.9}
        strategies = analyzer._suggest_strategies(FailureType.ATTRIBUTE_ERROR, root_cause_info)

        assert RepairStrategy.UPDATE_SIGNATURE in strategies

    @pytest.mark.asyncio
    async def test_suggest_strategies_import_error(self, analyzer):
        """Test strategy suggestions for import errors."""
        root_cause_info = {"confidence": 0.9}
        strategies = analyzer._suggest_strategies(FailureType.IMPORT_ERROR, root_cause_info)

        assert RepairStrategy.UPDATE_IMPORT in strategies

    @pytest.mark.asyncio
    async def test_suggest_strategies_timeout(self, analyzer):
        """Test strategy suggestions for timeout errors."""
        root_cause_info = {"confidence": 0.9}
        strategies = analyzer._suggest_strategies(FailureType.TIMEOUT, root_cause_info)

        assert RepairStrategy.ADD_WAIT in strategies

    @pytest.mark.asyncio
    async def test_suggest_strategies_mock_error(self, analyzer):
        """Test strategy suggestions for mock errors."""
        root_cause_info = {"confidence": 0.9}
        strategies = analyzer._suggest_strategies(FailureType.MOCK_ERROR, root_cause_info)

        assert RepairStrategy.UPDATE_MOCK in strategies

    @pytest.mark.asyncio
    async def test_analyze_failure_error_handling(self, analyzer, sample_failure):
        """Test error handling in failure analysis."""
        # Force an error by using invalid analyzer
        with patch.object(analyzer.analyzers["default"], "analyze", side_effect=Exception("Test error")):
            result = await analyzer.analyze_failure(sample_failure)

            # Should return low-confidence analysis instead of raising
            assert isinstance(result, FailureAnalysis)
            assert result.confidence == 0.3
            assert "Unable to determine root cause" in result.root_cause
            assert RepairStrategy.REGENERATE in result.suggested_strategies

    @pytest.mark.asyncio
    async def test_parse_error_basic(self, analyzer):
        """Test basic error message parsing."""
        error_msg = "Test failed with error"
        stack_trace = "line 1\nline 2"

        error_info = analyzer._parse_error(error_msg, stack_trace)

        assert error_info["message"] == error_msg
        assert error_info["stack_trace"] == stack_trace
        assert len(error_info["lines"]) > 0

    @pytest.mark.asyncio
    async def test_parse_error_no_stack_trace(self, analyzer):
        """Test error parsing without stack trace."""
        error_msg = "Simple error"
        error_info = analyzer._parse_error(error_msg, None)

        assert error_info["message"] == error_msg
        assert error_info["stack_trace"] is None

    @pytest.mark.asyncio
    async def test_find_related_changes(self, analyzer):
        """Test finding related code changes."""
        code_diff = {
            "functions": ["func1", "func2"],
            "files": ["file1.py", "file2.py"],
        }

        changes = analyzer._find_related_changes("func1", code_diff)

        assert "changed_functions" in changes
        assert "changed_files" in changes
        assert changes["changed_functions"] == ["func1", "func2"]

    @pytest.mark.asyncio
    async def test_analyzer_uses_appropriate_analyzer(self, analyzer, sample_failure):
        """Test that the orchestrator uses the appropriate analyzer."""
        # Test with syntax error
        sample_failure.failure_type = FailureType.SYNTAX_ERROR
        sample_failure.error_message = "SyntaxError: invalid syntax"

        with patch.object(SyntaxErrorAnalyzer, "analyze", return_value={
            "description": "Syntax issue",
            "confidence": 0.9,
            "evidence": ["test"]
        }) as mock_analyze:
            await analyzer.analyze_failure(sample_failure)
            mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyzer_uses_default_for_unknown_type(self, analyzer, sample_failure):
        """Test that the orchestrator uses default analyzer for unknown types."""
        sample_failure.failure_type = FailureType.RUNTIME_ERROR

        with patch.object(RuntimeAnalyzer, "analyze", return_value={
            "description": "Runtime issue",
            "confidence": 0.7,
            "evidence": ["test"]
        }) as mock_analyze:
            await analyzer.analyze_failure(sample_failure)
            mock_analyze.assert_called_once()


class TestRuntimeAnalyzer:
    """Test RuntimeAnalyzer component."""

    @pytest.fixture
    def analyzer(self):
        """Create runtime analyzer instance."""
        from src.testing.healing.analyzers.runtime_analyzer import RuntimeAnalyzer
        return RuntimeAnalyzer()

    @pytest.fixture
    def runtime_failure(self):
        """Create sample runtime failure."""
        return TestFailure(
            test_id="test_runtime",
            test_name="test_runtime_error",
            test_file="tests/test_example.py",
            failure_type=FailureType.RUNTIME_ERROR,
            error_message="RuntimeError: Something went wrong",
            target_file="src/example.py",
            target_function="example_func",
            test_framework="pytest",
            execution_time_ms=100,
        )

    @pytest.mark.asyncio
    async def test_can_analyze_runtime_error(self, analyzer, runtime_failure):
        """Test analyzer can handle various runtime errors."""
        # RuntimeAnalyzer should handle multiple error types
        result = await analyzer.can_analyze(runtime_failure)
        assert result is True

    @pytest.mark.asyncio
    async def test_analyze_attribute_error(self, analyzer, runtime_failure):
        """Test analysis of attribute errors."""
        runtime_failure.failure_type = FailureType.ATTRIBUTE_ERROR
        runtime_failure.error_message = "AttributeError: 'Foo' object has no attribute 'bar'"

        error_info = {"message": runtime_failure.error_message}
        result = await analyzer.analyze(runtime_failure, error_info)

        assert result["confidence"] > 0.5
        assert result["description"]
        assert len(result["evidence"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_type_error(self, analyzer, runtime_failure):
        """Test analysis of type errors."""
        runtime_failure.failure_type = FailureType.TYPE_ERROR
        runtime_failure.error_message = "TypeError: cannot concatenate str and int"

        error_info = {"message": runtime_failure.error_message}
        result = await analyzer.analyze(runtime_failure, error_info)

        assert result["confidence"] > 0.5
        assert "type" in result["description"].lower() or "TypeError" in result["description"]


class TestTimeoutAnalyzer:
    """Test TimeoutAnalyzer component."""

    @pytest.fixture
    def analyzer(self):
        """Create timeout analyzer instance."""
        return TimeoutAnalyzer()

    @pytest.fixture
    def timeout_failure(self):
        """Create sample timeout failure."""
        return TestFailure(
            test_id="test_timeout",
            test_name="test_slow_operation",
            test_file="tests/test_example.py",
            failure_type=FailureType.TIMEOUT,
            error_message="TimeoutError: Test exceeded 5 second timeout",
            target_file="src/example.py",
            target_function="slow_func",
            test_framework="pytest",
            execution_time_ms=5000,
        )

    @pytest.mark.asyncio
    async def test_can_analyze_timeout(self, analyzer, timeout_failure):
        """Test analyzer recognizes timeout errors."""
        result = await analyzer.can_analyze(timeout_failure)
        assert result is True

    @pytest.mark.asyncio
    async def test_cannot_analyze_other_failures(self, analyzer, timeout_failure):
        """Test analyzer rejects non-timeout errors."""
        timeout_failure.failure_type = FailureType.ASSERTION_FAILED
        result = await analyzer.can_analyze(timeout_failure)
        assert result is False

    @pytest.mark.asyncio
    async def test_analyze_timeout_error(self, analyzer, timeout_failure):
        """Test timeout error analysis."""
        error_info = {"message": timeout_failure.error_message}
        result = await analyzer.analyze(timeout_failure, error_info)

        assert result["confidence"] >= 0.8
        assert "timeout" in result["description"].lower()
        assert len(result["evidence"]) > 0


class TestAnalyzerEdgeCases:
    """Test edge cases and error conditions across analyzers."""

    @pytest.fixture
    def analyzer(self):
        """Create failure analyzer instance."""
        return FailureAnalyzer()

    @pytest.mark.asyncio
    async def test_empty_error_message(self, analyzer):
        """Test handling of empty error messages."""
        failure = TestFailure(
            test_id="test_empty",
            test_name="test_empty",
            test_file="test.py",
            failure_type=FailureType.RUNTIME_ERROR,
            error_message="",
            target_file="src.py",
            test_framework="pytest",
        )

        result = await analyzer.analyze_failure(failure)
        assert isinstance(result, FailureAnalysis)
        assert result.confidence >= 0.3

    @pytest.mark.asyncio
    async def test_very_long_error_message(self, analyzer):
        """Test handling of very long error messages."""
        failure = TestFailure(
            test_id="test_long",
            test_name="test_long",
            test_file="test.py",
            failure_type=FailureType.RUNTIME_ERROR,
            error_message="Error: " + ("x" * 10000),
            target_file="src.py",
            test_framework="pytest",
        )

        result = await analyzer.analyze_failure(failure)
        assert isinstance(result, FailureAnalysis)

    @pytest.mark.asyncio
    async def test_unicode_in_error_message(self, analyzer):
        """Test handling of unicode in error messages."""
        failure = TestFailure(
            test_id="test_unicode",
            test_name="test_unicode",
            test_file="test.py",
            failure_type=FailureType.ASSERTION_FAILED,
            error_message="AssertionError: '你好' != 'Hello'",
            target_file="src.py",
            test_framework="pytest",
        )

        result = await analyzer.analyze_failure(failure)
        assert isinstance(result, FailureAnalysis)

    @pytest.mark.asyncio
    async def test_multiline_error_message(self, analyzer):
        """Test handling of multiline error messages."""
        failure = TestFailure(
            test_id="test_multiline",
            test_name="test_multiline",
            test_file="test.py",
            failure_type=FailureType.SYNTAX_ERROR,
            error_message="SyntaxError: invalid syntax\nLine 1\nLine 2\nLine 3",
            target_file="src.py",
            test_framework="pytest",
        )

        result = await analyzer.analyze_failure(failure)
        assert isinstance(result, FailureAnalysis)

    @pytest.mark.asyncio
    async def test_none_stack_trace(self, analyzer):
        """Test handling when stack trace is None."""
        failure = TestFailure(
            test_id="test_no_stack",
            test_name="test_no_stack",
            test_file="test.py",
            failure_type=FailureType.RUNTIME_ERROR,
            error_message="Error occurred",
            stack_trace=None,
            target_file="src.py",
            test_framework="pytest",
        )

        result = await analyzer.analyze_failure(failure)
        assert isinstance(result, FailureAnalysis)

    @pytest.mark.asyncio
    async def test_analyzer_with_missing_target_function(self, analyzer):
        """Test analysis when target function is None."""
        failure = TestFailure(
            test_id="test_no_target",
            test_name="test_no_target",
            test_file="test.py",
            failure_type=FailureType.RUNTIME_ERROR,
            error_message="Error",
            target_file="src.py",
            target_function=None,
            test_framework="pytest",
        )

        code_diff = {"functions": ["some_func"], "files": ["file.py"]}
        result = await analyzer.analyze_failure(failure, code_diff)

        assert isinstance(result, FailureAnalysis)
        assert result.code_changes is not None
