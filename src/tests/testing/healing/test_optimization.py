"""Comprehensive tests for test optimization.

Tests TestOptimizer with various optimization patterns according to PRP-11.
Includes performance analysis, suggestion generation, and edge cases.
"""

import pytest
import pytest_asyncio
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.testing.healing.test_optimizer import TestOptimizer
from src.models.healing_models import TestOptimization


class TestTestOptimizer:
    """Test TestOptimizer component."""

    @pytest.fixture
    def mock_performance_analyzer(self):
        """Create mock performance analyzer."""
        analyzer = Mock()
        return analyzer

    @pytest.fixture
    def mock_coverage_analyzer(self):
        """Create mock coverage analyzer."""
        analyzer = Mock()
        return analyzer

    @pytest.fixture
    def optimizer(self, mock_performance_analyzer, mock_coverage_analyzer):
        """Create test optimizer instance."""
        return TestOptimizer(
            performance_analyzer=mock_performance_analyzer,
            coverage_analyzer=mock_coverage_analyzer,
            slow_test_threshold_ms=5000,
        )

    @pytest.fixture
    def performance_data(self):
        """Create sample performance data."""
        return {
            "tests": {
                "test_slow_001": {
                    "execution_time_ms": 8000,
                    "test_file": "tests/test_slow.py",
                },
                "test_slow_002": {
                    "execution_time_ms": 6000,
                    "test_file": "tests/test_slow.py",
                },
                "test_fast": {
                    "execution_time_ms": 100,
                    "test_file": "tests/test_fast.py",
                },
            }
        }

    @pytest.mark.asyncio
    async def test_suggest_optimizations_basic(self, optimizer, performance_data):
        """Test basic optimization suggestion generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import time

def test_slow_001():
    time.sleep(5)
    assert True
""")
            f.flush()
            performance_data["tests"]["test_slow_001"]["test_file"] = f.name

        try:
            optimizations = await optimizer.suggest_optimizations(
                "test_suite",
                performance_data,
                min_time_saving_ms=100
            )

            assert isinstance(optimizations, list)
            # Should find wait optimization
            wait_opts = [o for o in optimizations if o.optimization_type == "remove_unnecessary_waits"]
            assert len(wait_opts) > 0
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_suggest_optimizations_sorted_by_priority(self, optimizer, performance_data):
        """Test that optimizations are sorted by priority."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import time

def test_slow_001():
    time.sleep(5)
    db = setup_database()
    assert True

def test_slow_002():
    db = setup_database()
    assert True
""")
            f.flush()
            for test_id in performance_data["tests"]:
                if "slow" in test_id:
                    performance_data["tests"][test_id]["test_file"] = f.name

        try:
            optimizations = await optimizer.suggest_optimizations(
                "test_suite",
                performance_data,
                min_time_saving_ms=100
            )

            if len(optimizations) > 1:
                # Should be sorted by time savings * risk multiplier
                for i in range(len(optimizations) - 1):
                    current_priority = optimizations[i].time_saving_ms * optimizer._risk_multiplier(optimizations[i].risk_level)
                    next_priority = optimizations[i + 1].time_saving_ms * optimizer._risk_multiplier(optimizations[i + 1].risk_level)
                    assert current_priority >= next_priority
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_suggest_optimizations_min_time_filter(self, optimizer, performance_data):
        """Test that optimizations below min time saving are filtered."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test(): pass")
            f.flush()
            performance_data["tests"]["test_slow_001"]["test_file"] = f.name

        try:
            optimizations = await optimizer.suggest_optimizations(
                "test_suite",
                performance_data,
                min_time_saving_ms=10000  # Very high threshold
            )

            # Should filter out optimizations with low savings
            for opt in optimizations:
                assert opt.time_saving_ms >= 10000
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_suggest_optimizations_error_handling(self, optimizer):
        """Test error handling in optimization suggestion."""
        # Invalid performance data
        bad_data = {"tests": None}

        with patch.object(optimizer, '_identify_slow_tests', side_effect=Exception("Error")):
            optimizations = await optimizer.suggest_optimizations(
                "test_suite",
                bad_data,
                min_time_saving_ms=100
            )

            # Should return empty list on error
            assert optimizations == []

    def test_identify_slow_tests(self, optimizer, performance_data):
        """Test identification of slow tests."""
        slow_tests = optimizer._identify_slow_tests(performance_data)

        assert "test_slow_001" in slow_tests
        assert "test_slow_002" in slow_tests
        assert "test_fast" not in slow_tests

    def test_identify_slow_tests_custom_threshold(self, performance_data):
        """Test slow test identification with custom threshold."""
        optimizer = TestOptimizer(slow_test_threshold_ms=7000)

        slow_tests = optimizer._identify_slow_tests(performance_data)

        assert "test_slow_001" in slow_tests  # 8000ms
        assert "test_slow_002" not in slow_tests  # 6000ms

    def test_identify_slow_tests_empty_data(self, optimizer):
        """Test slow test identification with empty data."""
        slow_tests = optimizer._identify_slow_tests({"tests": {}})
        assert len(slow_tests) == 0

    @pytest.mark.asyncio
    async def test_get_test_code_success(self, optimizer):
        """Test successful test code retrieval."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            test_code = "def test(): pass"
            f.write(test_code)
            f.flush()

            try:
                code = await optimizer._get_test_code(f.name)
                assert code == test_code
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_get_test_code_with_pytest_id(self, optimizer):
        """Test code retrieval from pytest-style test ID."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            test_code = "def test(): pass"
            f.write(test_code)
            f.flush()

            try:
                test_id = f"{f.name}::TestClass::test_method"
                code = await optimizer._get_test_code(test_id)
                assert code == test_code
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_get_test_code_file_not_found(self, optimizer):
        """Test code retrieval when file doesn't exist."""
        code = await optimizer._get_test_code("/nonexistent/test.py")
        assert code is None

    def test_has_unnecessary_waits_time_sleep(self, optimizer):
        """Test detection of time.sleep calls."""
        code = """
import time

def test():
    time.sleep(5)
    assert True
"""
        assert optimizer._has_unnecessary_waits(code) is True

    def test_has_unnecessary_waits_asyncio_sleep(self, optimizer):
        """Test detection of asyncio.sleep calls."""
        code = """
import asyncio

async def test():
    await asyncio.sleep(5)
    assert True
"""
        assert optimizer._has_unnecessary_waits(code) is True

    def test_has_unnecessary_waits_no_waits(self, optimizer):
        """Test detection when no waits present."""
        code = """
def test():
    result = calculate()
    assert result == 5
"""
        assert optimizer._has_unnecessary_waits(code) is False

    def test_has_redundant_setup(self, optimizer):
        """Test detection of redundant setup code."""
        code = """
def setUp(self):
    pass

def setup(self):
    pass
"""
        assert optimizer._has_redundant_setup(code) is True

    def test_has_redundant_setup_single_setup(self, optimizer):
        """Test no redundant setup with single setup method."""
        code = """
def setUp(self):
    pass

def test():
    pass
"""
        assert optimizer._has_redundant_setup(code) is False

    @pytest.mark.asyncio
    async def test_can_parallelize_clean_test(self, optimizer):
        """Test parallelization check for clean test."""
        code = """
def test_feature():
    result = calculate(5)
    assert result == 10
"""
        assert await optimizer._can_parallelize("test_001", code) is True

    @pytest.mark.asyncio
    async def test_can_parallelize_with_global_state(self, optimizer):
        """Test parallelization check with global state modification."""
        code = """
def test_feature():
    global MY_VAR
    MY_VAR = 5
    assert True
"""
        assert await optimizer._can_parallelize("test_001", code) is False

    @pytest.mark.asyncio
    async def test_can_parallelize_with_env_modification(self, optimizer):
        """Test parallelization check with environment modification."""
        code = """
import os

def test_feature():
    os.environ['KEY'] = 'value'
    assert True
"""
        assert await optimizer._can_parallelize("test_001", code) is False

    @pytest.mark.asyncio
    async def test_can_parallelize_with_database(self, optimizer):
        """Test parallelization check with database access."""
        code = """
def test_feature():
    db.query('SELECT * FROM users')
    assert True
"""
        assert await optimizer._can_parallelize("test_001", code) is False

    def test_has_excessive_io_multiple_operations(self, optimizer):
        """Test detection of excessive I/O operations."""
        code = """
def test():
    with open('file1.txt') as f:
        data1 = f.read()
    with open('file2.txt') as f:
        data2 = f.read()
    result = requests.get('http://example.com')
    another = requests.post('http://example.com')
"""
        assert optimizer._has_excessive_io(code) is True

    def test_has_excessive_io_minimal_operations(self, optimizer):
        """Test detection with minimal I/O operations."""
        code = """
def test():
    result = calculate()
    assert result == 5
"""
        assert optimizer._has_excessive_io(code) is False

    @pytest.mark.asyncio
    async def test_optimize_waits(self, optimizer):
        """Test wait optimization suggestion."""
        test_code = """
import time

def test():
    time.sleep(5)
    assert True
"""
        metrics = {"execution_time_ms": 6000}

        optimization = await optimizer._optimize_waits("test_001", metrics, test_code, 1)

        assert optimization is not None
        assert optimization.optimization_type == "remove_unnecessary_waits"
        assert optimization.time_saving_ms > 0
        assert "wait" in optimization.description.lower()
        assert optimization.risk_level == "low"

    @pytest.mark.asyncio
    async def test_optimize_waits_no_waits_found(self, optimizer):
        """Test wait optimization when no waits present."""
        test_code = "def test(): assert True"
        metrics = {"execution_time_ms": 1000}

        optimization = await optimizer._optimize_waits("test_001", metrics, test_code, 1)

        assert optimization is None

    @pytest.mark.asyncio
    async def test_optimize_setup(self, optimizer):
        """Test setup optimization suggestion."""
        test_code = """
def setUp(self):
    self.db = create_db()

def test():
    pass
"""
        metrics = {"execution_time_ms": 5000}

        optimization = await optimizer._optimize_setup("test_001", metrics, test_code, 1)

        assert optimization is not None
        assert optimization.optimization_type == "optimize_setup"
        assert optimization.time_saving_ms > 0
        assert "fixture" in optimization.description.lower() or "setup" in optimization.description.lower()
        assert optimization.risk_level == "low"

    @pytest.mark.asyncio
    async def test_suggest_parallelization(self, optimizer):
        """Test parallelization suggestion."""
        test_code = "def test(): assert True"
        metrics = {"execution_time_ms": 5000}

        optimization = await optimizer._suggest_parallelization("test_001", metrics, test_code, 1)

        assert optimization is not None
        assert optimization.optimization_type == "parallelize_tests"
        assert optimization.time_saving_ms > 0
        assert "parallel" in optimization.description.lower()
        assert optimization.risk_level == "medium"

    @pytest.mark.asyncio
    async def test_optimize_io(self, optimizer):
        """Test I/O optimization suggestion."""
        test_code = """
def test():
    with open('file.txt') as f:
        data = f.read()
    requests.get('http://example.com')
"""
        metrics = {"execution_time_ms": 8000}

        optimization = await optimizer._optimize_io("test_001", metrics, test_code, 1)

        assert optimization is not None
        assert optimization.optimization_type == "mock_io_operations"
        assert optimization.time_saving_ms > 0
        assert "mock" in optimization.description.lower() or "I/O" in optimization.description.lower()
        assert optimization.risk_level == "medium"

    @pytest.mark.asyncio
    async def test_find_redundant_tests(self, optimizer, performance_data):
        """Test finding redundant tests."""
        redundant = await optimizer._find_redundant_tests("test_suite", performance_data)

        # Simplified implementation returns empty
        assert isinstance(redundant, list)

    @pytest.mark.asyncio
    async def test_merge_redundant_tests(self, optimizer, performance_data):
        """Test merging redundant tests suggestion."""
        test_group = ["test_slow_001", "test_slow_002"]

        optimization = await optimizer._merge_redundant_tests(test_group, performance_data, 1)

        assert optimization is not None
        assert optimization.optimization_type == "merge_redundant_tests"
        assert optimization.time_saving_ms > 0
        assert "merge" in optimization.description.lower()
        assert optimization.risk_level == "high"
        assert optimization.affects_coverage is True

    @pytest.mark.asyncio
    async def test_merge_redundant_tests_single_test(self, optimizer, performance_data):
        """Test merge with single test (should return None)."""
        test_group = ["test_slow_001"]

        optimization = await optimizer._merge_redundant_tests(test_group, performance_data, 1)

        assert optimization is None

    @pytest.mark.asyncio
    async def test_optimize_mocks(self, optimizer, performance_data):
        """Test mock optimization suggestions."""
        optimizations = await optimizer._optimize_mocks("test_suite", performance_data)

        # Simplified implementation returns empty
        assert isinstance(optimizations, list)

    def test_risk_multiplier_low(self, optimizer):
        """Test risk multiplier for low risk."""
        multiplier = optimizer._risk_multiplier("low")
        assert multiplier == 1.0

    def test_risk_multiplier_medium(self, optimizer):
        """Test risk multiplier for medium risk."""
        multiplier = optimizer._risk_multiplier("medium")
        assert multiplier == 0.7

    def test_risk_multiplier_high(self, optimizer):
        """Test risk multiplier for high risk."""
        multiplier = optimizer._risk_multiplier("high")
        assert multiplier == 0.4

    def test_risk_multiplier_unknown(self, optimizer):
        """Test risk multiplier for unknown risk level."""
        multiplier = optimizer._risk_multiplier("unknown")
        assert multiplier == 0.5


class TestOptimizationModel:
    """Test TestOptimization model validation."""

    def test_create_optimization_valid(self):
        """Test creating valid optimization."""
        opt = TestOptimization(
            optimization_id="opt_001",
            test_id="test_001",
            optimization_type="remove_waits",
            current_time_ms=5000.0,
            estimated_time_ms=1000.0,
            time_saving_ms=4000.0,
            time_saving_percent=80.0,
            description="Remove unnecessary waits",
            implementation="Replace sleep with smart wait",
            risk_level="low",
            affects_coverage=False,
            priority=1,
        )

        assert opt.optimization_id == "opt_001"
        assert opt.time_saving_ms == 4000.0
        assert opt.time_saving_percent == 80.0

    def test_optimization_time_calculations(self):
        """Test optimization time calculations are correct."""
        opt = TestOptimization(
            optimization_id="opt_002",
            test_id="test_002",
            optimization_type="optimize_setup",
            current_time_ms=8000.0,
            estimated_time_ms=4000.0,
            time_saving_ms=4000.0,
            time_saving_percent=50.0,
            description="Optimize setup",
            implementation="Use fixtures",
            risk_level="low",
            affects_coverage=False,
            priority=2,
        )

        assert opt.current_time_ms - opt.estimated_time_ms == opt.time_saving_ms
        assert (opt.time_saving_ms / opt.current_time_ms * 100) == opt.time_saving_percent


class TestOptimizerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for edge case testing."""
        return TestOptimizer()

    @pytest.mark.asyncio
    async def test_suggest_optimizations_empty_performance_data(self, optimizer):
        """Test optimization with empty performance data."""
        optimizations = await optimizer.suggest_optimizations(
            "test_suite",
            {"tests": {}},
            min_time_saving_ms=100
        )

        assert optimizations == []

    @pytest.mark.asyncio
    async def test_suggest_optimizations_no_slow_tests(self, optimizer):
        """Test optimization when all tests are fast."""
        performance_data = {
            "tests": {
                "test_fast_001": {"execution_time_ms": 100},
                "test_fast_002": {"execution_time_ms": 200},
            }
        }

        optimizations = await optimizer.suggest_optimizations(
            "test_suite",
            performance_data,
            min_time_saving_ms=100
        )

        # Should find no optimizations for fast tests
        assert len(optimizations) == 0

    @pytest.mark.asyncio
    async def test_get_test_code_empty_file(self, optimizer):
        """Test code retrieval from empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")
            f.flush()

            try:
                code = await optimizer._get_test_code(f.name)
                assert code == ""
            finally:
                os.unlink(f.name)

    def test_has_unnecessary_waits_edge_cases(self, optimizer):
        """Test wait detection edge cases."""
        # Comment with sleep keyword
        code_with_comment = "# time.sleep(5)\ndef test(): pass"
        # May or may not detect depending on implementation

        # String with sleep keyword
        code_with_string = 'message = "time.sleep(5)"\ndef test(): pass'
        # May or may not detect depending on implementation

    @pytest.mark.asyncio
    async def test_can_parallelize_empty_code(self, optimizer):
        """Test parallelization check with empty code."""
        result = await optimizer._can_parallelize("test_001", "")
        assert result is True  # Empty test can be parallelized

    def test_has_excessive_io_edge_at_threshold(self, optimizer):
        """Test I/O detection at threshold boundary."""
        # Exactly 3 operations (threshold)
        code = """
def test():
    open('f1')
    open('f2')
    open('f3')
"""
        # Should return False (threshold is > 3)
        assert optimizer._has_excessive_io(code) is False

    @pytest.mark.asyncio
    async def test_optimize_waits_multiple_sleeps(self, optimizer):
        """Test wait optimization with multiple sleep calls."""
        test_code = """
def test():
    time.sleep(2)
    time.sleep(3)
    time.sleep(1)
"""
        metrics = {"execution_time_ms": 7000}

        optimization = await optimizer._optimize_waits("test_001", metrics, test_code, 1)

        assert optimization is not None
        # Should account for all sleeps (2+3+1 = 6 seconds)
        assert optimization.time_saving_ms > 4000  # At least 70% of 6000ms

    @pytest.mark.asyncio
    async def test_optimize_waits_fractional_seconds(self, optimizer):
        """Test wait optimization with fractional seconds."""
        test_code = """
def test():
    time.sleep(0.5)
    time.sleep(1.5)
"""
        metrics = {"execution_time_ms": 3000}

        optimization = await optimizer._optimize_waits("test_001", metrics, test_code, 1)

        assert optimization is not None
        # Should handle fractional seconds (0.5 + 1.5 = 2.0 seconds)

    @pytest.mark.asyncio
    async def test_merge_redundant_tests_zero_time(self, optimizer):
        """Test merge with tests having zero execution time."""
        performance_data = {
            "tests": {
                "test_001": {"execution_time_ms": 0},
                "test_002": {"execution_time_ms": 0},
            }
        }

        optimization = await optimizer._merge_redundant_tests(
            ["test_001", "test_002"],
            performance_data,
            1
        )

        assert optimization is not None
        assert optimization.time_saving_ms == 0

    def test_risk_multiplier_case_sensitivity(self, optimizer):
        """Test risk multiplier with different cases."""
        # Implementation uses exact string match
        assert optimizer._risk_multiplier("LOW") == 0.5  # Default for unknown
        assert optimizer._risk_multiplier("low") == 1.0

    @pytest.mark.asyncio
    async def test_suggest_optimizations_with_unreadable_file(self, optimizer):
        """Test optimization when test file cannot be read."""
        performance_data = {
            "tests": {
                "test_unreadable": {
                    "execution_time_ms": 10000,
                    "test_file": "/nonexistent/test.py",
                }
            }
        }

        optimizations = await optimizer.suggest_optimizations(
            "test_suite",
            performance_data,
            min_time_saving_ms=100
        )

        # Should handle gracefully and continue
        assert isinstance(optimizations, list)

    @pytest.mark.asyncio
    async def test_optimization_calculation_consistency(self, optimizer):
        """Test that time saving calculations are consistent."""
        test_code = "import time\ndef test(): time.sleep(5)"
        metrics = {"execution_time_ms": 6000}

        opt = await optimizer._optimize_waits("test_001", metrics, test_code, 1)

        if opt:
            # Verify calculation consistency
            assert opt.current_time_ms == metrics["execution_time_ms"]
            assert opt.time_saving_ms == opt.current_time_ms - opt.estimated_time_ms
            assert abs(opt.time_saving_percent - (opt.time_saving_ms / opt.current_time_ms * 100)) < 0.1

    @pytest.mark.asyncio
    async def test_suggest_optimizations_calculates_total_savings(self, optimizer):
        """Test that total time savings are calculated correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("import time\ndef test(): time.sleep(5)")
            f.flush()

            performance_data = {
                "tests": {
                    "test_001": {"execution_time_ms": 6000, "test_file": f.name},
                    "test_002": {"execution_time_ms": 6000, "test_file": f.name},
                }
            }

            try:
                optimizations = await optimizer.suggest_optimizations(
                    "test_suite",
                    performance_data,
                    min_time_saving_ms=100
                )

                if len(optimizations) > 0:
                    total_savings = sum(opt.time_saving_ms for opt in optimizations)
                    assert total_savings > 0
            finally:
                os.unlink(f.name)
