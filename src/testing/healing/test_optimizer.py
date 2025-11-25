"""Test optimization for performance improvements.

PATTERN: Identify and suggest test performance improvements
CRITICAL: Balance speed with test effectiveness
"""

import logging
import re
from typing import List, Dict, Any, Optional

from ...models.healing_models import TestOptimization

logger = logging.getLogger(__name__)


class TestOptimizer:
    """
    Analyze and suggest test performance optimizations.

    PATTERN: Multi-dimensional optimization analysis
    CRITICAL: Balance performance with test quality
    GOTCHA: Don't sacrifice test effectiveness for speed
    """

    def __init__(
        self,
        performance_analyzer=None,
        coverage_analyzer=None,
        slow_test_threshold_ms: int = 5000,
    ):
        """
        Initialize test optimizer.

        Args:
            performance_analyzer: Service for performance profiling
            coverage_analyzer: Service for coverage analysis
            slow_test_threshold_ms: Threshold for slow tests (default: 5000ms)
        """
        self.performance_analyzer = performance_analyzer
        self.coverage_analyzer = coverage_analyzer
        self.slow_test_threshold_ms = slow_test_threshold_ms
        self.logger = logger

    async def suggest_optimizations(
        self,
        test_suite: str,
        performance_data: Dict[str, Any],
        min_time_saving_ms: float = 100,
    ) -> List[TestOptimization]:
        """
        Analyze and suggest test optimizations.

        PATTERN: Multi-dimensional optimization analysis
        CRITICAL: Prioritize by time savings and risk

        Args:
            test_suite: Path to test suite or test IDs
            performance_data: Test performance metrics
            min_time_saving_ms: Minimum time saving to suggest (default: 100ms)

        Returns:
            List of optimization suggestions
        """
        optimizations: List[TestOptimization] = []
        optimization_id_counter = 0

        try:
            # Identify slow tests
            slow_tests = self._identify_slow_tests(performance_data)

            for test_id, metrics in slow_tests.items():
                logger.debug(f"Analyzing optimization for slow test: {test_id}")

                # Check for various optimization opportunities
                test_code = await self._get_test_code(test_id)

                if not test_code:
                    continue

                # Check for unnecessary waits/sleeps
                if self._has_unnecessary_waits(test_code):
                    opt = await self._optimize_waits(
                        test_id, metrics, test_code, optimization_id_counter
                    )
                    if opt and opt.time_saving_ms >= min_time_saving_ms:
                        optimizations.append(opt)
                        optimization_id_counter += 1

                # Check for redundant setup
                if self._has_redundant_setup(test_code):
                    opt = await self._optimize_setup(
                        test_id, metrics, test_code, optimization_id_counter
                    )
                    if opt and opt.time_saving_ms >= min_time_saving_ms:
                        optimizations.append(opt)
                        optimization_id_counter += 1

                # Check for parallelization opportunities
                if await self._can_parallelize(test_id, test_code):
                    opt = await self._suggest_parallelization(
                        test_id, metrics, test_code, optimization_id_counter
                    )
                    if opt and opt.time_saving_ms >= min_time_saving_ms:
                        optimizations.append(opt)
                        optimization_id_counter += 1

                # Check for excessive I/O
                if self._has_excessive_io(test_code):
                    opt = await self._optimize_io(
                        test_id, metrics, test_code, optimization_id_counter
                    )
                    if opt and opt.time_saving_ms >= min_time_saving_ms:
                        optimizations.append(opt)
                        optimization_id_counter += 1

            # Analyze test redundancy across suite
            redundant_tests = await self._find_redundant_tests(
                test_suite, performance_data
            )
            for test_group in redundant_tests:
                opt = await self._merge_redundant_tests(
                    test_group, performance_data, optimization_id_counter
                )
                if opt and opt.time_saving_ms >= min_time_saving_ms:
                    optimizations.append(opt)
                    optimization_id_counter += 1

            # Analyze mock usage
            mock_optimizations = await self._optimize_mocks(
                test_suite, performance_data
            )
            optimizations.extend(
                [
                    opt
                    for opt in mock_optimizations
                    if opt.time_saving_ms >= min_time_saving_ms
                ]
            )

            # Sort by priority (time savings * inverse risk)
            optimizations.sort(
                key=lambda o: o.time_saving_ms * self._risk_multiplier(o.risk_level),
                reverse=True,
            )

            logger.info(
                f"Generated {len(optimizations)} optimization suggestions "
                f"with potential {sum(o.time_saving_ms for o in optimizations):.0f}ms savings"
            )

            return optimizations

        except Exception as e:
            logger.error(f"Optimization analysis error: {e}", exc_info=True)
            return []

    def _identify_slow_tests(self, performance_data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Identify slow tests from performance data.

        Args:
            performance_data: Performance metrics

        Returns:
            Dictionary of slow tests with their metrics
        """
        slow_tests = {}

        tests = performance_data.get("tests", {})
        for test_id, metrics in tests.items():
            exec_time = metrics.get("execution_time_ms", 0)
            if exec_time >= self.slow_test_threshold_ms:
                slow_tests[test_id] = metrics

        return slow_tests

    async def _get_test_code(self, test_id: str) -> Optional[str]:
        """
        Get test code from test ID.

        Args:
            test_id: Test identifier

        Returns:
            Test code or None
        """
        try:
            # Extract file path from test ID (format: file::class::method)
            if "::" in test_id:
                file_path = test_id.split("::")[0]
            else:
                file_path = test_id

            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read test code for {test_id}: {e}")
            return None

    def _has_unnecessary_waits(self, test_code: str) -> bool:
        """
        Check for unnecessary wait/sleep calls.

        Args:
            test_code: Test code to check

        Returns:
            True if unnecessary waits found
        """
        # Look for sleep, wait, or delay calls
        patterns = [
            r"time\.sleep\(",
            r"asyncio\.sleep\(",
            r"await.*sleep\(",
            r"Thread\.sleep\(",
            r"setTimeout\(",
        ]

        for pattern in patterns:
            if re.search(pattern, test_code):
                return True

        return False

    def _has_redundant_setup(self, test_code: str) -> bool:
        """
        Check for redundant setup code.

        Args:
            test_code: Test code to check

        Returns:
            True if redundant setup found
        """
        # Look for repeated setup patterns
        # This is simplified - real implementation would use AST analysis
        setup_indicators = ["setUp", "setup", "beforeEach", "before_each"]

        # Count setup occurrences
        setup_count = sum(1 for indicator in setup_indicators if indicator in test_code)

        return setup_count > 1

    async def _can_parallelize(self, test_id: str, test_code: str) -> bool:
        """
        Check if test can be parallelized.

        Args:
            test_id: Test ID
            test_code: Test code

        Returns:
            True if test can be parallelized
        """
        # Tests that modify shared state or have dependencies can't be parallelized
        blocking_patterns = [
            r"global\s+\w+\s*=",
            r"os\.environ\[",
            r"sys\.",
            r"@pytest\.mark\.serial",
            r"database",
            r"db\.",
        ]

        for pattern in blocking_patterns:
            if re.search(pattern, test_code, re.IGNORECASE):
                return False

        return True

    def _has_excessive_io(self, test_code: str) -> bool:
        """
        Check for excessive I/O operations.

        Args:
            test_code: Test code

        Returns:
            True if excessive I/O found
        """
        io_patterns = [
            r"open\(",
            r"\.read\(",
            r"\.write\(",
            r"requests\.",
            r"urllib\.",
            r"fetch\(",
        ]

        io_count = sum(len(re.findall(pattern, test_code)) for pattern in io_patterns)

        return io_count > 3

    async def _optimize_waits(
        self, test_id: str, metrics: Dict, test_code: str, opt_id: int
    ) -> Optional[TestOptimization]:
        """
        Optimize unnecessary waits and sleeps.

        PATTERN: Replace fixed waits with smart waits

        Args:
            test_id: Test ID
            metrics: Test metrics
            test_code: Test code
            opt_id: Optimization ID

        Returns:
            TestOptimization or None
        """
        current_time = metrics.get("execution_time_ms", 0)

        # Estimate time spent in waits (rough heuristic)
        wait_patterns = re.findall(r"sleep\((\d+(?:\.\d+)?)\)", test_code)
        total_wait_time = sum(float(t) for t in wait_patterns) * 1000  # Convert to ms

        if total_wait_time == 0:
            return None

        # Assume we can eliminate 70% of wait time with smart waits
        estimated_time = current_time - (total_wait_time * 0.7)
        time_saving = current_time - estimated_time

        return TestOptimization(
            optimization_id=f"opt_{opt_id}_{test_id}",
            test_id=test_id,
            optimization_type="remove_unnecessary_waits",
            current_time_ms=float(current_time),
            estimated_time_ms=float(estimated_time),
            time_saving_ms=float(time_saving),
            time_saving_percent=float(
                (time_saving / current_time * 100) if current_time > 0 else 0
            ),
            description="Replace fixed waits with condition-based waits",
            implementation="""Replace fixed sleep calls with smart waits:

# Before:
await asyncio.sleep(5)  # Wait for operation

# After:
await wait_for_condition(
    condition=lambda: operation_complete(),
    timeout=5,
    poll_interval=0.1
)

This reduces wait time when operations complete early.""",
            risk_level="low",
            affects_coverage=False,
            priority=1,
        )

    async def _optimize_setup(
        self, test_id: str, metrics: Dict, test_code: str, opt_id: int
    ) -> Optional[TestOptimization]:
        """
        Optimize redundant setup code.

        Args:
            test_id: Test ID
            metrics: Test metrics
            test_code: Test code
            opt_id: Optimization ID

        Returns:
            TestOptimization or None
        """
        current_time = metrics.get("execution_time_ms", 0)

        # Estimate setup overhead (assume 20% of test time)
        setup_time = current_time * 0.2
        estimated_time = current_time - (setup_time * 0.5)  # 50% reduction
        time_saving = current_time - estimated_time

        return TestOptimization(
            optimization_id=f"opt_{opt_id}_{test_id}",
            test_id=test_id,
            optimization_type="optimize_setup",
            current_time_ms=float(current_time),
            estimated_time_ms=float(estimated_time),
            time_saving_ms=float(time_saving),
            time_saving_percent=float(
                (time_saving / current_time * 100) if current_time > 0 else 0
            ),
            description="Use fixtures or move setup to class/module level",
            implementation="""Move repeated setup to shared fixtures:

# Before: Setup in each test
def test_feature_1():
    db = setup_database()
    user = create_test_user()
    # test logic

def test_feature_2():
    db = setup_database()
    user = create_test_user()
    # test logic

# After: Shared fixture
@pytest.fixture(scope="class")
def test_db():
    db = setup_database()
    yield db
    teardown_database(db)

def test_feature_1(test_db):
    # test logic

def test_feature_2(test_db):
    # test logic""",
            risk_level="low",
            affects_coverage=False,
            priority=2,
        )

    async def _suggest_parallelization(
        self, test_id: str, metrics: Dict, test_code: str, opt_id: int
    ) -> Optional[TestOptimization]:
        """
        Suggest test parallelization.

        Args:
            test_id: Test ID
            metrics: Test metrics
            test_code: Test code
            opt_id: Optimization ID

        Returns:
            TestOptimization or None
        """
        current_time = metrics.get("execution_time_ms", 0)

        # Parallelization can reduce suite time, not individual test time
        # Estimate based on CPU cores (assume 4 cores)
        cores = 4
        estimated_time = current_time / cores
        time_saving = current_time - estimated_time

        return TestOptimization(
            optimization_id=f"opt_{opt_id}_{test_id}",
            test_id=test_id,
            optimization_type="parallelize_tests",
            current_time_ms=float(current_time),
            estimated_time_ms=float(estimated_time),
            time_saving_ms=float(time_saving),
            time_saving_percent=float(
                (time_saving / current_time * 100) if current_time > 0 else 0
            ),
            description="Enable parallel test execution",
            implementation="""Enable test parallelization:

# Pytest with xdist
pytest -n 4  # Use 4 cores

# Or in pytest.ini
[pytest]
addopts = -n auto

# For specific tests
@pytest.mark.parallel
def test_can_run_parallel():
    # test logic

Note: Ensure tests are isolated and don't share state.""",
            risk_level="medium",
            affects_coverage=False,
            priority=3,
        )

    async def _optimize_io(
        self, test_id: str, metrics: Dict, test_code: str, opt_id: int
    ) -> Optional[TestOptimization]:
        """
        Optimize excessive I/O operations.

        Args:
            test_id: Test ID
            metrics: Test metrics
            test_code: Test code
            opt_id: Optimization ID

        Returns:
            TestOptimization or None
        """
        current_time = metrics.get("execution_time_ms", 0)

        # Estimate I/O overhead (assume 40% for I/O-heavy tests)
        io_time = current_time * 0.4
        estimated_time = current_time - (io_time * 0.8)  # 80% reduction with mocking
        time_saving = current_time - estimated_time

        return TestOptimization(
            optimization_id=f"opt_{opt_id}_{test_id}",
            test_id=test_id,
            optimization_type="mock_io_operations",
            current_time_ms=float(current_time),
            estimated_time_ms=float(estimated_time),
            time_saving_ms=float(time_saving),
            time_saving_percent=float(
                (time_saving / current_time * 100) if current_time > 0 else 0
            ),
            description="Mock file I/O and network operations",
            implementation="""Replace real I/O with mocks:

# Before: Real file operations
def test_file_processing():
    with open('data.txt', 'r') as f:
        data = f.read()
    result = process(data)
    assert result == expected

# After: Mock file operations
@patch('builtins.open', mock_open(read_data='test data'))
def test_file_processing():
    result = process_file('data.txt')
    assert result == expected

This eliminates I/O overhead while maintaining test behavior.""",
            risk_level="medium",
            affects_coverage=False,
            priority=2,
        )

    async def _find_redundant_tests(
        self, test_suite: str, performance_data: Dict[str, Any]
    ) -> List[List[str]]:
        """
        Find redundant tests that test the same functionality.

        Args:
            test_suite: Test suite path
            performance_data: Performance data

        Returns:
            List of test groups (each group contains redundant tests)
        """
        # Simplified implementation
        # Real implementation would use coverage analysis and AST comparison
        redundant_groups: List[List[str]] = []

        # This would analyze test coverage and identify overlapping tests
        # For now, return empty list
        return redundant_groups

    async def _merge_redundant_tests(
        self, test_group: List[str], performance_data: Dict[str, Any], opt_id: int
    ) -> Optional[TestOptimization]:
        """
        Suggest merging redundant tests.

        Args:
            test_group: Group of redundant tests
            performance_data: Performance data
            opt_id: Optimization ID

        Returns:
            TestOptimization or None
        """
        if len(test_group) < 2:
            return None

        # Calculate time savings from merging
        tests = performance_data.get("tests", {})
        total_time = sum(
            tests.get(test_id, {}).get("execution_time_ms", 0) for test_id in test_group
        )

        # Merged test would take roughly the time of the longest test
        longest_time = max(
            tests.get(test_id, {}).get("execution_time_ms", 0) for test_id in test_group
        )

        time_saving = total_time - longest_time

        return TestOptimization(
            optimization_id=f"opt_{opt_id}_merge",
            test_id=",".join(test_group),
            optimization_type="merge_redundant_tests",
            current_time_ms=float(total_time),
            estimated_time_ms=float(longest_time),
            time_saving_ms=float(time_saving),
            time_saving_percent=float(
                (time_saving / total_time * 100) if total_time > 0 else 0
            ),
            description=f"Merge {len(test_group)} redundant tests",
            implementation=f"""Merge redundant tests into one comprehensive test:

Tests to merge:
{chr(10).join(f"  - {test_id}" for test_id in test_group)}

These tests cover overlapping functionality and can be combined.""",
            risk_level="high",
            affects_coverage=True,
            priority=5,
        )

    async def _optimize_mocks(
        self, test_suite: str, performance_data: Dict[str, Any]
    ) -> List[TestOptimization]:
        """
        Optimize mock usage in tests.

        Args:
            test_suite: Test suite path
            performance_data: Performance data

        Returns:
            List of mock optimizations
        """
        optimizations: List[TestOptimization] = []

        # This would analyze mock patterns and suggest improvements
        # For now, return empty list
        return optimizations

    def _risk_multiplier(self, risk_level: str) -> float:
        """
        Get risk multiplier for prioritization.

        Args:
            risk_level: Risk level (low/medium/high)

        Returns:
            Multiplier for priority calculation
        """
        multipliers = {
            "low": 1.0,
            "medium": 0.7,
            "high": 0.4,
        }
        return multipliers.get(risk_level, 0.5)
