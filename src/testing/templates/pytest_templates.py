"""Pytest test templates."""

# Pytest unit test template with AAA pattern
PYTEST_UNIT_TEMPLATE = """import pytest
from unittest.mock import Mock, patch
{imports}

class Test{class_name}:
    \"\"\"Test suite for {target_function}\"\"\"

{fixtures}

    def test_{test_name}(self{fixture_params}):
        \"\"\"Test: {test_description}\"\"\"
        # Arrange
{setup_code}

        # Act
{action_code}

        # Assert
{assertions}
{teardown_code}
"""

# Pytest fixture template
PYTEST_FIXTURE_TEMPLATE = """
    @pytest.fixture{scope_decorator}
    def {fixture_name}(self):
        \"\"\"Fixture for {fixture_description}\"\"\"
{setup_code}
        yield {yield_value}
{teardown_code}
"""

# Pytest parametrize template
PYTEST_PARAMETRIZE_TEMPLATE = """
    @pytest.mark.parametrize("{param_names}", [
{param_values}
    ])
    def test_{test_name}_parametrized(self, {param_names}{fixture_params}):
        \"\"\"Parametrized test: {test_description}\"\"\"
        # Arrange
{setup_code}

        # Act
{action_code}

        # Assert
{assertions}
"""

# Pytest property-based test template (Hypothesis)
PYTEST_PROPERTY_TEMPLATE = """import pytest
from hypothesis import given, strategies as st
{imports}

class Test{class_name}Properties:
    \"\"\"Property-based tests for {target_function}\"\"\"

    @given({strategies})
    def test_{property_name}(self, {param_names}):
        \"\"\"Property: {property_description}\"\"\"
        # Arrange
{setup_code}

        # Act
        result = {function_call}

        # Assert property
{property_assertions}
"""

# Pytest integration test template
PYTEST_INTEGRATION_TEMPLATE = """import pytest
{imports}

class Test{class_name}Integration:
    \"\"\"Integration tests for {target_module}\"\"\"

    @pytest.fixture(scope="module")
    def setup_integration(self):
        \"\"\"Setup integration test environment\"\"\"
{setup_code}
        yield
{teardown_code}

    def test_{test_name}(self, setup_integration):
        \"\"\"Integration test: {test_description}\"\"\"
        # Arrange
{arrange_code}

        # Act
{act_code}

        # Assert
{assertions}
"""

# Pytest async test template
PYTEST_ASYNC_TEMPLATE = """import pytest
{imports}

class Test{class_name}Async:
    \"\"\"Async tests for {target_function}\"\"\"

    @pytest.mark.asyncio
    async def test_{test_name}(self{fixture_params}):
        \"\"\"Test: {test_description}\"\"\"
        # Arrange
{setup_code}

        # Act
{action_code}

        # Assert
{assertions}
"""

# Pytest mock template
PYTEST_MOCK_TEMPLATE = """
    @patch('{mock_target}')
    def test_{test_name}_with_mock(self, mock_{mock_name}{fixture_params}):
        \"\"\"Test with mocked {mock_name}: {test_description}\"\"\"
        # Arrange mock
        mock_{mock_name}.return_value = {return_value}

        # Act
{action_code}

        # Assert
{assertions}
        # Verify mock calls
        mock_{mock_name}.assert_called_once_with({expected_args})
"""

# Pytest edge case test template
PYTEST_EDGE_CASE_TEMPLATE = """
    def test_{test_name}_edge_case_{case_name}(self{fixture_params}):
        \"\"\"Edge case test: {test_description}\"\"\"
        # Arrange edge case
{setup_code}

        # Act
{action_code}

        # Assert
{assertions}
"""
