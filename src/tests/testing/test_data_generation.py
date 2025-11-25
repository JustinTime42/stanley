"""Tests for test data generation."""

import pytest

from src.testing.data_generator import TestDataGenerator
from src.models.testing_models import TestDataSpec, TestDataStrategy


class TestDataGenerationTests:
    """Test data generation functionality."""

    @pytest.fixture
    def generator(self):
        """Create data generator instance."""
        return TestDataGenerator()

    @pytest.mark.asyncio
    async def test_generate_int_boundaries(self, generator):
        """Test integer boundary value generation."""
        spec = TestDataSpec(
            parameter_name="value",
            data_type="int",
            constraints={"min": 0, "max": 100},
            strategy=TestDataStrategy.BOUNDARY,
        )

        values = await generator.generate_test_data(spec)

        # Should include boundaries
        assert 0 in values  # Min
        assert 100 in values  # Max
        assert 1 in values  # Min + 1
        assert 99 in values  # Max - 1

        # Should have invalid values
        assert -1 in spec.invalid_values
        assert 101 in spec.invalid_values

    @pytest.mark.asyncio
    async def test_generate_string_boundaries(self, generator):
        """Test string boundary value generation."""
        spec = TestDataSpec(
            parameter_name="text",
            data_type="str",
            constraints={"min_length": 1, "max_length": 10},
            strategy=TestDataStrategy.BOUNDARY,
        )

        values = await generator.generate_test_data(spec)

        # Should include empty string
        assert "" in values

        # Should have strings of various lengths
        has_min_length = any(len(v) == 1 for v in values if isinstance(v, str))
        has_max_length = any(len(v) == 10 for v in values if isinstance(v, str))

        assert has_min_length
        assert has_max_length

    @pytest.mark.asyncio
    async def test_generate_edge_cases(self, generator):
        """Test edge case generation for different types."""
        # Integer edge cases
        int_cases = generator.generate_edge_cases("int")
        assert 0 in int_cases
        assert -1 in int_cases
        assert 1 in int_cases

        # String edge cases
        str_cases = generator.generate_edge_cases("str")
        assert "" in str_cases
        assert " " in str_cases

        # List edge cases
        list_cases = generator.generate_edge_cases("list")
        assert [] in list_cases

    @pytest.mark.asyncio
    async def test_random_values_generation(self, generator):
        """Test random value generation."""
        spec = TestDataSpec(
            parameter_name="value",
            data_type="int",
            constraints={"min": 0, "max": 100},
            strategy=TestDataStrategy.RANDOM,
        )

        values = await generator.generate_test_data(spec)

        # Should generate multiple values
        assert len(values) > 0

        # All values should be within constraints
        for value in values:
            assert 0 <= value <= 100
