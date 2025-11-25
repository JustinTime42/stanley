"""Test data generation engine."""

import logging
import sys
import random
import string
from typing import List, Any, Dict, Optional

from ..models.testing_models import TestDataSpec, TestDataStrategy

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """
    Generate intelligent test data based on types and constraints.

    PATTERN: Type-aware generation with multiple strategies
    CRITICAL: Cover edge cases systematically
    GOTCHA: Must respect type constraints and boundaries
    """

    def __init__(self):
        """Initialize test data generator."""
        self.logger = logger
        self.random = random.Random()  # Use instance for reproducibility

    async def generate_test_data(
        self,
        parameter_spec: TestDataSpec,
        strategy: Optional[TestDataStrategy] = None,
    ) -> List[Any]:
        """
        Generate test data for parameter.

        PATTERN: Type-aware generation with strategies

        Args:
            parameter_spec: Parameter specification
            strategy: Optional override for generation strategy

        Returns:
            List of generated test values
        """
        strategy = strategy or parameter_spec.strategy

        if strategy == TestDataStrategy.BOUNDARY:
            return await self._generate_boundary_values(parameter_spec)
        elif strategy == TestDataStrategy.EQUIVALENCE:
            return await self._generate_equivalence_partitions(parameter_spec)
        elif strategy == TestDataStrategy.RANDOM:
            return await self._generate_random_values(parameter_spec)
        elif strategy == TestDataStrategy.PROPERTY:
            return await self._generate_property_based(parameter_spec)
        else:
            # Default to boundary
            return await self._generate_boundary_values(parameter_spec)

    async def _generate_boundary_values(
        self, spec: TestDataSpec
    ) -> List[Any]:
        """
        Generate boundary test values.

        PATTERN: Type-specific boundaries
        CRITICAL: Include min, max, zero, edge cases

        Args:
            spec: Test data specification

        Returns:
            List of boundary values
        """
        values = []
        data_type = spec.data_type.lower()

        if data_type in ["int", "integer", "long"]:
            values.extend(await self._generate_int_boundaries(spec))

        elif data_type in ["float", "double", "decimal"]:
            values.extend(await self._generate_float_boundaries(spec))

        elif data_type in ["str", "string", "text"]:
            values.extend(await self._generate_string_boundaries(spec))

        elif data_type in ["list", "array"]:
            values.extend(await self._generate_list_boundaries(spec))

        elif data_type in ["dict", "map", "object"]:
            values.extend(await self._generate_dict_boundaries(spec))

        elif data_type in ["bool", "boolean"]:
            values.extend([True, False])

        elif data_type in ["any", "object", "unknown"]:
            # For untyped/Any parameters, default to integer values
            # This is safest as most functions accept numeric inputs
            values.extend([1, 0, -1, 5, 10])

        else:
            # Truly unknown type - use sensible defaults (integers first, not None)
            values.extend([1, 0, -1, 5, 10])

        # Store generated values in spec
        spec.values = values

        return values

    async def _generate_int_boundaries(self, spec: TestDataSpec) -> List[int]:
        """Generate integer boundary values."""
        values = []
        min_val = spec.constraints.get("min", -sys.maxsize)
        max_val = spec.constraints.get("max", sys.maxsize)

        # Boundary values
        values.extend([
            min_val,  # Lower boundary
            min_val + 1,  # Just above lower
            max_val - 1,  # Just below upper
            max_val,  # Upper boundary
        ])

        # Special values
        if min_val <= 0 <= max_val:
            values.append(0)  # Zero
        if min_val < 0:
            values.append(-1)  # Negative
        if max_val > 0:
            values.append(1)  # Positive

        # Add invalid values for negative testing
        spec.invalid_values = [min_val - 1, max_val + 1]

        return values

    async def _generate_float_boundaries(self, spec: TestDataSpec) -> List[float]:
        """Generate float boundary values."""
        values = []
        min_val = spec.constraints.get("min", -sys.float_info.max)
        max_val = spec.constraints.get("max", sys.float_info.max)

        # Boundary values
        values.extend([
            min_val,
            min_val + 0.1,
            max_val - 0.1,
            max_val,
        ])

        # Special values
        if min_val <= 0.0 <= max_val:
            values.extend([0.0, -0.0])
        if min_val < 0:
            values.append(-1.0)
        if max_val > 0:
            values.append(1.0)

        # Edge cases
        values.extend([
            0.1,
            0.9,
            -0.1,
            -0.9,
        ])

        # Invalid values
        spec.invalid_values = [
            float('inf'),
            float('-inf'),
            float('nan'),
        ]

        return values

    async def _generate_string_boundaries(self, spec: TestDataSpec) -> List[str]:
        """Generate string boundary values."""
        values = []
        min_len = spec.constraints.get("min_length", 0)
        max_len = spec.constraints.get("max_length", 1000)

        # Length boundaries
        values.extend([
            "",  # Empty
            "a" * min_len,  # Minimum length
            "a" * max_len,  # Maximum length
            "a" * (max_len // 2),  # Medium length
        ])

        # Special characters
        values.extend([
            "Special!@#$%^&*()_+",
            "Unicode: Ω≈ç√∫˜µ≤≥÷",
            "Escaped\\n\\t\\r",
            "Quotes'\"",
            "Spaces   multiple",
            "\n\r\t",  # Whitespace
        ])

        # Pattern-specific if provided
        pattern = spec.constraints.get("pattern")
        if pattern == "email":
            values.extend([
                "test@example.com",
                "invalid-email",
                "@example.com",
                "test@",
            ])
        elif pattern == "url":
            values.extend([
                "https://example.com",
                "http://example.com/path",
                "invalid-url",
                "ftp://example.com",
            ])

        # Invalid values
        spec.invalid_values = [
            "a" * (max_len + 1),  # Too long
        ]
        if min_len > 0:
            spec.invalid_values.append("a" * (min_len - 1))  # Too short

        return values

    async def _generate_list_boundaries(self, spec: TestDataSpec) -> List[List]:
        """Generate list boundary values."""
        values = []
        min_len = spec.constraints.get("min_length", 0)
        max_len = spec.constraints.get("max_length", 100)

        # Length boundaries
        values.extend([
            [],  # Empty list
            [1],  # Single element
            list(range(min_len)) if min_len > 0 else [1],  # Minimum length
            list(range(max_len)),  # Maximum length
            list(range(max_len // 2)),  # Medium length
        ])

        # Special cases
        values.extend([
            [None],  # Single None
            [1, 2, 3, 2, 1],  # Duplicates
            [1, None, 3],  # Mixed with None
        ])

        # Invalid values
        spec.invalid_values = [
            list(range(max_len + 1)),  # Too long
        ]

        return values

    async def _generate_dict_boundaries(self, spec: TestDataSpec) -> List[Dict]:
        """Generate dictionary boundary values."""
        values = []

        # Dictionary edge cases
        values.extend([
            {},  # Empty dict
            {"key": "value"},  # Single entry
            {"key1": "value1", "key2": "value2"},  # Multiple entries
            {"nested": {"key": "value"}},  # Nested
            {"special!@#": "value"},  # Special key
            {"key": None},  # None value
        ])

        return values

    async def _generate_equivalence_partitions(
        self, spec: TestDataSpec
    ) -> List[Any]:
        """
        Generate values from equivalence partitions.

        PATTERN: Partition input space into equivalent classes

        Args:
            spec: Test data specification

        Returns:
            Representative values from each partition
        """
        values = []
        data_type = spec.data_type.lower()

        if data_type in ["int", "integer"]:
            # Partitions: negative, zero, positive
            min_val = spec.constraints.get("min", -100)
            max_val = spec.constraints.get("max", 100)

            if min_val < 0:
                values.append(min_val // 2)  # Representative negative
            values.append(0)
            if max_val > 0:
                values.append(max_val // 2)  # Representative positive

        elif data_type in ["str", "string"]:
            # Partitions: empty, short, medium, long
            max_len = spec.constraints.get("max_length", 100)
            values.extend([
                "",
                "short",
                "medium" * 5,
                "long" * (max_len // 20) if max_len > 20 else "long",
            ])

        return values

    async def _generate_random_values(
        self, spec: TestDataSpec, count: int = 10
    ) -> List[Any]:
        """
        Generate random test values.

        Args:
            spec: Test data specification
            count: Number of random values

        Returns:
            List of random values
        """
        values = []
        data_type = spec.data_type.lower()

        for _ in range(count):
            if data_type in ["int", "integer"]:
                min_val = spec.constraints.get("min", -1000)
                max_val = spec.constraints.get("max", 1000)
                values.append(self.random.randint(min_val, max_val))

            elif data_type in ["float", "double"]:
                min_val = spec.constraints.get("min", -1000.0)
                max_val = spec.constraints.get("max", 1000.0)
                values.append(self.random.uniform(min_val, max_val))

            elif data_type in ["str", "string"]:
                min_len = spec.constraints.get("min_length", 0)
                max_len = spec.constraints.get("max_length", 50)
                length = self.random.randint(min_len, max_len)
                values.append(
                    "".join(
                        self.random.choices(string.ascii_letters + string.digits, k=length)
                    )
                )

            elif data_type in ["bool", "boolean"]:
                values.append(self.random.choice([True, False]))

        return values

    async def _generate_property_based(
        self, spec: TestDataSpec
    ) -> List[Any]:
        """
        Generate values for property-based testing.

        PATTERN: Use hypothesis-style strategies

        Args:
            spec: Test data specification

        Returns:
            Generated values
        """
        # For now, use boundary + random
        boundary_values = await self._generate_boundary_values(spec)
        random_values = await self._generate_random_values(spec, count=5)

        return boundary_values + random_values

    def generate_edge_cases(self, data_type: str) -> List[Any]:
        """
        Generate common edge cases for a data type.

        Args:
            data_type: Type name

        Returns:
            List of edge case values
        """
        data_type = data_type.lower()

        if data_type in ["int", "integer"]:
            return [0, -1, 1, sys.maxsize, -sys.maxsize]

        elif data_type in ["float", "double"]:
            return [0.0, -0.0, 1.0, -1.0, float('inf'), float('-inf')]

        elif data_type in ["str", "string"]:
            return ["", " ", "\n", "\t", "null", "undefined"]

        elif data_type in ["list", "array"]:
            return [[], [None], [1], [[]], [1, 2, 3]]

        elif data_type in ["dict", "object"]:
            return [{}, {"key": None}, {"": ""}, {"nested": {}}]

        elif data_type in ["bool", "boolean"]:
            return [True, False]

        elif data_type in ["any", "unknown"]:
            # For Any/unknown types, default to integers
            return [0, 1, -1, sys.maxsize]

        else:
            # Default to integer edge cases
            return [0, 1, -1]
