"""Boundary value analysis testing strategy."""

import logging
from typing import List, Dict, Any, Optional

from ...models.testing_models import TestCase, TestType, TestFramework
from ...models.analysis_models import CodeEntity
from ..data_generator import TestDataGenerator, TestDataSpec

logger = logging.getLogger(__name__)


class BoundaryTestStrategy:
    """
    Generate tests using boundary value analysis.

    PATTERN: Strategy pattern for test generation
    CRITICAL: Identify parameter boundaries and generate edge cases
    GOTCHA: Must respect type constraints and domains
    """

    def __init__(self):
        """Initialize boundary test strategy."""
        self.logger = logger
        self.data_generator = TestDataGenerator()

    async def generate_boundary_tests(
        self,
        function: CodeEntity,
        framework: TestFramework,
    ) -> List[TestCase]:
        """
        Generate boundary value tests for function.

        Args:
            function: Function to test
            framework: Testing framework

        Returns:
            List of boundary test cases
        """
        test_cases = []

        # Extract parameters from function signature
        parameters = await self._extract_parameters(function)

        for param in parameters:
            # Create test data specification
            spec = TestDataSpec(
                parameter_name=param["name"],
                data_type=param["type"],
                constraints=param.get("constraints", {}),
            )

            # Generate boundary values
            boundary_values = await self.data_generator._generate_boundary_values(spec)

            # Create test case for each boundary value
            for idx, value in enumerate(boundary_values):
                test_case = await self._create_boundary_test(
                    function=function,
                    parameter_name=param["name"],
                    boundary_value=value,
                    boundary_index=idx,
                    framework=framework,
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    async def _extract_parameters(
        self, function: CodeEntity
    ) -> List[Dict[str, Any]]:
        """
        Extract function parameters from signature.

        Args:
            function: Function to analyze

        Returns:
            List of parameter specifications
        """
        # Simplified parameter extraction
        # In production, would parse actual function signature

        parameters = []

        # Example parameter extraction
        signature = function.signature or ""

        # Very simple parsing - in production would use AST
        if "int" in signature.lower():
            parameters.append({
                "name": "value",
                "type": "int",
                "constraints": {"min": -1000, "max": 1000},
            })
        elif "str" in signature.lower():
            parameters.append({
                "name": "text",
                "type": "str",
                "constraints": {"min_length": 0, "max_length": 100},
            })

        return parameters

    async def _create_boundary_test(
        self,
        function: CodeEntity,
        parameter_name: str,
        boundary_value: Any,
        boundary_index: int,
        framework: TestFramework,
    ) -> Optional[TestCase]:
        """
        Create test case for boundary value.

        Args:
            function: Function being tested
            parameter_name: Parameter name
            boundary_value: Boundary value to test
            boundary_index: Index of boundary
            framework: Testing framework

        Returns:
            TestCase or None
        """
        test_id = f"boundary_{function.name}_{parameter_name}_{boundary_index}"

        # Build test body based on framework
        if framework == TestFramework.PYTEST:
            test_body = f"""
def test_{function.name}_boundary_{parameter_name}_{boundary_index}():
    \"\"\"Test {function.name} with boundary value for {parameter_name}.\"\"\"
    result = {function.name}({parameter_name}={boundary_value!r})
    assert result is not None
"""
        elif framework == TestFramework.JEST:
            test_body = f"""
    it('should handle boundary value for {parameter_name}', () => {{
        const result = {function.name}({boundary_value!r});
        expect(result).toBeDefined();
    }});
"""
        else:
            return None

        test_case = TestCase(
            id=test_id,
            name=f"test_{function.name}_boundary_{parameter_name}_{boundary_index}",
            description=f"Boundary test for {parameter_name} = {boundary_value}",
            type=TestType.EDGE_CASE,
            target_function=function.name,
            target_file=function.file_path,
            test_file="",  # Will be set by generator
            test_body=test_body,
            framework=framework,
            language="python" if framework == TestFramework.PYTEST else "javascript",
            inputs=[{parameter_name: boundary_value}],
        )

        return test_case
