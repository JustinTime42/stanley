"""Jest test generator."""

import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..base import BaseTestGenerator
from ...models.testing_models import (
    TestSuite,
    TestCase,
    TestGenerationRequest,
    TestFramework,
    TestType,
    PropertyTest,
)
from ...models.analysis_models import CodeEntity, ASTNode
from ..data_generator import TestDataGenerator
from ..mock_generator import MockGenerator
from ..property_generator import PropertyTestGenerator

logger = logging.getLogger(__name__)


class JestGenerator(BaseTestGenerator):
    """
    Jest test generator for JavaScript/TypeScript.

    PATTERN: Generate Jest-style tests with mocks and spies
    CRITICAL: Handle both sync and async tests
    GOTCHA: Different import syntax for ES6 modules vs CommonJS
    """

    def __init__(self):
        """Initialize Jest generator."""
        super().__init__(framework=TestFramework.JEST)
        self.data_generator = TestDataGenerator()
        self.mock_generator = MockGenerator()
        self.property_generator = PropertyTestGenerator()

    async def generate_test_suite(
        self,
        target_file: str,
        request: TestGenerationRequest,
        ast: Optional[ASTNode] = None,
    ) -> TestSuite:
        """
        Generate complete Jest test suite.

        Args:
            target_file: File to generate tests for
            request: Test generation parameters
            ast: Optional pre-parsed AST

        Returns:
            TestSuite with Jest tests
        """
        start_time = datetime.now()

        # Extract testable units
        functions = await self._extract_testable_functions(target_file, ast)

        # Generate test file path
        test_file = self._get_test_file_path(target_file)

        # Generate test cases
        test_cases = []
        property_tests = []

        for function in functions[: request.max_test_cases]:
            # Generate unit test
            if TestType.UNIT in request.test_types:
                unit_test = await self.generate_unit_test(
                    function, request.include_mocks
                )
                test_cases.append(unit_test)

            # Generate property test if applicable
            if (
                request.include_property_tests
                and self._is_pure_function(function)
            ):
                prop_test = await self.generate_property_test(function)
                if prop_test:
                    property_tests.append(prop_test)

            # Generate edge case tests
            if request.include_edge_cases:
                edge_tests = await self._generate_edge_case_tests(function)
                test_cases.extend(edge_tests)

        # Create test suite
        suite = TestSuite(
            id=f"jest_suite_{os.path.basename(target_file)}",
            name=f"Test suite for {os.path.basename(target_file)}",
            target_module=target_file,
            framework=TestFramework.JEST,
            test_cases=test_cases,
            property_tests=property_tests,
        )

        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds() * 1000
        suite.generation_time_ms = int(generation_time)

        # Estimate coverage
        if ast:
            suite.total_coverage = await self._estimate_coverage(suite, ast)

        return suite

    async def generate_unit_test(
        self, function: CodeEntity, include_mocks: bool = True
    ) -> TestCase:
        """
        Generate unit test for a single function.

        Args:
            function: Function to test
            include_mocks: Whether to generate mocks

        Returns:
            TestCase for the function
        """
        # Create test ID
        test_id = self._create_test_id(function.file_path, function.name)

        # Generate test data
        test_inputs = await self._generate_test_inputs(function)

        # Generate mocks if needed
        mocks = []
        if include_mocks and function.dependencies:
            mocks = await self.mock_generator.generate_mocks_for_function(
                function, TestFramework.JEST
            )

        # Build test code
        test_body = await self._build_test_body(function, test_inputs, mocks)

        # Create test case
        test_case = TestCase(
            id=test_id,
            name=f"{function.name}",
            description=f"should work with valid inputs",
            type=TestType.UNIT,
            target_function=function.name,
            target_file=function.file_path,
            test_file=self._get_test_file_path(function.file_path),
            test_body=test_body,
            framework=TestFramework.JEST,
            language="javascript",
            inputs=test_inputs,
            mocks=[
                {
                    "target": mock.target,
                    "type": mock.mock_type.value,
                    "return_value": mock.return_value,
                }
                for mock in mocks
            ],
        )

        return test_case

    async def generate_property_test(
        self, function: CodeEntity
    ) -> Optional[PropertyTest]:
        """
        Generate property-based test using fast-check.

        Args:
            function: Function to test

        Returns:
            PropertyTest if applicable
        """
        return await self.property_generator.generate_properties(
            function, TestFramework.JEST
        )

    async def validate_test_syntax(self, test_suite: TestSuite) -> bool:
        """
        Validate Jest test syntax.

        Args:
            test_suite: Test suite to validate

        Returns:
            True if all tests are valid
        """
        # For JavaScript, we'd need a JS parser
        # For now, just check basic structure
        for test_case in test_suite.test_cases:
            if not test_case.test_body:
                return False

            # Check for required Jest patterns
            if "describe(" not in test_case.test_body and "it(" not in test_case.test_body:
                self.logger.error(
                    f"Test {test_case.name} missing describe() or it()"
                )
                return False

        return True

    async def _extract_testable_functions(
        self, target_file: str, ast_node: Optional[ASTNode] = None
    ) -> List[CodeEntity]:
        """
        Extract testable functions from file.

        Args:
            target_file: File to analyze
            ast_node: Optional pre-parsed AST

        Returns:
            List of testable functions
        """
        # Simplified extraction
        functions = []

        # Sample function for demonstration
        sample_function = CodeEntity(
            name="sampleFunction",
            type="function",
            file_path=target_file,
            line_start=1,
            line_end=10,
            signature="function sampleFunction(x: number): number",
            docstring="Sample function for testing",
            dependencies=[],
        )

        functions.append(sample_function)

        return functions

    async def _generate_test_inputs(
        self, function: CodeEntity
    ) -> List[Dict[str, Any]]:
        """
        Generate test inputs for function.

        Args:
            function: Function to generate inputs for

        Returns:
            List of input dictionaries
        """
        inputs = []

        # Simple input generation
        from ..data_generator import TestDataSpec

        spec = TestDataSpec(
            parameter_name="x",
            data_type="int",
            constraints={"min": -100, "max": 100},
        )

        values = await self.data_generator.generate_test_data(spec)

        for value in values[:5]:
            inputs.append({"x": value})

        return inputs

    async def _build_test_body(
        self,
        function: CodeEntity,
        inputs: List[Dict[str, Any]],
        mocks: List[Any],
    ) -> str:
        """
        Build test function body.

        Args:
            function: Function being tested
            inputs: Test inputs
            mocks: Mock specifications

        Returns:
            Test body string
        """
        # Determine import path
        module_path = self._get_module_path(function.file_path)

        # Build imports
        imports = function.name

        # Build test
        test_input = inputs[0] if inputs else {}
        input_str = ", ".join(f"{v!r}" for v in test_input.values())

        test_body = f"""
import {{ {imports} }} from '{module_path}';

describe('{function.name}', () => {{
    it('should work with valid inputs', () => {{
        // Arrange
        const input = {input_str};

        // Act
        const result = {function.name}(input);

        // Assert
        expect(result).toBeDefined();
    }});
}});
"""

        return test_body

    async def _generate_edge_case_tests(
        self, function: CodeEntity
    ) -> List[TestCase]:
        """
        Generate edge case tests for function.

        Args:
            function: Function to test

        Returns:
            List of edge case test cases
        """
        edge_tests = []

        # Generate edge case inputs
        edge_inputs = [
            {"name": "zero", "value": {"x": 0}},
            {"name": "negative", "value": {"x": -1}},
            {"name": "null", "value": {"x": None}},
        ]

        for edge_case in edge_inputs:
            test_id = self._create_test_id(
                function.file_path, f"{function.name}_{edge_case['name']}"
            )

            value_str = "null" if edge_case["value"]["x"] is None else str(edge_case["value"]["x"])

            test_body = f"""
    it('should handle edge case: {edge_case['name']}', () => {{
        const result = {function.name}({value_str});
        expect(result).toBeDefined();
    }});
"""

            test_case = TestCase(
                id=test_id,
                name=f"{function.name}_{edge_case['name']}",
                description=f"should handle edge case: {edge_case['name']}",
                type=TestType.EDGE_CASE,
                target_function=function.name,
                target_file=function.file_path,
                test_file=self._get_test_file_path(function.file_path),
                test_body=test_body,
                framework=TestFramework.JEST,
                language="javascript",
                inputs=[edge_case["value"]],
            )

            edge_tests.append(test_case)

        return edge_tests

    def _get_test_file_path(self, source_file: str) -> str:
        """
        Get test file path for source file.

        Args:
            source_file: Source file path

        Returns:
            Test file path
        """
        # Convert src/module.js -> src/__tests__/module.test.js
        dirname = os.path.dirname(source_file)
        basename = os.path.basename(source_file)
        name, ext = os.path.splitext(basename)

        # Jest convention: __tests__ directory or .test.js suffix
        test_dir = os.path.join(dirname, "__tests__")
        test_filename = f"{name}.test{ext}"

        return os.path.join(test_dir, test_filename)

    def _get_module_path(self, file_path: str) -> str:
        """
        Get module import path.

        Args:
            file_path: File path

        Returns:
            Module path for imports
        """
        # Convert file path to module path
        # e.g., src/utils/helper.js -> '../utils/helper'
        name, _ = os.path.splitext(os.path.basename(file_path))
        dirname = os.path.dirname(file_path)

        if dirname:
            return f"./{os.path.join(dirname, name).replace(os.sep, '/')}"
        else:
            return f"./{name}"
