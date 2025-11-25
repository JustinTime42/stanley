"""Pytest test generator."""

import logging
import ast
import os
from typing import Optional, List, Dict, Any, Union
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
from ..templates.pytest_templates import (
    PYTEST_UNIT_TEMPLATE,
    PYTEST_FIXTURE_TEMPLATE,
    PYTEST_PROPERTY_TEMPLATE,
)
from ..data_generator import TestDataGenerator
from ..mock_generator import MockGenerator
from ..property_generator import PropertyTestGenerator

logger = logging.getLogger(__name__)


class PytestGenerator(BaseTestGenerator):
    """
    Pytest test generator.

    PATTERN: Generate pytest-style tests with fixtures and parametrization
    CRITICAL: Validate syntax using Python AST
    GOTCHA: Must handle async tests with pytest-asyncio
    """

    def __init__(self):
        """Initialize pytest generator."""
        super().__init__(framework=TestFramework.PYTEST)
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
        Generate complete pytest test suite.

        Args:
            target_file: File to generate tests for
            request: Test generation parameters
            ast: Optional pre-parsed AST

        Returns:
            TestSuite with pytest tests
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
            id=f"pytest_suite_{os.path.basename(target_file)}",
            name=f"Test suite for {os.path.basename(target_file)}",
            target_module=target_file,
            framework=TestFramework.PYTEST,
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
                function, TestFramework.PYTEST
            )

        # Build test code
        test_body = await self._build_test_body(function, test_inputs, mocks)

        # Create test case
        test_case = TestCase(
            id=test_id,
            name=f"test_{function.name}",
            description=f"Test {function.name} with valid inputs",
            type=TestType.UNIT,
            target_function=function.name,
            target_file=function.file_path,
            test_file=self._get_test_file_path(function.file_path),
            test_body=test_body,
            framework=TestFramework.PYTEST,
            language="python",
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
        Generate property-based test using Hypothesis.

        Args:
            function: Function to test

        Returns:
            PropertyTest if applicable
        """
        return await self.property_generator.generate_properties(
            function, TestFramework.PYTEST
        )

    async def validate_test_syntax(self, test_suite: TestSuite) -> bool:
        """
        Validate pytest test syntax using Python AST.

        Args:
            test_suite: Test suite to validate

        Returns:
            True if all tests are valid
        """
        for test_case in test_suite.test_cases:
            try:
                # Try to parse test body as Python code
                ast.parse(test_case.test_body)
            except SyntaxError as e:
                self.logger.error(
                    f"Syntax error in test {test_case.name}: {e}"
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
        functions = []

        # Read and parse the actual file
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                source_code = f.read()
        except FileNotFoundError:
            logger.warning(f"Target file not found: {target_file}")
            return functions
        except Exception as e:
            logger.error(f"Error reading target file {target_file}: {e}")
            return functions

        # Parse using Python AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in target file {target_file}: {e}")
            return functions

        # Extract TOP-LEVEL functions only (not methods inside classes)
        # We use tree.body to only get direct children of the module
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Skip private functions (starting with _) unless they're __init__
                if node.name.startswith("_") and node.name != "__init__":
                    continue

                # Skip functions that already look like tests
                if node.name.startswith("test_"):
                    continue

                # Skip CLI entry point functions - they require special mocking
                if node.name in ("main", "__main__", "cli", "run", "entry_point"):
                    continue

                # Skip pytest fixtures (functions that return a test object)
                # Common fixture names or functions decorated with @pytest.fixture
                is_fixture = False
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        if decorator.attr == "fixture":
                            is_fixture = True
                            break
                    elif isinstance(decorator, ast.Name):
                        if decorator.id == "fixture":
                            is_fixture = True
                            break
                if is_fixture:
                    continue

                # Build signature
                signature = self._build_signature(node)

                # Get docstring
                docstring = ast.get_docstring(node) or ""

                # Extract dependencies (function calls within the function)
                dependencies = self._extract_dependencies(node)

                function_entity = CodeEntity(
                    name=node.name,
                    type="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                    file_path=target_file,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    signature=signature,
                    docstring=docstring,
                    dependencies=dependencies,
                )

                functions.append(function_entity)

        # Also extract class methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Skip private methods except __init__
                        if item.name.startswith("_") and item.name != "__init__":
                            continue

                        signature = self._build_signature(item, class_name)
                        docstring = ast.get_docstring(item) or ""
                        dependencies = self._extract_dependencies(item)

                        method_entity = CodeEntity(
                            name=f"{class_name}.{item.name}",
                            type="method",
                            file_path=target_file,
                            line_start=item.lineno,
                            line_end=item.end_lineno or item.lineno,
                            signature=signature,
                            docstring=docstring,
                            dependencies=dependencies,
                        )

                        functions.append(method_entity)

        logger.info(f"Extracted {len(functions)} testable functions from {target_file}")
        return functions

    def _build_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], class_name: str = None) -> str:
        """Build function signature string from AST node."""
        args = []

        # Process regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            args.append(arg_str)

        # Process keyword-only arguments
        for arg in node.args.kwonlyargs:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            args.append(arg_str)

        # Build return annotation
        return_annotation = ""
        if node.returns:
            try:
                return_annotation = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        func_type = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        func_name = f"{class_name}.{node.name}" if class_name else node.name

        return f"{func_type} {func_name}({', '.join(args)}){return_annotation}"

    def _extract_dependencies(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract function call dependencies from AST node."""
        dependencies = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # Handle method calls like obj.method()
                    try:
                        dep_name = ast.unparse(child.func)
                        dependencies.append(dep_name)
                    except Exception:
                        pass

        # Remove duplicates while preserving order
        seen = set()
        unique_deps = []
        for dep in dependencies:
            if dep not in seen:
                seen.add(dep)
                unique_deps.append(dep)

        return unique_deps

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
        from ..data_generator import TestDataSpec

        # Parse parameters from signature
        params = self._parse_parameters_from_signature(function.signature)

        if not params:
            # No parameters - return empty input dict
            return [{}]

        # Generate test data for each parameter
        param_values = {}
        for param_name, param_type in params.items():
            # Skip 'self' for methods
            if param_name == "self":
                continue

            # Map type to data generator type
            data_type = self._map_type_to_generator(param_type)

            spec = TestDataSpec(
                parameter_name=param_name,
                data_type=data_type,
                constraints=self._get_constraints_for_type(data_type),
            )

            values = await self.data_generator.generate_test_data(spec)
            # If data generator returns empty/None, use sensible defaults based on type
            if not values:
                if data_type == "int" or data_type == "Any":
                    values = [1, 0, -1, 10, 100]
                elif data_type == "float":
                    values = [1.0, 0.0, -1.0, 3.14, 100.0]
                elif data_type == "str":
                    values = ["test", "hello", "world", "", "a"]
                elif data_type == "bool":
                    values = [True, False]
                elif data_type == "list":
                    values = [[], [1], [1, 2, 3]]
                elif data_type == "dict":
                    values = [{}, {"key": "value"}]
                else:
                    values = [1]  # Default to int as safest fallback
            param_values[param_name] = values[:5]

        # Create test input combinations (simplified - just pair up values)
        if param_values:
            max_inputs = max(len(v) for v in param_values.values())
            for i in range(min(max_inputs, 5)):
                input_dict = {}
                for param_name, values in param_values.items():
                    input_dict[param_name] = values[i % len(values)]
                inputs.append(input_dict)
        else:
            inputs.append({})

        return inputs

    def _parse_parameters_from_signature(self, signature: str) -> Dict[str, str]:
        """Parse parameter names and types from function signature."""
        import re

        params = {}

        # Extract parameters between parentheses
        match = re.search(r'\((.*?)\)', signature)
        if not match:
            return params

        params_str = match.group(1).strip()
        if not params_str:
            return params

        # Split by comma, handling nested brackets
        param_parts = []
        bracket_depth = 0
        current_param = ""

        for char in params_str:
            if char in "([{":
                bracket_depth += 1
            elif char in ")]}":
                bracket_depth -= 1
            elif char == "," and bracket_depth == 0:
                param_parts.append(current_param.strip())
                current_param = ""
                continue
            current_param += char

        if current_param.strip():
            param_parts.append(current_param.strip())

        # Parse each parameter
        for part in param_parts:
            part = part.strip()
            if not part:
                continue

            # Handle default values
            if "=" in part:
                part = part.split("=")[0].strip()

            # Handle type annotations
            if ":" in part:
                name_type = part.split(":", 1)
                param_name = name_type[0].strip()
                param_type = name_type[1].strip() if len(name_type) > 1 else "Any"
            else:
                param_name = part
                param_type = "Any"

            # Skip *args and **kwargs
            if param_name.startswith("*"):
                continue

            params[param_name] = param_type

        return params

    def _map_type_to_generator(self, type_str: str) -> str:
        """Map Python type annotation to data generator type."""
        type_str_lower = type_str.lower()

        if "int" in type_str_lower:
            return "int"
        elif "float" in type_str_lower:
            return "float"
        elif "str" in type_str_lower:
            return "str"
        elif "bool" in type_str_lower:
            return "bool"
        elif "list" in type_str_lower:
            return "list"
        elif "dict" in type_str_lower:
            return "dict"
        elif "optional" in type_str_lower:
            # Extract inner type from Optional[X]
            import re
            inner_match = re.search(r'Optional\[(.*?)\]', type_str, re.IGNORECASE)
            if inner_match:
                return self._map_type_to_generator(inner_match.group(1))
            return "Any"
        else:
            return "Any"

    def _get_constraints_for_type(self, data_type: str) -> Dict[str, Any]:
        """Get default constraints for a data type."""
        if data_type == "int":
            return {"min": -100, "max": 100}
        elif data_type == "float":
            return {"min": -100.0, "max": 100.0}
        elif data_type == "str":
            return {"min_length": 1, "max_length": 50}
        elif data_type == "list":
            return {"min_length": 0, "max_length": 10}
        elif data_type == "Any":
            # For untyped parameters, default to int constraints
            return {"min": -100, "max": 100}
        else:
            return {}

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
        # Build imports - handle class methods vs functions
        module_name = os.path.splitext(os.path.basename(function.file_path))[0]

        is_method = "." in function.name
        if is_method:
            # Method: import class, test method
            class_name, method_name = function.name.rsplit(".", 1)
            imports_section = f"from {module_name} import {class_name}"
            test_func_name = f"test_{class_name}_{method_name}".replace(".", "_")
            call_target = f"instance.{method_name}"
        else:
            # Function: import directly
            imports_section = f"from {module_name} import {function.name}"
            test_func_name = f"test_{function.name}"
            call_target = function.name

        # Build setup (mocks) - use 4 spaces for function body indentation
        setup_lines = []
        if mocks:
            setup_lines = [f"    # Mock: {mock.target}" for mock in mocks]

        # For methods, add instance creation
        if is_method:
            setup_lines.append(f"    instance = {class_name}()")

        # Build action - 4 spaces indentation
        if inputs:
            input_str = ", ".join(f"{k}={v!r}" for k, v in inputs[0].items())
            action_line = f"    result = {call_target}({input_str})"
        else:
            action_line = f"    result = {call_target}()"

        # Build assertions - 4 spaces indentation
        assertion_line = "    assert result is not None"

        # Combine into test with proper indentation
        test_lines = [
            imports_section,
            "",
            f"def {test_func_name}():",
            f'    """Test {function.name} with valid inputs."""',
            "    # Arrange",
        ]

        # Only add setup lines if they exist
        if setup_lines:
            test_lines.extend(setup_lines)
        else:
            test_lines.append("    pass  # No setup needed")

        test_lines.extend([
            "",
            "    # Act",
            action_line,
            "",
            "    # Assert",
            assertion_line,
        ])

        test_body = "\n".join(test_lines)
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

        # Parse function parameters to generate appropriate edge cases
        params = self._parse_parameters_from_signature(function.signature)

        # Filter out 'self' for methods
        params = {k: v for k, v in params.items() if k != "self"}

        if not params:
            return edge_tests  # No parameters to test edge cases for

        # Generate edge case values based on parameter types
        edge_inputs = self._generate_edge_case_inputs(params)

        # Setup for methods vs functions
        module_name = os.path.splitext(os.path.basename(function.file_path))[0]
        is_method = "." in function.name

        if is_method:
            class_name, method_name = function.name.rsplit(".", 1)
            import_line = f"from {module_name} import {class_name}"
            call_target = f"instance.{method_name}"
            setup_line = f"    instance = {class_name}()"
            safe_name = f"{class_name}_{method_name}"
        else:
            import_line = f"from {module_name} import {function.name}"
            call_target = function.name
            setup_line = ""
            safe_name = function.name

        for edge_case in edge_inputs:
            test_id = self._create_test_id(
                function.file_path, f"{safe_name}_{edge_case['name']}"
            )

            # Build test body with proper imports and setup
            input_str = ', '.join(f"{k}={v!r}" for k, v in edge_case['value'].items())

            if setup_line:
                test_body = f'''{import_line}

def test_{safe_name}_{edge_case['name']}():
    """Test {function.name} with edge case: {edge_case['name']}."""
{setup_line}
    result = {call_target}({input_str})
    assert result is not None
'''
            else:
                test_body = f'''{import_line}

def test_{safe_name}_{edge_case['name']}():
    """Test {function.name} with edge case: {edge_case['name']}."""
    result = {call_target}({input_str})
    assert result is not None
'''

            test_case = TestCase(
                id=test_id,
                name=f"test_{safe_name}_{edge_case['name']}",
                description=f"Edge case: {edge_case['name']}",
                type=TestType.EDGE_CASE,
                target_function=function.name,
                target_file=function.file_path,
                test_file=self._get_test_file_path(function.file_path),
                test_body=test_body,
                framework=TestFramework.PYTEST,
                language="python",
                inputs=[edge_case["value"]],
            )

            edge_tests.append(test_case)

        return edge_tests

    def _generate_edge_case_inputs(self, params: Dict[str, str]) -> List[Dict[str, Any]]:
        """Generate edge case input values based on parameter types."""
        edge_cases = []

        # Determine primary parameter types
        has_int = any("int" in t.lower() for t in params.values())
        has_float = any("float" in t.lower() for t in params.values())
        has_str = any("str" in t.lower() for t in params.values())

        # Generate edge cases for each parameter
        for param_name, param_type in params.items():
            type_lower = param_type.lower()

            if "int" in type_lower:
                edge_cases.extend([
                    {"name": f"{param_name}_zero", "value": {param_name: 0}},
                    {"name": f"{param_name}_negative", "value": {param_name: -1}},
                    {"name": f"{param_name}_large", "value": {param_name: 2147483647}},
                ])
            elif "float" in type_lower:
                edge_cases.extend([
                    {"name": f"{param_name}_zero", "value": {param_name: 0.0}},
                    {"name": f"{param_name}_negative", "value": {param_name: -1.0}},
                    {"name": f"{param_name}_small", "value": {param_name: 0.0001}},
                ])
            elif "str" in type_lower:
                edge_cases.extend([
                    {"name": f"{param_name}_empty", "value": {param_name: ""}},
                    {"name": f"{param_name}_whitespace", "value": {param_name: "   "}},
                ])
            elif "bool" in type_lower:
                edge_cases.extend([
                    {"name": f"{param_name}_true", "value": {param_name: True}},
                    {"name": f"{param_name}_false", "value": {param_name: False}},
                ])

        # Fill in other parameters with default values for multi-param functions
        if len(params) > 1:
            for edge_case in edge_cases:
                for param_name, param_type in params.items():
                    if param_name not in edge_case["value"]:
                        edge_case["value"][param_name] = self._get_default_value(param_type)

        return edge_cases[:6]  # Limit to 6 edge case tests

    def _get_default_value(self, type_str: str) -> Any:
        """Get a safe default value for a type."""
        type_lower = type_str.lower()
        if "int" in type_lower:
            return 1
        elif "float" in type_lower:
            return 1.0
        elif "str" in type_lower:
            return "test"
        elif "bool" in type_lower:
            return True
        elif "list" in type_lower:
            return []
        elif "dict" in type_lower:
            return {}
        else:
            # For Any/unknown types, default to integer (safest for arithmetic)
            return 1

    def _get_test_file_path(self, source_file: str) -> str:
        """
        Get test file path for source file.

        Args:
            source_file: Source file path

        Returns:
            Test file path
        """
        # Convert src/module.py -> tests/test_module.py
        dirname = os.path.dirname(source_file)
        basename = os.path.basename(source_file)
        name, ext = os.path.splitext(basename)

        # Replace src with tests
        if "src" in dirname:
            test_dir = dirname.replace("src", "tests")
        else:
            test_dir = os.path.join(dirname, "tests")

        test_filename = f"test_{name}{ext}"
        return os.path.join(test_dir, test_filename)
