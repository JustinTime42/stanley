"""Python-specific code analyzer."""

import logging
from typing import List, Optional

from ..base import BaseAnalyzer
from ...models.analysis_models import (
    Language,
    ASTNode,
    CodeEntity,
    NodeType,
)

logger = logging.getLogger(__name__)


class PythonAnalyzer(BaseAnalyzer):
    """
    Python-specific code analyzer.

    PATTERN: BaseAnalyzer inheritance for language-specific analysis
    CRITICAL: Extract Python-specific constructs (decorators, async, etc.)
    """

    def __init__(self):
        """Initialize Python analyzer."""
        super().__init__(Language.PYTHON)

    def get_language(self) -> Language:
        """Get the language this analyzer handles."""
        return Language.PYTHON

    async def analyze(self, file_path: str, ast: ASTNode, **kwargs) -> List[CodeEntity]:
        """
        Analyze Python code and extract entities.

        Args:
            file_path: Path to Python file
            ast: Parsed AST
            **kwargs: Additional parameters

        Returns:
            List of code entities (functions, classes, etc.)
        """
        entities = []

        # Extract functions
        functions = await self.extract_functions(ast)
        entities.extend(functions)

        # Extract classes
        classes = await self.extract_classes(ast)
        entities.extend(classes)

        # Add file path to all entities
        for entity in entities:
            entity.file_path = file_path

        return entities

    async def extract_functions(self, ast: ASTNode) -> List[CodeEntity]:
        """
        Extract function definitions from Python AST.

        PATTERN: Find function_definition nodes
        GOTCHA: Must handle async functions, methods, lambdas

        Args:
            ast: Parsed AST

        Returns:
            List of function entities
        """
        functions = []

        def visit(node: ASTNode, parent_class: Optional[str] = None):
            # Python function definition node
            if node.node_type == "function_definition":
                func_entity = self._extract_function_entity(node, parent_class)
                if func_entity:
                    functions.append(func_entity)

            # Recurse into children (for nested functions)
            for child in node.children:
                # Track if we're inside a class
                current_class = parent_class
                if node.node_type == "class_definition":
                    # Extract class name for method context
                    class_name = self._get_class_name(node)
                    current_class = class_name if class_name else parent_class

                visit(child, current_class)

        visit(ast)
        return functions

    async def extract_classes(self, ast: ASTNode) -> List[CodeEntity]:
        """
        Extract class definitions from Python AST.

        Args:
            ast: Parsed AST

        Returns:
            List of class entities
        """
        classes = []

        def visit(node: ASTNode):
            if node.node_type == "class_definition":
                class_entity = self._extract_class_entity(node)
                if class_entity:
                    classes.append(class_entity)

            for child in node.children:
                visit(child)

        visit(ast)
        return classes

    async def extract_imports(self, ast: ASTNode) -> List[str]:
        """
        Extract import statements from Python AST.

        PATTERN: Find import_statement and import_from_statement nodes
        GOTCHA: Handle both 'import x' and 'from x import y'

        Args:
            ast: Parsed AST

        Returns:
            List of imported module names
        """
        imports = []

        def visit(node: ASTNode):
            if node.node_type in ["import_statement", "import_from_statement"]:
                # Extract module name from import
                module_name = self._extract_module_name(node)
                if module_name:
                    imports.append(module_name)

            for child in node.children:
                visit(child)

        visit(ast)
        return imports

    async def extract_docstring(self, node: ASTNode) -> Optional[str]:
        """
        Extract docstring from a Python function or class.

        PATTERN: First string literal in body is docstring
        GOTCHA: Must be expression_statement > string

        Args:
            node: Function or class AST node

        Returns:
            Docstring text if found
        """
        # Find body node
        for child in node.children:
            if child.node_type == "block":
                # First statement in block
                for stmt in child.children:
                    if stmt.node_type == "expression_statement":
                        # Check if it's a string
                        for expr in stmt.children:
                            if expr.node_type == "string":
                                # Extract and clean docstring
                                docstring = expr.text
                                if docstring:
                                    # Remove quotes
                                    docstring = docstring.strip()
                                    for quote in ['"""', "'''", '"', "'"]:
                                        if docstring.startswith(quote):
                                            docstring = docstring[len(quote) :]
                                        if docstring.endswith(quote):
                                            docstring = docstring[: -len(quote)]
                                    return docstring.strip()
                # Only check first statement
                break

        return None

    def _extract_function_entity(
        self, node: ASTNode, parent_class: Optional[str] = None
    ) -> Optional[CodeEntity]:
        """
        Extract function entity from function_definition node.

        Args:
            node: function_definition AST node
            parent_class: Optional parent class name (for methods)

        Returns:
            CodeEntity or None
        """
        # Find function name
        name = None
        parameters = []

        for child in node.children:
            if child.node_type == "identifier":
                name = child.text
            elif child.node_type == "parameters":
                # Extract parameter list
                parameters = self._extract_parameters(child)

        if not name:
            return None

        # Determine if it's a method or function
        entity_type = NodeType.METHOD if parent_class else NodeType.FUNCTION

        # Build signature
        signature = f"def {name}({', '.join(parameters)})"

        return CodeEntity(
            name=name,
            type=entity_type,
            file_path="",  # Will be set by analyze()
            line_start=node.start_point[0] + 1,  # Tree-sitter is 0-indexed
            line_end=node.end_point[0] + 1,
            signature=signature,
            docstring=None,  # Will be extracted separately
            metadata={
                "parent_class": parent_class,
                "is_async": self._is_async(node),
                "decorators": self._extract_decorators(node),
            },
        )

    def _extract_class_entity(self, node: ASTNode) -> Optional[CodeEntity]:
        """
        Extract class entity from class_definition node.

        Args:
            node: class_definition AST node

        Returns:
            CodeEntity or None
        """
        # Find class name
        name = self._get_class_name(node)

        if not name:
            return None

        # Extract base classes
        bases = self._extract_base_classes(node)

        return CodeEntity(
            name=name,
            type=NodeType.CLASS,
            file_path="",  # Will be set by analyze()
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"class {name}({', '.join(bases)})" if bases else f"class {name}",
            docstring=None,  # Will be extracted separately
            metadata={
                "base_classes": bases,
                "decorators": self._extract_decorators(node),
            },
        )

    def _get_class_name(self, node: ASTNode) -> Optional[str]:
        """Extract class name from class_definition node."""
        for child in node.children:
            if child.node_type == "identifier":
                return child.text
        return None

    def _extract_parameters(self, params_node: ASTNode) -> List[str]:
        """Extract parameter names from parameters node."""
        params = []

        def visit(node: ASTNode):
            if node.node_type == "identifier":
                # Simple parameter
                if node.text and node.text not in params:
                    params.append(node.text)
            elif node.node_type == "typed_parameter":
                # Parameter with type hint
                for child in node.children:
                    if child.node_type == "identifier":
                        if child.text:
                            params.append(child.text)
                        break

            for child in node.children:
                visit(child)

        visit(params_node)

        # Filter out 'self' and 'cls' for readability
        return [p for p in params if p not in ["self", "cls"]]

    def _extract_base_classes(self, class_node: ASTNode) -> List[str]:
        """Extract base class names from class definition."""
        bases = []

        for child in class_node.children:
            if child.node_type == "argument_list":
                for arg_child in child.children:
                    if arg_child.node_type == "identifier":
                        if arg_child.text:
                            bases.append(arg_child.text)

        return bases

    def _is_async(self, func_node: ASTNode) -> bool:
        """Check if function is async."""
        # Look for 'async' keyword before 'def'
        for child in func_node.children:
            if child.node_type == "async" or (child.text and "async" in child.text):
                return True
        return False

    def _extract_decorators(self, node: ASTNode) -> List[str]:
        """Extract decorator names from function or class."""
        decorators = []

        # Decorators appear before the definition
        for child in node.children:
            if child.node_type == "decorator":
                # Extract decorator name
                for dec_child in child.children:
                    if dec_child.node_type == "identifier":
                        if dec_child.text:
                            decorators.append(dec_child.text)

        return decorators

    def _extract_module_name(self, import_node: ASTNode) -> Optional[str]:
        """Extract module name from import statement."""
        # Handle 'import x' and 'from x import y'
        for child in import_node.children:
            if child.node_type == "dotted_name":
                return child.text
            elif child.node_type == "aliased_import":
                # Extract actual module name
                for alias_child in child.children:
                    if alias_child.node_type == "dotted_name":
                        return alias_child.text

        return None
