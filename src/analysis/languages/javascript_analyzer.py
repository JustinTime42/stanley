"""JavaScript/TypeScript code analyzer."""

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


class JavaScriptAnalyzer(BaseAnalyzer):
    """JavaScript/TypeScript code analyzer."""

    def __init__(self, language: Language = Language.JAVASCRIPT):
        """Initialize JavaScript analyzer."""
        super().__init__(language)

    def get_language(self) -> Language:
        """Get the language this analyzer handles."""
        return self.language

    async def analyze(self, file_path: str, ast: ASTNode, **kwargs) -> List[CodeEntity]:
        """Analyze JavaScript/TypeScript code and extract entities."""
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
        """Extract function definitions from JavaScript/TypeScript AST."""
        functions = []

        # JS/TS function node types
        function_types = [
            "function_declaration",
            "function",
            "arrow_function",
            "method_definition",
        ]

        def visit(node: ASTNode, parent_class: Optional[str] = None):
            if node.node_type in function_types:
                func_entity = self._extract_function_entity(node, parent_class)
                if func_entity:
                    functions.append(func_entity)

            # Track class context
            for child in node.children:
                current_class = parent_class
                if node.node_type == "class_declaration":
                    class_name = self._get_class_name(node)
                    current_class = class_name if class_name else parent_class

                visit(child, current_class)

        visit(ast)
        return functions

    async def extract_classes(self, ast: ASTNode) -> List[CodeEntity]:
        """Extract class definitions from JavaScript/TypeScript AST."""
        classes = []

        def visit(node: ASTNode):
            if node.node_type == "class_declaration":
                class_entity = self._extract_class_entity(node)
                if class_entity:
                    classes.append(class_entity)

            for child in node.children:
                visit(child)

        visit(ast)
        return classes

    async def extract_imports(self, ast: ASTNode) -> List[str]:
        """Extract import statements from JavaScript/TypeScript AST."""
        imports = []

        def visit(node: ASTNode):
            if node.node_type in ["import_statement", "import_clause"]:
                module = self._extract_module_name(node)
                if module:
                    imports.append(module)

            for child in node.children:
                visit(child)

        visit(ast)
        return imports

    def _extract_function_entity(
        self, node: ASTNode, parent_class: Optional[str] = None
    ) -> Optional[CodeEntity]:
        """Extract function entity from AST node."""
        name = self._get_function_name(node)

        if not name:
            name = "<anonymous>"

        entity_type = NodeType.METHOD if parent_class else NodeType.FUNCTION

        parameters = self._extract_parameters(node)
        signature = f"function {name}({', '.join(parameters)})"

        return CodeEntity(
            name=name,
            type=entity_type,
            file_path="",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=signature,
            metadata={
                "parent_class": parent_class,
                "is_async": self._is_async(node),
                "is_arrow": node.node_type == "arrow_function",
            },
        )

    def _extract_class_entity(self, node: ASTNode) -> Optional[CodeEntity]:
        """Extract class entity from AST node."""
        name = self._get_class_name(node)

        if not name:
            return None

        return CodeEntity(
            name=name,
            type=NodeType.CLASS,
            file_path="",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"class {name}",
        )

    def _get_function_name(self, node: ASTNode) -> Optional[str]:
        """Extract function name from node."""
        for child in node.children:
            if child.node_type == "identifier":
                return child.text
            elif child.node_type == "property_identifier":
                return child.text
        return None

    def _get_class_name(self, node: ASTNode) -> Optional[str]:
        """Extract class name from node."""
        for child in node.children:
            if child.node_type == "identifier" or child.node_type == "type_identifier":
                return child.text
        return None

    def _extract_parameters(self, node: ASTNode) -> List[str]:
        """Extract parameter names from function node."""
        params = []

        for child in node.children:
            if child.node_type == "formal_parameters":
                self._collect_params(child, params)

        return params

    def _collect_params(self, node: ASTNode, params: List[str]):
        """Recursively collect parameter names."""
        if node.node_type in ["identifier", "parameter"]:
            if node.text and node.text not in ["(", ")", ","]:
                params.append(node.text)

        for child in node.children:
            self._collect_params(child, params)

    def _is_async(self, node: ASTNode) -> bool:
        """Check if function is async."""
        return node.text and node.text.strip().startswith("async")

    def _extract_module_name(self, node: ASTNode) -> Optional[str]:
        """Extract module name from import statement."""
        for child in node.children:
            if child.node_type == "string":
                # Remove quotes from module name
                module = child.text
                if module:
                    return module.strip("'\"")
        return None
