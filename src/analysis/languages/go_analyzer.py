"""Go code analyzer."""

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


class GoAnalyzer(BaseAnalyzer):
    """Go code analyzer."""

    def __init__(self):
        """Initialize Go analyzer."""
        super().__init__(Language.GO)

    def get_language(self) -> Language:
        """Get the language this analyzer handles."""
        return Language.GO

    async def analyze(self, file_path: str, ast: ASTNode, **kwargs) -> List[CodeEntity]:
        """Analyze Go code and extract entities."""
        entities = []

        functions = await self.extract_functions(ast)
        entities.extend(functions)

        # Go doesn't have classes, but has types with methods
        # Treat type definitions as "classes"
        types = await self.extract_types(ast)
        entities.extend(types)

        for entity in entities:
            entity.file_path = file_path

        return entities

    async def extract_functions(self, ast: ASTNode) -> List[CodeEntity]:
        """Extract function definitions from Go AST."""
        functions = []

        def visit(node: ASTNode, receiver_type: Optional[str] = None):
            if node.node_type == "function_declaration":
                func_entity = self._extract_function_entity(node, receiver_type)
                if func_entity:
                    functions.append(func_entity)
            elif node.node_type == "method_declaration":
                # Go methods have receivers
                func_entity = self._extract_method_entity(node)
                if func_entity:
                    functions.append(func_entity)

            for child in node.children:
                visit(child, receiver_type)

        visit(ast)
        return functions

    async def extract_types(self, ast: ASTNode) -> List[CodeEntity]:
        """Extract type definitions from Go AST."""
        types = []

        def visit(node: ASTNode):
            if node.node_type == "type_declaration":
                type_entity = self._extract_type_entity(node)
                if type_entity:
                    types.append(type_entity)

            for child in node.children:
                visit(child)

        visit(ast)
        return types

    def _extract_function_entity(
        self, node: ASTNode, receiver_type: Optional[str] = None
    ) -> Optional[CodeEntity]:
        """Extract function entity from AST node."""
        name = None

        for child in node.children:
            if child.node_type == "identifier":
                name = child.text
                break

        if not name:
            return None

        parameters = self._extract_parameters(node)
        signature = f"func {name}({', '.join(parameters)})"

        return CodeEntity(
            name=name,
            type=NodeType.FUNCTION,
            file_path="",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=signature,
        )

    def _extract_method_entity(self, node: ASTNode) -> Optional[CodeEntity]:
        """Extract method entity from Go AST node."""
        name = None
        receiver = None

        for child in node.children:
            if child.node_type == "identifier":
                name = child.text
            elif child.node_type == "parameter_list":
                # First parameter list is the receiver
                receiver = self._extract_receiver(child)

        if not name:
            return None

        parameters = self._extract_parameters(node)
        signature = f"func ({receiver}) {name}({', '.join(parameters)})"

        return CodeEntity(
            name=name,
            type=NodeType.METHOD,
            file_path="",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=signature,
            metadata={"receiver": receiver},
        )

    def _extract_type_entity(self, node: ASTNode) -> Optional[CodeEntity]:
        """Extract type entity from Go AST node."""
        name = None

        for child in node.children:
            if child.node_type == "type_spec":
                for spec_child in child.children:
                    if spec_child.node_type == "type_identifier":
                        name = spec_child.text
                        break
                if name:
                    break

        if not name:
            return None

        return CodeEntity(
            name=name,
            type=NodeType.CLASS,  # Use CLASS for Go types
            file_path="",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"type {name}",
        )

    def _extract_parameters(self, node: ASTNode) -> List[str]:
        """Extract parameter names from function/method node."""
        params = []

        for child in node.children:
            if child.node_type == "parameter_list":
                self._collect_params(child, params)

        return params

    def _collect_params(self, node: ASTNode, params: List[str]):
        """Recursively collect parameter names."""
        if node.node_type == "parameter_declaration":
            for child in node.children:
                if child.node_type == "identifier":
                    if child.text:
                        params.append(child.text)

        for child in node.children:
            self._collect_params(child, params)

    def _extract_receiver(self, node: ASTNode) -> Optional[str]:
        """Extract receiver type from parameter list."""
        for child in node.children:
            if child.node_type == "parameter_declaration":
                for param_child in child.children:
                    if param_child.node_type in ["type_identifier", "pointer_type"]:
                        return param_child.text
        return None
