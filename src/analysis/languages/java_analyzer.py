"""Java code analyzer."""

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


class JavaAnalyzer(BaseAnalyzer):
    """Java code analyzer."""

    def __init__(self):
        """Initialize Java analyzer."""
        super().__init__(Language.JAVA)

    def get_language(self) -> Language:
        """Get the language this analyzer handles."""
        return Language.JAVA

    async def analyze(self, file_path: str, ast: ASTNode, **kwargs) -> List[CodeEntity]:
        """Analyze Java code and extract entities."""
        entities = []

        functions = await self.extract_functions(ast)
        entities.extend(functions)

        classes = await self.extract_classes(ast)
        entities.extend(classes)

        for entity in entities:
            entity.file_path = file_path

        return entities

    async def extract_functions(self, ast: ASTNode) -> List[CodeEntity]:
        """Extract method definitions from Java AST."""
        functions = []

        def visit(node: ASTNode, parent_class: Optional[str] = None):
            if node.node_type == "method_declaration":
                func_entity = self._extract_method_entity(node, parent_class)
                if func_entity:
                    functions.append(func_entity)

            for child in node.children:
                current_class = parent_class
                if node.node_type == "class_declaration":
                    class_name = self._get_class_name(node)
                    current_class = class_name if class_name else parent_class

                visit(child, current_class)

        visit(ast)
        return functions

    async def extract_classes(self, ast: ASTNode) -> List[CodeEntity]:
        """Extract class definitions from Java AST."""
        classes = []

        def visit(node: ASTNode):
            if node.node_type in ["class_declaration", "interface_declaration"]:
                class_entity = self._extract_class_entity(node)
                if class_entity:
                    classes.append(class_entity)

            for child in node.children:
                visit(child)

        visit(ast)
        return classes

    def _extract_method_entity(
        self, node: ASTNode, parent_class: Optional[str] = None
    ) -> Optional[CodeEntity]:
        """Extract method entity from AST node."""
        name = None
        return_type = None

        for child in node.children:
            if child.node_type == "identifier":
                name = child.text
            elif child.node_type in ["type_identifier", "void_type"]:
                return_type = child.text

        if not name:
            return None

        parameters = self._extract_parameters(node)
        signature = f"{return_type or 'void'} {name}({', '.join(parameters)})"

        return CodeEntity(
            name=name,
            type=NodeType.METHOD,
            file_path="",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=signature,
            metadata={"parent_class": parent_class},
        )

    def _extract_class_entity(self, node: ASTNode) -> Optional[CodeEntity]:
        """Extract class entity from AST node."""
        name = self._get_class_name(node)

        if not name:
            return None

        entity_type = (
            NodeType.CLASS
            if node.node_type == "class_declaration"
            else NodeType.CLASS  # Interface treated as class for now
        )

        return CodeEntity(
            name=name,
            type=entity_type,
            file_path="",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"class {name}",
        )

    def _get_class_name(self, node: ASTNode) -> Optional[str]:
        """Extract class name from node."""
        for child in node.children:
            if child.node_type == "identifier":
                return child.text
        return None

    def _extract_parameters(self, node: ASTNode) -> List[str]:
        """Extract parameter names from method node."""
        params = []

        for child in node.children:
            if child.node_type == "formal_parameters":
                self._collect_params(child, params)

        return params

    def _collect_params(self, node: ASTNode, params: List[str]):
        """Recursively collect parameter names."""
        if node.node_type == "formal_parameter":
            for child in node.children:
                if child.node_type == "identifier":
                    if child.text:
                        params.append(child.text)

        for child in node.children:
            self._collect_params(child, params)
