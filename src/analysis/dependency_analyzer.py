"""Dependency graph analyzer."""

import logging
from typing import List, Dict

from ..models.analysis_models import (
    ASTNode,
    CodeEntity,
    DependencyGraph,
)

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """
    Analyzer for building code dependency graphs.

    PATTERN: Graph traversal algorithms
    CRITICAL: Track imports, calls, and inheritance relationships
    """

    def __init__(self):
        """Initialize dependency analyzer."""
        self.logger = logger

    async def analyze(
        self, entities: List[CodeEntity], ast: ASTNode
    ) -> DependencyGraph:
        """
        Build dependency graph from code entities.

        Args:
            entities: List of code entities
            ast: Parsed AST

        Returns:
            DependencyGraph with relationships
        """
        # Build entity lookup by name
        entity_map: Dict[str, CodeEntity] = {
            self._get_entity_id(e): e for e in entities
        }

        # Extract relationships
        imports = await self._extract_imports(ast)
        calls = await self._extract_calls(ast, entity_map)
        inheritance = await self._extract_inheritance(ast, entity_map)

        # Build edges from relationships
        edges = []

        # Add import edges
        for from_module, to_modules in imports.items():
            for to_module in to_modules:
                edges.append((from_module, to_module, "imports"))

        # Add call edges
        for caller, callees in calls.items():
            for callee in callees:
                edges.append((caller, callee, "calls"))

        # Add inheritance edges
        for child, parents in inheritance.items():
            for parent in parents:
                edges.append((child, parent, "inherits"))

        return DependencyGraph(
            nodes=entity_map,
            edges=edges,
            imports=imports,
            calls=calls,
            inheritance=inheritance,
        )

    async def _extract_imports(self, ast: ASTNode) -> Dict[str, List[str]]:
        """
        Extract import relationships from AST.

        Args:
            ast: AST root node

        Returns:
            Dict mapping files to imported modules
        """
        imports: Dict[str, List[str]] = {}

        import_types = {
            "import_statement",
            "import_from_statement",
            "import_declaration",
        }

        def visit(node: ASTNode, current_file: str = "<module>"):
            if node.node_type in import_types:
                module_name = self._extract_import_name(node)
                if module_name:
                    if current_file not in imports:
                        imports[current_file] = []
                    if module_name not in imports[current_file]:
                        imports[current_file].append(module_name)

            for child in node.children:
                visit(child, current_file)

        visit(ast)
        return imports

    async def _extract_calls(
        self, ast: ASTNode, entity_map: Dict[str, CodeEntity]
    ) -> Dict[str, List[str]]:
        """
        Extract function call relationships.

        Args:
            ast: AST root node
            entity_map: Map of entity IDs to entities

        Returns:
            Dict mapping callers to callees
        """
        calls: Dict[str, List[str]] = {}

        call_types = {"call_expression", "call"}

        current_function = None

        def visit(node: ASTNode, func_name: str = None):
            nonlocal current_function

            # Track current function context
            if node.node_type in {"function_definition", "function_declaration"}:
                current_function = self._get_function_name(node)

            # Extract function calls
            elif node.node_type in call_types and current_function:
                callee = self._get_called_function(node)
                if callee:
                    if current_function not in calls:
                        calls[current_function] = []
                    if callee not in calls[current_function]:
                        calls[current_function].append(callee)

            for child in node.children:
                visit(child, func_name)

        visit(ast)
        return calls

    async def _extract_inheritance(
        self, ast: ASTNode, entity_map: Dict[str, CodeEntity]
    ) -> Dict[str, List[str]]:
        """
        Extract class inheritance relationships.

        Args:
            ast: AST root node
            entity_map: Map of entity IDs to entities

        Returns:
            Dict mapping child classes to parent classes
        """
        inheritance: Dict[str, List[str]] = {}

        class_types = {"class_definition", "class_declaration"}

        def visit(node: ASTNode):
            if node.node_type in class_types:
                class_name = self._get_class_name(node)
                bases = self._get_base_classes(node)

                if class_name and bases:
                    inheritance[class_name] = bases

            for child in node.children:
                visit(child)

        visit(ast)
        return inheritance

    def _get_entity_id(self, entity: CodeEntity) -> str:
        """Generate unique ID for entity."""
        return f"{entity.file_path}:{entity.name}"

    def _extract_import_name(self, node: ASTNode) -> str:
        """Extract module name from import node."""
        # Look for module identifier
        for child in node.children:
            if child.node_type in {"dotted_name", "identifier", "string"}:
                text = child.text
                if text:
                    return text.strip("'\"")

        return ""

    def _get_function_name(self, node: ASTNode) -> str:
        """Extract function name from function definition node."""
        for child in node.children:
            if child.node_type == "identifier":
                return child.text or ""
        return ""

    def _get_called_function(self, node: ASTNode) -> str:
        """Extract called function name from call expression."""
        for child in node.children:
            if child.node_type in {"identifier", "member_expression"}:
                return child.text or ""
        return ""

    def _get_class_name(self, node: ASTNode) -> str:
        """Extract class name from class definition."""
        for child in node.children:
            if child.node_type in {"identifier", "type_identifier"}:
                return child.text or ""
        return ""

    def _get_base_classes(self, node: ASTNode) -> List[str]:
        """Extract base class names from class definition."""
        bases = []

        for child in node.children:
            if child.node_type == "argument_list":
                for arg in child.children:
                    if arg.node_type == "identifier":
                        if arg.text:
                            bases.append(arg.text)

        return bases

    def find_cycles(self, graph: DependencyGraph) -> List[List[str]]:
        """
        Find circular dependencies in the graph.

        Args:
            graph: Dependency graph

        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Find neighbors
            neighbors = []
            for edge in graph.edges:
                if edge[0] == node:
                    neighbors.append(edge[1])

            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    cycles.append(cycle)

            rec_stack.remove(node)

        for node_id in graph.nodes.keys():
            if node_id not in visited:
                dfs(node_id, [])

        return cycles
