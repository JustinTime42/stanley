"""Code complexity metrics analyzer."""

import logging
from typing import Dict
import math

from ..models.analysis_models import (
    ASTNode,
    ComplexityMetrics,
    Language,
)

logger = logging.getLogger(__name__)


class ComplexityAnalyzer:
    """
    Analyzer for calculating code complexity metrics.

    PATTERN: Visitor pattern for AST traversal
    GOTCHA: Different languages have different control flow constructs
    """

    # Language-specific control flow keywords
    DECISION_NODES = {
        "if_statement",
        "while_statement",
        "for_statement",
        "case_statement",
        "switch_statement",
        "catch_clause",
        "conditional_expression",
        "and",
        "or",
        "&&",
        "||",
    }

    LOOP_NODES = {
        "for_statement",
        "while_statement",
        "do_statement",
        "for_in_statement",
        "for_of_statement",
    }

    def __init__(self):
        """Initialize complexity analyzer."""
        self.logger = logger

    async def analyze(
        self, file_path: str, ast: ASTNode, language: Language, source_code: str = ""
    ) -> ComplexityMetrics:
        """
        Calculate complexity metrics for code.

        Args:
            file_path: Path to file
            ast: Parsed AST
            language: Programming language
            source_code: Optional source code for additional metrics

        Returns:
            ComplexityMetrics with all calculated metrics
        """
        # Calculate different complexity metrics
        cyclomatic = self.calculate_cyclomatic(ast)
        cognitive = self.calculate_cognitive(ast)
        halstead = self.calculate_halstead(ast)

        # Calculate code statistics
        lines = self._count_lines(ast, source_code)
        functions = self._count_functions(ast)
        classes = self._count_classes(ast)
        max_depth = self._calculate_max_nesting(ast)

        # Calculate average function complexity
        avg_complexity = cyclomatic / functions if functions > 0 else 0.0

        return ComplexityMetrics(
            file_path=file_path,
            language=language,
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            halstead_metrics=halstead,
            lines_of_code=lines["total"],
            lines_of_code_without_comments=lines["code"],
            function_count=functions,
            class_count=classes,
            max_nesting_depth=max_depth,
            average_function_complexity=avg_complexity,
        )

    def calculate_cyclomatic(self, ast: ASTNode) -> int:
        """
        Calculate McCabe cyclomatic complexity.

        Formula: M = E - N + 2P
        Where E = edges, N = nodes, P = connected components

        Simplified: Count decision points + 1

        Args:
            ast: AST root node

        Returns:
            Cyclomatic complexity score
        """
        complexity = 1  # Start with 1 for linear flow

        def visit(node: ASTNode):
            nonlocal complexity

            # Count decision points that create branches
            if node.node_type in self.DECISION_NODES:
                complexity += 1

            # Recurse through children
            for child in node.children:
                visit(child)

        visit(ast)
        return complexity

    def calculate_cognitive(self, ast: ASTNode, depth: int = 0) -> int:
        """
        Calculate cognitive complexity.

        PATTERN: Increment for nesting and flow breaks
        CRITICAL: Penalizes nested structures more heavily

        Args:
            ast: AST root node
            depth: Current nesting depth

        Returns:
            Cognitive complexity score
        """
        complexity = 0

        for node in ast.children:
            # Control flow structures increase complexity
            if node.node_type in {
                "if_statement",
                "switch_statement",
                "for_statement",
                "while_statement",
            }:
                # Base complexity + nesting penalty
                complexity += 1 + depth
                # Recursively calculate nested complexity
                complexity += self.calculate_cognitive(node, depth + 1)

            elif node.node_type in {"catch_clause", "else_clause", "elif_clause"}:
                # Else/catch adds complexity but doesn't increase nesting
                complexity += 1
                complexity += self.calculate_cognitive(node, depth)

            elif node.node_type in {"and", "or", "&&", "||"}:
                # Logical operators add to complexity
                complexity += 1
                complexity += self.calculate_cognitive(node, depth)

            else:
                # Continue traversal without penalty
                complexity += self.calculate_cognitive(node, depth)

        return complexity

    def calculate_halstead(self, ast: ASTNode) -> Dict[str, float]:
        """
        Calculate Halstead complexity metrics.

        Metrics:
        - n1: Number of distinct operators
        - n2: Number of distinct operands
        - N1: Total number of operators
        - N2: Total number of operands
        - Vocabulary: n = n1 + n2
        - Length: N = N1 + N2
        - Volume: V = N * log2(n)
        - Difficulty: D = (n1 / 2) * (N2 / n2)
        - Effort: E = V * D

        Args:
            ast: AST root node

        Returns:
            Dictionary of Halstead metrics
        """
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0

        # Operator node types
        operator_types = {
            "binary_expression",
            "unary_expression",
            "assignment_expression",
            "comparison_operator",
            "binary_operator",
            "unary_operator",
        }

        # Operand node types
        operand_types = {
            "identifier",
            "number",
            "string",
            "boolean",
            "null",
            "true",
            "false",
        }

        def visit(node: ASTNode):
            nonlocal operator_count, operand_count

            if node.node_type in operator_types:
                if node.text:
                    operators.add(node.text)
                    operator_count += 1

            elif node.node_type in operand_types:
                if node.text:
                    operands.add(node.text)
                    operand_count += 1

            for child in node.children:
                visit(child)

        visit(ast)

        # Calculate Halstead metrics
        n1 = len(operators)  # Distinct operators
        n2 = len(operands)  # Distinct operands
        N1 = operator_count  # Total operators
        N2 = operand_count  # Total operands

        # Avoid division by zero
        if n1 == 0 or n2 == 0 or N2 == 0:
            return {
                "vocabulary": 0,
                "length": 0,
                "volume": 0,
                "difficulty": 0,
                "effort": 0,
            }

        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2)
        effort = volume * difficulty

        return {
            "vocabulary": vocabulary,
            "length": length,
            "volume": round(volume, 2),
            "difficulty": round(difficulty, 2),
            "effort": round(effort, 2),
        }

    def _count_lines(self, ast: ASTNode, source_code: str = "") -> Dict[str, int]:
        """
        Count lines of code.

        Args:
            ast: AST root node
            source_code: Optional source code string

        Returns:
            Dictionary with total and code-only line counts
        """
        if source_code:
            lines = source_code.split("\n")
            total_lines = len(lines)

            # Count non-comment, non-blank lines
            code_lines = sum(
                1
                for line in lines
                if line.strip() and not line.strip().startswith(("#", "//", "/*"))
            )

            return {"total": total_lines, "code": code_lines}
        else:
            # Use AST end point as approximation
            total_lines = ast.end_point[0] + 1 if ast.end_point else 0
            return {"total": total_lines, "code": total_lines}

    def _count_functions(self, ast: ASTNode) -> int:
        """Count function definitions in AST."""
        function_types = {
            "function_definition",
            "function_declaration",
            "method_definition",
            "method_declaration",
        }

        count = 0

        def visit(node: ASTNode):
            nonlocal count
            if node.node_type in function_types:
                count += 1

            for child in node.children:
                visit(child)

        visit(ast)
        return count

    def _count_classes(self, ast: ASTNode) -> int:
        """Count class definitions in AST."""
        class_types = {
            "class_definition",
            "class_declaration",
            "interface_declaration",
        }

        count = 0

        def visit(node: ASTNode):
            nonlocal count
            if node.node_type in class_types:
                count += 1

            for child in node.children:
                visit(child)

        visit(ast)
        return count

    def _calculate_max_nesting(self, ast: ASTNode, current_depth: int = 0) -> int:
        """
        Calculate maximum nesting depth.

        Args:
            ast: AST node
            current_depth: Current depth

        Returns:
            Maximum nesting depth found
        """
        max_depth = current_depth

        # Nesting structures
        nesting_types = {
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
            "with_statement",
            "function_definition",
            "class_definition",
        }

        for child in ast.children:
            if child.node_type in nesting_types:
                child_depth = self._calculate_max_nesting(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_max_nesting(child, current_depth)
                max_depth = max(max_depth, child_depth)

        return max_depth
