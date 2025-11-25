"""Code complexity analyzer using AST analysis.

PATTERN: AST-based cyclomatic and cognitive complexity calculation
CRITICAL: Extends patterns from src/analysis/ast_parser.py
GOTCHA: Complexity metrics vary by language, Python-focused implementation
"""

import asyncio
import ast
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class ComplexityAnalyzer:
    """
    Code complexity analyzer for cyclomatic and cognitive complexity.

    PATTERN: AST traversal for complexity metrics calculation
    CRITICAL: Uses AST analysis patterns from existing ast_parser.py
    GOTCHA: Different languages have different complexity definitions
    """

    def __init__(self):
        """Initialize complexity analyzer."""
        self.logger = logger

    async def analyze_complexity(
        self,
        source_path: str,
        include_cognitive: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze code complexity metrics.

        PATTERN: Calculate multiple complexity metrics in parallel
        CRITICAL: Includes cyclomatic, cognitive, and maintainability index

        Args:
            source_path: Path to source code
            include_cognitive: Calculate cognitive complexity (more expensive)

        Returns:
            Complexity analysis results
        """
        start_time = datetime.now()

        # Get all Python files
        files = await self._get_python_files(source_path)

        if not files:
            return self._empty_report()

        # Analyze each file
        tasks = []
        for file_path in files:
            tasks.append(self._analyze_file(file_path, include_cognitive))

        file_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        total_cyclomatic = 0
        total_cognitive = 0
        total_functions = 0
        total_lines = 0
        files_analyzed = 0
        complex_functions = []

        file_details = {}

        for i, result in enumerate(file_results):
            if isinstance(result, Exception):
                self.logger.error(f"Error analyzing {files[i]}: {result}")
                continue

            file_path = files[i]
            file_details[file_path] = result

            total_cyclomatic += result["total_cyclomatic"]
            total_cognitive += result["total_cognitive"]
            total_functions += result["function_count"]
            total_lines += result["line_count"]
            files_analyzed += 1

            # Collect complex functions
            for func in result["functions"]:
                if func["cyclomatic_complexity"] > 10:  # Threshold for complexity
                    complex_functions.append({
                        "file": file_path,
                        "function": func["name"],
                        "cyclomatic": func["cyclomatic_complexity"],
                        "cognitive": func.get("cognitive_complexity", 0),
                        "line": func["line_number"],
                    })

        # Calculate averages
        avg_cyclomatic = total_cyclomatic / total_functions if total_functions > 0 else 0
        avg_cognitive = total_cognitive / total_functions if total_functions > 0 else 0

        # Calculate maintainability index
        maintainability_index = self._calculate_maintainability_index(
            total_lines,
            avg_cyclomatic,
            total_functions
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "timestamp": datetime.now().isoformat(),
            "source_path": source_path,
            "files_analyzed": files_analyzed,
            "total_functions": total_functions,
            "total_lines": total_lines,
            "cyclomatic_complexity": round(avg_cyclomatic, 2),
            "cognitive_complexity": round(avg_cognitive, 2),
            "maintainability_index": round(maintainability_index, 2),
            "complex_functions": sorted(
                complex_functions,
                key=lambda x: x["cyclomatic"],
                reverse=True
            )[:20],  # Top 20 most complex
            "files": file_details,
            "execution_time_seconds": round(execution_time, 2),
        }

    async def _analyze_file(
        self,
        file_path: str,
        include_cognitive: bool
    ) -> Dict[str, Any]:
        """
        Analyze complexity for a single file.

        Args:
            file_path: Path to Python file
            include_cognitive: Calculate cognitive complexity

        Returns:
            File complexity metrics
        """
        try:
            # Read file
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            # Parse AST
            tree = ast.parse(source_code)

            # Count lines
            line_count = len(source_code.splitlines())

            # Analyze functions and methods
            functions = []
            total_cyclomatic = 0
            total_cognitive = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_complexity = self._analyze_function(
                        node,
                        source_code,
                        include_cognitive
                    )
                    functions.append(func_complexity)
                    total_cyclomatic += func_complexity["cyclomatic_complexity"]
                    total_cognitive += func_complexity.get("cognitive_complexity", 0)

            return {
                "line_count": line_count,
                "function_count": len(functions),
                "total_cyclomatic": total_cyclomatic,
                "total_cognitive": total_cognitive,
                "functions": functions,
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            raise

    def _analyze_function(
        self,
        func_node: ast.FunctionDef,
        source_code: str,
        include_cognitive: bool
    ) -> Dict[str, Any]:
        """
        Analyze complexity of a single function.

        PATTERN: Calculate cyclomatic and optionally cognitive complexity
        CRITICAL: Cyclomatic = number of decision points + 1

        Args:
            func_node: AST node for function
            source_code: Full source code
            include_cognitive: Calculate cognitive complexity

        Returns:
            Function complexity metrics
        """
        # Calculate cyclomatic complexity
        cyclomatic = self._calculate_cyclomatic_complexity(func_node)

        # Calculate cognitive complexity if requested
        cognitive = 0
        if include_cognitive:
            cognitive = self._calculate_cognitive_complexity(func_node)

        # Get function signature
        func_name = func_node.name
        line_number = func_node.lineno

        # Count lines in function
        func_lines = 0
        if hasattr(func_node, "end_lineno") and func_node.end_lineno:
            func_lines = func_node.end_lineno - func_node.lineno + 1

        return {
            "name": func_name,
            "line_number": line_number,
            "line_count": func_lines,
            "cyclomatic_complexity": cyclomatic,
            "cognitive_complexity": cognitive if include_cognitive else None,
        }

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        Calculate cyclomatic complexity for an AST node.

        PATTERN: Count decision points (if, for, while, and, or, except)
        CRITICAL: Cyclomatic complexity = decision points + 1

        Args:
            node: AST node to analyze

        Returns:
            Cyclomatic complexity score
        """
        complexity = 1  # Base complexity

        # Decision point node types
        decision_nodes = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.ExceptHandler,
        )

        # Walk the AST and count decision points
        for child in ast.walk(node):
            if isinstance(child, decision_nodes):
                complexity += 1

            # Count boolean operators (and, or)
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

            # Count comprehensions
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # Each comprehension adds complexity
                for generator in child.generators:
                    complexity += 1
                    # Count if clauses in comprehensions
                    complexity += len(generator.ifs)

        return complexity

    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """
        Calculate cognitive complexity for an AST node.

        PATTERN: Weight decision points by nesting level
        CRITICAL: Cognitive complexity penalizes nesting more heavily
        GOTCHA: More expensive to calculate than cyclomatic

        Args:
            node: AST node to analyze

        Returns:
            Cognitive complexity score
        """
        complexity = 0

        def visit_node(n: ast.AST, nesting_level: int, parent_node: Optional[ast.AST] = None):
            nonlocal complexity

            # Increment for control structures
            increment_nodes = (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.ExceptHandler,
            )

            if isinstance(n, increment_nodes):
                # Add base complexity + nesting penalty
                complexity += 1 + nesting_level

                # Increase nesting level for children
                new_nesting = nesting_level + 1
            else:
                new_nesting = nesting_level

            # Boolean operators add complexity
            if isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1

            # Recursion adds complexity
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if function calls itself
                func_name = n.name
                for child in ast.walk(n):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == func_name:
                            complexity += 1
                            break

            # Visit children
            for child in ast.iter_child_nodes(n):
                visit_node(child, new_nesting, n)

        # Start traversal
        for child in ast.iter_child_nodes(node):
            visit_node(child, 0, node)

        return complexity

    def _calculate_maintainability_index(
        self,
        lines_of_code: int,
        avg_cyclomatic: float,
        num_functions: int
    ) -> float:
        """
        Calculate maintainability index.

        PATTERN: Microsoft maintainability index formula (simplified)
        CRITICAL: Scale 0-100, higher is better
        Formula: 171 - 5.2 * ln(Halstead Volume) - 0.23 * Cyclomatic - 16.2 * ln(LOC)

        Args:
            lines_of_code: Total lines of code
            avg_cyclomatic: Average cyclomatic complexity
            num_functions: Number of functions

        Returns:
            Maintainability index (0-100)
        """
        import math

        if lines_of_code == 0:
            return 100.0

        # Simplified formula without Halstead volume
        # MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
        # Approximating without full Halstead metrics

        try:
            # Approximate Halstead volume based on LOC and functions
            approx_volume = lines_of_code * math.log(max(num_functions, 1))

            mi = 171
            mi -= 5.2 * math.log(max(approx_volume, 1))
            mi -= 0.23 * avg_cyclomatic
            mi -= 16.2 * math.log(max(lines_of_code, 1))

            # Scale to 0-100
            mi = max(0, min(100, mi))

            # Normalize: values above 85 = good, 65-85 = moderate, below 65 = difficult
            return mi

        except Exception as e:
            self.logger.error(f"Failed to calculate maintainability index: {e}")
            return 50.0  # Return neutral score on error

    async def _get_python_files(self, source_path: str) -> List[str]:
        """
        Get all Python files in source path.

        Args:
            source_path: Path to search

        Returns:
            List of Python file paths
        """
        path = Path(source_path)
        files = []

        if path.is_file() and path.suffix == ".py":
            files.append(str(path))
        elif path.is_dir():
            # Recursively find all .py files
            files = [str(f) for f in path.rglob("*.py")]

        return files

    def _empty_report(self) -> Dict[str, Any]:
        """
        Create empty complexity report.

        Returns:
            Empty report structure
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "files_analyzed": 0,
            "total_functions": 0,
            "total_lines": 0,
            "cyclomatic_complexity": 0.0,
            "cognitive_complexity": 0.0,
            "maintainability_index": 100.0,
            "complex_functions": [],
            "files": {},
        }

    async def get_complex_hotspots(
        self,
        source_path: str,
        threshold: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify complexity hotspots that need refactoring.

        Args:
            source_path: Path to analyze
            threshold: Complexity threshold for hotspots

        Returns:
            List of complex functions exceeding threshold
        """
        results = await self.analyze_complexity(source_path)

        hotspots = [
            func for func in results.get("complex_functions", [])
            if func["cyclomatic"] >= threshold
        ]

        return hotspots

    def generate_complexity_recommendations(
        self,
        complexity_data: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on complexity analysis.

        Args:
            complexity_data: Complexity analysis results

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        avg_cyclomatic = complexity_data.get("cyclomatic_complexity", 0)
        avg_cognitive = complexity_data.get("cognitive_complexity", 0)
        maintainability = complexity_data.get("maintainability_index", 100)
        complex_functions = complexity_data.get("complex_functions", [])

        # Average cyclomatic complexity recommendations
        if avg_cyclomatic > 10:
            recommendations.append(
                f"Average cyclomatic complexity ({avg_cyclomatic:.1f}) is high. "
                "Consider refactoring complex functions into smaller units."
            )
        elif avg_cyclomatic > 5:
            recommendations.append(
                f"Average cyclomatic complexity ({avg_cyclomatic:.1f}) is moderate. "
                "Monitor complexity as codebase grows."
            )

        # Cognitive complexity recommendations
        if avg_cognitive > 15:
            recommendations.append(
                f"Average cognitive complexity ({avg_cognitive:.1f}) indicates code may be hard to understand. "
                "Focus on reducing nesting and improving readability."
            )

        # Maintainability index recommendations
        if maintainability < 65:
            recommendations.append(
                f"Maintainability index ({maintainability:.1f}) is low. "
                "Code may be difficult to maintain. Consider refactoring and adding documentation."
            )
        elif maintainability < 85:
            recommendations.append(
                f"Maintainability index ({maintainability:.1f}) is moderate. "
                "Some areas may benefit from simplification."
            )

        # Specific function recommendations
        if complex_functions:
            top_complex = complex_functions[0]
            recommendations.append(
                f"Function '{top_complex['function']}' in {top_complex['file']} "
                f"has cyclomatic complexity of {top_complex['cyclomatic']}. "
                "This should be refactored as a priority."
            )

            if len(complex_functions) > 5:
                recommendations.append(
                    f"Found {len(complex_functions)} functions with high complexity. "
                    "Consider a refactoring sprint to address technical debt."
                )

        if not recommendations:
            recommendations.append(
                "Code complexity is within acceptable ranges. Continue monitoring as codebase evolves."
            )

        return recommendations
