"""Code analysis tools for agents."""

import logging
from datetime import datetime
from typing import Optional

from ..base import BaseTool
from ...models.tool_models import (
    ToolCategory,
    ToolSchema,
    ToolParameter,
    ToolResult,
)
from ...models.analysis_models import (
    AnalysisRequest,
    SemanticSearchRequest,
)
from ...services.analysis_service import AnalysisOrchestrator

logger = logging.getLogger(__name__)


class AnalyzeCodeTool(BaseTool):
    """
    Tool for analyzing code files.

    PATTERN: Tool with analysis service integration
    GOTCHA: Requires analysis service dependency
    """

    def __init__(self, analysis_service: Optional[AnalysisOrchestrator] = None):
        """
        Initialize code analysis tool.

        Args:
            analysis_service: Analysis orchestrator service
        """
        super().__init__(
            name="analyze_code",
            category=ToolCategory.CODE_GENERATION,
        )
        self.analysis_service = analysis_service or AnalysisOrchestrator()

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Analyze code file for structure, complexity, and patterns",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to file to analyze",
                    required=True,
                ),
                ToolParameter(
                    name="analysis_types",
                    type="array",
                    description="Types of analysis to perform",
                    required=False,
                    default=["ast", "complexity", "patterns"],
                ),
                ToolParameter(
                    name="include_dependencies",
                    type="boolean",
                    description="Include dependency graph",
                    required=False,
                    default=False,
                ),
            ],
            returns="Analysis results with AST, complexity, and patterns",
            timeout_seconds=120,
        )

    async def execute(
        self,
        file_path: str,
        analysis_types: list[str] = None,
        include_dependencies: bool = False,
        **kwargs,
    ) -> ToolResult:
        """
        Execute code analysis.

        Args:
            file_path: Path to file
            analysis_types: Types of analysis
            include_dependencies: Include dependency analysis

        Returns:
            ToolResult with analysis results
        """
        start_time = datetime.now()

        try:
            # Build analysis request
            if analysis_types is None:
                analysis_types = ["ast", "complexity", "patterns"]

            if include_dependencies:
                analysis_types.append("dependencies")

            request = AnalysisRequest(
                file_paths=[file_path],
                analysis_types=analysis_types,
                cache_enabled=True,
            )

            # Analyze file
            result = await self.analysis_service.analyze_file(file_path, request)

            # Convert to JSON-serializable format
            result_dict = {
                "file_path": result.file_path,
                "language": result.language.value,
                "entities": [
                    {
                        "name": e.name,
                        "type": e.type.value,
                        "line_start": e.line_start,
                        "line_end": e.line_end,
                        "signature": e.signature,
                    }
                    for e in result.entities
                ],
                "complexity": (
                    {
                        "cyclomatic": result.complexity.cyclomatic_complexity,
                        "cognitive": result.complexity.cognitive_complexity,
                        "lines_of_code": result.complexity.lines_of_code,
                        "function_count": result.complexity.function_count,
                        "class_count": result.complexity.class_count,
                    }
                    if result.complexity
                    else None
                ),
                "patterns": [
                    {
                        "name": p.name,
                        "type": p.type.value,
                        "confidence": p.confidence,
                        "description": p.description,
                        "recommendation": p.recommendation,
                    }
                    for p in result.patterns
                ],
                "errors": result.errors,
                "analysis_time_ms": result.analysis_time_ms,
                "cache_hit": result.cache_hit,
            }

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result=result_dict,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Code analysis failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class CalculateComplexityTool(BaseTool):
    """Tool for calculating code complexity metrics."""

    def __init__(self, analysis_service: Optional[AnalysisOrchestrator] = None):
        """Initialize complexity calculation tool."""
        super().__init__(
            name="calculate_complexity",
            category=ToolCategory.CODE_GENERATION,
        )
        self.analysis_service = analysis_service or AnalysisOrchestrator()

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Calculate complexity metrics for code",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to file to analyze",
                    required=True,
                ),
            ],
            returns="Complexity metrics (cyclomatic, cognitive, Halstead)",
            timeout_seconds=60,
        )

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        """Execute complexity calculation."""
        start_time = datetime.now()

        try:
            request = AnalysisRequest(
                file_paths=[file_path],
                analysis_types=["complexity"],
                cache_enabled=True,
            )

            result = await self.analysis_service.analyze_file(file_path, request)

            if result.complexity:
                complexity_dict = {
                    "cyclomatic_complexity": result.complexity.cyclomatic_complexity,
                    "cognitive_complexity": result.complexity.cognitive_complexity,
                    "halstead_metrics": result.complexity.halstead_metrics,
                    "lines_of_code": result.complexity.lines_of_code,
                    "function_count": result.complexity.function_count,
                    "class_count": result.complexity.class_count,
                    "max_nesting_depth": result.complexity.max_nesting_depth,
                    "average_function_complexity": result.complexity.average_function_complexity,
                }
            else:
                complexity_dict = {"error": "Failed to calculate complexity"}

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result=complexity_dict,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Complexity calculation failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class FindDependenciesTool(BaseTool):
    """Tool for finding code dependencies."""

    def __init__(self, analysis_service: Optional[AnalysisOrchestrator] = None):
        """Initialize dependency finder tool."""
        super().__init__(
            name="find_dependencies",
            category=ToolCategory.CODE_GENERATION,
        )
        self.analysis_service = analysis_service or AnalysisOrchestrator()

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Find code dependencies (imports, calls, inheritance)",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to file to analyze",
                    required=True,
                ),
            ],
            returns="Dependency graph with imports, calls, and inheritance",
            timeout_seconds=60,
        )

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        """Execute dependency analysis."""
        start_time = datetime.now()

        try:
            request = AnalysisRequest(
                file_paths=[file_path],
                analysis_types=["dependencies"],
                cache_enabled=True,
            )

            result = await self.analysis_service.analyze_file(file_path, request)

            if result.dependencies:
                deps_dict = {
                    "imports": result.dependencies.imports,
                    "calls": result.dependencies.calls,
                    "inheritance": result.dependencies.inheritance,
                    "edge_count": len(result.dependencies.edges),
                }
            else:
                deps_dict = {"error": "No dependencies found"}

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result=deps_dict,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Dependency analysis failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class SearchCodeTool(BaseTool):
    """Tool for semantic code search."""

    def __init__(self, analysis_service: Optional[AnalysisOrchestrator] = None):
        """Initialize code search tool."""
        super().__init__(
            name="search_code",
            category=ToolCategory.CODE_GENERATION,
        )
        self.analysis_service = analysis_service or AnalysisOrchestrator()

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Search code semantically using natural language",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query in natural language",
                    required=True,
                ),
                ToolParameter(
                    name="scope_paths",
                    type="array",
                    description="Paths to search within",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=10,
                ),
            ],
            returns="List of matching code snippets with similarity scores",
            timeout_seconds=60,
        )

    async def execute(
        self, query: str, scope_paths: list[str], max_results: int = 10, **kwargs
    ) -> ToolResult:
        """Execute code search."""
        start_time = datetime.now()

        try:
            request = SemanticSearchRequest(
                query=query,
                scope_paths=scope_paths,
                max_results=max_results,
            )

            results = await self.analysis_service.search_code(request)

            results_list = [
                {
                    "code_snippet": r.code_snippet,
                    "file_path": r.file_path,
                    "line_start": r.line_start,
                    "line_end": r.line_end,
                    "similarity_score": r.similarity_score,
                }
                for r in results
            ]

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={"results": results_list, "total": len(results_list)},
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Code search failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )
