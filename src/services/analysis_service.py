"""High-level code analysis orchestration service."""

import logging
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path

from ..analysis import (
    ASTParser,
    ComplexityAnalyzer,
    DependencyAnalyzer,
    PatternDetector,
    SemanticSearch,
    ASTCache,
)
from ..analysis.languages import (
    PythonAnalyzer,
    JavaScriptAnalyzer,
    JavaAnalyzer,
    GoAnalyzer,
)
from ..models.analysis_models import (
    Language,
    AnalysisRequest,
    AnalysisResult,
    SemanticSearchRequest,
    SemanticSearchResult,
    CodeEntity,
)

logger = logging.getLogger(__name__)


class AnalysisOrchestrator:
    """
    High-level code analysis service.

    PATTERN: Service facade for all analysis operations
    CRITICAL: Single entry point for code analysis
    GOTCHA: Initialize analyzers once, reuse for performance
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        llm_service=None,
    ):
        """
        Initialize analysis orchestrator.

        Args:
            cache_enabled: Whether to enable AST caching
            cache_ttl: Cache time-to-live in seconds
            llm_service: Optional LLM service for semantic search
        """
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.parser = ASTParser()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.pattern_detector = PatternDetector()
        self.semantic_search = SemanticSearch(llm_service)

        # Initialize cache
        self.cache = ASTCache(ttl_seconds=cache_ttl) if cache_enabled else None

        # Initialize language-specific analyzers
        self.language_analyzers = {
            Language.PYTHON: PythonAnalyzer(),
            Language.JAVASCRIPT: JavaScriptAnalyzer(Language.JAVASCRIPT),
            Language.TYPESCRIPT: JavaScriptAnalyzer(Language.TYPESCRIPT),
            Language.JAVA: JavaAnalyzer(),
            Language.GO: GoAnalyzer(),
        }

        self.logger.info(
            f"Analysis Orchestrator initialized with {len(self.language_analyzers)} language analyzers"
        )

    async def analyze_file(
        self, file_path: str, request: Optional[AnalysisRequest] = None
    ) -> AnalysisResult:
        """
        Analyze a single file.

        CRITICAL: This is the main entry point for file analysis
        PATTERN: Parse -> Extract Entities -> Analyze Components

        Args:
            file_path: Path to file to analyze
            request: Optional analysis request with options

        Returns:
            AnalysisResult with all requested analyses
        """
        start_time = datetime.now()
        errors = []

        # Create default request if not provided
        if request is None:
            request = AnalysisRequest(file_paths=[file_path])

        # Read file content
        try:
            with open(file_path, "rb") as f:
                content = f.read()
        except Exception as e:
            error_msg = f"Failed to read file {file_path}: {e}"
            self.logger.error(error_msg)
            return self._create_error_result(file_path, error_msg, start_time)

        # Check cache if enabled
        cache_hit = False
        if self.cache and request.cache_enabled:
            cached_result = self.cache.get_cached_result(file_path, content)
            if cached_result:
                cached_result.cache_hit = True
                return cached_result

        # Detect language
        language = request.language or self.parser.detect_language(file_path)

        if language == Language.UNKNOWN:
            error_msg = f"Unknown language for file: {file_path}"
            self.logger.error(error_msg)
            return self._create_error_result(file_path, error_msg, start_time)

        # Parse AST
        ast = None
        if self.cache and request.cache_enabled:
            ast = self.cache.get_cached_ast(file_path, content)
            cache_hit = ast is not None

        if ast is None:
            ast = await self.parser.parse_file(file_path, language)
            if ast and self.cache:
                self.cache.store_ast(file_path, content, ast)

        if ast is None:
            error_msg = f"Failed to parse file: {file_path}"
            self.logger.error(error_msg)
            return self._create_error_result(file_path, error_msg, start_time)

        # Extract entities
        entities = []
        if "ast" in request.analysis_types or "dependencies" in request.analysis_types:
            entities = await self._extract_entities(file_path, ast, language)

        # Analyze complexity
        complexity = None
        if "complexity" in request.analysis_types:
            try:
                complexity = await self.complexity_analyzer.analyze(
                    file_path, ast, language, content.decode("utf-8", errors="ignore")
                )
            except Exception as e:
                error_msg = f"Complexity analysis failed: {e}"
                self.logger.warning(error_msg)
                errors.append(error_msg)

        # Analyze dependencies
        dependencies = None
        if "dependencies" in request.analysis_types and entities:
            try:
                dependencies = await self.dependency_analyzer.analyze(entities, ast)
            except Exception as e:
                error_msg = f"Dependency analysis failed: {e}"
                self.logger.warning(error_msg)
                errors.append(error_msg)

        # Detect patterns
        patterns = []
        if "patterns" in request.analysis_types and entities:
            try:
                patterns = await self.pattern_detector.detect_patterns(entities, ast)
            except Exception as e:
                error_msg = f"Pattern detection failed: {e}"
                self.logger.warning(error_msg)
                errors.append(error_msg)

        # Calculate analysis time
        analysis_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Create result
        result = AnalysisResult(
            file_path=file_path,
            language=language,
            ast=ast if "ast" in request.analysis_types else None,
            entities=entities,
            dependencies=dependencies,
            complexity=complexity,
            patterns=patterns,
            errors=errors,
            analysis_time_ms=analysis_time_ms,
            cache_hit=cache_hit,
        )

        # Cache result if enabled
        if self.cache and request.cache_enabled:
            self.cache.store_result(file_path, content, result)

        return result

    async def analyze_codebase(
        self, directory: str, request: Optional[AnalysisRequest] = None
    ) -> List[AnalysisResult]:
        """
        Analyze entire codebase directory.

        Args:
            directory: Directory path to analyze
            request: Optional analysis request

        Returns:
            List of AnalysisResults for all files
        """
        # Find all supported files
        files = self._find_source_files(directory)

        self.logger.info(f"Found {len(files)} files to analyze in {directory}")

        # Analyze each file
        results = []
        for file_path in files:
            try:
                result = await self.analyze_file(file_path, request)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to analyze {file_path}: {e}")

        return results

    async def search_code(
        self, request: SemanticSearchRequest
    ) -> List[SemanticSearchResult]:
        """
        Search code using semantic search.

        Args:
            request: Search request

        Returns:
            List of search results
        """
        return await self.semantic_search.search(request)

    async def index_codebase(self, directory: str):
        """
        Index codebase for semantic search.

        Args:
            directory: Directory path to index
        """
        # Analyze all files
        results = await self.analyze_codebase(directory)

        # Extract all entities
        all_entities = []
        for result in results:
            all_entities.extend(result.entities)

        # Index entities
        await self.semantic_search.index_code(all_entities)

        self.logger.info(f"Indexed {len(all_entities)} entities from {directory}")

    async def _extract_entities(
        self, file_path: str, ast, language: Language
    ) -> List[CodeEntity]:
        """
        Extract code entities using language-specific analyzer.

        Args:
            file_path: File path
            ast: Parsed AST
            language: Programming language

        Returns:
            List of code entities
        """
        analyzer = self.language_analyzers.get(language)

        if not analyzer:
            self.logger.warning(f"No analyzer for language: {language}")
            return []

        try:
            entities = await analyzer.analyze(file_path, ast)
            return entities
        except Exception as e:
            self.logger.error(f"Entity extraction failed for {file_path}: {e}")
            return []

    def _find_source_files(self, directory: str) -> List[str]:
        """
        Find all source code files in directory.

        Args:
            directory: Directory path

        Returns:
            List of file paths
        """
        dir_path = Path(directory)
        files = []

        # Supported extensions
        extensions = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go"}

        for ext in extensions:
            files.extend([str(f) for f in dir_path.rglob(f"*{ext}")])

        return files

    def _create_error_result(
        self, file_path: str, error: str, start_time: datetime
    ) -> AnalysisResult:
        """Create an error result."""
        analysis_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return AnalysisResult(
            file_path=file_path,
            language=Language.UNKNOWN,
            errors=[error],
            analysis_time_ms=analysis_time_ms,
            cache_hit=False,
        )

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {}

    def clear_cache(self):
        """Clear all caches."""
        if self.cache:
            self.cache.clear()
        self.semantic_search.clear_index()
        self.logger.info("Cleared all caches")
