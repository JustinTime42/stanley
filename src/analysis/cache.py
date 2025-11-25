"""AST caching layer."""

import logging
import hashlib
from typing import Optional, Dict
from datetime import datetime, timedelta

from ..models.analysis_models import ASTNode, AnalysisResult

logger = logging.getLogger(__name__)


class ASTCache:
    """
    Cache for parsed ASTs and analysis results.

    PATTERN: In-memory cache with TTL
    CRITICAL: Cache keys must include file hash
    GOTCHA: File changes must invalidate cache entries
    """

    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize AST cache.

        Args:
            ttl_seconds: Time-to-live for cache entries
        """
        self.logger = logger
        self.ttl_seconds = ttl_seconds

        # In-memory caches
        self.ast_cache: Dict[str, tuple[ASTNode, datetime]] = {}
        self.result_cache: Dict[str, tuple[AnalysisResult, datetime]] = {}

    def get_cached_ast(self, file_path: str, file_content: bytes) -> Optional[ASTNode]:
        """
        Get cached AST for file.

        Args:
            file_path: Path to file
            file_content: File content bytes

        Returns:
            Cached ASTNode if found and valid, None otherwise
        """
        cache_key = self._get_cache_key(file_path, file_content)

        if cache_key in self.ast_cache:
            ast, cached_at = self.ast_cache[cache_key]

            # Check if cache is still valid
            if self._is_cache_valid(cached_at):
                self.logger.debug(f"AST cache hit for {file_path}")
                return ast
            else:
                # Cache expired
                del self.ast_cache[cache_key]
                self.logger.debug(f"AST cache expired for {file_path}")

        return None

    def store_ast(self, file_path: str, file_content: bytes, ast: ASTNode):
        """
        Store AST in cache.

        Args:
            file_path: Path to file
            file_content: File content bytes
            ast: Parsed AST to cache
        """
        cache_key = self._get_cache_key(file_path, file_content)
        self.ast_cache[cache_key] = (ast, datetime.now())
        self.logger.debug(f"Stored AST in cache for {file_path}")

    def get_cached_result(
        self, file_path: str, file_content: bytes
    ) -> Optional[AnalysisResult]:
        """
        Get cached analysis result.

        Args:
            file_path: Path to file
            file_content: File content bytes

        Returns:
            Cached AnalysisResult if found and valid, None otherwise
        """
        cache_key = self._get_cache_key(file_path, file_content)

        if cache_key in self.result_cache:
            result, cached_at = self.result_cache[cache_key]

            if self._is_cache_valid(cached_at):
                self.logger.debug(f"Result cache hit for {file_path}")
                return result
            else:
                del self.result_cache[cache_key]
                self.logger.debug(f"Result cache expired for {file_path}")

        return None

    def store_result(self, file_path: str, file_content: bytes, result: AnalysisResult):
        """
        Store analysis result in cache.

        Args:
            file_path: Path to file
            file_content: File content bytes
            result: Analysis result to cache
        """
        cache_key = self._get_cache_key(file_path, file_content)
        self.result_cache[cache_key] = (result, datetime.now())
        self.logger.debug(f"Stored result in cache for {file_path}")

    def invalidate(self, file_path: str):
        """
        Invalidate all cache entries for a file.

        Args:
            file_path: Path to file
        """
        # Remove all entries with this file path
        keys_to_remove = [key for key in self.ast_cache.keys() if file_path in key] + [
            key for key in self.result_cache.keys() if file_path in key
        ]

        for key in keys_to_remove:
            self.ast_cache.pop(key, None)
            self.result_cache.pop(key, None)

        self.logger.debug(f"Invalidated cache for {file_path}")

    def clear(self):
        """Clear all cache entries."""
        self.ast_cache.clear()
        self.result_cache.clear()
        self.logger.info("Cleared all cache entries")

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        return {
            "ast_cache_size": len(self.ast_cache),
            "result_cache_size": len(self.result_cache),
            "total_entries": len(self.ast_cache) + len(self.result_cache),
        }

    def _get_cache_key(self, file_path: str, file_content: bytes) -> str:
        """
        Generate cache key from file path and content hash.

        CRITICAL: Include content hash to detect changes

        Args:
            file_path: Path to file
            file_content: File content bytes

        Returns:
            Cache key string
        """
        # Hash file content
        content_hash = hashlib.sha256(file_content).hexdigest()[:16]

        # Combine file path and content hash
        return f"{file_path}:{content_hash}"

    def _is_cache_valid(self, cached_at: datetime) -> bool:
        """
        Check if cache entry is still valid.

        Args:
            cached_at: When the entry was cached

        Returns:
            True if still valid, False if expired
        """
        age = datetime.now() - cached_at
        return age < timedelta(seconds=self.ttl_seconds)
