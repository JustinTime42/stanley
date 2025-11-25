"""File system scanner with intelligent filtering."""

import logging
import hashlib
import fnmatch
from pathlib import Path
from typing import Iterator, Optional, List, Set
from datetime import datetime

from .models import FileInfo

logger = logging.getLogger(__name__)


class FileScanner:
    """
    Intelligent file system scanner for codebase analysis.

    PATTERN: Streaming iteration with filtering
    CRITICAL: Skip binary files, node_modules, __pycache__, etc.
    GOTCHA: Must respect .gitignore patterns
    """

    # Default patterns to skip
    DEFAULT_SKIP_PATTERNS: Set[str] = {
        # Version control
        ".git",
        ".svn",
        ".hg",
        # Dependencies
        "node_modules",
        "venv",
        ".venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        # Build outputs
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        "out",
        # IDE
        ".idea",
        ".vscode",
        ".vs",
        # Package locks (too large, not useful for understanding)
        "package-lock.json",
        "yarn.lock",
        "poetry.lock",
        "Pipfile.lock",
        "pnpm-lock.yaml",
        # Binary/compiled
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.dll",
        "*.exe",
        "*.class",
        "*.jar",
        "*.war",
        # Bundled/minified
        "*.min.js",
        "*.min.css",
        "*.bundle.js",
        "*.chunk.js",
        # Data files
        "*.sqlite",
        "*.db",
        "*.pickle",
        # Media
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.gif",
        "*.ico",
        "*.svg",
        "*.mp3",
        "*.mp4",
        "*.wav",
        "*.pdf",
        # Other
        ".DS_Store",
        "Thumbs.db",
        "*.log",
        "*.tmp",
        "*.bak",
    }

    # File extensions to language mapping
    EXTENSION_TO_LANGUAGE = {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".vue": "vue",
        ".svelte": "svelte",
        # Config/data files
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".less": "less",
        ".md": "markdown",
        ".rst": "rst",
        ".txt": "text",
        ".sql": "sql",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".fish": "shell",
        ".ps1": "powershell",
        ".dockerfile": "dockerfile",
        ".graphql": "graphql",
        ".proto": "protobuf",
    }

    def __init__(
        self,
        root_path: str | Path,
        skip_patterns: Optional[Set[str]] = None,
        include_patterns: Optional[Set[str]] = None,
        respect_gitignore: bool = True,
        max_file_size_mb: float = 10.0,
    ):
        """
        Initialize scanner.

        Args:
            root_path: Project root directory
            skip_patterns: Additional patterns to skip
            include_patterns: Only include files matching these patterns
            respect_gitignore: Whether to respect .gitignore
            max_file_size_mb: Maximum file size to scan in MB
        """
        self.root_path = Path(root_path).resolve()
        self.skip_patterns = self.DEFAULT_SKIP_PATTERNS.copy()
        if skip_patterns:
            self.skip_patterns.update(skip_patterns)
        self.include_patterns = include_patterns
        self.respect_gitignore = respect_gitignore
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

        # Load .gitignore patterns
        self._gitignore_patterns: List[str] = []
        if respect_gitignore:
            self._load_gitignore()

    def _load_gitignore(self) -> None:
        """Load patterns from .gitignore file."""
        gitignore_path = self.root_path / ".gitignore"
        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith("#"):
                            self._gitignore_patterns.append(line)
                logger.debug(f"Loaded {len(self._gitignore_patterns)} gitignore patterns")
            except Exception as e:
                logger.warning(f"Failed to read .gitignore: {e}")

    def _should_skip(self, path: Path) -> bool:
        """
        Check if path should be skipped.

        Args:
            path: Path to check

        Returns:
            True if should skip
        """
        name = path.name
        rel_path = str(path.relative_to(self.root_path))

        # Check skip patterns
        for pattern in self.skip_patterns:
            # Pattern matches name
            if fnmatch.fnmatch(name, pattern):
                return True
            # Pattern matches relative path
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            # Pattern is in path (for directories like node_modules)
            if pattern in rel_path.split("/") or pattern in rel_path.split("\\"):
                return True

        # Check gitignore patterns
        for pattern in self._gitignore_patterns:
            # Handle negation
            if pattern.startswith("!"):
                continue  # TODO: Handle negation properly

            # Handle directory-specific patterns
            if pattern.endswith("/"):
                if path.is_dir() and fnmatch.fnmatch(name, pattern.rstrip("/")):
                    return True
            else:
                if fnmatch.fnmatch(name, pattern):
                    return True
                if fnmatch.fnmatch(rel_path, pattern):
                    return True

        return False

    def _should_include(self, path: Path) -> bool:
        """
        Check if file should be included.

        Args:
            path: File path

        Returns:
            True if should include
        """
        if not self.include_patterns:
            return True

        name = path.name
        for pattern in self.include_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _detect_language(self, path: Path) -> str:
        """Detect programming language from file extension."""
        suffix = path.suffix.lower()
        language = self.EXTENSION_TO_LANGUAGE.get(suffix, "unknown")

        # Special cases
        if path.name.lower() == "dockerfile":
            return "dockerfile"
        if path.name.lower() in ("makefile", "gnumakefile"):
            return "makefile"
        if path.name.lower() == ".env":
            return "env"

        return language

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of file content."""
        try:
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:32]
        except Exception as e:
            logger.warning(f"Failed to hash {path}: {e}")
            return ""

    def _count_lines(self, path: Path) -> int:
        """Count lines in file."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.warning(f"Failed to count lines in {path}: {e}")
            return 0

    def scan(
        self,
        additional_skip_patterns: Optional[List[str]] = None,
    ) -> Iterator[FileInfo]:
        """
        Scan directory and yield FileInfo for each source file.

        PATTERN: Generator for memory efficiency
        CRITICAL: Don't load file contents into memory

        Args:
            additional_skip_patterns: Extra patterns to skip

        Yields:
            FileInfo for each discovered source file
        """
        skip_set = self.skip_patterns.copy()
        if additional_skip_patterns:
            skip_set.update(additional_skip_patterns)

        # Use rglob for recursive iteration
        for path in self.root_path.rglob("*"):
            # Skip directories
            if path.is_dir():
                continue

            # Skip based on patterns
            if self._should_skip(path):
                continue

            # Skip based on include patterns
            if not self._should_include(path):
                continue

            # Skip files too large
            try:
                size = path.stat().st_size
                if size > self.max_file_size_bytes:
                    logger.debug(f"Skipping large file: {path} ({size} bytes)")
                    continue
            except OSError:
                continue

            # Skip binary files (heuristic)
            if self._is_binary(path):
                continue

            # Detect language
            language = self._detect_language(path)

            # Create FileInfo
            try:
                stat = path.stat()
                yield FileInfo(
                    path=str(path),
                    relative_path=str(path.relative_to(self.root_path)),
                    language=language,
                    size_bytes=stat.st_size,
                    line_count=self._count_lines(path),
                    last_modified=datetime.fromtimestamp(stat.st_mtime),
                    content_hash=self._compute_file_hash(path),
                )
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")
                continue

    def _is_binary(self, path: Path) -> bool:
        """
        Check if file appears to be binary.

        PATTERN: Read first chunk and check for null bytes
        """
        try:
            with open(path, "rb") as f:
                chunk = f.read(1024)
                # Check for null bytes (common in binary files)
                if b"\x00" in chunk:
                    return True
                # Check if most bytes are printable ASCII or whitespace
                text_chars = set(range(32, 127)) | {9, 10, 13}  # printable + tab, newline, cr
                non_text = sum(1 for byte in chunk if byte not in text_chars)
                if len(chunk) > 0 and non_text / len(chunk) > 0.3:
                    return True
                return False
        except Exception:
            return True  # Assume binary if can't read

    def scan_quick(self) -> List[FileInfo]:
        """
        Quick scan returning list of files without content analysis.

        Returns:
            List of FileInfo objects
        """
        return list(self.scan())

    def get_source_files(self, languages: Optional[List[str]] = None) -> List[FileInfo]:
        """
        Get source files filtered by language.

        Args:
            languages: List of languages to include (None for all)

        Returns:
            List of FileInfo for matching files
        """
        files = []
        for file_info in self.scan():
            if languages is None or file_info.language in languages:
                files.append(file_info)
        return files

    def get_statistics(self) -> dict:
        """
        Get scanning statistics.

        Returns:
            Dict with file counts by language, total lines, etc.
        """
        stats = {
            "total_files": 0,
            "total_lines": 0,
            "total_bytes": 0,
            "by_language": {},
        }

        for file_info in self.scan():
            stats["total_files"] += 1
            stats["total_lines"] += file_info.line_count
            stats["total_bytes"] += file_info.size_bytes

            lang = file_info.language
            if lang not in stats["by_language"]:
                stats["by_language"][lang] = {"files": 0, "lines": 0}
            stats["by_language"][lang]["files"] += 1
            stats["by_language"][lang]["lines"] += file_info.line_count

        return stats
