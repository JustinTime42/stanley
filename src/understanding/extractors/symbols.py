"""Symbol extraction using AST parsing."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .base import BaseExtractor
from ..models import Symbol, SymbolKind, FileInfo, ConfidenceLevel

logger = logging.getLogger(__name__)


class SymbolExtractor(BaseExtractor[Dict[str, Symbol]]):
    """
    Extract code symbols using AST parsing.

    PATTERN: Integrate with existing ASTParser from PRP-05
    CRITICAL: Capture signatures, docstrings, type annotations
    GOTCHA: Different AST node types per language
    """

    # Python AST node types to symbol kinds
    PYTHON_NODE_MAPPING = {
        "function_definition": SymbolKind.FUNCTION,
        "async_function_definition": SymbolKind.FUNCTION,
        "class_definition": SymbolKind.CLASS,
        "decorated_definition": None,  # Check inner node
        "assignment": SymbolKind.VARIABLE,
        "import_statement": SymbolKind.IMPORT,
        "import_from_statement": SymbolKind.IMPORT,
    }

    # JavaScript/TypeScript node types
    JS_NODE_MAPPING = {
        "function_declaration": SymbolKind.FUNCTION,
        "arrow_function": SymbolKind.FUNCTION,
        "method_definition": SymbolKind.METHOD,
        "class_declaration": SymbolKind.CLASS,
        "variable_declaration": SymbolKind.VARIABLE,
        "lexical_declaration": SymbolKind.VARIABLE,
        "import_statement": SymbolKind.IMPORT,
        "interface_declaration": SymbolKind.TYPE,
        "type_alias_declaration": SymbolKind.TYPE,
    }

    def __init__(self):
        """Initialize symbol extractor."""
        super().__init__()
        self._ast_parser = None

    async def _get_ast_parser(self):
        """Lazy load AST parser."""
        if self._ast_parser is None:
            try:
                from ...analysis.ast_parser import ASTParser

                self._ast_parser = ASTParser()
            except ImportError as e:
                self.logger.warning(f"AST parser not available: {e}")
        return self._ast_parser

    async def extract(self, files: List[FileInfo]) -> Dict[str, Symbol]:
        """
        Extract symbols from files.

        Args:
            files: List of files to analyze

        Returns:
            Dict mapping symbol IDs to Symbol objects
        """
        symbols = {}

        for file_info in files:
            file_symbols = await self.extract_file(file_info)
            symbols.update(file_symbols)

        return symbols

    async def extract_file(
        self,
        file_info: FileInfo | str,
    ) -> Dict[str, Symbol]:
        """
        Extract symbols from a single file.

        Args:
            file_info: FileInfo object or file path

        Returns:
            Dict of symbols from this file
        """
        if isinstance(file_info, str):
            file_path = file_info
            language = self._detect_language(file_path)
        else:
            file_path = file_info.path
            language = file_info.language

        symbols = {}

        # Try AST-based extraction first
        ast_parser = await self._get_ast_parser()
        if ast_parser and language in ("python", "javascript", "typescript"):
            try:
                symbols = await self._extract_with_ast(file_path, language, ast_parser)
            except Exception as e:
                self.logger.warning(f"AST extraction failed for {file_path}: {e}")

        # Fallback to regex-based extraction
        if not symbols:
            symbols = await self._extract_with_regex(file_path, language)

        return symbols

    async def extract_batch(
        self,
        files: List[FileInfo],
    ) -> Tuple[Dict[str, Symbol], Dict[str, FileInfo]]:
        """
        Extract symbols from a batch of files.

        Args:
            files: List of files to analyze

        Returns:
            Tuple of (symbols dict, updated file infos dict)
        """
        symbols = {}
        file_infos = {}

        for file_info in files:
            try:
                file_symbols = await self.extract_file(file_info)
                symbols.update(file_symbols)

                # Update file info with symbol IDs
                file_info.symbols = list(file_symbols.keys())
                file_info.analyzed = True
                file_info.confidence = ConfidenceLevel.VERIFIED
                file_infos[file_info.path] = file_info
            except Exception as e:
                self.logger.warning(f"Failed to extract symbols from {file_info.path}: {e}")
                file_info.analysis_errors.append(str(e))
                file_infos[file_info.path] = file_info

        return symbols, file_infos

    async def _extract_with_ast(
        self,
        file_path: str,
        language: str,
        ast_parser,
    ) -> Dict[str, Symbol]:
        """Extract symbols using AST parser."""
        from ...models.analysis_models import Language as LangEnum

        # Map language string to enum
        lang_map = {
            "python": LangEnum.PYTHON,
            "javascript": LangEnum.JAVASCRIPT,
            "typescript": LangEnum.TYPESCRIPT,
        }

        lang_enum = lang_map.get(language)
        if not lang_enum:
            return {}

        # Parse file
        ast = await ast_parser.parse_file(file_path, lang_enum)
        if not ast:
            return {}

        # Extract symbols based on language
        if language == "python":
            return self._extract_python_symbols(ast, file_path)
        elif language in ("javascript", "typescript"):
            return self._extract_js_symbols(ast, file_path)

        return {}

    def _extract_python_symbols(self, ast, file_path: str) -> Dict[str, Symbol]:
        """Extract symbols from Python AST."""
        symbols = {}
        content = self._read_file_content(file_path) or ""
        lines = content.split("\n")

        def get_name_from_node(node) -> Optional[str]:
            """Extract name from AST node."""
            for child in node.children:
                if child.node_type == "identifier":
                    return child.text
            return None

        def get_docstring(node) -> Optional[str]:
            """Extract docstring from function/class."""
            for child in node.children:
                if child.node_type == "block":
                    for block_child in child.children:
                        if block_child.node_type == "expression_statement":
                            for expr_child in block_child.children:
                                if expr_child.node_type == "string":
                                    text = expr_child.text
                                    # Remove quotes
                                    if text.startswith('"""') or text.startswith("'''"):
                                        return text[3:-3].strip()
                                    return text[1:-1].strip()
                    break
            return None

        def get_parameters(node) -> List[Dict]:
            """Extract function parameters."""
            params = []
            for child in node.children:
                if child.node_type == "parameters":
                    for param_child in child.children:
                        if param_child.node_type == "identifier":
                            params.append({"name": param_child.text, "type": None})
                        elif param_child.node_type == "typed_parameter":
                            name = None
                            param_type = None
                            for p in param_child.children:
                                if p.node_type == "identifier":
                                    name = p.text
                                elif p.node_type == "type":
                                    param_type = p.text
                            if name:
                                params.append({"name": name, "type": param_type})
            return params

        def get_return_type(node) -> Optional[str]:
            """Extract return type annotation."""
            for child in node.children:
                if child.node_type == "type":
                    return child.text
            return None

        def get_signature(node, name: str, kind: SymbolKind) -> str:
            """Build signature string."""
            if kind == SymbolKind.FUNCTION:
                params = get_parameters(node)
                param_str = ", ".join(p["name"] for p in params)
                return_type = get_return_type(node)
                sig = f"def {name}({param_str})"
                if return_type:
                    sig += f" -> {return_type}"
                return sig
            elif kind == SymbolKind.CLASS:
                return f"class {name}"
            return name

        def visit(node, parent_name: Optional[str] = None, parent_id: Optional[str] = None):
            """Visit AST nodes recursively."""
            kind = self.PYTHON_NODE_MAPPING.get(node.node_type)

            if kind:
                name = get_name_from_node(node)
                if name:
                    line_start = node.start_point[0] + 1
                    line_end = node.end_point[0] + 1

                    qualified_name = f"{parent_name}.{name}" if parent_name else name
                    symbol_id = Symbol.generate_id(file_path, qualified_name, line_start)

                    # Get symbol content for hashing
                    start_line = node.start_point[0]
                    end_line = node.end_point[0] + 1
                    symbol_content = "\n".join(lines[start_line:end_line])

                    symbol = Symbol(
                        id=symbol_id,
                        name=name,
                        qualified_name=qualified_name,
                        kind=kind,
                        file_path=file_path,
                        line_start=line_start,
                        line_end=line_end,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        signature=get_signature(node, name, kind),
                        docstring=get_docstring(node) if kind in (SymbolKind.FUNCTION, SymbolKind.CLASS) else None,
                        parameters=get_parameters(node) if kind == SymbolKind.FUNCTION else [],
                        return_type=get_return_type(node) if kind == SymbolKind.FUNCTION else None,
                        parent_symbol=parent_id,
                        content_hash=Symbol.compute_content_hash(symbol_content),
                        confidence=ConfidenceLevel.VERIFIED,
                        last_verified=datetime.now(),
                    )
                    symbols[symbol_id] = symbol

                    # Visit children with this as parent (for methods in classes)
                    if kind == SymbolKind.CLASS:
                        for child in node.children:
                            visit(child, qualified_name, symbol_id)
                        return

            # Visit children
            for child in node.children:
                visit(child, parent_name, parent_id)

        visit(ast)
        return symbols

    def _extract_js_symbols(self, ast, file_path: str) -> Dict[str, Symbol]:
        """Extract symbols from JavaScript/TypeScript AST."""
        symbols = {}
        content = self._read_file_content(file_path) or ""
        lines = content.split("\n")

        def get_name_from_node(node) -> Optional[str]:
            """Extract name from AST node."""
            for child in node.children:
                if child.node_type == "identifier":
                    return child.text
                if child.node_type == "property_identifier":
                    return child.text
            return None

        def visit(node, parent_name: Optional[str] = None, parent_id: Optional[str] = None):
            """Visit AST nodes recursively."""
            kind = self.JS_NODE_MAPPING.get(node.node_type)

            if kind:
                name = get_name_from_node(node)
                if name:
                    line_start = node.start_point[0] + 1
                    line_end = node.end_point[0] + 1

                    qualified_name = f"{parent_name}.{name}" if parent_name else name
                    symbol_id = Symbol.generate_id(file_path, qualified_name, line_start)

                    # Get symbol content
                    start_line = node.start_point[0]
                    end_line = node.end_point[0] + 1
                    symbol_content = "\n".join(lines[start_line:end_line])

                    symbol = Symbol(
                        id=symbol_id,
                        name=name,
                        qualified_name=qualified_name,
                        kind=kind,
                        file_path=file_path,
                        line_start=line_start,
                        line_end=line_end,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        parent_symbol=parent_id,
                        content_hash=Symbol.compute_content_hash(symbol_content),
                        confidence=ConfidenceLevel.VERIFIED,
                        last_verified=datetime.now(),
                    )
                    symbols[symbol_id] = symbol

                    if kind == SymbolKind.CLASS:
                        for child in node.children:
                            visit(child, qualified_name, symbol_id)
                        return

            for child in node.children:
                visit(child, parent_name, parent_id)

        visit(ast)
        return symbols

    async def _extract_with_regex(
        self,
        file_path: str,
        language: str,
    ) -> Dict[str, Symbol]:
        """
        Fallback regex-based symbol extraction.

        PATTERN: Works when AST not available
        CRITICAL: Less accurate than AST
        """
        content = self._read_file_content(file_path)
        if not content:
            return {}

        symbols = {}
        lines = content.split("\n")

        if language == "python":
            symbols = self._extract_python_regex(file_path, lines)
        elif language in ("javascript", "typescript"):
            symbols = self._extract_js_regex(file_path, lines)

        return symbols

    def _extract_python_regex(
        self,
        file_path: str,
        lines: List[str],
    ) -> Dict[str, Symbol]:
        """Extract Python symbols using regex."""
        symbols = {}

        # Patterns for Python
        function_pattern = re.compile(r"^\s*(async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(\S+))?\s*:")
        class_pattern = re.compile(r"^\s*class\s+(\w+)\s*(?:\([^)]*\))?\s*:")

        current_class = None
        class_indent = 0

        for i, line in enumerate(lines):
            line_num = i + 1

            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                name = class_match.group(1)
                indent = len(line) - len(line.lstrip())

                symbol_id = Symbol.generate_id(file_path, name, line_num)
                symbol = Symbol(
                    id=symbol_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.CLASS,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,  # Will be approximate
                    signature=f"class {name}",
                    content_hash=Symbol.compute_content_hash(line),
                    confidence=ConfidenceLevel.INFERRED,
                    last_verified=datetime.now(),
                )
                symbols[symbol_id] = symbol
                current_class = name
                class_indent = indent
                continue

            # Check for function/method
            func_match = function_pattern.match(line)
            if func_match:
                is_async = func_match.group(1) is not None
                name = func_match.group(2)
                params_str = func_match.group(3)
                return_type = func_match.group(4)

                indent = len(line) - len(line.lstrip())

                # Determine if method or function
                if current_class and indent > class_indent:
                    kind = SymbolKind.METHOD
                    qualified_name = f"{current_class}.{name}"
                else:
                    kind = SymbolKind.FUNCTION
                    qualified_name = name
                    current_class = None

                # Parse parameters
                params = []
                if params_str.strip():
                    for param in params_str.split(","):
                        param = param.strip()
                        if param and param != "self" and param != "cls":
                            param_parts = param.split(":")
                            param_name = param_parts[0].strip().split("=")[0].strip()
                            param_type = param_parts[1].strip() if len(param_parts) > 1 else None
                            params.append({"name": param_name, "type": param_type})

                prefix = "async def" if is_async else "def"
                signature = f"{prefix} {name}({params_str})"
                if return_type:
                    signature += f" -> {return_type}"

                symbol_id = Symbol.generate_id(file_path, qualified_name, line_num)
                symbol = Symbol(
                    id=symbol_id,
                    name=name,
                    qualified_name=qualified_name,
                    kind=kind,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    signature=signature,
                    parameters=params,
                    return_type=return_type,
                    content_hash=Symbol.compute_content_hash(line),
                    confidence=ConfidenceLevel.INFERRED,
                    last_verified=datetime.now(),
                )
                symbols[symbol_id] = symbol

        return symbols

    def _extract_js_regex(
        self,
        file_path: str,
        lines: List[str],
    ) -> Dict[str, Symbol]:
        """Extract JavaScript/TypeScript symbols using regex."""
        symbols = {}

        # Patterns
        function_pattern = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(")
        class_pattern = re.compile(r"^\s*(?:export\s+)?class\s+(\w+)")
        arrow_pattern = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>")
        interface_pattern = re.compile(r"^\s*(?:export\s+)?interface\s+(\w+)")
        type_pattern = re.compile(r"^\s*(?:export\s+)?type\s+(\w+)")

        for i, line in enumerate(lines):
            line_num = i + 1

            # Check class
            class_match = class_pattern.match(line)
            if class_match:
                name = class_match.group(1)
                symbol_id = Symbol.generate_id(file_path, name, line_num)
                symbol = Symbol(
                    id=symbol_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.CLASS,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    signature=f"class {name}",
                    content_hash=Symbol.compute_content_hash(line),
                    confidence=ConfidenceLevel.INFERRED,
                    last_verified=datetime.now(),
                )
                symbols[symbol_id] = symbol
                continue

            # Check function
            func_match = function_pattern.match(line)
            if func_match:
                name = func_match.group(1)
                symbol_id = Symbol.generate_id(file_path, name, line_num)
                symbol = Symbol(
                    id=symbol_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.FUNCTION,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    signature=f"function {name}()",
                    content_hash=Symbol.compute_content_hash(line),
                    confidence=ConfidenceLevel.INFERRED,
                    last_verified=datetime.now(),
                )
                symbols[symbol_id] = symbol
                continue

            # Check arrow function
            arrow_match = arrow_pattern.match(line)
            if arrow_match:
                name = arrow_match.group(1)
                symbol_id = Symbol.generate_id(file_path, name, line_num)
                symbol = Symbol(
                    id=symbol_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.FUNCTION,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    signature=f"const {name} = () =>",
                    content_hash=Symbol.compute_content_hash(line),
                    confidence=ConfidenceLevel.INFERRED,
                    last_verified=datetime.now(),
                )
                symbols[symbol_id] = symbol
                continue

            # Check interface
            interface_match = interface_pattern.match(line)
            if interface_match:
                name = interface_match.group(1)
                symbol_id = Symbol.generate_id(file_path, name, line_num)
                symbol = Symbol(
                    id=symbol_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.TYPE,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    signature=f"interface {name}",
                    content_hash=Symbol.compute_content_hash(line),
                    confidence=ConfidenceLevel.INFERRED,
                    last_verified=datetime.now(),
                )
                symbols[symbol_id] = symbol
                continue

            # Check type alias
            type_match = type_pattern.match(line)
            if type_match:
                name = type_match.group(1)
                symbol_id = Symbol.generate_id(file_path, name, line_num)
                symbol = Symbol(
                    id=symbol_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.TYPE,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    signature=f"type {name}",
                    content_hash=Symbol.compute_content_hash(line),
                    confidence=ConfidenceLevel.INFERRED,
                    last_verified=datetime.now(),
                )
                symbols[symbol_id] = symbol

        return symbols

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
        }
        return mapping.get(ext, "unknown")
