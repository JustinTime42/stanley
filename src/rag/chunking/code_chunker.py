"""Code-aware chunker that preserves syntactic boundaries."""

import logging
from typing import List, Optional
from dataclasses import dataclass

from ...models.document_models import Document, Chunk
from ...analysis.ast_parser import ASTParser
from ...models.analysis_models import Language, ASTNode
from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)


@dataclass
class CodeEntity:
    """Represents a code entity (function, class, etc.)."""

    type: str  # "function", "class", "method", etc.
    name: str
    content: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int


class CodeChunker(BaseChunker):
    """
    Chunk code preserving AST boundaries.

    PATTERN: Use AST to identify natural boundaries
    CRITICAL: Never split in middle of functions/classes
    """

    def __init__(self, ast_parser: Optional[ASTParser] = None):
        """
        Initialize code chunker.

        Args:
            ast_parser: Optional AST parser (creates if None)
        """
        self.ast_parser = ast_parser or ASTParser()
        self.logger = logger

    async def chunk_document(
        self,
        document: Document,
        target_chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[Chunk]:
        """
        Chunk code preserving function/class boundaries.

        PATTERN: Extract code entities from AST, chunk appropriately
        GOTCHA: Large entities may exceed chunk size

        Args:
            document: Code document to chunk
            target_chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks (minimal for code)

        Returns:
            List of code chunks with preserved syntax
        """
        # Get language from metadata
        language_str = document.metadata.get("language", "unknown")

        # Map string to Language enum
        try:
            language = Language(language_str.lower())
        except ValueError:
            language = Language.UNKNOWN

        if language == Language.UNKNOWN:
            self.logger.warning(
                f"Unknown language for document {document.id}, using simple chunking"
            )
            return await self._simple_chunk(document, target_chunk_size)

        # Parse AST
        try:
            ast = await self.ast_parser.parse_code(
                document.content.encode("utf-8"),
                language,
                document.source,
            )

            if not ast:
                self.logger.warning(
                    f"Failed to parse AST for {document.id}, using simple chunking"
                )
                return await self._simple_chunk(document, target_chunk_size)

        except Exception as e:
            self.logger.error(f"Error parsing AST: {e}, using simple chunking")
            return await self._simple_chunk(document, target_chunk_size)

        # Extract code entities
        entities = self._extract_entities(ast, document.content.encode("utf-8"))

        if not entities:
            self.logger.warning(
                f"No entities found in {document.id}, using simple chunking"
            )
            return await self._simple_chunk(document, target_chunk_size)

        # Create chunks from entities
        chunks = []
        current_group = []
        current_tokens = 0

        for entity in entities:
            entity_tokens = self._estimate_tokens(entity.content)

            # If entity alone exceeds chunk size
            if entity_tokens > target_chunk_size:
                # Create chunk from current group if any
                if current_group:
                    chunk = self._create_chunk_from_entities(
                        document, current_group, len(chunks)
                    )
                    chunks.append(chunk)
                    current_group = []
                    current_tokens = 0

                # Split large entity into sub-chunks
                sub_chunks = self._split_large_entity(
                    document, entity, target_chunk_size, len(chunks)
                )
                chunks.extend(sub_chunks)

            # If adding entity would exceed chunk size
            elif current_tokens + entity_tokens > target_chunk_size and current_group:
                # Create chunk from current group
                chunk = self._create_chunk_from_entities(
                    document, current_group, len(chunks)
                )
                chunks.append(chunk)
                current_group = [entity]
                current_tokens = entity_tokens

            # Add entity to current group
            else:
                current_group.append(entity)
                current_tokens += entity_tokens

        # Create final chunk if any entities remain
        if current_group:
            chunk = self._create_chunk_from_entities(
                document, current_group, len(chunks)
            )
            chunks.append(chunk)

        self.logger.info(
            f"Code-aware chunked document {document.id} into {len(chunks)} chunks"
        )
        return chunks

    def _extract_entities(
        self,
        ast: ASTNode,
        source_code: bytes,
    ) -> List[CodeEntity]:
        """
        Extract code entities (functions, classes) from AST.

        Args:
            ast: Parsed AST
            source_code: Original source code

        Returns:
            List of code entities
        """
        entities = []

        # Define node types to extract for each language
        entity_types = [
            "function_definition",  # Python
            "function_declaration",  # JavaScript, TypeScript, Go
            "method_definition",  # JavaScript, TypeScript
            "method_declaration",  # Java
            "class_definition",  # Python
            "class_declaration",  # JavaScript, TypeScript, Java
            "type_declaration",  # Go (structs)
        ]

        # Find all entity nodes
        nodes = self.ast_parser.find_nodes_by_type(ast, entity_types)

        for node in nodes:
            # Extract entity name
            name = self._extract_entity_name(node)

            # Extract content
            try:
                content = source_code[node.start_byte : node.end_byte].decode("utf-8")
            except Exception as e:
                self.logger.warning(f"Failed to extract entity content: {e}")
                continue

            # Determine entity type
            entity_type = self._classify_entity_type(node.node_type)

            entity = CodeEntity(
                type=entity_type,
                name=name or "<anonymous>",
                content=content,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                start_byte=node.start_byte,
                end_byte=node.end_byte,
            )
            entities.append(entity)

        return entities

    def _extract_entity_name(self, node: ASTNode) -> str:
        """Extract name from entity node."""
        # Look for identifier child
        for child in node.children:
            if child.node_type == "identifier":
                return child.text.strip()

        return "<anonymous>"

    def _classify_entity_type(self, node_type: str) -> str:
        """Classify AST node type to entity type."""
        if "function" in node_type:
            return "function"
        elif "method" in node_type:
            return "method"
        elif "class" in node_type:
            return "class"
        elif "type" in node_type:
            return "type"
        else:
            return "code"

    def _create_chunk_from_entities(
        self,
        document: Document,
        entities: List[CodeEntity],
        chunk_index: int,
    ) -> Chunk:
        """Create chunk from list of entities."""
        # Combine entity contents
        content = "\n\n".join(entity.content for entity in entities)

        # Extract metadata
        entity_names = [entity.name for entity in entities]
        function_names = [e.name for e in entities if e.type in ("function", "method")]
        class_names = [e.name for e in entities if e.type == "class"]

        chunk = Chunk(
            id=f"{document.id}_chunk_{chunk_index}",
            document_id=document.id,
            content=content,
            start_index=entities[0].start_byte,
            end_index=entities[-1].end_byte,
            chunk_index=chunk_index,
            chunk_type="code",
            language=document.metadata.get("language"),
            function_name=function_names[0] if function_names else None,
            class_name=class_names[0] if class_names else None,
            token_count=self._estimate_tokens(content),
            metadata={
                **document.metadata,
                "entity_names": entity_names,
                "entity_count": len(entities),
            },
        )

        return chunk

    def _split_large_entity(
        self,
        document: Document,
        entity: CodeEntity,
        target_size: int,
        start_index: int,
    ) -> List[Chunk]:
        """Split large entity into smaller chunks."""
        # For very large entities, split by lines
        lines = entity.content.splitlines(keepends=True)
        chunks = []
        current_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = self._estimate_tokens(line)

            if current_tokens + line_tokens > target_size and current_lines:
                # Create chunk
                content = "".join(current_lines)
                chunk = Chunk(
                    id=f"{document.id}_chunk_{start_index + len(chunks)}",
                    document_id=document.id,
                    content=content,
                    start_index=entity.start_byte,
                    end_index=entity.start_byte + len(content),
                    chunk_index=start_index + len(chunks),
                    chunk_type="code",
                    language=document.metadata.get("language"),
                    function_name=entity.name
                    if entity.type in ("function", "method")
                    else None,
                    class_name=entity.name if entity.type == "class" else None,
                    token_count=current_tokens,
                    metadata={
                        **document.metadata,
                        "is_partial": True,
                        "entity_name": entity.name,
                    },
                )
                chunks.append(chunk)

                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        # Add remaining lines
        if current_lines:
            content = "".join(current_lines)
            chunk = Chunk(
                id=f"{document.id}_chunk_{start_index + len(chunks)}",
                document_id=document.id,
                content=content,
                start_index=entity.start_byte,
                end_index=entity.end_byte,
                chunk_index=start_index + len(chunks),
                chunk_type="code",
                language=document.metadata.get("language"),
                function_name=entity.name
                if entity.type in ("function", "method")
                else None,
                class_name=entity.name if entity.type == "class" else None,
                token_count=current_tokens,
                metadata={
                    **document.metadata,
                    "is_partial": True,
                    "entity_name": entity.name,
                },
            )
            chunks.append(chunk)

        return chunks

    async def _simple_chunk(
        self,
        document: Document,
        target_chunk_size: int,
    ) -> List[Chunk]:
        """Fallback to simple line-based chunking."""
        lines = document.content.splitlines(keepends=True)
        chunks = []
        current_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = self._estimate_tokens(line)

            if current_tokens + line_tokens > target_chunk_size and current_lines:
                content = "".join(current_lines)
                chunk = Chunk(
                    id=f"{document.id}_chunk_{len(chunks)}",
                    document_id=document.id,
                    content=content,
                    start_index=0,
                    end_index=len(content),
                    chunk_index=len(chunks),
                    token_count=current_tokens,
                    metadata=document.metadata.copy(),
                )
                chunks.append(chunk)
                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        # Add remaining
        if current_lines:
            content = "".join(current_lines)
            chunk = Chunk(
                id=f"{document.id}_chunk_{len(chunks)}",
                document_id=document.id,
                content=content,
                start_index=0,
                end_index=len(content),
                chunk_index=len(chunks),
                token_count=current_tokens,
                metadata=document.metadata.copy(),
            )
            chunks.append(chunk)

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token)."""
        return max(1, len(text) // 4)
