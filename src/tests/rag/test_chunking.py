"""Tests for chunking strategies."""

import pytest
from datetime import datetime

from ...models.document_models import Document, DocumentType, Chunk
from ...rag.chunking.semantic_chunker import SemanticChunker
from ...rag.chunking.code_chunker import CodeChunker


@pytest.mark.asyncio
async def test_semantic_chunker_simple():
    """Test semantic chunker with simple text."""
    chunker = SemanticChunker()

    document = Document(
        id="test-doc-1",
        content="This is sentence one. This is sentence two. This is sentence three. "
                "This is sentence four. This is sentence five.",
        type=DocumentType.TEXT,
        source="test.txt",
    )

    chunks = await chunker.chunk_document(document, target_chunk_size=20, overlap=5)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.document_id == document.id for c in chunks)


@pytest.mark.asyncio
async def test_code_chunker_python():
    """Test code chunker with Python code."""
    chunker = CodeChunker()

    python_code = '''
def function_one():
    """First function."""
    return 1

def function_two():
    """Second function."""
    return 2

class MyClass:
    """A test class."""
    def method_one(self):
        return "method"
'''

    document = Document(
        id="test-code-1",
        content=python_code,
        type=DocumentType.CODE,
        source="test.py",
        metadata={"language": "python"},
    )

    chunks = await chunker.chunk_document(document, target_chunk_size=100)

    assert len(chunks) > 0
    # Should preserve function/class boundaries
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.chunk_type == "code" for c in chunks)


@pytest.mark.asyncio
async def test_chunker_preserves_metadata():
    """Test that chunkers preserve document metadata."""
    chunker = SemanticChunker()

    document = Document(
        id="test-doc-2",
        content="Some content here. More content here.",
        type=DocumentType.TEXT,
        source="test.txt",
        metadata={"author": "test", "version": "1.0"},
    )

    chunks = await chunker.chunk_document(document)

    assert len(chunks) > 0
    for chunk in chunks:
        assert "author" in chunk.metadata
        assert chunk.metadata["author"] == "test"
