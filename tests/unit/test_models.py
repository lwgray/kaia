"""
Unit tests for Kaia data models.
"""

from datetime import datetime

import pytest

from kaia.models import (
    BaseChunk,
    ChunkMetadata,
    ChunkType,
    CodeChunk,
    CommitChunk,
    DocChunk,
)


class TestChunkMetadata:
    """Test suite for ChunkMetadata model"""

    def test_create_metadata_with_defaults(self) -> None:
        """Test creating metadata with default values"""
        metadata = ChunkMetadata()

        assert metadata.file_path is None
        assert isinstance(metadata.last_updated, datetime)
        assert metadata.importance == 0.5
        assert metadata.section_hierarchy == []
        assert metadata.keywords == []

    def test_create_metadata_with_code_fields(self) -> None:
        """Test creating metadata with code-specific fields"""
        metadata = ChunkMetadata(
            file_path="src/core/coordinator.py",
            class_name="TaskCoordinator",
            function_name="request_next_task",
            line_start=145,
            line_end=167,
            importance=0.8,
        )

        assert metadata.file_path == "src/core/coordinator.py"
        assert metadata.class_name == "TaskCoordinator"
        assert metadata.function_name == "request_next_task"
        assert metadata.line_start == 145
        assert metadata.line_end == 167
        assert metadata.importance == 0.8

    def test_importance_validation(self) -> None:
        """Test that importance is validated to be between 0 and 1"""
        # Valid values
        ChunkMetadata(importance=0.0)
        ChunkMetadata(importance=1.0)
        ChunkMetadata(importance=0.5)

        # Invalid values should raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            ChunkMetadata(importance=-0.1)

        with pytest.raises(Exception):
            ChunkMetadata(importance=1.1)


class TestCodeChunk:
    """Test suite for CodeChunk model"""

    def test_create_code_chunk_minimal(self) -> None:
        """Test creating a minimal code chunk"""
        chunk = CodeChunk(content="def foo(): pass")

        assert chunk.content == "def foo(): pass"
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert isinstance(chunk.id, str)
        assert len(chunk.id) > 0
        assert chunk.embedding is None

    def test_create_code_chunk_with_metadata(self) -> None:
        """Test creating code chunk with full metadata"""
        metadata = ChunkMetadata(
            file_path="src/core/test.py",
            function_name="foo",
            line_start=10,
            line_end=12,
        )
        chunk = CodeChunk(content="def foo(): pass", metadata=metadata)

        assert chunk.metadata.file_path == "src/core/test.py"
        assert chunk.metadata.function_name == "foo"
        assert chunk.metadata.line_start == 10

    def test_token_counting(self) -> None:
        """Test automatic token counting on chunk creation"""
        chunk = CodeChunk(content="def foo(): pass")

        assert chunk.token_count > 0
        assert chunk.token_count < 10  # Short function should have few tokens

    def test_token_counting_longer_content(self) -> None:
        """Test token counting with longer content"""
        long_content = """
def calculate_fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
        chunk = CodeChunk(content=long_content)

        assert chunk.token_count > 20  # Longer function has more tokens

    def test_to_chromadb_document(self) -> None:
        """Test conversion to ChromaDB document format"""
        metadata = ChunkMetadata(
            file_path="test.py", function_name="foo", importance=0.7
        )
        chunk = CodeChunk(content="def foo(): pass", metadata=metadata)

        doc = chunk.to_chromadb_document()

        assert "id" in doc
        assert doc["id"] == chunk.id
        assert "document" in doc
        assert doc["document"] == "def foo(): pass"
        assert "metadata" in doc
        assert doc["metadata"]["chunk_type"] == "function"
        assert doc["metadata"]["file_path"] == "test.py"
        assert doc["metadata"]["function_name"] == "foo"
        assert doc["metadata"]["importance"] == 0.7
        assert doc["metadata"]["token_count"] == chunk.token_count

    def test_chromadb_document_excludes_none_values(self) -> None:
        """Test that None values are excluded from metadata in ChromaDB format"""
        chunk = CodeChunk(content="def foo(): pass")
        doc = chunk.to_chromadb_document()

        # None values should not be in metadata
        assert "file_path" not in doc["metadata"]
        assert "class_name" not in doc["metadata"]

    def test_chromadb_document_datetime_serialization(self) -> None:
        """Test that datetime is serialized to ISO string"""
        chunk = CodeChunk(content="def foo(): pass")
        doc = chunk.to_chromadb_document()

        # last_updated should be ISO string
        assert "last_updated" in doc["metadata"]
        assert isinstance(doc["metadata"]["last_updated"], str)
        # Should be parseable as ISO format
        datetime.fromisoformat(doc["metadata"]["last_updated"])


class TestDocChunk:
    """Test suite for DocChunk model"""

    def test_create_doc_chunk(self) -> None:
        """Test creating a documentation chunk"""
        chunk = DocChunk(
            content="# Architecture\n\nMarcus uses board-mediated coordination.",
            chunk_type=ChunkType.DOC_SECTION,
        )

        assert chunk.chunk_type == ChunkType.DOC_SECTION
        assert "Architecture" in chunk.content

    def test_doc_chunk_with_hierarchy(self) -> None:
        """Test doc chunk with section hierarchy metadata"""
        metadata = ChunkMetadata(
            section_hierarchy=["ERROR_HANDLING", "RETRY_PATTERNS"],
            keywords=["error", "retry", "resilience"],
        )
        chunk = DocChunk(content="Use retry decorators...", metadata=metadata)

        assert chunk.metadata.section_hierarchy == ["ERROR_HANDLING", "RETRY_PATTERNS"]
        assert "retry" in chunk.metadata.keywords


class TestCommitChunk:
    """Test suite for CommitChunk model"""

    def test_create_commit_chunk(self) -> None:
        """Test creating a git commit chunk"""
        metadata = ChunkMetadata(
            commit_hash="960e518",
            author="lwgray",
            file_path="src/integrations/planka.py",
        )
        chunk = CommitChunk(
            content="Fix: Parse acceptance criteria from Planka checklists",
            metadata=metadata,
        )

        assert chunk.chunk_type == ChunkType.GIT_COMMIT
        assert chunk.metadata.commit_hash == "960e518"
        assert chunk.metadata.author == "lwgray"


class TestChunkType:
    """Test suite for ChunkType enum"""

    def test_chunk_types_exist(self) -> None:
        """Test that all expected chunk types exist"""
        assert ChunkType.FUNCTION == "function"
        assert ChunkType.CLASS == "class"
        assert ChunkType.MODULE == "module"
        assert ChunkType.DOC_SECTION == "doc_section"
        assert ChunkType.GIT_COMMIT == "git_commit"

    def test_chunk_type_as_string(self) -> None:
        """Test that chunk types can be used as strings"""
        chunk_type = ChunkType.FUNCTION
        # Enum values can be accessed directly
        assert chunk_type.value == "function"
        assert chunk_type == "function"  # Direct comparison works
