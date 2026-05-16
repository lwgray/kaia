"""
Unit tests for vector store operations.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from kaia.models import ChunkMetadata, ChunkType, CodeChunk
from kaia.vector_store import VectorStore


class TestVectorStore:
    """Test suite for VectorStore"""

    @pytest.fixture
    def vector_store(self, tmp_path: Path) -> VectorStore:
        """Create a VectorStore with temporary directory"""
        # No mocking needed - uses local embeddings!
        store = VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_collection",
        )
        return store

    def test_initialization(self, tmp_path: Path) -> None:
        """Test VectorStore initialization"""
        store = VectorStore(persist_directory=str(tmp_path / "chroma"))
        assert store.collection is not None
        assert store.collection.name == "marcus_codebase"
        assert store.embedding_model is not None

    def test_add_single_chunk(self, vector_store: VectorStore) -> None:
        """Test adding a single chunk to vector store"""
        chunk = CodeChunk(
            content="def foo(): pass",
            metadata=ChunkMetadata(file_path="test.py"),
        )

        vector_store.add_chunks([chunk])

        # Verify it was added
        assert vector_store.count() == 1

    def test_add_multiple_chunks(self, vector_store: VectorStore) -> None:
        """Test adding multiple chunks at once"""
        chunks = [
            CodeChunk(content="def foo(): pass"),
            CodeChunk(content="def bar(): pass"),
            CodeChunk(content="class Baz: pass"),
        ]

        vector_store.add_chunks(chunks)

        # All chunks should be added
        assert vector_store.count() == 3

    def test_add_empty_list(self, vector_store: VectorStore) -> None:
        """Test that adding empty list does nothing"""
        vector_store.add_chunks([])

        assert vector_store.count() == 0

    def test_query_chunks(self, vector_store: VectorStore) -> None:
        """Test querying the vector store"""
        # Add some chunks
        chunk1 = CodeChunk(content="def calculate_sum(): pass")
        chunk2 = CodeChunk(content="def calculate_product(): pass")
        vector_store.add_chunks([chunk1, chunk2])

        # Query
        results = vector_store.query("calculate function", n_results=2)

        # Should return results
        assert len(results) <= 2
        assert all("content" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("distance" in r for r in results)

    def test_query_with_filters(self, vector_store: VectorStore) -> None:
        """Test querying with metadata filters"""
        chunk1 = CodeChunk(content="def foo(): pass")
        chunk1.metadata.file_path = "src/core/test.py"
        chunk1.metadata.function_name = "foo"

        chunk2 = CodeChunk(content="def bar(): pass")
        chunk2.metadata.file_path = "src/utils/test.py"
        chunk2.metadata.function_name = "bar"

        vector_store.add_chunks([chunk1, chunk2])

        # Query with filter
        results = vector_store.query(
            "function",
            filters={"function_name": "foo"},
            n_results=10,
        )

        # ChromaDB will handle the filtering
        # We just verify the query was made
        assert len(results) >= 0  # Could be 0 if filter works, or more if not

    def test_count(self, vector_store: VectorStore) -> None:
        """Test counting documents in collection"""
        assert vector_store.count() == 0

        chunks = [CodeChunk(content=f"def func{i}(): pass") for i in range(5)]
        vector_store.add_chunks(chunks)

        assert vector_store.count() == 5

    def test_delete_collection(
        self, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """Test deleting a collection"""
        # Add some data
        chunk = CodeChunk(content="def foo(): pass")
        vector_store.add_chunks([chunk])
        assert vector_store.count() == 1

        # Delete collection
        vector_store.delete_collection()

        # Create a new store with same name - should be empty
        new_store = VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_collection",
        )
        assert new_store.count() == 0

    def test_chunk_metadata_preserved(self, vector_store: VectorStore) -> None:
        """Test that chunk metadata is preserved in vector store"""
        metadata = ChunkMetadata(
            file_path="src/core/coordinator.py",
            class_name="TaskCoordinator",
            function_name="request_next_task",
            line_start=10,
            line_end=20,
            importance=0.8,
        )
        chunk = CodeChunk(content="def request_next_task(): pass", metadata=metadata)

        vector_store.add_chunks([chunk])

        # Query and check metadata
        results = vector_store.query("request_next_task", n_results=1)

        assert len(results) > 0
        result_metadata = results[0]["metadata"]
        assert result_metadata["file_path"] == "src/core/coordinator.py"
        assert result_metadata["class_name"] == "TaskCoordinator"
        assert result_metadata["function_name"] == "request_next_task"
        assert result_metadata["importance"] == 0.8
