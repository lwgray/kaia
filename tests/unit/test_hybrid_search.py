"""
Unit tests for hybrid search (BM25 + semantic) and content truncation.
"""

from pathlib import Path

import pytest

from kaia.models import ChunkMetadata, ChunkType, CodeChunk, PdfChunk
from kaia.vector_store import BM25Index, VectorStore


class TestBM25Index:
    """Test the BM25 keyword index in isolation."""

    def test_add_and_score(self) -> None:
        idx = BM25Index()
        idx.add("doc1", "multi agent coordination system")
        idx.add("doc2", "single agent reinforcement learning")
        idx.add("doc3", "multi agent reinforcement learning coordination")

        scores = idx.score("multi agent coordination")
        assert "doc1" in scores
        assert "doc3" in scores
        # doc1 and doc3 should score higher than doc2
        assert scores.get("doc1", 0) > scores.get("doc2", 0)

    def test_score_with_doc_ids_filter(self) -> None:
        idx = BM25Index()
        idx.add("doc1", "alpha beta gamma")
        idx.add("doc2", "alpha delta epsilon")

        scores = idx.score("alpha", doc_ids=["doc1"])
        assert "doc1" in scores
        assert "doc2" not in scores

    def test_empty_query(self) -> None:
        idx = BM25Index()
        idx.add("doc1", "some text")
        scores = idx.score("")
        assert scores == {}

    def test_no_match(self) -> None:
        idx = BM25Index()
        idx.add("doc1", "hello world")
        scores = idx.score("zzzzz")
        assert scores == {}

    def test_clear(self) -> None:
        idx = BM25Index()
        idx.add("doc1", "hello world")
        idx.clear()
        scores = idx.score("hello")
        assert scores == {}

    def test_tokenize(self) -> None:
        tokens = BM25Index.tokenize("Hello, World! Test-123.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "123" in tokens


class TestHybridSearch:
    """Test hybrid search in VectorStore."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> VectorStore:
        return VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_hybrid",
        )

    def test_hybrid_query_returns_results(self, store: VectorStore) -> None:
        """Hybrid search should return results with hybrid_score."""
        chunks = [
            PdfChunk(
                content="Multi-agent coordination involves multiple agents working together.",
                metadata=ChunkMetadata(file_path="paper1.pdf"),
            ),
            PdfChunk(
                content="Reinforcement learning trains agents via rewards.",
                metadata=ChunkMetadata(file_path="paper2.pdf"),
            ),
        ]
        store.add_chunks(chunks)

        results = store.query("multi-agent coordination", n_results=2)
        assert len(results) > 0
        assert "hybrid_score" in results[0]

    def test_keyword_match_boosts_ranking(self, store: VectorStore) -> None:
        """A document with exact keyword match should rank higher with hybrid."""
        chunks = [
            PdfChunk(
                content="The BM25 algorithm is a bag-of-words retrieval function.",
                metadata=ChunkMetadata(file_path="a.pdf"),
            ),
            PdfChunk(
                content="Information retrieval uses various ranking functions for search.",
                metadata=ChunkMetadata(file_path="b.pdf"),
            ),
        ]
        store.add_chunks(chunks)

        results = store.query("BM25 algorithm", n_results=2)
        # The document mentioning "BM25" explicitly should rank first
        assert "BM25" in results[0]["content"]

    def test_bm25_lazy_load(self, store: VectorStore) -> None:
        """BM25 index should lazy-load from ChromaDB on first query."""
        chunk = CodeChunk(content="def hello(): pass")
        store.add_chunks([chunk])

        # Reset BM25 to simulate a fresh server start
        store.bm25.clear()
        store._bm25_loaded = False

        results = store.query("hello function", n_results=1)
        assert store._bm25_loaded
        assert len(results) > 0


class TestContentTruncation:
    """Test the MCP server's content truncation."""

    def test_short_content_not_truncated(self) -> None:
        from kaia.mcp_server import KaiaMCPServer

        content = "Short content."
        result = KaiaMCPServer._truncate_content(content, 5000)
        assert result == content
        assert "[... truncated]" not in result

    def test_long_content_truncated(self) -> None:
        from kaia.mcp_server import KaiaMCPServer

        content = "First paragraph.\n\n" + "Word " * 2000 + "\n\nLast paragraph."
        result = KaiaMCPServer._truncate_content(content, 500)
        assert len(result) <= 600  # some tolerance for the truncation marker
        assert "[... truncated]" in result

    def test_truncation_at_paragraph_boundary(self) -> None:
        from kaia.mcp_server import KaiaMCPServer

        content = "Paragraph one with content.\n\n" * 20
        result = KaiaMCPServer._truncate_content(content, 200)
        # Should end at a paragraph boundary, not mid-word
        assert result.endswith("[... truncated]")
        # Content before truncation marker should end cleanly
        before_marker = result.replace("\n\n[... truncated]", "")
        assert not before_marker.endswith(" ")

    def test_truncation_at_sentence_boundary(self) -> None:
        from kaia.mcp_server import KaiaMCPServer

        # One big paragraph with no double-newlines
        content = "Sentence one. " * 100
        result = KaiaMCPServer._truncate_content(content, 200)
        assert "[... truncated]" in result
