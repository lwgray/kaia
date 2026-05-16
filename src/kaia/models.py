"""
Core data models for Kaia RAG system.

This module defines the data structures for representing different types of
content chunks (code, documentation, git commits) and their associated metadata.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import tiktoken
from pydantic import BaseModel, Field, computed_field


class ChunkType(str, Enum):
    """Types of content chunks that can be indexed."""

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    DOC_SECTION = "doc_section"
    DOC_CODE_EXAMPLE = "doc_code_example"
    GIT_COMMIT = "git_commit"
    FILE_EVOLUTION = "file_evolution"
    ARCHITECTURE_DECISION = "architecture_decision"
    TEST_EXAMPLE = "test_example"
    PDF_SECTION = "pdf_section"


class ChunkMetadata(BaseModel):
    """
    Metadata for any chunk type.

    Attributes
    ----------
    repository : str
        Repository name (e.g., "marcus", "cato")
    file_path : str | None
        Path to source file relative to repository root
    last_updated : datetime
        Timestamp of last update
    importance : float
        Importance score (0.0-1.0) for ranking results
    class_name : str | None
        Name of class (for code chunks)
    function_name : str | None
        Name of function/method (for code chunks)
    line_start : int | None
        Starting line number in source file
    line_end : int | None
        Ending line number in source file
    section_hierarchy : list[str]
        Section path for documentation (e.g., ["ERROR_HANDLING", "RETRY_PATTERNS"])
    keywords : list[str]
        Keywords for filtering and search enhancement
    commit_hash : str | None
        Git commit hash (for git chunks)
    author : str | None
        Author name (for git chunks)
    related_chunks : list[str]
        IDs of related chunks for cross-referencing
    """

    # Repository identification
    repository: str = "unknown"

    # Common fields
    file_path: str | None = None
    last_updated: datetime = Field(default_factory=datetime.now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    # Code-specific
    class_name: str | None = None
    function_name: str | None = None
    line_start: int | None = None
    line_end: int | None = None

    # Doc-specific
    section_hierarchy: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    # Git-specific
    commit_hash: str | None = None
    author: str | None = None

    # Cross-references
    related_chunks: list[str] = Field(default_factory=list)


class BaseChunk(BaseModel):
    """
    Base class for all chunk types.

    Attributes
    ----------
    id : str
        Unique identifier for the chunk
    content : str
        The actual content (code, text, etc.)
    chunk_type : ChunkType
        Type of content this chunk represents
    metadata : ChunkMetadata
        Associated metadata
    embedding : list[float] | None
        Vector embedding (computed during indexing)
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    chunk_type: ChunkType
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    embedding: list[float] | None = None

    @computed_field  # type: ignore[misc]
    @property
    def token_count(self) -> int:
        """
        Count tokens in content using tiktoken.

        Returns
        -------
        int
            Number of tokens in content
        """
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(self.content))

    def to_chromadb_document(self) -> dict[str, Any]:
        """
        Convert to ChromaDB document format.

        Returns
        -------
        dict[str, Any]
            Document in ChromaDB format with id, document, metadata, embedding
        """
        # Convert metadata to dict, filtering out None values
        metadata_dict = self.metadata.model_dump(exclude_none=True)

        # Convert datetime to ISO string for ChromaDB compatibility
        if "last_updated" in metadata_dict:
            metadata_dict["last_updated"] = metadata_dict["last_updated"].isoformat()

        # ChromaDB doesn't allow empty lists in metadata - filter them out
        filtered_metadata = {
            k: v
            for k, v in metadata_dict.items()
            if not (isinstance(v, list) and len(v) == 0)
        }

        return {
            "id": self.id,
            "document": self.content,
            "metadata": {
                "chunk_type": self.chunk_type.value,
                "token_count": self.token_count,
                **filtered_metadata,
            },
            "embedding": self.embedding,
        }


class CodeChunk(BaseChunk):
    """Chunk representing Python code (function, class, or module)."""

    chunk_type: ChunkType = ChunkType.FUNCTION


class DocChunk(BaseChunk):
    """Chunk representing documentation section or example."""

    chunk_type: ChunkType = ChunkType.DOC_SECTION


class CommitChunk(BaseChunk):
    """Chunk representing a git commit with its changes."""

    chunk_type: ChunkType = ChunkType.GIT_COMMIT


class PdfChunk(BaseChunk):
    """Chunk representing a section from a PDF research paper."""

    chunk_type: ChunkType = ChunkType.PDF_SECTION
