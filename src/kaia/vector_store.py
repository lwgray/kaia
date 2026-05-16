"""
Vector database operations using ChromaDB with hybrid search.

This module handles all vector database operations including indexing chunks,
generating embeddings, and performing hybrid search (semantic + keyword BM25).
"""

import math
import os
import re
from collections import Counter
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from kaia.models import BaseChunk


class BM25Index:
    """
    Lightweight BM25 keyword index that lives alongside the vector store.

    Stores tokenized documents and computes BM25 scores at query time.
    This is intentionally simple — no external dependencies, no persistence
    beyond what the VectorStore already provides.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # doc_id -> list of tokens
        self._docs: dict[str, list[str]] = {}
        # token -> set of doc_ids containing it
        self._inverted: dict[str, set[str]] = {}
        self._avg_dl: float = 0.0
        self._N: int = 0

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Simple lowercase alphanumeric tokenizer."""
        return cls._TOKEN_RE.findall(text.lower())

    def add(self, doc_id: str, text: str) -> None:
        """Add a document to the index."""
        tokens = self.tokenize(text)
        self._docs[doc_id] = tokens
        for token in set(tokens):
            self._inverted.setdefault(token, set()).add(doc_id)
        # Recompute average document length
        self._N = len(self._docs)
        total = sum(len(t) for t in self._docs.values())
        self._avg_dl = total / self._N if self._N else 0.0

    def score(self, query: str, doc_ids: list[str] | None = None) -> dict[str, float]:
        """
        Compute BM25 scores for the given query.

        Parameters
        ----------
        query : str
            Search query
        doc_ids : list[str] | None
            If given, only score these documents (for re-ranking)

        Returns
        -------
        dict[str, float]
            doc_id -> BM25 score
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return {}

        candidates = set(doc_ids) if doc_ids else None
        scores: dict[str, float] = {}

        for token in query_tokens:
            posting = self._inverted.get(token, set())
            if not posting:
                continue
            df = len(posting)
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id in posting:
                if candidates is not None and doc_id not in candidates:
                    continue
                doc_tokens = self._docs[doc_id]
                tf = doc_tokens.count(token)
                dl = len(doc_tokens)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl) if self._avg_dl else tf + self.k1
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        return scores

    def clear(self) -> None:
        """Clear the index."""
        self._docs.clear()
        self._inverted.clear()
        self._avg_dl = 0.0
        self._N = 0


class VectorStore:
    """
    Manages ChromaDB vector database for Marcus codebase with hybrid search.

    This class handles:
    - Creating and managing ChromaDB collections
    - Generating embeddings using SentenceTransformer
    - Adding chunks to the vector store
    - Querying with hybrid search (semantic + BM25 keyword scoring)
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "marcus_codebase",
    ):
        """
        Initialize vector store.

        Parameters
        ----------
        persist_directory : str | None
            Directory to persist ChromaDB data. If None, uses in-memory storage.
        collection_name : str
            Name of the collection to use
        """
        if persist_directory is None:
            persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize local embedding model (no API key needed!)
        # Using all-MiniLM-L6-v2: fast, good quality, runs locally
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # BM25 keyword index for hybrid search
        self.bm25 = BM25Index()
        self._bm25_loaded = False

    def add_chunks(self, chunks: list[BaseChunk]) -> None:
        """
        Add chunks to vector store with embeddings.

        Batches chunks to avoid exceeding ChromaDB's max batch size.

        Parameters
        ----------
        chunks : list[BaseChunk]
            Chunks to index
        """
        if not chunks:
            return

        # ChromaDB has a max batch size - batch to avoid errors
        BATCH_SIZE = 1000

        # Process in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]

            # Generate embeddings for batch
            texts = [chunk.content for chunk in batch]
            embeddings = self._generate_embeddings(texts)

            # Convert to ChromaDB format
            ids = []
            documents = []
            metadatas = []

            for chunk, embedding in zip(batch, embeddings):
                doc = chunk.to_chromadb_document()
                ids.append(doc["id"])
                documents.append(doc["document"])
                metadatas.append(doc["metadata"])

                # Also index in BM25
                self.bm25.add(doc["id"], doc["document"])

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        self._bm25_loaded = True

    def _ensure_bm25_loaded(self) -> None:
        """Lazily load BM25 index from ChromaDB if not already populated."""
        if self._bm25_loaded:
            return

        count = self.collection.count()
        if count == 0:
            self._bm25_loaded = True
            return

        # Load all documents from ChromaDB into BM25
        BATCH = 1000
        for offset in range(0, count, BATCH):
            results = self.collection.get(
                limit=BATCH,
                offset=offset,
                include=["documents"],
            )
            for doc_id, doc_text in zip(results["ids"], results["documents"]):
                if doc_text:
                    self.bm25.add(doc_id, doc_text)

        self._bm25_loaded = True

    def query(
        self,
        query: str,
        n_results: int = 10,
        filters: dict[str, Any] | None = None,
        semantic_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Query vector store with hybrid search (semantic + keyword).

        Retrieves 2x candidates via semantic search, re-ranks with a
        weighted combination of cosine similarity and BM25 keyword score.

        Parameters
        ----------
        query : str
            Search query
        n_results : int
            Number of results to return
        filters : dict[str, Any] | None
            Metadata filters (e.g., {"file_path": "src/core/coordinator.py"})
        semantic_weight : float
            Weight for semantic score (0-1). Keyword weight = 1 - semantic_weight.

        Returns
        -------
        list[dict[str, Any]]
            Retrieved chunks with metadata and relevance scores
        """
        # Fetch more candidates for re-ranking
        n_candidates = min(n_results * 3, max(n_results, 30))

        query_embedding = self._generate_embeddings([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            where=filters,
        )

        formatted = self._format_results(results)

        if not formatted:
            return []

        # Re-rank with BM25
        self._ensure_bm25_loaded()
        doc_ids = [r["id"] for r in formatted]
        bm25_scores = self.bm25.score(query, doc_ids)

        if bm25_scores:
            # Normalize BM25 scores to 0-1 range
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
            if max_bm25 > 0:
                norm_bm25 = {k: v / max_bm25 for k, v in bm25_scores.items()}
            else:
                norm_bm25 = bm25_scores

            keyword_weight = 1.0 - semantic_weight

            for result in formatted:
                semantic_score = 1.0 - result["distance"]  # cosine distance -> similarity
                keyword_score = norm_bm25.get(result["id"], 0.0)
                result["hybrid_score"] = (
                    semantic_weight * semantic_score
                    + keyword_weight * keyword_score
                )

            formatted.sort(key=lambda r: r["hybrid_score"], reverse=True)
        else:
            # No BM25 matches — use pure semantic ranking (already sorted by distance)
            for result in formatted:
                result["hybrid_score"] = 1.0 - result["distance"]

        return formatted[:n_results]

    def delete_collection(self) -> None:
        """Delete the current collection."""
        self.client.delete_collection(name=self.collection.name)
        self.bm25.clear()
        self._bm25_loaded = False

    def delete_by_repository(self, repository: str) -> int:
        """
        Delete all chunks tagged with the given repository.

        Parameters
        ----------
        repository : str
            Repository name (e.g., "marcus", "cato", "papers")

        Returns
        -------
        int
            Number of chunks deleted
        """
        existing = self.collection.get(where={"repository": repository})
        ids = existing.get("ids", [])
        if not ids:
            return 0
        self.collection.delete(where={"repository": repository})
        # Force BM25 reload on next query
        self.bm25.clear()
        self._bm25_loaded = False
        return len(ids)

    def count(self) -> int:
        """
        Get count of documents in the collection.

        Returns
        -------
        int
            Number of documents
        """
        return self.collection.count()

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using local SentenceTransformer model.

        Parameters
        ----------
        texts : list[str]
            Texts to embed

        Returns
        -------
        list[list[float]]
            List of embedding vectors (384 dimensions)
        """
        # Generate embeddings locally - no API calls!
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @staticmethod
    def _format_results(raw_results: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Format ChromaDB results.

        Parameters
        ----------
        raw_results : dict[str, Any]
            Raw results from ChromaDB

        Returns
        -------
        list[dict[str, Any]]
            Formatted results with id, content, metadata, distance
        """
        formatted = []
        for i, doc_id in enumerate(raw_results["ids"][0]):
            formatted.append(
                {
                    "id": doc_id,
                    "content": raw_results["documents"][0][i],
                    "metadata": raw_results["metadatas"][0][i],
                    "distance": raw_results["distances"][0][i],
                }
            )
        return formatted
