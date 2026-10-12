
"""
FAISS vector store for the RAG pipeline.

Stores document embeddings and their associated chunks,
and provides persistent similarity search functionality.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

import faiss
import numpy as np

from config import settings
from src.rag.chunker import DocumentChunk
from src.rag.embeddings import get_embedding_service
from src.utils.logger import get_logger


logger = get_logger(__name__)


class FAISSVectorStore:
    """Persistent FAISS vector store for document chunks."""

    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "metadata.json"

    def __init__(self, store_path: Path | str | None = None) -> None:
        """Initialize the vector store."""

        configured_path = (
            store_path
            if store_path is not None
            else settings.VECTOR_STORE_PATH
        )

        self.store_path = Path(configured_path).expanduser().resolve()
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.index: faiss.Index | None = None
        self.chunks: list[DocumentChunk] = []

        self.embedding_service = get_embedding_service()
        self._lock = RLock()

    # -----------------------------------------------------
    # Validation Helpers
    # -----------------------------------------------------

    @staticmethod
    def _validate_chunks(chunks: list[DocumentChunk]) -> None:
        """Validate a collection of document chunks."""

        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list.")

        if not chunks:
            raise ValueError("Cannot process an empty chunk list.")

        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                raise TypeError(
                    f"Item at position {position} must be a DocumentChunk."
                )

            if (
                not isinstance(chunk.content, str)
                or not chunk.content.strip()
            ):
                raise ValueError(
                    f"Chunk at position {position} has empty content."
                )

            if not isinstance(chunk.metadata, dict):
                raise TypeError(
                    f"Chunk at position {position} has invalid metadata."
                )

    def _prepare_vectors(
        self,
        embeddings: list[list[float]],
        expected_count: int,
    ) -> np.ndarray:
        """Validate embeddings and convert them to float32 vectors."""

        if expected_count <= 0:
            raise ValueError("expected_count must be greater than zero.")

        if not isinstance(embeddings, (list, tuple)) or not embeddings:
            raise ValueError("No embeddings were generated.")

        try:
            vectors = np.asarray(embeddings, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Embeddings cannot be converted to float32 vectors."
            ) from exc

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a two-dimensional array."
            )

        if vectors.shape[0] != expected_count:
            raise ValueError(
                "Embedding count does not match the number of chunks."
            )

        expected_dimension = self.embedding_service.get_dimension()

        if (
            isinstance(expected_dimension, bool)
            or not isinstance(expected_dimension, int)
            or expected_dimension <= 0
        ):
            raise ValueError(
                "Embedding service returned an invalid dimension."
            )

        if vectors.shape[1] != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected "
                f"{expected_dimension}, received {vectors.shape[1]}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain non-finite values.")

        return np.ascontiguousarray(vectors, dtype=np.float32)

    @staticmethod
    def _copy_chunks(
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Create independent chunk objects with copied metadata."""

        return [
            DocumentChunk(
                content=chunk.content,
                metadata=dict(chunk.metadata),
            )
            for chunk in chunks
        ]

    # -----------------------------------------------------
    # Build Index
    # -----------------------------------------------------

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Build a new in-memory FAISS index."""

        self._validate_chunks(chunks)

        logger.info("Building FAISS index from %d chunks.", len(chunks))

        embeddings = self.embedding_service.embed_chunks(chunks)
        vectors = self._prepare_vectors(embeddings, len(chunks))

        # IndexFlatIP ranks vectors by inner product.
        # Embedding normalization must be handled consistently by
        # the embedding service for scores to represent cosine similarity.
        new_index = faiss.IndexFlatIP(vectors.shape[1])
        new_index.add(vectors)

        new_chunks = self._copy_chunks(chunks)

        # Replace the active state only after building succeeds.
        with self._lock:
            self.index = new_index
            self.chunks = new_chunks

        logger.info(
            "FAISS index created with %d vectors.",
            new_index.ntotal,
        )

    # -----------------------------------------------------
    # Add Chunks
    # -----------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to the existing index, creating one if needed."""

        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list.")

        if not chunks:
            return

        self._validate_chunks(chunks)

        embeddings = self.embedding_service.embed_chunks(chunks)
        vectors = self._prepare_vectors(embeddings, len(chunks))
        new_chunks = self._copy_chunks(chunks)

        with self._lock:
            if self.index is None:
                updated_index = faiss.IndexFlatIP(vectors.shape[1])
                updated_index.add(vectors)

                self.index = updated_index
                self.chunks = new_chunks

            else:
                if vectors.shape[1] != self.index.d:
                    raise ValueError(
                        "Embedding dimension does not match "
                        "the existing FAISS index."
                    )

                if self.index.ntotal != len(self.chunks):
                    raise RuntimeError(
                        "FAISS index and chunk metadata are inconsistent."
                    )

                # Update a clone so readers never see a partially
                # modified active index.
                updated_index = faiss.clone_index(self.index)
                updated_index.add(vectors)

                self.index = updated_index
                self.chunks = self.chunks + new_chunks

        logger.info("Added %d chunks to FAISS index.", len(chunks))

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return relevant chunks and their inner-product scores."""

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        normalized_query = query.strip()

        if not normalized_query:
            return []

        requested_k = settings.TOP_K if top_k is None else top_k

        if isinstance(requested_k, bool) or not isinstance(
            requested_k, int
        ):
            raise TypeError("top_k must be an integer.")

        if requested_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Capture a consistent snapshot. Updates replace the active
        # index instead of mutating an index already captured by readers.
        with self._lock:
            index = self.index
            chunks = tuple(self.chunks)

        if index is None:
            raise RuntimeError(
                "Vector store has not been built or loaded."
            )

        if index.ntotal != len(chunks):
            raise RuntimeError(
                "FAISS index and chunk metadata are inconsistent."
            )

        if not chunks:
            return []

        query_embedding = self.embedding_service.embed_query(
            normalized_query
        )

        query_vector = self._prepare_vectors(
            [query_embedding],
            expected_count=1,
        )

        if query_vector.shape[1] != index.d:
            raise ValueError(
                "Query embedding dimension does not match the FAISS index."
            )

        actual_k = min(requested_k, index.ntotal)
        scores, indices = index.search(query_vector, actual_k)

        results: list[tuple[DocumentChunk, float]] = []

        for score, chunk_index in zip(scores[0], indices[0]):
            if chunk_index < 0 or chunk_index >= len(chunks):
                continue

            if not np.isfinite(score):
                continue

            results.append(
                (chunks[int(chunk_index)], float(score))
            )

        return results

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    def save(self) -> None:
        """Persist the index and metadata to disk."""

        index_path = self.store_path / self.INDEX_FILENAME
        metadata_path = self.store_path / self.METADATA_FILENAME

        # Unique temporary filenames prevent concurrent saves from
        # overwriting one another's temporary files.
        suffix = uuid.uuid4().hex
        temporary_index = self.store_path / f"index.{suffix}.tmp"
        temporary_metadata = self.store_path / f"metadata.{suffix}.tmp"

        try:
            with self._lock:
                if self.index is None:
                    raise RuntimeError(
                        "Cannot save an empty vector store."
                    )

                if self.index.ntotal != len(self.chunks):
                    raise RuntimeError(
                        "Cannot save: index and chunk counts do not match."
                    )

                # Snapshot the matching index and chunk metadata.
                index_snapshot = faiss.clone_index(self.index)
                metadata = [
                    {
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                    }
                    for chunk in self.chunks
                ]

            faiss.write_index(index_snapshot, str(temporary_index))

            with temporary_metadata.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    metadata,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                file.flush()
                os.fsync(file.fileno())

            # Replace each destination only after both temporary files
            # have been written successfully.
            temporary_index.replace(index_path)
            temporary_metadata.replace(metadata_path)

            logger.info(
                "Vector store saved successfully to %s.",
                self.store_path,
            )

        except Exception as exc:
            logger.exception("Failed to save vector store.")

            for temporary_file in (
                temporary_index,
                temporary_metadata,
            ):
                try:
                    temporary_file.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Could not remove temporary file: %s",
                        temporary_file,
                    )

            raise RuntimeError(
                "Unable to save vector store."
            ) from exc

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    def load(self) -> None:
        """Load a persisted FAISS index and its chunk metadata."""

        index_path = self.store_path / self.INDEX_FILENAME
        metadata_path = self.store_path / self.METADATA_FILENAME

        if not index_path.is_file():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Vector metadata not found: {metadata_path}"
            )

        try:
            loaded_index = faiss.read_index(str(index_path))

            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                metadata: Any = json.load(file)

            if not isinstance(metadata, list):
                raise ValueError(
                    "Vector metadata must be a JSON list."
                )

            loaded_chunks: list[DocumentChunk] = []

            for position, item in enumerate(metadata):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Invalid metadata entry at position {position}."
                    )

                content = item.get("content")
                chunk_metadata = item.get("metadata")

                if (
                    not isinstance(content, str)
                    or not content.strip()
                ):
                    raise ValueError(
                        f"Invalid chunk content at position {position}."
                    )

                if not isinstance(chunk_metadata, dict):
                    raise ValueError(
                        f"Invalid chunk metadata at position {position}."
                    )

                loaded_chunks.append(
                    DocumentChunk(
                        content=content,
                        metadata=chunk_metadata,
                    )
                )

            if loaded_index.ntotal != len(loaded_chunks):
                raise ValueError(
                    "FAISS index and metadata count do not match."
                )

            if not isinstance(loaded_index, faiss.IndexFlatIP):
                raise ValueError(
                    "Unsupported FAISS index type. "
                    "Expected an IndexFlatIP index."
                )

            expected_dimension = self.embedding_service.get_dimension()

            if loaded_index.d != expected_dimension:
                raise ValueError(
                    "Saved index dimension does not match the "
                    "configured embedding model. Rebuild the vector store."
                )

            with self._lock:
                self.index = loaded_index
                self.chunks = loaded_chunks

            logger.info(
                "Vector store loaded successfully. Vectors: %d",
                loaded_index.ntotal,
            )

        except Exception as exc:
            logger.exception("Failed to load vector store.")
            raise RuntimeError(
                "Unable to load vector store."
            ) from exc

    # -----------------------------------------------------
    # Exists
    # -----------------------------------------------------

    def exists(self) -> bool:
        """Check whether both persisted store files exist."""

        return (
            (self.store_path / self.INDEX_FILENAME).is_file()
            and (self.store_path / self.METADATA_FILENAME).is_file()
        )

    # -----------------------------------------------------
    # Clear
    # -----------------------------------------------------

    def clear(self) -> None:
        """Clear the in-memory index and chunks."""

        with self._lock:
            self.index = None
            self.chunks = []

        logger.info("Vector store cleared from memory.")

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return basic vector store statistics."""

        with self._lock:
            index = self.index
            chunk_count = len(self.chunks)

            return {
                "vectors": index.ntotal if index is not None else 0,
                "chunks": chunk_count,
                "dimension": index.d if index is not None else 0,
                "path": str(self.store_path),
            }


# =========================================================
# Convenience Functions
# =========================================================

def create_vector_store(
    chunks: list[DocumentChunk],
) -> FAISSVectorStore:
    """Build and save a new vector store."""

    vector_store = FAISSVectorStore()
    vector_store.build(chunks)
    vector_store.save()

    return vector_store


def load_vector_store() -> FAISSVectorStore:
    """Load the configured persistent vector store."""

    vector_store = FAISSVectorStore()
    vector_store.load()

    return vector_store
