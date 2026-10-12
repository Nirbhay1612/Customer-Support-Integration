
"""
Embedding utilities for the RAG pipeline.

Generates finite, normalized vector embeddings for document chunks
and user queries. Normalized vectors are suitable for inner-product
similarity search when used with FAISS IndexFlatIP.
"""

from __future__ import annotations

import math
from functools import lru_cache
from threading import Lock
from typing import Any

from sentence_transformers import SentenceTransformer

from config import settings
from src.rag.chunker import DocumentChunk
from src.utils.logger import get_logger


logger = get_logger(__name__)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

EMBEDDING_BATCH_SIZE = 32


# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the configured sentence-transformer model."""
    model_name = getattr(settings, "EMBEDDING_MODEL", None)

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            "EMBEDDING_MODEL must be a non-empty string."
        )

    model_name = model_name.strip()

    try:
        logger.info("Loading embedding model: %s", model_name)

        model = SentenceTransformer(model_name)
        dimension = model.get_sentence_embedding_dimension()

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise RuntimeError(
                "Embedding model returned an invalid dimension."
            )

        logger.info(
            "Embedding model loaded successfully; dimension=%d.",
            dimension,
        )

        return model

    except Exception as exc:
        logger.exception(
            "Failed to load embedding model: %s",
            model_name,
        )
        raise RuntimeError(
            f"Unable to load embedding model: {model_name}"
        ) from exc


# ---------------------------------------------------------
# Embedding Service
# ---------------------------------------------------------


class EmbeddingService:
    """Generate normalized embeddings for text, chunks, and queries."""

    def __init__(self) -> None:
        self.model = get_embedding_model()
        self._encode_lock = Lock()

        dimension = self.model.get_sentence_embedding_dimension()

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise RuntimeError(
                "Could not determine the embedding model dimension."
            )

        self._dimension = dimension

    # -----------------------------------------------------
    # Input Validation
    # -----------------------------------------------------

    @staticmethod
    def _validate_texts(texts: list[str]) -> list[str]:
        """Validate text inputs and strip surrounding whitespace."""
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings.")

        cleaned_texts: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Text at index {index} must be a string."
                )

            normalized_text = text.strip()

            if not normalized_text:
                raise ValueError(
                    f"Text at index {index} cannot be empty."
                )

            cleaned_texts.append(normalized_text)

        return cleaned_texts

    # -----------------------------------------------------
    # Embedding Validation
    # -----------------------------------------------------

    def _validate_embeddings(
        self,
        embeddings: Any,
        expected_count: int,
    ) -> list[list[float]]:
        """Validate embedding shape, dimensions, and numeric values."""
        if getattr(embeddings, "ndim", None) != 2:
            raise RuntimeError(
                "Embedding model returned an invalid array shape."
            )

        if embeddings.shape[0] != expected_count:
            raise RuntimeError(
                "Embedding count does not match input text count."
            )

        if embeddings.shape[1] != self._dimension:
            raise RuntimeError(
                "Embedding dimension does not match the model."
            )

        result: list[list[float]] = []

        for index, embedding in enumerate(embeddings):
            vector: list[float] = []

            for value in embedding:
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        f"Invalid embedding value at index {index}."
                    ) from exc

                if not math.isfinite(number):
                    raise RuntimeError(
                        f"Non-finite embedding value at index {index}."
                    )

                vector.append(number)

            if len(vector) != self._dimension:
                raise RuntimeError(
                    f"Unexpected embedding dimension at index {index}."
                )

            result.append(vector)

        return result

    # -----------------------------------------------------
    # Embed Texts
    # -----------------------------------------------------

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate normalized embeddings for multiple text inputs."""
        cleaned_texts = self._validate_texts(texts)

        if not cleaned_texts:
            return []

        try:
            # Serialize model inference for safer shared-model access.
            with self._encode_lock:
                embeddings = self.model.encode(
                    cleaned_texts,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

            return self._validate_embeddings(
                embeddings,
                expected_count=len(cleaned_texts),
            )

        except (TypeError, ValueError):
            # Preserve input/configuration errors raised by the model.
            raise

        except Exception as exc:
            logger.exception("Failed to generate text embeddings.")
            raise RuntimeError(
                "Unable to generate text embeddings."
            ) from exc

    # -----------------------------------------------------
    # Embed Document Chunks
    # -----------------------------------------------------

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[list[float]]:
        """Generate embeddings for document chunks."""
        if not isinstance(chunks, list):
            raise TypeError(
                "chunks must be a list of DocumentChunk objects."
            )

        if not chunks:
            return []

        texts: list[str] = []

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                raise TypeError(
                    f"Chunk at index {index} must be a DocumentChunk."
                )

            texts.append(chunk.content)

        return self.embed_texts(texts)

    # -----------------------------------------------------
    # Embed Query
    # -----------------------------------------------------

    def embed_query(self, query: str) -> list[float]:
        """Generate a normalized embedding for a single query."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embeddings = self.embed_texts([query])

        if len(embeddings) != 1:
            raise RuntimeError(
                "Failed to generate query embedding."
            )

        return embeddings[0]

    # -----------------------------------------------------
    # Embedding Dimension
    # -----------------------------------------------------

    def get_dimension(self) -> int:
        """Return the embedding model's vector dimension."""
        return self._dimension


# ---------------------------------------------------------
# Global Embedding Service
# ---------------------------------------------------------

_embedding_service: EmbeddingService | None = None
_service_lock = Lock()


def get_embedding_service() -> EmbeddingService:
    """Return the lazily initialized, thread-safe embedding service."""
    global _embedding_service

    if _embedding_service is None:
        with _service_lock:
            if _embedding_service is None:
                _embedding_service = EmbeddingService()

    return _embedding_service


# ---------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple text inputs."""
    return get_embedding_service().embed_texts(texts)


def embed_chunks(
    chunks: list[DocumentChunk],
) -> list[list[float]]:
    """Generate embeddings for document chunks."""
    return get_embedding_service().embed_chunks(chunks)


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a single query."""
    return get_embedding_service().embed_query(query)
