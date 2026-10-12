
"""
Retriever module for the RAG pipeline.

Retrieves relevant knowledge-base chunks from the FAISS vector store
and formats the retrieved content for downstream LLM processing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from config import settings
from src.rag.chunker import DocumentChunk
from src.rag.vector_store import FAISSVectorStore
from src.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_SCORE_THRESHOLD = 0.30


# =========================================================
# Retrieval Result
# =========================================================

@dataclass
class RetrievalResult:
    """Represent a retrieved document chunk and its similarity score."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize retrieval result fields."""

        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError(
                "Retrieval result content cannot be empty."
            )

        if isinstance(self.score, bool) or not isinstance(
            self.score, (int, float)
        ):
            raise TypeError("Retrieval score must be numeric.")

        if not math.isfinite(self.score):
            raise ValueError("Retrieval score must be finite.")

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "Retrieval metadata must be a dictionary."
            )

        self.content = self.content.strip()
        self.score = float(self.score)
        self.metadata = dict(self.metadata)


# =========================================================
# Retriever
# =========================================================

class Retriever:
    """Retrieve relevant chunks from a vector store."""

    def __init__(
        self,
        vector_store: FAISSVectorStore | None = None,
        top_k: int | None = None,
    ) -> None:
        """Initialize the retriever with a vector store and result limit."""

        configured_top_k = settings.TOP_K if top_k is None else top_k
        self._validate_top_k(configured_top_k)

        self.top_k = configured_top_k
        self.vector_store = (
            vector_store
            if vector_store is not None
            else self._load_vector_store()
        )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """Validate the number of requested results."""

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

    @staticmethod
    def _validate_threshold(score_threshold: float) -> None:
        """Validate a similarity-score threshold."""

        if isinstance(score_threshold, bool) or not isinstance(
            score_threshold, (int, float)
        ):
            raise TypeError("score_threshold must be numeric.")

        if not math.isfinite(score_threshold):
            raise ValueError("score_threshold must be finite.")

        if not -1.0 <= score_threshold <= 1.0:
            raise ValueError(
                "score_threshold must be between -1.0 and 1.0."
            )

    # -----------------------------------------------------
    # Load Vector Store
    # -----------------------------------------------------

    @staticmethod
    def _load_vector_store() -> FAISSVectorStore:
        """Load the persisted vector store."""

        vector_store = FAISSVectorStore()

        if not vector_store.exists():
            raise FileNotFoundError(
                "Vector store not found. "
                "Run scripts/ingest_documents.py first."
            )

        vector_store.load()

        logger.info("FAISS vector store loaded successfully.")

        return vector_store

    # -----------------------------------------------------
    # Retrieve
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a user query."""

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        normalized_query = query.strip()

        if not normalized_query:
            return []

        number_of_results = self.top_k if top_k is None else top_k
        self._validate_top_k(number_of_results)

        try:
            results = self.vector_store.search(
                query=normalized_query,
                top_k=number_of_results,
            )

            if not isinstance(results, (list, tuple)):
                raise TypeError(
                    "Vector store search must return a list or tuple."
                )

            retrieval_results: list[RetrievalResult] = []

            for item in results:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError(
                        "Each search result must contain a chunk and score."
                    )

                chunk, score = item

                if not isinstance(chunk, DocumentChunk):
                    raise TypeError(
                        "Vector store returned an invalid document chunk."
                    )

                retrieval_results.append(
                    RetrievalResult(
                        content=chunk.content,
                        score=score,
                        metadata=chunk.metadata,
                    )
                )

            logger.debug(
                "Retrieved %d chunks for a query.",
                len(retrieval_results),
            )

            return retrieval_results

        except Exception as exc:
            logger.exception(
                "Failed to retrieve relevant documents."
            )
            raise RuntimeError(
                "Unable to retrieve knowledge-base context."
            ) from exc

    # -----------------------------------------------------
    # Filter by Score
    # -----------------------------------------------------

    def retrieve_with_threshold(
        self,
        query: str,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Filter retrieved chunks by their similarity score."""

        self._validate_threshold(score_threshold)

        results = self.retrieve(
            query=query,
            top_k=top_k,
        )

        filtered_results = [
            result
            for result in results
            if result.score >= score_threshold
        ]

        logger.debug(
            "Similarity filtering retained %d of %d chunks.",
            len(filtered_results),
            len(results),
        )

        return filtered_results

    # -----------------------------------------------------
    # Build Context
    # -----------------------------------------------------

    @staticmethod
    def _get_source(metadata: dict[str, Any]) -> str:
        """Return a safe, single-line source label."""

        source: Any = (
            metadata.get("relative_path")
            or metadata.get("filename")
            or metadata.get("source")
            or metadata.get("Source")
            or "unknown"
        )

        if not isinstance(source, str):
            return "unknown"

        # Keep metadata from breaking the source header.
        source = " ".join(source.split())

        return source[:200] or "unknown"

    def get_context(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> str:
        """Format retrieved chunks into context for the LLM."""

        results = self.retrieve_with_threshold(
            query=query,
            score_threshold=score_threshold,
            top_k=top_k,
        )

        if not results:
            return ""

        context_parts: list[str] = []

        for index, result in enumerate(results, start=1):
            source = self._get_source(result.metadata)

            context_parts.append(
                f"[Source {index}: {source}]\n"
                f"{result.content}"
            )

        return "\n\n".join(context_parts)

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[DocumentChunk]:
        """Return retrieved results as DocumentChunk objects."""

        results = self.retrieve(
            query=query,
            top_k=top_k,
        )

        return [
            DocumentChunk(
                content=result.content,
                metadata=result.metadata,
            )
            for result in results
        ]


# =========================================================
# Global Retriever
# =========================================================

_retriever: Retriever | None = None
_retriever_lock = Lock()


def get_retriever() -> Retriever:
    """Return a reusable, lazily initialized retriever."""

    global _retriever

    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = Retriever()

    return _retriever


# =========================================================
# Convenience Functions
# =========================================================

def retrieve_documents(
    query: str,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Retrieve relevant documents for a query."""

    return get_retriever().retrieve(
        query=query,
        top_k=top_k,
    )


def retrieve_context(
    query: str,
    top_k: int | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> str:
    """Retrieve and format context for the LLM."""

    return get_retriever().get_context(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
    )
