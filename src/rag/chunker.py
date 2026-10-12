
"""
Document chunking utilities for the RAG pipeline.

Splits documents into smaller overlapping chunks suitable for
embedding generation and vector retrieval while preserving metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.rag.document_loader import Document
from src.utils.logger import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    ", ",
    " ",
    "",
]


# ---------------------------------------------------------
# Chunk Model
# ---------------------------------------------------------


@dataclass
class DocumentChunk:
    """Represent a text chunk extracted from a source document."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate chunk content and copy metadata."""
        if not isinstance(self.content, str):
            raise TypeError("Chunk content must be a string.")

        self.content = self.content.strip()

        if not self.content:
            raise ValueError("Chunk content cannot be empty.")

        if not isinstance(self.metadata, dict):
            raise TypeError("Chunk metadata must be a dictionary.")

        # Prevent modifications to the original top-level dictionary.
        self.metadata = dict(self.metadata)


# ---------------------------------------------------------
# Document Chunker
# ---------------------------------------------------------


class DocumentChunker:
    """Split documents into overlapping text chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initialize the splitter using validated configuration."""
        self.chunk_size = (
            settings.CHUNK_SIZE
            if chunk_size is None
            else chunk_size
        )
        self.chunk_overlap = (
            settings.CHUNK_OVERLAP
            if chunk_overlap is None
            else chunk_overlap
        )

        self._validate_configuration()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=DEFAULT_SEPARATORS,
            length_function=len,
            strip_whitespace=True,
            add_start_index=True,
        )

    # -----------------------------------------------------
    # Configuration Validation
    # -----------------------------------------------------

    def _validate_configuration(self) -> None:
        """Validate chunk size and overlap settings."""
        if (
            isinstance(self.chunk_size, bool)
            or not isinstance(self.chunk_size, int)
        ):
            raise TypeError("chunk_size must be an integer.")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if (
            isinstance(self.chunk_overlap, bool)
            or not isinstance(self.chunk_overlap, int)
        ):
            raise TypeError("chunk_overlap must be an integer.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

    # -----------------------------------------------------
    # Chunk Multiple Documents
    # -----------------------------------------------------

    def chunk_documents(
        self,
        documents: list[Document],
    ) -> list[DocumentChunk]:
        """Chunk multiple documents without stopping on one failure."""
        if not isinstance(documents, list):
            raise TypeError("documents must be a list.")

        if not documents:
            logger.info("No documents provided for chunking.")
            return []

        all_chunks: list[DocumentChunk] = []
        failed_documents = 0

        for index, document in enumerate(documents):
            try:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)

            except (TypeError, ValueError, RuntimeError, AttributeError):
                failed_documents += 1
                logger.exception(
                    "Failed to chunk document at input index %d.",
                    index,
                )

        logger.info(
            "Chunking completed: %d chunks from %d documents; "
            "%d documents failed.",
            len(all_chunks),
            len(documents),
            failed_documents,
        )

        return all_chunks

    # -----------------------------------------------------
    # Chunk Single Document
    # -----------------------------------------------------

    def chunk_document(
        self,
        document: Document,
    ) -> list[DocumentChunk]:
        """Split one document and preserve its source metadata."""
        if not isinstance(document, Document):
            raise TypeError("document must be a Document instance.")

        if not isinstance(document.content, str):
            raise TypeError("Document content must be a string.")

        if not isinstance(document.metadata, dict):
            raise TypeError(
                "Document metadata must be a dictionary."
            )

        if (
            not isinstance(document.source, str)
            or not document.source.strip()
        ):
            raise ValueError("Document source cannot be empty.")

        source = document.source.strip()
        content = document.content.strip()

        if not content:
            logger.debug("Skipping empty document: %s", source)
            return []

        # Normalize path separators for consistent filename extraction.
        normalized_source = source.replace("\\", "/")
        filename = normalized_source.rsplit("/", maxsplit=1)[-1]

        try:
            split_documents = self.splitter.create_documents(
                texts=[content],
                metadatas=[dict(document.metadata)],
            )
        except Exception as exc:
            logger.exception(
                "Failed to split document: %s",
                source,
            )
            raise RuntimeError(
                f"Document chunking failed for source: {source}"
            ) from exc

        document_chunks: list[DocumentChunk] = []

        for index, split_document in enumerate(split_documents):
            chunk_content = split_document.page_content.strip()

            if not chunk_content:
                continue

            raw_start_index = split_document.metadata.get(
                "start_index",
                -1,
            )

            start_index = (
                raw_start_index
                if isinstance(raw_start_index, int)
                and not isinstance(raw_start_index, bool)
                and raw_start_index >= 0
                else -1
            )

            metadata: dict[str, Any] = dict(document.metadata)

            # Standardize fields required by downstream RAG components.
            metadata.update(
                {
                    "source": source,
                    "Source": source,
                    "filename": filename,
                    "chunk_id": index,
                    "chunk_size": len(chunk_content),
                    "start_index": start_index,
                }
            )

            document_chunks.append(
                DocumentChunk(
                    content=chunk_content,
                    metadata=metadata,
                )
            )

        logger.debug(
            "Created %d chunks from %s.",
            len(document_chunks),
            source,
        )

        return document_chunks


# ---------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------


def chunk_documents(
    documents: list[Document],
) -> list[DocumentChunk]:
    """Chunk documents using the default configuration."""
    return DocumentChunker().chunk_documents(documents)
