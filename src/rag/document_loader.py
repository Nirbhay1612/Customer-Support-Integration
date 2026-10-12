
"""
Document loading utilities for the AI Customer Support RAG system.

Loads Markdown and TXT files from the configured knowledge base
and converts them into Document objects with source metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".md", ".txt"})
SUPPORTED_ENCODING = "utf-8"
MAX_DOCUMENT_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


# ---------------------------------------------------------
# Document Model
# ---------------------------------------------------------


@dataclass
class Document:
    """Represent a single loaded knowledge-base document."""

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate document fields and isolate the metadata dictionary."""
        if not isinstance(self.content, str):
            raise TypeError("Document content must be a string.")

        if not isinstance(self.source, str):
            raise TypeError("Document source must be a string.")

        self.source = self.source.strip()

        if not self.source:
            raise ValueError("Document source cannot be empty.")

        if not isinstance(self.metadata, dict):
            raise TypeError("Document metadata must be a dictionary.")

        self.metadata = dict(self.metadata)


# ---------------------------------------------------------
# Document Loader
# ---------------------------------------------------------


class DocumentLoader:
    """Load supported documents from a knowledge-base directory."""

    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS

    def __init__(
        self,
        knowledge_base_path: Path | str | None = None,
    ) -> None:
        """Initialize the loader with a configured or custom path."""
        configured_path = (
            settings.KNOWLEDGE_BASE_PATH
            if knowledge_base_path is None
            else knowledge_base_path
        )

        if not isinstance(configured_path, (str, Path)):
            raise TypeError(
                "knowledge_base_path must be a string or Path."
            )

        if isinstance(configured_path, str) and not configured_path.strip():
            raise ValueError(
                "knowledge_base_path cannot be empty."
            )

        self.knowledge_base_path = Path(
            configured_path
        ).expanduser()

    # -----------------------------------------------------
    # Load Documents
    # -----------------------------------------------------

    def load_documents(self) -> list[Document]:
        """Load supported files recursively in deterministic order."""
        root = self.knowledge_base_path

        try:
            resolved_root = root.resolve(strict=True)
        except FileNotFoundError:
            logger.warning(
                "Knowledge-base directory does not exist: %s",
                root,
            )
            return []
        except OSError:
            logger.exception(
                "Unable to resolve the knowledge-base directory."
            )
            raise RuntimeError(
                "Unable to access the knowledge-base directory."
            ) from None

        if not resolved_root.is_dir():
            logger.warning(
                "Knowledge-base path is not a directory: %s",
                root,
            )
            return []

        try:
            file_paths = sorted(
                resolved_root.rglob("*"),
                key=lambda path: str(path).casefold(),
            )
        except OSError:
            logger.exception(
                "Unable to enumerate the knowledge-base directory."
            )
            raise RuntimeError(
                "Unable to read the knowledge-base directory."
            ) from None

        documents: list[Document] = []
        failed_files = 0

        for file_path in file_paths:
            if file_path.suffix.casefold() not in self.SUPPORTED_EXTENSIONS:
                continue

            try:
                # Do not ingest symbolic links.
                if file_path.is_symlink():
                    logger.warning(
                        "Skipping symbolic link: %s",
                        file_path.name,
                    )
                    failed_files += 1
                    continue

                resolved_file = file_path.resolve(strict=True)

                # Keep ingestion confined to the knowledge-base root.
                try:
                    relative_path = resolved_file.relative_to(
                        resolved_root
                    )
                except ValueError:
                    logger.warning(
                        "Skipping file outside the knowledge base: %s",
                        file_path.name,
                    )
                    failed_files += 1
                    continue

                if not resolved_file.is_file():
                    continue

                document = self._load_file(
                    resolved_file,
                    relative_path=relative_path,
                    resolved_root=resolved_root,
                )

                if not document.content.strip():
                    logger.debug(
                        "Skipping empty document: %s",
                        relative_path.as_posix(),
                    )
                    continue

                documents.append(document)

            except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
                failed_files += 1
                logger.warning(
                    "Unable to load document %s; error type: %s",
                    file_path.name,
                    type(exc).__name__,
                )

        logger.info(
            "Knowledge-base loading completed: %d loaded, "
            "%d skipped or failed.",
            len(documents),
            failed_files,
        )

        return documents

    # -----------------------------------------------------
    # Backward-Compatible Alias
    # -----------------------------------------------------

    def load_knowledge_base(self) -> list[Document]:
        """Backward-compatible alias for load_documents()."""
        return self.load_documents()

    # -----------------------------------------------------
    # Load Single File
    # -----------------------------------------------------

    def _load_file(
        self,
        file_path: Path,
        relative_path: Path | None = None,
        resolved_root: Path | None = None,
    ) -> Document:
        """Load one supported file and construct its metadata."""
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be a Path instance.")

        if file_path.suffix.casefold() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported document file type.")

        if file_path.is_symlink():
            raise ValueError("Symbolic links are not supported.")

        try:
            root = (
                resolved_root
                if resolved_root is not None
                else self.knowledge_base_path.resolve(strict=True)
            )
            resolved_file = file_path.resolve(strict=True)

            # Verify containment even when a relative path is supplied.
            actual_relative_path = resolved_file.relative_to(root)

        except (OSError, ValueError) as exc:
            raise ValueError(
                "Document must exist inside the knowledge base."
            ) from exc

        if not root.is_dir() or not resolved_file.is_file():
            raise ValueError(
                "Knowledge-base root or document is not a valid file."
            )

        if relative_path is not None and relative_path != actual_relative_path:
            raise ValueError(
                "Provided relative path does not match the document path."
            )

        relative_path = actual_relative_path
        file_size = resolved_file.stat().st_size

        if file_size > MAX_DOCUMENT_SIZE_BYTES:
            raise ValueError(
                "Document exceeds the maximum allowed file size."
            )

        content = resolved_file.read_text(
            encoding=SUPPORTED_ENCODING
        ).strip()

        normalized_source = relative_path.as_posix()
        metadata: dict[str, Any] = {
            "source": normalized_source,
            "filename": resolved_file.name,
            "file_type": resolved_file.suffix.casefold().lstrip("."),
            "category": resolved_file.stem.casefold(),
            "relative_path": normalized_source,
            "file_size_bytes": file_size,
        }

        if len(relative_path.parts) > 1:
            metadata["directory"] = Path(
                *relative_path.parts[:-1]
            ).as_posix()

        return Document(
            content=content,
            source=normalized_source,
            metadata=metadata,
        )


# ---------------------------------------------------------
# Public Convenience Functions
# ---------------------------------------------------------


def load_documents(
    knowledge_base_path: Path | str | None = None,
) -> list[Document]:
    """Load documents using a new DocumentLoader instance."""
    return DocumentLoader(
        knowledge_base_path=knowledge_base_path
    ).load_documents()


def load_knowledge_base(
    knowledge_base_path: Path | str | None = None,
) -> list[Document]:
    """Load the knowledge base using the public convenience API."""
    return load_documents(knowledge_base_path)
