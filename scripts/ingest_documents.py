
"""
Knowledge Base Ingestion Script.

Pipeline:
    Markdown/TXT Documents
            -> Document Loader
            -> Chunker
            -> Embeddings
            -> FAISS Vector Store

Usage:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --force
    python scripts/ingest_documents.py --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Support direct execution from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.rag.chunker import DocumentChunker
from src.rag.document_loader import Document, DocumentLoader
from src.rag.chunker import DocumentChunk
from src.rag.vector_store import FAISSVectorStore
from src.utils.logger import get_logger


logger = get_logger(__name__)

KNOWLEDGE_BASE_PATH = Path(settings.KNOWLEDGE_BASE_PATH)
VECTOR_STORE_PATH = Path(settings.VECTOR_STORE_PATH)


# ---------------------------------------------------------
# Load Documents
# ---------------------------------------------------------

def load_documents() -> list[Document]:
    """Load supported documents from the knowledge base."""

    logger.info("Loading knowledge base from: %s", KNOWLEDGE_BASE_PATH)

    loader = DocumentLoader(
        knowledge_base_path=KNOWLEDGE_BASE_PATH
    )
    documents = loader.load_documents()

    logger.info("Loaded %d documents.", len(documents))
    return documents


# ---------------------------------------------------------
# Chunk Documents
# ---------------------------------------------------------

def chunk_documents(
    documents: list[Document],
) -> list[DocumentChunk]:
    """Split loaded documents into smaller chunks."""

    if not documents:
        return []

    logger.info("Chunking %d documents.", len(documents))

    chunker = DocumentChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_documents(documents)

    logger.info("Created %d chunks.", len(chunks))
    return chunks


# ---------------------------------------------------------
# Build Vector Store
# ---------------------------------------------------------

def build_vector_store(
    chunks: list[DocumentChunk],
) -> FAISSVectorStore:
    """Build and persist a FAISS vector store."""

    if not chunks:
        raise ValueError("Cannot build a vector store without chunks.")

    logger.info("Building FAISS vector store.")

    vector_store = FAISSVectorStore(
        store_path=VECTOR_STORE_PATH
    )
    vector_store.build(chunks)
    vector_store.save()

    logger.info("Vector store saved to: %s", VECTOR_STORE_PATH)
    return vector_store


# ---------------------------------------------------------
# Ingestion Pipeline
# ---------------------------------------------------------

def ingest_documents(
    force: bool = False,
) -> FAISSVectorStore:
    """Load, chunk, embed, and persist the knowledge base."""

    logger.info("Starting knowledge-base ingestion.")

    vector_store = FAISSVectorStore(
        store_path=VECTOR_STORE_PATH
    )

    if vector_store.exists() and not force:
        logger.info("Existing vector store found; loading it.")
        logger.info("Use --force to rebuild the store.")
        vector_store.load()
        return vector_store

    documents = load_documents()

    if not documents:
        raise RuntimeError(
            f"No supported documents found in {KNOWLEDGE_BASE_PATH}."
        )

    chunks = chunk_documents(documents)

    if not chunks:
        raise RuntimeError(
            "Document chunking produced no chunks."
        )

    return build_vector_store(chunks)


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def print_statistics(
    vector_store: FAISSVectorStore,
) -> None:
    """Display vector-store statistics."""

    stats = vector_store.stats()

    print("\n" + "=" * 60)
    print("RAG VECTOR STORE STATISTICS")
    print("=" * 60)
    print(f"Vectors / Chunks    : {stats.get('chunks', 0)}")
    print(f"Embedding Dimension : {stats.get('dimension', 0)}")
    print(f"Vector Store Path   : {stats.get('path', VECTOR_STORE_PATH)}")
    print("=" * 60)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the FAISS vector store from the "
            "customer-support knowledge base."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the vector store even if one already exists.",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display vector-store statistics without rebuilding.",
    )

    args = parser.parse_args(argv)

    if args.force and args.stats:
        parser.error("--force and --stats cannot be used together.")

    return args


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    """Run the ingestion CLI."""

    try:
        args = parse_arguments(argv)

        if args.stats:
            vector_store = FAISSVectorStore(
                store_path=VECTOR_STORE_PATH
            )

            if not vector_store.exists():
                print(
                    "Vector store does not exist yet. "
                    "Run: python scripts/ingest_documents.py"
                )
                return 1

            vector_store.load()
            print_statistics(vector_store)
            return 0

        vector_store = ingest_documents(force=args.force)
        stats = vector_store.stats()

        print("\n" + "=" * 60)
        print("RAG INGESTION COMPLETED")
        print("=" * 60)
        print(f"Knowledge Base : {KNOWLEDGE_BASE_PATH}")
        print(f"Vector Store   : {VECTOR_STORE_PATH}")
        print(f"Vectors / Chunks: {stats.get('chunks', 0)}")
        print(f"Embedding Dimension: {stats.get('dimension', 0)}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user.")
        print("\nIngestion interrupted.")
        return 130

    except Exception as exc:
        logger.exception("Document ingestion failed.")
        print(f"\nIngestion failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
