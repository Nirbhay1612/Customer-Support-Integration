"""
Tests for the RAG pipeline.

Run:
    python -m pytest tests/test_rag.py -v

These tests use deterministic fake embeddings so the test suite does not
need to download a SentenceTransformer model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.rag.document_loader import Document, DocumentLoader
from src.rag.chunker import DocumentChunk, DocumentChunker
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import FAISSVectorStore


# ---------------------------------------------------------
# Fake Embedding Service and Model
# ---------------------------------------------------------


class FakeEmbeddingService:
    """Small deterministic embedding service for FAISS unit tests."""

    dimension = 8

    @staticmethod
    def _embedding_from_text(text: str) -> list[float]:
        vector = np.zeros(FakeEmbeddingService.dimension, dtype=np.float32)

        for index, char in enumerate(text):
            vector[index % FakeEmbeddingService.dimension] += ord(char)

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm

        return vector.tolist()

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        return [self._embedding_from_text(chunk.content) for chunk in chunks]

    def embed_query(self, query: str) -> list[float]:
        return self._embedding_from_text(query)

    def get_dimension(self) -> int:
        return self.dimension


class FakeSentenceTransformer:
    """Stand-in model for testing EmbeddingService without model downloads."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        vectors = [
            FakeEmbeddingService._embedding_from_text(text)
            for text in texts
        ]
        return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------


@pytest.fixture
def fake_embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            content="Our return policy allows eligible products to be returned.",
            metadata={"source": "returns.md", "category": "returns"},
        ),
        DocumentChunk(
            content="Orders are usually delivered within several business days.",
            metadata={"source": "shipping.md", "category": "shipping"},
        ),
        DocumentChunk(
            content="Refunds are processed after the return is approved.",
            metadata={"source": "refunds.md", "category": "refunds"},
        ),
    ]


@pytest.fixture
def fake_store(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedding_service: FakeEmbeddingService,
) -> None:
    monkeypatch.setattr(
        "src.rag.vector_store.get_embedding_service",
        lambda: fake_embedding_service,
    )


# ---------------------------------------------------------
# Document Loader Tests
# ---------------------------------------------------------


class TestDocumentLoader:
    def test_load_single_markdown_file(self, tmp_path: Path) -> None:
        (tmp_path / "faq.md").write_text(
            "# FAQ\n\nWhat is your return policy?",
            encoding="utf-8",
        )

        documents = DocumentLoader(tmp_path).load_documents()

        assert len(documents) == 1
        assert documents[0].source == "faq.md"
        assert "return policy" in documents[0].content

    def test_load_text_file(self, tmp_path: Path) -> None:
        (tmp_path / "shipping.txt").write_text(
            "Shipping usually takes several business days.",
            encoding="utf-8",
        )

        documents = DocumentLoader(tmp_path).load_documents()

        assert len(documents) == 1
        assert documents[0].source == "shipping.txt"
        assert "several business days" in documents[0].content

    def test_load_multiple_supported_files(self, tmp_path: Path) -> None:
        (tmp_path / "faq.md").write_text("FAQ information.", encoding="utf-8")
        (tmp_path / "shipping.txt").write_text(
            "Shipping information.",
            encoding="utf-8",
        )

        documents = DocumentLoader(tmp_path).load_documents()

        assert len(documents) == 2
        assert {doc.source for doc in documents} == {
            "faq.md",
            "shipping.txt",
        }

    def test_ignores_unsupported_files(self, tmp_path: Path) -> None:
        (tmp_path / "faq.md").write_text("FAQ information.", encoding="utf-8")
        (tmp_path / "image.jpg").write_bytes(b"fake image")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")

        documents = DocumentLoader(tmp_path).load_documents()

        assert [doc.source for doc in documents] == ["faq.md"]

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert DocumentLoader(tmp_path).load_documents() == []

    def test_missing_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert DocumentLoader(tmp_path / "missing").load_documents() == []

    def test_nested_directory_preserves_relative_path(
        self,
        tmp_path: Path,
    ) -> None:
        nested_dir = tmp_path / "policies"
        nested_dir.mkdir()
        (nested_dir / "returns.md").write_text(
            "Return policy information.",
            encoding="utf-8",
        )

        documents = DocumentLoader(tmp_path).load_documents()

        assert len(documents) == 1
        assert documents[0].source == "policies/returns.md"
        assert documents[0].metadata["relative_path"] == "policies/returns.md"

    def test_blank_document_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "blank.md").write_text("   \n  ", encoding="utf-8")

        assert DocumentLoader(tmp_path).load_documents() == []

    def test_document_metadata_is_independent(self) -> None:
        original = {"category": "faq"}
        document = Document(
            content="A useful answer.",
            source="faq.md",
            metadata=original,
        )

        document.metadata["category"] = "changed"

        assert original["category"] == "faq"


# ---------------------------------------------------------
# Chunker Tests
# ---------------------------------------------------------


class TestDocumentChunker:
    def test_chunks_document(self) -> None:
        document = Document(
            content=(
                "This is a long document. It contains information about "
                "returns, shipping, payments and refunds."
            ),
            source="faq.md",
            metadata={"category": "faq"},
        )

        chunks = DocumentChunker(
            chunk_size=50,
            chunk_overlap=10,
        ).chunk_documents([document])

        assert chunks
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.content.strip() for chunk in chunks)
        assert all(len(chunk.content) <= 50 for chunk in chunks)

    def test_chunk_preserves_metadata(self) -> None:
        document = Document(
            content="Return policy information.",
            source="returns.md",
            metadata={"category": "returns"},
        )

        chunks = DocumentChunker(
            chunk_size=100,
            chunk_overlap=10,
        ).chunk_documents([document])

        assert chunks
        assert chunks[0].metadata["source"] == "returns.md"
        assert chunks[0].metadata["category"] == "returns"
        assert chunks[0].metadata["filename"] == "returns.md"
        assert chunks[0].metadata["chunk_id"] == 0

    def test_empty_document_content_returns_no_chunks(self) -> None:
        document = Document(content="   ", source="empty.md")

        chunks = DocumentChunker().chunk_document(document)

        assert chunks == []

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            DocumentChunker(chunk_size=0, chunk_overlap=0)

    def test_overlap_must_be_smaller_than_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=10, chunk_overlap=10)

    def test_non_document_input_is_rejected(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            DocumentChunker().chunk_document("not a Document")  # type: ignore[arg-type]


# ---------------------------------------------------------
# Embedding Service Tests
# ---------------------------------------------------------


class TestEmbeddingService:
    @pytest.fixture
    def service(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> EmbeddingService:
        fake_model = FakeSentenceTransformer(dimension=8)
        monkeypatch.setattr(
            "src.rag.embeddings.get_embedding_model",
            lambda: fake_model,
        )
        return EmbeddingService()

    def test_embed_query_returns_expected_dimension(
        self,
        service: EmbeddingService,
    ) -> None:
        vector = service.embed_query("Where is my order?")

        assert len(vector) == 8
        assert all(np.isfinite(value) for value in vector)

    def test_embed_texts_returns_one_vector_per_text(
        self,
        service: EmbeddingService,
    ) -> None:
        vectors = service.embed_texts(["shipping policy", "return policy"])

        assert len(vectors) == 2
        assert all(len(vector) == 8 for vector in vectors)

    def test_vectors_are_normalized(
        self,
        service: EmbeddingService,
    ) -> None:
        vector = service.embed_query("shipping information")

        assert np.linalg.norm(np.asarray(vector)) == pytest.approx(1.0)

    def test_empty_text_list_returns_empty_list(
        self,
        service: EmbeddingService,
    ) -> None:
        assert service.embed_texts([]) == []

    def test_blank_query_raises_value_error(
        self,
        service: EmbeddingService,
    ) -> None:
        with pytest.raises(ValueError):
            service.embed_query("   ")

    def test_non_string_query_raises_type_error(
        self,
        service: EmbeddingService,
    ) -> None:
        with pytest.raises(TypeError):
            service.embed_query(123)  # type: ignore[arg-type]

    def test_empty_text_inside_batch_is_rejected(
        self,
        service: EmbeddingService,
    ) -> None:
        with pytest.raises(ValueError):
            service.embed_texts(["valid text", "   "])

    def test_dimension_is_reported(self, service: EmbeddingService) -> None:
        assert service.get_dimension() == 8


# ---------------------------------------------------------
# FAISS Vector Store Tests
# ---------------------------------------------------------


class TestFAISSVectorStore:
    def test_build_vector_store(
        self,
        tmp_path: Path,
        sample_chunks: list[DocumentChunk],
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")
        store.build(sample_chunks)

        assert store.index is not None
        assert len(store.chunks) == 3
        assert store.index.ntotal == 3

    def test_build_empty_chunks_raises_error(
        self,
        tmp_path: Path,
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")

        with pytest.raises(ValueError):
            store.build([])

    def test_search_returns_chunks_and_float_scores(
        self,
        tmp_path: Path,
        sample_chunks: list[DocumentChunk],
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")
        store.build(sample_chunks)

        results = store.search(query="return policy", top_k=2)

        assert isinstance(results, list)
        assert 0 < len(results) <= 2

        for chunk, score in results:
            assert isinstance(chunk, DocumentChunk)
            assert isinstance(score, float)

    def test_search_empty_query_returns_empty_list(
        self,
        tmp_path: Path,
        sample_chunks: list[DocumentChunk],
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")
        store.build(sample_chunks)

        assert store.search(query="", top_k=3) == []

    def test_search_before_build_raises_error(
        self,
        tmp_path: Path,
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")

        with pytest.raises(RuntimeError):
            store.search("shipping")

    def test_add_chunks(
        self,
        tmp_path: Path,
        sample_chunks: list[DocumentChunk],
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")
        store.build(sample_chunks[:2])
        store.add_chunks(sample_chunks[2:])

        assert len(store.chunks) == 3
        assert store.index is not None
        assert store.index.ntotal == 3

    def test_add_empty_chunks_does_nothing(
        self,
        tmp_path: Path,
        sample_chunks: list[DocumentChunk],
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")
        store.build(sample_chunks)
        store.add_chunks([])

        assert len(store.chunks) == 3

    def test_save_and_load_vector_store(
        self,
        tmp_path: Path,
        sample_chunks: list[DocumentChunk],
        fake_store: None,
    ) -> None:
        store_path = tmp_path / "vector_store"
        store = FAISSVectorStore(store_path)
        store.build(sample_chunks)
        store.save()

        assert (store_path / "index.faiss").exists()
        assert (store_path / "metadata.json").exists()

        loaded_store = FAISSVectorStore(store_path)
        loaded_store.load()

        assert loaded_store.index is not None
        assert len(loaded_store.chunks) == 3
        assert loaded_store.index.ntotal == 3

        results = loaded_store.search("refund policy", top_k=2)
        assert results

    def test_save_before_build_raises_error(
        self,
        tmp_path: Path,
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")

        with pytest.raises(RuntimeError):
            store.save()

    def test_load_missing_store_raises_file_not_found(
        self,
        tmp_path: Path,
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "missing_store")

        with pytest.raises(FileNotFoundError):
            store.load()

    def test_exists_reflects_persisted_state(
        self,
        tmp_path: Path,
        sample_chunks: list[DocumentChunk],
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")

        assert store.exists() is False

        store.build(sample_chunks)
        store.save()

        assert store.exists() is True

    def test_clear_removes_in_memory_index(
        self,
        tmp_path: Path,
        sample_chunks: list[DocumentChunk],
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")
        store.build(sample_chunks)

        store.clear()

        assert store.index is None
        assert store.chunks == []

    def test_stats_before_build(
        self,
        tmp_path: Path,
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")
        stats = store.stats()

        assert stats["vectors"] == 0
        assert stats["chunks"] == 0
        assert stats["dimension"] == 0

    def test_stats_after_build(
        self,
        tmp_path: Path,
        sample_chunks: list[DocumentChunk],
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")
        store.build(sample_chunks)

        stats = store.stats()

        assert stats["vectors"] == 3
        assert stats["chunks"] == 3
        assert stats["dimension"] == 8


# ---------------------------------------------------------
# End-to-End RAG Pipeline Test
# ---------------------------------------------------------


class TestRAGPipeline:
    def test_document_to_vector_store_pipeline(
        self,
        tmp_path: Path,
        fake_store: None,
    ) -> None:
        knowledge_base = tmp_path / "knowledge_base"
        knowledge_base.mkdir()

        (knowledge_base / "shipping.md").write_text(
            "# Shipping Policy\n\n"
            "Standard shipping usually takes several business days.\n\n"
            "Customers can track orders using the tracking number.",
            encoding="utf-8",
        )

        documents = DocumentLoader(knowledge_base).load_documents()
        assert len(documents) == 1

        chunks = DocumentChunker(
            chunk_size=100,
            chunk_overlap=20,
        ).chunk_documents(documents)
        assert chunks

        store = FAISSVectorStore(tmp_path / "vector_store")
        store.build(chunks)

        results = store.search(
            query="How can I track my order?",
            top_k=2,
        )

        assert results
        for chunk, score in results:
            assert chunk.content
            assert isinstance(score, float)


# ---------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------


class TestRAGValidation:
    def test_chunk_content_is_not_empty(
        self,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        assert all(chunk.content.strip() for chunk in sample_chunks)

    def test_chunk_metadata_is_dictionary(
        self,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        assert all(isinstance(chunk.metadata, dict) for chunk in sample_chunks)

    def test_vector_count_matches_chunk_count(
        self,
        tmp_path: Path,
        sample_chunks: list[DocumentChunk],
        fake_store: None,
    ) -> None:
        store = FAISSVectorStore(tmp_path / "vector_store")
        store.build(sample_chunks)

        assert store.index is not None
        assert store.index.ntotal == len(store.chunks)
