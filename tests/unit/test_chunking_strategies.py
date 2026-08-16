"""Unit tests for advanced chunking strategies."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.chunking.config import ChunkingConfig
from app.services.chunking.fixed import FixedSizeChunker
from app.services.chunking.recursive import RecursiveChunker
from app.services.chunking.semantic import SemanticChunker
from app.services.chunking.structure import StructureAwareChunker
from app.services.ingestion.base import ExtractedSection
from app.services.ingestion.service import DocumentIngestionService
from tests.conftest import make_settings
from tests.helpers.document_fixtures import build_markdown_bytes, build_txt_bytes
from tests.helpers.pdf_fixtures import build_pdf_bytes


def _config(**overrides: object) -> ChunkingConfig:
    base = {
        "strategy": "fixed",
        "chunk_size": 80,
        "chunk_overlap": 10,
        "min_chunk_size": 5,
        "max_chunk_size": 160,
        "semantic_similarity_threshold": 0.5,
    }
    base.update(overrides)
    return ChunkingConfig(**base)  # type: ignore[arg-type]


def _sections(*texts: str) -> list[ExtractedSection]:
    return [
        ExtractedSection(
            text=text,
            section_index=index,
            page_number=1,
            section=f"Section {index + 1}" if index else None,
        )
        for index, text in enumerate(texts)
    ]


def test_fixed_chunker_respects_size_and_overlap() -> None:
    chunker = FixedSizeChunker(_config(strategy="fixed", chunk_size=30, chunk_overlap=5))
    chunks = chunker.chunk_sections(
        _sections("word " * 40),
        document_id="doc-fixed",
        filename="fixed.txt",
        file_type="txt",
        source="fixed.txt",
    )
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 30 for chunk in chunks)
    assert chunks[0].chunk_id == "doc-fixed:00000"


def test_recursive_chunker_splits_on_paragraph_boundaries() -> None:
    text = "Paragraph one about cats.\n\nParagraph two about dogs.\n\nParagraph three about birds."
    chunker = RecursiveChunker(_config(strategy="recursive", chunk_size=60, chunk_overlap=0))
    chunks = chunker.chunk_sections(
        _sections(text),
        document_id="doc-rec",
        filename="notes.txt",
        file_type="txt",
        source="notes.txt",
    )
    assert len(chunks) >= 2
    assert any("cats" in chunk.text for chunk in chunks)
    assert any("dogs" in chunk.text for chunk in chunks)


def test_semantic_chunker_groups_similar_sentences() -> None:
    embedding = MagicMock()

    def _embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if "cats" in text or "felines" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "dogs" in text or "canines" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    embedding.embed_documents.side_effect = _embed
    text = "Cats are felines. Dogs are canines. Birds can fly."
    chunker = SemanticChunker(
        _config(strategy="semantic", chunk_size=120, semantic_similarity_threshold=0.8),
        embedding,
    )
    chunks = chunker.chunk_sections(
        _sections(text),
        document_id="doc-sem",
        filename="animals.txt",
        file_type="txt",
        source="animals.txt",
    )
    assert len(chunks) >= 2
    assert embedding.embed_documents.called


def test_structure_aware_chunker_keeps_small_sections_whole() -> None:
    chunker = StructureAwareChunker(_config(strategy="structure", chunk_size=100, chunk_overlap=0))
    chunks = chunker.chunk_sections(
        [
            ExtractedSection(text="Short intro.", section_index=0, page_number=1, section="Intro"),
            ExtractedSection(
                text="A much longer section that should be split internally without crossing boundaries.",
                section_index=1,
                page_number=1,
                section="Details",
            ),
        ],
        document_id="doc-struct",
        filename="readme.md",
        file_type="markdown",
        source="readme.md",
    )
    assert len(chunks) >= 2
    assert {chunk.section for chunk in chunks} == {"Intro", "Details"}


def test_metadata_preserved_for_all_strategies() -> None:
    section = ExtractedSection(
        text="Metadata check paragraph.",
        section_index=0,
        page_number=2,
        section="Overview",
    )
    for chunker in (
        FixedSizeChunker(_config(strategy="fixed")),
        RecursiveChunker(_config(strategy="recursive")),
        StructureAwareChunker(_config(strategy="structure")),
    ):
        chunks = chunker.chunk_sections(
            [section],
            document_id="doc-meta",
            filename="file.json",
            file_type="json",
            source="file.json",
        )
        assert chunks
        chunk = chunks[0]
        assert chunk.document_id == "doc-meta"
        assert chunk.filename == "file.json"
        assert chunk.file_type == "json"
        assert chunk.source == "file.json"
        assert chunk.page_number == 2
        assert chunk.section == "Overview"
        assert chunk.chunk_id.startswith("doc-meta:")


def test_empty_sections_produce_no_chunks() -> None:
    chunker = FixedSizeChunker(_config())
    assert (
        chunker.chunk_sections(
            [ExtractedSection(text="   ", section_index=0, page_number=1, section=None)],
            document_id="doc-empty",
            filename="blank.txt",
            file_type="txt",
            source="blank.txt",
        )
        == []
    )


def test_small_document_single_chunk() -> None:
    chunker = FixedSizeChunker(_config(chunk_size=200, min_chunk_size=20))
    chunks = chunker.chunk_sections(
        _sections("Tiny note."),
        document_id="doc-small",
        filename="tiny.txt",
        file_type="txt",
        source="tiny.txt",
    )
    assert len(chunks) == 1
    assert chunks[0].text == "Tiny note."


def test_multi_format_ingestion_with_recursive_strategy() -> None:
    vector_store = MagicMock()
    vector_store.find_by_payload.return_value = []
    embedding = MagicMock()
    embedding.provider_name = "huggingface"
    embedding.model_name = "test-model"

    def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] for _ in texts]

    embedding.embed_documents.side_effect = _embed

    settings = make_settings(
        embedding_dimension=8,
        chunking_strategy="recursive",
        chunk_size=40,
        chunk_overlap=5,
        chunk_min_size=10,
        chunk_max_size=120,
    )
    service = DocumentIngestionService(settings, vector_store, embedding)
    result = service.ingest_document(
        filename="notes.txt",
        content=build_txt_bytes("Paragraph one.\n\nParagraph two with more detail."),
        content_type="text/plain",
    )
    assert result.file_type == "txt"
    records = vector_store.add_vectors.call_args.args[1]
    assert records[0].payload["chunking_strategy"] == "recursive"
    assert records[0].id == records[0].payload["chunk_id"]

    pdf_result = service.ingest_document(
        filename="sample.pdf",
        content=build_pdf_bytes(["PDF page one", "PDF page two"]),
        content_type="application/pdf",
    )
    assert pdf_result.file_type == "pdf"

    md_result = service.ingest_document(
        filename="readme.md",
        content=build_markdown_bytes("# Title\nBody text"),
        content_type="text/markdown",
    )
    assert md_result.file_type == "markdown"
