"""Unit tests for multi-document retrieval and citation metadata."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.rag.service import RAGService
from app.services.retrieval.service import RetrievalService, RetrievedChunk
from tests.conftest import make_settings


def test_retrieval_maps_multi_document_metadata() -> None:
    vector_store = MagicMock()
    embedding_service = MagicMock()
    embedding_service.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    vector_store.search.return_value = [
        MagicMock(
            id="p1",
            score=0.91,
            payload={
                "text": "Alpha chunk",
                "document_id": "doc-a",
                "filename": "a.txt",
                "file_type": "txt",
                "source": "a.txt",
                "page_number": 1,
                "section": None,
                "chunk_index": 0,
            },
        ),
        MagicMock(
            id="p2",
            score=0.88,
            payload={
                "text": "Beta chunk",
                "document_id": "doc-b",
                "filename": "b.md",
                "file_type": "markdown",
                "source": "b.md",
                "page_number": 1,
                "section": "Overview",
                "chunk_index": 1,
            },
        ),
    ]

    retrieval = RetrievalService(
        make_settings(retrieval_top_k=5, embedding_dimension=4),
        embedding_service,
        vector_store,
    )
    chunks = retrieval.retrieve("cross document question")
    assert len(chunks) == 2
    assert {chunk.document_id for chunk in chunks} == {"doc-a", "doc-b"}
    assert chunks[1].section == "Overview"
    assert chunks[1].file_type == "markdown"


def test_rag_citations_include_source_document_fields() -> None:
    retrieval = MagicMock()
    llm = MagicMock()
    llm.generate.return_value = "Combined answer [S1][S2]"
    retrieval.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            text="From doc A",
            document_id="doc-a",
            filename="a.txt",
            file_type="txt",
            source="a.txt",
            page_number=1,
            section=None,
            chunk_index=0,
            chunking_strategy="fixed",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="From doc B",
            document_id="doc-b",
            filename="b.csv",
            file_type="csv",
            source="b.csv",
            page_number=1,
            section="row:2",
            chunk_index=0,
            chunking_strategy="fixed",
            score=0.85,
        ),
    ]

    result = RAGService(retrieval_service=retrieval, llm_service=llm).answer("question")
    assert len(result.citations) == 2
    assert result.citations[0].source == "a.txt"
    assert result.citations[1].file_type == "csv"
    assert result.citations[1].section == "row:2"
