"""Unit tests for RAG integration with multi-query retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.query_transformation.service import TransformedQuery
from app.services.rag.service import RAGService
from app.services.retrieval.multi_query import MultiQueryRetrievalService
from app.services.retrieval.service import RetrievalService, RetrievedChunk
from tests.conftest import make_settings


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        text="RAG combines retrieval and generation.",
        document_id="doc-1",
        filename="guide.pdf",
        file_type="pdf",
        source="guide.pdf",
        page_number=1,
        section=None,
        chunk_index=0,
        chunking_strategy="fixed",
        score=0.9,
    )


def test_phase_2d_transformed_query_used_for_multi_query_basis() -> None:
    inner_retrieval = MagicMock(spec=RetrievalService)
    inner_retrieval.retrieve.return_value = [_chunk()]
    llm = MagicMock()
    llm.generate.return_value = "Final answer"

    multi_query = MultiQueryRetrievalService(
        make_settings(multi_query_enabled=True, multi_query_count=3),
        inner_retrieval,
        llm,
    )
    generator = MagicMock()
    generator.generate.return_value = MagicMock(
        basis_query="RAG architecture",
        queries=("RAG architecture", "retrieval augmented generation", "vector RAG"),
    )
    multi_query._generator = generator  # noqa: SLF001

    transformer = MagicMock()
    original = "Can you please explain what RAG is?"
    transformer.transform.return_value = TransformedQuery(
        original_query=original,
        transformed_query="RAG architecture",
        was_transformed=True,
    )

    rag = RAGService(
        retrieval_service=multi_query,
        llm_service=llm,
        query_transformer=transformer,
    )
    rag.answer(original)

    generator.generate.assert_called_once_with("RAG architecture")
    assert "Question: Can you please explain what RAG is?" in llm.generate.call_args.args[0]
