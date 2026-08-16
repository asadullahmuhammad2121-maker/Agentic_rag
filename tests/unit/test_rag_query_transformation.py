"""Unit tests for RAG query transformation integration."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.query_transformation.service import TransformedQuery
from app.services.rag.service import RAGService
from app.services.retrieval.service import RetrievedChunk


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


def test_disabled_transformation_uses_original_query_for_retrieval() -> None:
    retrieval = MagicMock()
    llm = MagicMock()
    llm.generate.return_value = "Answer"
    retrieval.retrieve.return_value = [_chunk()]

    rag = RAGService(retrieval_service=retrieval, llm_service=llm, query_transformer=None)
    rag.answer("What is RAG?")

    retrieval.retrieve.assert_called_once_with("What is RAG?", top_k=None, filters=None)
    assert "Question: What is RAG?" in llm.generate.call_args.args[0]


def test_enabled_transformation_uses_transformed_query_for_retrieval() -> None:
    retrieval = MagicMock()
    llm = MagicMock()
    llm.generate.return_value = "Final answer"
    retrieval.retrieve.return_value = [_chunk()]

    transformer = MagicMock()
    original = "Can you please explain what RAG is?"
    transformed = TransformedQuery(
        original_query=original,
        transformed_query="RAG architecture",
        was_transformed=True,
    )
    transformer.transform.return_value = transformed

    rag = RAGService(retrieval_service=retrieval, llm_service=llm, query_transformer=transformer)
    result = rag.answer(original)

    transformer.transform.assert_called_once_with(original)
    retrieval.retrieve.assert_called_once_with("RAG architecture", top_k=None, filters=None)
    assert "Question: Can you please explain what RAG is?" in llm.generate.call_args.args[0]
    assert result.answer == "Final answer"


def test_generation_preserves_original_query_when_transformed() -> None:
    retrieval = MagicMock()
    llm = MagicMock()
    llm.generate.return_value = "Answer with citations"
    retrieval.retrieve.return_value = [_chunk()]

    transformer = MagicMock()
    original = "Could you tell me about vector retrieval please?"
    transformer.transform.return_value = TransformedQuery(
        original_query=original,
        transformed_query="vector retrieval basics",
        was_transformed=True,
    )

    rag = RAGService(retrieval_service=retrieval, llm_service=llm, query_transformer=transformer)
    rag.answer(original)

    prompt_arg = llm.generate.call_args.args[0]
    assert "Question: Could you tell me about vector retrieval please?" in prompt_arg
    assert "Question: vector retrieval basics" not in prompt_arg
