"""Unit tests for multi-query retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import ProviderError
from app.services.retrieval.combiner import combine_retrieved_chunks
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.multi_query import MultiQueryGenerator, MultiQueryRetrievalService
from app.services.retrieval.service import RetrievalService, RetrievedChunk
from tests.conftest import make_settings


def _chunk(
    *,
    chunk_id: str,
    score: float,
    document_id: str = "doc-1",
    text: str = "chunk",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        filename="file.pdf",
        file_type="pdf",
        source="file.pdf",
        page_number=1,
        section=None,
        chunk_index=0,
        chunking_strategy="fixed",
        score=score,
    )


@pytest.fixture
def retrieval() -> MagicMock:
    return MagicMock(spec=RetrievalService)


@pytest.fixture
def llm() -> MagicMock:
    service = MagicMock()
    service.generate.return_value = (
        "RAG pipeline overview\n"
        "retrieval augmented generation architecture\n"
        "vector search with LLM answers"
    )
    return service


@pytest.fixture
def multi_query(retrieval: MagicMock, llm: MagicMock) -> MultiQueryRetrievalService:
    settings = make_settings(multi_query_enabled=True, multi_query_count=3, retrieval_top_k=5)
    return MultiQueryRetrievalService(settings, retrieval, llm)


def test_disabled_mode_delegates_to_single_retrieval(retrieval: MagicMock, llm: MagicMock) -> None:
    settings = make_settings(multi_query_enabled=False)
    service = MultiQueryRetrievalService(settings, retrieval, llm)
    retrieval.retrieve.return_value = [_chunk(chunk_id="c1", score=0.9)]

    chunks = service.retrieve("What is RAG?")

    assert len(chunks) == 1
    retrieval.retrieve.assert_called_once_with(
        "What is RAG?",
        top_k=None,
        filters=None,
        score_threshold=None,
    )
    llm.generate.assert_not_called()


def test_enabled_generates_three_queries_and_retrieves_each(
    multi_query: MultiQueryRetrievalService,
    retrieval: MagicMock,
    llm: MagicMock,
) -> None:
    retrieval.retrieve.side_effect = [
        [_chunk(chunk_id="c1", score=0.9)],
        [_chunk(chunk_id="c2", score=0.8)],
        [_chunk(chunk_id="c3", score=0.7)],
    ]

    chunks = multi_query.retrieve("What is RAG?")

    assert llm.generate.call_count == 1
    assert retrieval.retrieve.call_count == 3
    assert {chunk.chunk_id for chunk in chunks} == {"c1", "c2", "c3"}
    assert [chunk.score for chunk in chunks] == pytest.approx([0.9, 0.8, 0.7])


def test_configurable_query_count(retrieval: MagicMock, llm: MagicMock) -> None:
    llm.generate.return_value = "query one\nquery two\nquery three\nquery four"
    settings = make_settings(multi_query_enabled=True, multi_query_count=2, retrieval_top_k=5)
    service = MultiQueryRetrievalService(settings, retrieval, llm)
    retrieval.retrieve.side_effect = [[_chunk(chunk_id="c1", score=0.5)], [_chunk(chunk_id="c2", score=0.4)]]

    service.retrieve("Explain embeddings")

    assert retrieval.retrieve.call_count == 2


def test_duplicate_query_removal(retrieval: MagicMock, llm: MagicMock) -> None:
    llm.generate.return_value = "RAG basics\nrag basics\nvector search for RAG"
    settings = make_settings(multi_query_enabled=True, multi_query_count=3, retrieval_top_k=5)
    service = MultiQueryRetrievalService(settings, retrieval, llm)
    retrieval.retrieve.return_value = []

    service.retrieve("What is RAG?")

    called_queries = [call.args[0] for call in retrieval.retrieve.call_args_list]
    assert len(called_queries) == len(set(query.casefold() for query in called_queries))


def test_duplicate_chunk_removal_keeps_highest_score() -> None:
    combined = combine_retrieved_chunks(
        [
            [_chunk(chunk_id="c1", score=0.7, text="first")],
            [_chunk(chunk_id="c1", score=0.95, text="second")],
            [_chunk(chunk_id="c2", score=0.6, text="other")],
        ],
        limit=5,
    )

    assert len(combined) == 2
    assert combined[0].chunk_id == "c1"
    assert combined[0].score == pytest.approx(0.95)
    assert combined[0].text == "second"
    assert combined[1].chunk_id == "c2"


def test_groq_failure_falls_back_to_single_retrieval(
    retrieval: MagicMock,
    llm: MagicMock,
) -> None:
    settings = make_settings(multi_query_enabled=True, multi_query_count=3)
    service = MultiQueryRetrievalService(settings, retrieval, llm)
    llm.generate.side_effect = ProviderError("fail", provider="groq")
    retrieval.retrieve.return_value = [_chunk(chunk_id="c1", score=0.5)]

    chunks = service.retrieve("What is RAG?")

    assert len(chunks) == 1
    retrieval.retrieve.assert_called_once()


def test_metadata_preservation(multi_query: MultiQueryRetrievalService, retrieval: MagicMock) -> None:
    retrieval.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="doc-1:00001",
            text="Detailed answer",
            document_id="doc-1",
            filename="guide.pdf",
            file_type="pdf",
            source="guide.pdf",
            page_number=2,
            section="Intro",
            chunk_index=1,
            chunking_strategy="fixed",
            score=0.88,
        )
    ]

    chunks = multi_query.retrieve("Explain RAG")

    chunk = chunks[0]
    assert chunk.document_id == "doc-1"
    assert chunk.filename == "guide.pdf"
    assert chunk.page_number == 2
    assert chunk.section == "Intro"
    assert chunk.chunking_strategy == "fixed"


def test_document_filters_passed_to_each_retrieval(
    multi_query: MultiQueryRetrievalService,
    retrieval: MagicMock,
) -> None:
    retrieval.retrieve.return_value = []
    filters = RetrievalFilters.from_query(document_ids=["doc-1"], file_types=["pdf"])

    multi_query.retrieve("What is RAG?", filters=filters)

    for call in retrieval.retrieve.call_args_list:
        assert call.kwargs["filters"] == filters


def test_uses_transformed_basis_query(retrieval: MagicMock, llm: MagicMock) -> None:
    settings = make_settings(multi_query_enabled=True, multi_query_count=3)
    generator = MagicMock()
    generator.generate.return_value = MagicMock(
        basis_query="RAG architecture",
        queries=("RAG architecture", "retrieval augmented generation", "vector RAG pipeline"),
    )
    service = MultiQueryRetrievalService(settings, retrieval, llm, generator=generator)
    retrieval.retrieve.return_value = []

    service.retrieve("RAG architecture")

    generator.generate.assert_called_once_with("RAG architecture")
    assert all(call.args[0] for call in retrieval.retrieve.call_args_list)


def test_generator_normalizes_numbered_output(llm: MagicMock) -> None:
    llm.generate.return_value = "1. first query\n2) second query\n- third query"
    generator = MultiQueryGenerator(make_settings(multi_query_count=3), llm)
    result = generator.generate("basis question")

    assert len(result.queries) == 3
    assert result.queries[0] == "basis question"
