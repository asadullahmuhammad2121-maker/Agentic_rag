"""Unit tests for context optimization (Phase 2H)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.context_optimization.service import ContextOptimizationService
from app.services.context_optimization.tokens import estimate_chunk_prompt_tokens, estimate_tokens
from app.services.rag.service import RAGService
from app.services.retrieval.service import RetrievedChunk
from tests.conftest import make_settings


def _chunk(
    *,
    chunk_id: str,
    text: str,
    score: float = 0.9,
    document_id: str = "doc-1",
    filename: str = "a.pdf",
    file_type: str = "pdf",
    page_number: int = 1,
    section: str | None = None,
    chunk_index: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        source=filename,
        page_number=page_number,
        section=section,
        chunk_index=chunk_index,
        chunking_strategy="fixed",
        score=score,
    )


@pytest.fixture
def optimizer_disabled() -> ContextOptimizationService:
    return ContextOptimizationService(make_settings(context_optimization_enabled=False))


@pytest.fixture
def optimizer_enabled() -> ContextOptimizationService:
    return ContextOptimizationService(
        make_settings(
            context_optimization_enabled=True,
            context_max_chunks=8,
            context_max_tokens=6000,
            context_min_score=0.0,
        )
    )


def test_disabled_passthrough_returns_same_chunks(
    optimizer_disabled: ContextOptimizationService,
) -> None:
    chunks = [
        _chunk(chunk_id="doc-1:00001", text="alpha"),
        _chunk(chunk_id="doc-1:00002", text="beta"),
    ]

    result = optimizer_disabled.optimize(chunks)

    assert result.chunks is chunks
    assert result.removed_count == 0
    assert result.metadata.total_removed == 0
    assert result.estimated_tokens == sum(estimate_chunk_prompt_tokens(c) for c in chunks)


def test_enabled_removes_duplicate_chunk_ids(
    optimizer_enabled: ContextOptimizationService,
) -> None:
    chunks = [
        _chunk(chunk_id="doc-1:00001", text="first copy", score=0.95),
        _chunk(chunk_id="doc-1:00001", text="duplicate copy", score=0.80),
        _chunk(chunk_id="doc-1:00002", text="unique", score=0.70),
    ]

    result = optimizer_enabled.optimize(chunks)

    assert [chunk.chunk_id for chunk in result.chunks] == ["doc-1:00001", "doc-1:00002"]
    assert result.chunks[0].text == "first copy"
    assert result.metadata.duplicate_removed == 1
    assert result.removed_count == 1


def test_enabled_removes_redundant_overlapping_chunks(
    optimizer_enabled: ContextOptimizationService,
) -> None:
    shared = "machine learning models require large training datasets for accuracy"
    chunks = [
        _chunk(chunk_id="doc-1:00001", text=shared, score=0.92),
        _chunk(
            chunk_id="doc-1:00002",
            text=f"{shared} and careful evaluation",
            score=0.88,
        ),
        _chunk(chunk_id="doc-2:00001", text="completely unrelated database indexing", score=0.75),
    ]

    result = optimizer_enabled.optimize(chunks)

    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_id == "doc-1:00001"
    assert result.chunks[1].chunk_id == "doc-2:00001"
    assert result.metadata.redundant_removed == 1


def test_enabled_filters_by_minimum_reliable_score() -> None:
    optimizer = ContextOptimizationService(
        make_settings(
            context_optimization_enabled=True,
            context_min_score=0.5,
        )
    )
    chunks = [
        _chunk(chunk_id="doc-1:00001", text="high score", score=0.91),
        _chunk(chunk_id="doc-1:00002", text="low score", score=0.30),
        _chunk(chunk_id="doc-1:00003", text="rrf score", score=0.008),
    ]

    result = optimizer.optimize(chunks)

    chunk_ids = [chunk.chunk_id for chunk in result.chunks]
    assert chunk_ids == ["doc-1:00001", "doc-1:00003"]
    assert result.metadata.score_filtered == 1


def test_missing_or_unreliable_scores_are_not_filtered() -> None:
    optimizer = ContextOptimizationService(
        make_settings(
            context_optimization_enabled=True,
            context_min_score=0.5,
        )
    )
    chunks = [
        _chunk(chunk_id="doc-1:00001", text="zero score", score=0.0),
        _chunk(chunk_id="doc-1:00002", text="bm25 score", score=12.5),
    ]

    result = optimizer.optimize(chunks)

    assert len(result.chunks) == 2
    assert result.metadata.score_filtered == 0


def test_enabled_enforces_max_chunk_limit() -> None:
    optimizer = ContextOptimizationService(
        make_settings(
            context_optimization_enabled=True,
            context_max_chunks=2,
            context_max_tokens=6000,
        )
    )
    chunks = [
        _chunk(chunk_id=f"doc-1:{index:05d}", text=f"chunk {index}", score=0.9 - index * 0.01)
        for index in range(4)
    ]

    result = optimizer.optimize(chunks)

    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_id == "doc-1:00000"
    assert result.chunks[1].chunk_id == "doc-1:00001"
    assert result.metadata.max_chunks_truncated == 2
    assert result.removed_count == 2


def test_enabled_enforces_token_budget_without_splitting_chunks() -> None:
    optimizer = ContextOptimizationService(
        make_settings(
            context_optimization_enabled=True,
            context_max_chunks=8,
            context_max_tokens=120,
        )
    )
    chunks = [
        _chunk(chunk_id="doc-1:00001", text="short chunk", score=0.95),
        _chunk(
            chunk_id="doc-1:00002",
            text="another chunk with more words than the budget allows after first",
            score=0.90,
        ),
        _chunk(chunk_id="doc-1:00003", text="third chunk", score=0.85),
    ]

    result = optimizer.optimize(chunks)

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "doc-1:00001"
    assert result.metadata.token_budget_truncated == 2
    assert result.estimated_tokens <= 120 or estimate_chunk_prompt_tokens(result.chunks[0]) > 120


def test_oversized_first_chunk_is_kept_whole() -> None:
    optimizer = ContextOptimizationService(
        make_settings(
            context_optimization_enabled=True,
            context_max_chunks=8,
            context_max_tokens=10,
        )
    )
    large_text = "word " * 200
    chunks = [_chunk(chunk_id="doc-1:00001", text=large_text, score=0.95)]

    result = optimizer.optimize(chunks)

    assert len(result.chunks) == 1
    assert result.chunks[0].text == large_text


def test_metadata_is_preserved_on_selected_chunks(
    optimizer_enabled: ContextOptimizationService,
) -> None:
    chunk = _chunk(
        chunk_id="doc-9:00004",
        text="metadata rich chunk",
        document_id="doc-9",
        filename="notes.md",
        file_type="markdown",
        page_number=7,
        section="Methods",
        chunk_index=4,
        score=0.87,
    )

    result = optimizer_enabled.optimize([chunk])
    selected = result.chunks[0]

    assert selected.chunk_id == chunk.chunk_id
    assert selected.document_id == "doc-9"
    assert selected.filename == "notes.md"
    assert selected.file_type == "markdown"
    assert selected.page_number == 7
    assert selected.section == "Methods"
    assert selected.chunk_index == 4
    assert selected.source == "notes.md"
    assert selected.score == 0.87


def test_optimizer_does_not_modify_chunk_content(
    optimizer_enabled: ContextOptimizationService,
) -> None:
    original = _chunk(chunk_id="doc-1:00001", text="immutable content", score=0.9)

    result = optimizer_enabled.optimize([original])

    assert result.chunks[0].text == "immutable content"
    assert result.chunks[0] == original


def test_empty_retrieval_results(optimizer_enabled: ContextOptimizationService) -> None:
    result = optimizer_enabled.optimize([])

    assert result.chunks == []
    assert result.removed_count == 0
    assert result.estimated_tokens == 0


def test_multi_document_chunks_preserve_order(
    optimizer_enabled: ContextOptimizationService,
) -> None:
    chunks = [
        _chunk(
            chunk_id="doc-a:00001",
            text="project atlas timeline",
            document_id="doc-a",
            filename="a.pdf",
            score=0.93,
        ),
        _chunk(
            chunk_id="doc-b:00001",
            text="project atlas budget",
            document_id="doc-b",
            filename="b.pdf",
            score=0.88,
        ),
    ]

    result = optimizer_enabled.optimize(chunks)

    assert [chunk.document_id for chunk in result.chunks] == ["doc-a", "doc-b"]


def test_rag_citations_match_optimized_chunks() -> None:
    retrieval = MagicMock()
    llm = MagicMock()
    llm.generate.return_value = "Answer [S1]"
    optimizer = ContextOptimizationService(
        make_settings(
            context_optimization_enabled=True,
            context_max_chunks=1,
            context_max_tokens=6000,
        )
    )
    rag = RAGService(
        retrieval_service=retrieval,
        llm_service=llm,
        context_optimizer=optimizer,
    )
    retrieval.retrieve.return_value = [
        _chunk(
            chunk_id="doc-1:00001",
            text="keep me",
            filename="keep.pdf",
            page_number=2,
            score=0.95,
        ),
        _chunk(
            chunk_id="doc-1:00002",
            text="drop me",
            filename="drop.pdf",
            page_number=3,
            score=0.50,
        ),
    ]

    result = rag.answer("question")

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.label == "S1"
    assert citation.chunk_id == "doc-1:00001"
    assert citation.filename == "keep.pdf"
    assert citation.page_number == 2
    assert citation.document_id == "doc-1"
    prompt = llm.generate.call_args.args[0]
    assert "keep me" in prompt
    assert "drop me" not in prompt


def test_rag_returns_empty_answer_when_optimization_removes_all_chunks() -> None:
    retrieval = MagicMock()
    llm = MagicMock()
    optimizer = ContextOptimizationService(
        make_settings(
            context_optimization_enabled=True,
            context_min_score=0.99,
        )
    )
    rag = RAGService(
        retrieval_service=retrieval,
        llm_service=llm,
        context_optimizer=optimizer,
    )
    retrieval.retrieve.return_value = [
        _chunk(chunk_id="doc-1:00001", text="low relevance", score=0.40),
    ]

    result = rag.answer("question")

    assert "could not find relevant information" in result.answer.lower()
    assert result.citations == []
    llm.generate.assert_not_called()


def test_token_estimation_is_lightweight() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 8) == 2
