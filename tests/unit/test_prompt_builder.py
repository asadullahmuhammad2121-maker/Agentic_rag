"""Unit tests for prompt builder and citations."""

from __future__ import annotations

from app.services.rag.prompt_builder import PromptBuilder
from app.services.retrieval.service import RetrievedChunk


def _chunk(**overrides: object) -> RetrievedChunk:
    base = {
        "chunk_id": "c1",
        "text": "Relevant text about cats.",
        "document_id": "doc-1",
        "filename": "pets.pdf",
        "page_number": 3,
        "chunk_index": 0,
        "score": 0.88,
    }
    base.update(overrides)
    return RetrievedChunk(**base)  # type: ignore[arg-type]


def test_prompt_construction_includes_context_and_question() -> None:
    built = PromptBuilder().build("What about cats?", [_chunk()])
    assert "What about cats?" in built.user_prompt
    assert "[S1]" in built.user_prompt
    assert "pets.pdf" in built.user_prompt
    assert "Relevant text about cats." in built.user_prompt
    assert (
        "only the context" in built.system_prompt.lower()
        or "only the provided context" in built.system_prompt.lower()
    )
    assert built.context_chunk_count == 1


def test_prompt_construction_handles_empty_context() -> None:
    built = PromptBuilder().build("Anything?", [])
    assert "No relevant context" in built.user_prompt
    assert "do not have enough information" in built.user_prompt.lower()
    assert built.context_chunk_count == 0
