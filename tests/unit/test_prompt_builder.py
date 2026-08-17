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
        "file_type": "pdf",
        "source": "pets.pdf",
        "page_number": 3,
        "section": None,
        "chunk_index": 0,
        "chunking_strategy": "fixed",
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


def test_combined_prompt_preserves_source_labels_across_sections() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="doc-1:00000",
            text="Document chunk",
            document_id="doc-1",
            filename="guide.pdf",
            file_type="pdf",
            source="guide.pdf",
            page_number=1,
            section=None,
            chunk_index=0,
            chunking_strategy="fixed",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="https://example.com/news",
            text="Web chunk",
            document_id="https://example.com/news",
            filename="News",
            file_type="web",
            source="https://example.com/news",
            page_number=0,
            section=None,
            chunk_index=1,
            chunking_strategy="web",
            score=0.8,
        ),
    ]
    built = PromptBuilder().build_combined("Compare docs and web", chunks)
    assert "[S1]" in built.user_prompt
    assert "[S2]" in built.user_prompt
    assert "guide.pdf" in built.user_prompt
    assert "https://example.com/news" in built.user_prompt
