"""Tests for agent run persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.services.agent.runs.store import AgentRunStore


@pytest.fixture
def run_store(tmp_path: Path) -> AgentRunStore:
    return AgentRunStore(tmp_path / "agent_runs.db")


def test_save_and_get_success_run(run_store: AgentRunStore) -> None:
    started = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    completed = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)
    payload = {
        "answer": "Done",
        "citations": [{"document_id": "d1", "filename": "a.pdf", "file_type": "pdf", "source": "a.pdf", "page_number": 1, "section": None, "chunk_index": 0, "chunk_id": "c1", "score": 0.9, "label": "S1"}],
        "tool_used": "rag_retrieval",
        "steps": [{"action": {"type": "call_tool", "tool_name": "rag_retrieval", "tool_names": ["rag_retrieval"], "reasoning": "Use RAG"}, "observation": {"tool_name": "rag_retrieval", "success": True, "citation_count": 1}}],
        "metadata": {"step_count": 1, "citation_count": 1, "finished": True},
    }
    run_store.save_success(
        run_id="run-1",
        query="What is RAG?",
        started_at=started,
        completed_at=completed,
        duration_ms=1200,
        response_payload=payload,
    )

    detail = run_store.get_run("run-1")
    assert detail is not None
    assert detail.status == "success"
    assert detail.answer == "Done"
    assert detail.step_count == 1
    assert detail.citation_count == 1
    assert detail.tool_used == "rag_retrieval"
    assert len(detail.steps) == 1


def test_save_and_list_failure_run(run_store: AgentRunStore) -> None:
    started = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    completed = datetime(2026, 1, 1, 12, 0, 2, tzinfo=UTC)
    run_store.save_failure(
        run_id="run-2",
        query="broken query",
        started_at=started,
        completed_at=completed,
        duration_ms=500,
        error_message="Provider failed",
        error_code="provider_error",
    )

    page = run_store.list_runs(status="failure")
    assert page.total == 1
    assert page.runs[0].run_id == "run-2"
    assert page.runs[0].error_code == "provider_error"


def test_list_runs_supports_search(run_store: AgentRunStore) -> None:
    started = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    completed = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)
    for run_id, query in [("a", "vector databases"), ("b", "web search news")]:
        run_store.save_failure(
            run_id=run_id,
            query=query,
            started_at=started,
            completed_at=completed,
            duration_ms=100,
            error_message="x",
            error_code="provider_error",
        )

    page = run_store.list_runs(search="vector")
    assert page.total == 1
    assert page.runs[0].query == "vector databases"
