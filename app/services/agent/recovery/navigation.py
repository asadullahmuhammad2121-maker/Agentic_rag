"""Document navigation recovery after insufficient RAG retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from app.services.agent.models import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentStep,
    RAGRetrievalOutput,
    RetrievedChunkOutput,
)
from app.services.agent.tools.document_navigation import DOCUMENT_NAVIGATION_TOOL_NAME
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.rag.service import EMPTY_RETRIEVAL_ANSWER

_INSUFFICIENT_ANSWER_MARKERS: tuple[str, ...] = (
    "do not have enough information",
    "don't have enough information",
    "could not find relevant information",
)


def maybe_document_navigation_recovery(
    history: Sequence[AgentStep],
    *,
    tools: ToolRegistry,
) -> AgentAction | None:
    """Return a navigation tool action when RAG context looks incomplete."""
    if DOCUMENT_NAVIGATION_TOOL_NAME not in tools:
        return None
    if navigation_already_attempted(history):
        return None

    rag_step = _last_rag_step(history)
    if rag_step is None:
        return None

    observation = rag_step.observation
    if observation is None or not observation.success:
        return None
    if observation.tool_name != RAG_RETRIEVAL_TOOL_NAME:
        return None
    if observation.metadata.get("multi_tool"):
        return None
    if not _should_recover_from_rag(observation):
        return None

    retrieval = _parse_rag_output(observation)
    if retrieval is None:
        return None

    anchor = _select_navigation_anchor(retrieval.chunks)
    if anchor is None:
        return None

    return AgentAction(
        type=AgentActionType.CALL_TOOL,
        tool_name=DOCUMENT_NAVIGATION_TOOL_NAME,
        tool_names=[DOCUMENT_NAVIGATION_TOOL_NAME],
        arguments={
            "document_id": anchor.document_id,
            "chunk_id": anchor.chunk_id,
        },
        reasoning="Retrieved context appears incomplete; fetching nearby document chunks.",
    )


def navigation_already_attempted(history: Sequence[AgentStep]) -> bool:
    for step in history:
        action = step.action
        if action.tool_name == DOCUMENT_NAVIGATION_TOOL_NAME:
            return True
        if DOCUMENT_NAVIGATION_TOOL_NAME in action.tool_names:
            return True
        observation = step.observation
        if observation is None:
            continue
        if observation.tool_name == DOCUMENT_NAVIGATION_TOOL_NAME:
            return True
        if DOCUMENT_NAVIGATION_TOOL_NAME in observation.tool_names:
            return True
    return False


def prior_rag_observation(history: Sequence[AgentStep]) -> AgentObservation | None:
    rag_step = _last_rag_step(history)
    if rag_step is None:
        return None
    return rag_step.observation


def is_insufficient_rag_answer(answer: str | None) -> bool:
    if not answer:
        return True
    normalized = answer.strip().casefold()
    if not normalized:
        return True
    if normalized == EMPTY_RETRIEVAL_ANSWER.casefold():
        return True
    return any(marker in normalized for marker in _INSUFFICIENT_ANSWER_MARKERS)


def _should_recover_from_rag(observation: AgentObservation) -> bool:
    return observation.metadata.get("empty_retrieval") or is_insufficient_rag_answer(
        observation.answer
    )


def _last_rag_step(history: Sequence[AgentStep]) -> AgentStep | None:
    for step in reversed(history):
        if step.action.tool_name == RAG_RETRIEVAL_TOOL_NAME:
            return step
        if step.action.tool_names == [RAG_RETRIEVAL_TOOL_NAME]:
            return step
    return None


def _parse_rag_output(observation: AgentObservation) -> RAGRetrievalOutput | None:
    if observation.tool_output is None:
        return None
    try:
        return RAGRetrievalOutput.model_validate(observation.tool_output)
    except Exception:
        return None


def _select_navigation_anchor(chunks: list[RetrievedChunkOutput]) -> RetrievedChunkOutput | None:
    candidates = [chunk for chunk in chunks if chunk.document_id.strip() and chunk.chunk_id.strip()]
    if not candidates:
        return None
    return max(candidates, key=lambda chunk: chunk.score)
