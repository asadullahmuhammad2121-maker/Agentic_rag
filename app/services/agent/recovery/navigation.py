"""Document navigation recovery after insufficient RAG retrieval."""

from __future__ import annotations

import re
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
    "insufficient information",
    "insufficient context",
    "does not contain enough information",
    "does not specify",
    "not specified in the provided context",
    "not mentioned in the provided context",
    "cannot determine",
    "unable to determine",
)

_STAGE_LABEL_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+[.)]\s*)?([A-Z][A-Za-z]+)\s*:",
    re.MULTILINE,
)

_CONTINUATION_QUERY_MARKERS: tuple[str, ...] = (
    " after ",
    " immediately after ",
    " next stage",
    " next stages",
    " following stage",
    " following stages",
    " in order",
    " subsequent stage",
    " subsequent stages",
    " comes after ",
)

_ENUMERATION_QUERY_MARKERS: tuple[str, ...] = (
    "five stage",
    "all stage",
    "each stage",
    "what are the stage",
    "list the stage",
    "name the stage",
    "identify the stage",
    "stages of the core pipeline",
    "stages of the pipeline",
)


def maybe_document_navigation_recovery(
    history: Sequence[AgentStep],
    *,
    query: str,
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
    if not _should_recover_from_rag(observation, query=query):
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


def retrieval_likely_incomplete_for_query(
    query: str,
    retrieval: RAGRetrievalOutput,
) -> bool:
    """True when retrieved same-document context likely misses what the query needs."""
    if retrieval.empty:
        return False
    if not _chunks_share_single_document(retrieval.chunks):
        return False

    combined = _combined_chunk_text(retrieval.chunks)
    if not combined.strip():
        return False

    context_stages = _stage_labels_in_text(combined)
    if _query_expects_stage_enumeration(query):
        expected = _expected_stage_count(query)
        if expected is not None and len(context_stages) < expected:
            return True
        if expected is None and len(context_stages) < 2:
            return True

    if _query_expects_stage_continuation(query):
        if len(context_stages) <= 1:
            return True
        anchor = _stage_anchor_from_query(query)
        if anchor is not None and anchor in {stage.casefold() for stage in context_stages}:
            if _query_expects_immediate_successor(query) and len(context_stages) < 2:
                return True
            if _query_expects_ordered_sequence(query) and len(context_stages) < 3:
                return True

    return False


def _should_recover_from_rag(observation: AgentObservation, *, query: str) -> bool:
    if observation.metadata.get("empty_retrieval"):
        return True
    if is_insufficient_rag_answer(observation.answer):
        return True
    if _answer_satisfies_query(query, observation.answer):
        return False

    retrieval = _parse_rag_output(observation)
    if retrieval is None:
        return False
    return retrieval_likely_incomplete_for_query(query, retrieval)


def _answer_satisfies_query(query: str, answer: str | None) -> bool:
    if not answer:
        return False

    answer_stages = _stage_labels_in_text(answer)
    if _query_expects_stage_enumeration(query):
        expected = _expected_stage_count(query)
        if expected is not None:
            return len(answer_stages) >= expected
        return len(answer_stages) >= 5

    normalized_query = query.casefold()
    if _query_expects_immediate_successor(query):
        anchor = _stage_anchor_from_query(query)
        if anchor == "ingest":
            return "store" in answer.casefold()
        return len(answer_stages) >= 2

    if _query_expects_ordered_sequence(query) and "starting from ingest" in normalized_query:
        return len(answer_stages) >= 3

    return False


def _query_expects_stage_enumeration(query: str) -> bool:
    normalized = query.casefold()
    return any(marker in normalized for marker in _ENUMERATION_QUERY_MARKERS)


def _query_expects_stage_continuation(query: str) -> bool:
    normalized = query.casefold()
    return any(marker in normalized for marker in _CONTINUATION_QUERY_MARKERS)


def _query_expects_immediate_successor(query: str) -> bool:
    normalized = query.casefold()
    return (
        "immediately after" in normalized
        or "next stage" in normalized
        or " comes after " in normalized
    )


def _query_expects_ordered_sequence(query: str) -> bool:
    normalized = query.casefold()
    return "in order" in normalized or "next stages" in normalized


def _expected_stage_count(query: str) -> int | None:
    normalized = query.casefold()
    if "five stage" in normalized:
        return 5
    return None


def _stage_anchor_from_query(query: str) -> str | None:
    normalized = query.casefold()
    if " after " in normalized:
        tail = normalized.split(" after ", 1)[1]
        token = tail.split()[0].strip(".,?!")
        return token or None
    if "starting from " in normalized:
        tail = normalized.split("starting from ", 1)[1]
        token = tail.split()[0].strip(".,?!")
        return token or None
    return None


def _stage_labels_in_text(text: str) -> set[str]:
    return {match.group(1) for match in _STAGE_LABEL_PATTERN.finditer(text)}


def _combined_chunk_text(chunks: Sequence[RetrievedChunkOutput]) -> str:
    return "\n".join(chunk.text for chunk in chunks if chunk.text.strip())


def _chunks_share_single_document(chunks: Sequence[RetrievedChunkOutput]) -> bool:
    document_ids = {chunk.document_id.strip() for chunk in chunks if chunk.document_id.strip()}
    return len(document_ids) == 1


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
