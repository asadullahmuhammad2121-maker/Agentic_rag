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

_NUMBERED_LIST_ITEM = re.compile(r"(?:^|\n)\s*\d+[.)]\s+\S", re.MULTILINE)
_LABEL_LIST_ITEM = re.compile(r"(?:^|\n)\s*[A-Za-z][\w\s\-]{0,80}:\s+\S", re.MULTILINE)
_STAGE_COUNT_PATTERN = re.compile(
    r"\b(?:"
    r"one|two|three|four|five|six|seven|eight|nine|ten|\d+"
    r")\s+stages?\b",
    re.IGNORECASE,
)

_WORD_STAGE_COUNTS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_ENUMERATION_QUERY_MARKERS: tuple[str, ...] = (
    "what are the stage",
    "what are all the stage",
    "list all",
    "list the stage",
    "name the stage",
    "name all",
    "identify the stage",
    "identify all",
    "each stage",
    "all stage",
    "every stage",
    "stages of the",
)

_CONTINUATION_QUERY_MARKERS: tuple[str, ...] = (
    " after ",
    " immediately after ",
    " before ",
    " immediately before ",
    " next stage",
    " next stages",
    " following stage",
    " following stages",
    " in order",
    " subsequent stage",
    " subsequent stages",
    " comes after ",
    " comes before ",
    " prior to ",
    " preceding ",
    " follows ",
    " follow ",
)

_REFERENCE_LABEL_AFTER = re.compile(
    r"comes\s+(?:immediately\s+)?after\s+(?:the\s+)?(.+?)(?:\s+stage)?\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_REFERENCE_LABEL_BEFORE = re.compile(
    r"comes\s+(?:immediately\s+)?before\s+(?:the\s+)?(.+?)(?:\s+stage)?\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_REFERENCE_AFTER = re.compile(
    r"(?:immediately\s+)?after\s+(?:the\s+)?(.+?)(?:\s+stage)?\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_REFERENCE_BEFORE = re.compile(
    r"(?:immediately\s+)?before\s+(?:the\s+)?(.+?)(?:\s+stage)?\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_REFERENCE_STARTING_FROM = re.compile(
    r"starting\s+from\s+(?:the\s+)?(.+?)(?:\s+stage)?\s*[?.!]?\s*$",
    re.IGNORECASE,
)

_ADJACENT_NAVIGATION_WINDOW = 2

_FIRST_ITEM_QUERY_MARKERS: tuple[str, ...] = (
    "first stage",
    "initial stage",
    "starting stage",
    "begin with",
)

_BROAD_NAVIGATION_LIMIT = 20


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

    anchor = _select_navigation_anchor(query, retrieval.chunks)
    if anchor is None:
        return None

    arguments: dict[str, str | int] = {
        "document_id": anchor.document_id,
    }
    if _query_needs_broad_navigation(query):
        arguments["page_number"] = anchor.page_number if anchor.page_number >= 1 else 1
        arguments["limit"] = _BROAD_NAVIGATION_LIMIT
    elif _query_needs_adjacent_navigation(query):
        arguments["chunk_id"] = anchor.chunk_id
        arguments["window"] = _adjacent_window_for_query(query)
    else:
        arguments["chunk_id"] = anchor.chunk_id

    return AgentAction(
        type=AgentActionType.CALL_TOOL,
        tool_name=DOCUMENT_NAVIGATION_TOOL_NAME,
        tool_names=[DOCUMENT_NAVIGATION_TOOL_NAME],
        arguments=arguments,
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

    list_items = _list_item_count_in_text(combined)

    if _query_expects_stage_enumeration(query):
        expected = _expected_stage_count(query)
        if expected is not None and list_items < expected:
            return True
        if expected is None and list_items < 2:
            return True

    if _query_expects_first_item(query) and list_items < 1:
        return True

    if _query_expects_stage_continuation(query):
        reference = _reference_label_from_query(query)
        if reference is not None:
            direction = _adjacent_direction_from_query(query)
            if not _has_adjacent_chunk_in_context(
                retrieval.chunks,
                reference=reference,
                direction=direction,
            ):
                return True
        if list_items <= 1:
            return True
        if reference is not None:
            if _query_expects_immediate_successor(query) and list_items < 2:
                return True
            if _query_expects_ordered_sequence(query) and list_items < 3:
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

    answer_items = _list_item_count_in_text(answer)
    if _query_expects_stage_enumeration(query):
        expected = _expected_stage_count(query)
        if expected is not None:
            return answer_items >= expected
        return answer_items >= 3

    if _query_expects_first_item(query):
        return answer_items >= 1 and not is_insufficient_rag_answer(answer)

    if _query_expects_immediate_successor(query):
        return answer_items >= 2

    if _query_expects_ordered_sequence(query):
        return answer_items >= 3

    return False


def _query_expects_stage_enumeration(query: str) -> bool:
    normalized = query.casefold()
    if _STAGE_COUNT_PATTERN.search(normalized):
        return True
    return any(marker in normalized for marker in _ENUMERATION_QUERY_MARKERS)


def _query_expects_first_item(query: str) -> bool:
    normalized = query.casefold()
    return any(marker in normalized for marker in _FIRST_ITEM_QUERY_MARKERS)


def _query_expects_stage_continuation(query: str) -> bool:
    normalized = query.casefold()
    return any(marker in normalized for marker in _CONTINUATION_QUERY_MARKERS)


def _query_expects_immediate_successor(query: str) -> bool:
    normalized = query.casefold()
    return (
        "immediately after" in normalized
        or "next stage" in normalized
        or " comes after " in normalized
        or " follows " in normalized
        or " follow " in normalized
    )


def _query_expects_immediate_predecessor(query: str) -> bool:
    normalized = query.casefold()
    return (
        "immediately before" in normalized
        or " comes before " in normalized
        or " prior to " in normalized
        or " preceding " in normalized
    )


def _query_expects_ordered_sequence(query: str) -> bool:
    normalized = query.casefold()
    return "in order" in normalized or "next stages" in normalized


def _query_needs_broad_navigation(query: str) -> bool:
    return (
        _query_expects_stage_enumeration(query)
        or _query_expects_first_item(query)
        or _query_expects_ordered_sequence(query)
    )


def _query_needs_adjacent_navigation(query: str) -> bool:
    return (
        _query_expects_immediate_successor(query)
        or _query_expects_immediate_predecessor(query)
    )


def _adjacent_window_for_query(query: str) -> int:
    normalized = query.casefold()
    if re.search(r"\b(?:two|2)\s+stages?\b", normalized):
        return 3
    return _ADJACENT_NAVIGATION_WINDOW


def _adjacent_direction_from_query(query: str) -> str:
    if _query_expects_immediate_predecessor(query):
        return "before"
    return "after"


def _expected_stage_count(query: str) -> int | None:
    normalized = query.casefold()
    match = _STAGE_COUNT_PATTERN.search(normalized)
    if match is None:
        return None
    token = match.group(0).split()[0]
    if token.isdigit():
        return int(token)
    return _WORD_STAGE_COUNTS.get(token)


def _reference_label_from_query(query: str) -> str | None:
    normalized = query.strip()
    for pattern in (
        _REFERENCE_LABEL_AFTER,
        _REFERENCE_LABEL_BEFORE,
        _REFERENCE_AFTER,
        _REFERENCE_BEFORE,
        _REFERENCE_STARTING_FROM,
    ):
        match = pattern.search(normalized)
        if match is None:
            continue
        label = match.group(1).strip(" .?,!")
        if label and label.casefold() not in {"the", "a", "an"}:
            return _normalize_reference_label(label)
    return None


def _normalize_reference_label(label: str) -> str:
    lowered = label.casefold()
    for separator in (" in the ", " in ", " of the ", " of "):
        if separator in lowered:
            index = lowered.index(separator)
            return label[:index].strip()
    return label.strip()


def _stage_anchor_from_query(query: str) -> str | None:
    return _reference_label_from_query(query)


def _list_item_count_in_text(text: str) -> int:
    numbered = len(_NUMBERED_LIST_ITEM.findall(text))
    labels = len(_LABEL_LIST_ITEM.findall(text))
    return max(numbered, labels)


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


def _select_navigation_anchor(
    query: str,
    chunks: list[RetrievedChunkOutput],
) -> RetrievedChunkOutput | None:
    candidates = [chunk for chunk in chunks if chunk.document_id.strip() and chunk.chunk_id.strip()]
    if not candidates:
        return None

    reference = _reference_label_from_query(query)
    if reference is not None:
        matched = _best_chunk_for_reference(candidates, reference)
        if matched is not None:
            return matched

    if _query_expects_first_item(query):
        return min(candidates, key=lambda chunk: chunk.chunk_index)

    if _query_expects_stage_enumeration(query) or _query_expects_ordered_sequence(query):
        return max(
            candidates,
            key=lambda chunk: (_list_item_count_in_text(chunk.text), -chunk.chunk_index),
        )

    return max(candidates, key=lambda chunk: chunk.score)


def _best_chunk_for_reference(
    chunks: Sequence[RetrievedChunkOutput],
    reference: str,
) -> RetrievedChunkOutput | None:
    reference_cf = reference.casefold()

    def rank(chunk: RetrievedChunkOutput) -> tuple[int, float, int]:
        text_cf = chunk.text.casefold()
        if f"{reference_cf}:" in text_cf:
            return (3, chunk.score, -chunk.chunk_index)
        if reference_cf in text_cf:
            return (2, chunk.score, -chunk.chunk_index)
        return (0, chunk.score, -chunk.chunk_index)

    best = max(chunks, key=rank)
    if rank(best)[0] > 0:
        return best
    return None


def _chunk_matches_reference(text: str, reference: str) -> bool:
    reference_cf = reference.casefold()
    text_cf = text.casefold()
    return f"{reference_cf}:" in text_cf or reference_cf in text_cf


def _has_adjacent_chunk_in_context(
    chunks: Sequence[RetrievedChunkOutput],
    *,
    reference: str,
    direction: str,
) -> bool:
    matching = [chunk for chunk in chunks if _chunk_matches_reference(chunk.text, reference)]
    if not matching:
        return False

    anchor_index = min(chunk.chunk_index for chunk in matching)
    offset = -1 if direction == "before" else 1
    target_index = anchor_index + offset
    return any(chunk.chunk_index == target_index for chunk in chunks)
