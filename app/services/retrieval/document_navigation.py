"""Metadata-based navigation around ingested document chunks."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.retrieval.chunk_mapping import payload_to_retrieved_chunk
from app.services.retrieval.models import RetrievedChunk
from app.vector_store.base import VectorStore

logger = get_logger(__name__)

DEFAULT_WINDOW = 2
DEFAULT_MAX_CHUNKS = 10
MAX_WINDOW = 5
MAX_LIMIT = 20


@dataclass(frozen=True, slots=True)
class DocumentNavigationResult:
    """Neighboring chunks resolved from stored document metadata."""

    document_id: str
    anchor_chunk_id: str | None
    anchor_chunk_index: int | None
    chunks: list[RetrievedChunk]


class DocumentNavigationService:
    """Retrieve nearby chunks from the same document using payload metadata."""

    def __init__(self, settings: Settings, vector_store: VectorStore) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._collection = settings.qdrant_collection_name

    def navigate(
        self,
        *,
        document_id: str,
        chunk_id: str | None = None,
        chunk_index: int | None = None,
        page_number: int | None = None,
        window: int | None = None,
        limit: int | None = None,
    ) -> DocumentNavigationResult:
        normalized_document_id = document_id.strip()
        if not normalized_document_id:
            return DocumentNavigationResult(
                document_id="",
                anchor_chunk_id=None,
                anchor_chunk_index=None,
                chunks=[],
            )

        resolved_window = _clamp(window if window is not None else DEFAULT_WINDOW, 0, MAX_WINDOW)
        resolved_limit = _clamp(limit if limit is not None else DEFAULT_MAX_CHUNKS, 1, MAX_LIMIT)

        if chunk_id is not None or chunk_index is not None:
            anchor = self._resolve_anchor(
                document_id=normalized_document_id,
                chunk_id=chunk_id.strip() if chunk_id else None,
                chunk_index=chunk_index,
            )
            if anchor is None:
                logger.info(
                    "document_navigation_anchor_missing",
                    extra={
                        "operation": "document_navigation",
                        "document_id": normalized_document_id,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                    },
                )
                return DocumentNavigationResult(
                    document_id=normalized_document_id,
                    anchor_chunk_id=chunk_id.strip() if chunk_id else None,
                    anchor_chunk_index=chunk_index,
                    chunks=[],
                )

            chunks = self._neighbor_chunks(
                document_id=normalized_document_id,
                anchor_index=anchor.chunk_index,
                window=resolved_window,
                limit=resolved_limit,
            )
            logger.info(
                "document_navigation_completed",
                extra={
                    "operation": "document_navigation",
                    "document_id": normalized_document_id,
                    "anchor_chunk_index": anchor.chunk_index,
                    "result_count": len(chunks),
                },
            )
            return DocumentNavigationResult(
                document_id=normalized_document_id,
                anchor_chunk_id=anchor.chunk_id,
                anchor_chunk_index=anchor.chunk_index,
                chunks=chunks,
            )

        if page_number is not None:
            chunks = self._page_chunks(
                document_id=normalized_document_id,
                page_number=page_number,
                limit=resolved_limit,
            )
            logger.info(
                "document_navigation_completed",
                extra={
                    "operation": "document_navigation",
                    "document_id": normalized_document_id,
                    "page_number": page_number,
                    "result_count": len(chunks),
                },
            )
            return DocumentNavigationResult(
                document_id=normalized_document_id,
                anchor_chunk_id=None,
                anchor_chunk_index=None,
                chunks=chunks,
            )

        return DocumentNavigationResult(
            document_id=normalized_document_id,
            anchor_chunk_id=None,
            anchor_chunk_index=None,
            chunks=[],
        )

    def _resolve_anchor(
        self,
        *,
        document_id: str,
        chunk_id: str | None,
        chunk_index: int | None,
    ) -> RetrievedChunk | None:
        if chunk_id:
            hits = self._vector_store.find_by_payload(
                self._collection,
                {"document_id": document_id, "chunk_id": chunk_id},
                limit=1,
            )
            if hits:
                return _to_chunk(hits[0].id, hits[0].payload, document_id=document_id)
        if chunk_index is not None:
            hits = self._vector_store.find_by_payload(
                self._collection,
                {"document_id": document_id, "chunk_index": chunk_index},
                limit=1,
            )
            if hits:
                return _to_chunk(hits[0].id, hits[0].payload, document_id=document_id)
        return None

    def _neighbor_chunks(
        self,
        *,
        document_id: str,
        anchor_index: int,
        window: int,
        limit: int,
    ) -> list[RetrievedChunk]:
        start_index = max(0, anchor_index - window)
        end_index = anchor_index + window
        chunks: list[RetrievedChunk] = []
        for index in range(start_index, end_index + 1):
            hits = self._vector_store.find_by_payload(
                self._collection,
                {"document_id": document_id, "chunk_index": index},
                limit=1,
            )
            if not hits:
                continue
            chunk = _to_chunk(hits[0].id, hits[0].payload, document_id=document_id)
            if chunk is not None:
                chunks.append(chunk)
        chunks.sort(key=lambda item: item.chunk_index)
        return _limit_around_anchor(chunks, anchor_index=anchor_index, limit=limit)

    def _page_chunks(
        self,
        *,
        document_id: str,
        page_number: int,
        limit: int,
    ) -> list[RetrievedChunk]:
        hits = self._vector_store.find_by_payload(
            self._collection,
            {"document_id": document_id, "page_number": page_number},
            limit=limit,
        )
        chunks = [
            chunk
            for hit in hits
            if (chunk := _to_chunk(hit.id, hit.payload, document_id=document_id)) is not None
        ]
        chunks.sort(key=lambda item: item.chunk_index)
        return chunks[:limit]


def _to_chunk(
    point_id: str,
    payload: dict[str, object],
    *,
    document_id: str,
) -> RetrievedChunk | None:
    payload_document_id = str(payload.get("document_id", ""))
    if payload_document_id != document_id:
        return None
    return payload_to_retrieved_chunk(point_id, 1.0, payload)


def _limit_around_anchor(
    chunks: list[RetrievedChunk],
    *,
    anchor_index: int,
    limit: int,
) -> list[RetrievedChunk]:
    if len(chunks) <= limit:
        return chunks

    anchor_pos = next(
        (index for index, chunk in enumerate(chunks) if chunk.chunk_index == anchor_index),
        len(chunks) // 2,
    )
    selected: list[RetrievedChunk] = [chunks[anchor_pos]]
    left = anchor_pos - 1
    right = anchor_pos + 1
    while len(selected) < limit and (left >= 0 or right < len(chunks)):
        if left >= 0:
            selected.append(chunks[left])
            left -= 1
            if len(selected) >= limit:
                break
        if right < len(chunks):
            selected.append(chunks[right])
            right += 1
    selected.sort(key=lambda item: item.chunk_index)
    return selected[:limit]


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))
