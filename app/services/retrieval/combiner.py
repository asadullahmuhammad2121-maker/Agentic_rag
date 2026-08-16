"""Combine and deduplicate multi-query retrieval results."""

from __future__ import annotations

from app.services.retrieval.service import RetrievedChunk


def combine_retrieved_chunks(
    chunk_groups: list[list[RetrievedChunk]],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    """
    Merge retrieval results, keeping the highest score per ``chunk_id``.

    Results are sorted by score descending and truncated to ``limit``.
    """
    best_by_chunk_id: dict[str, RetrievedChunk] = {}
    for chunks in chunk_groups:
        for chunk in chunks:
            existing = best_by_chunk_id.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best_by_chunk_id[chunk.chunk_id] = chunk

    combined = sorted(best_by_chunk_id.values(), key=lambda item: item.score, reverse=True)
    return combined[:limit]
