"""Rank fusion utilities for hybrid retrieval."""

from __future__ import annotations

from app.services.retrieval.service import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    weights: list[float],
    limit: int,
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Fuse multiple ranked result lists using weighted Reciprocal Rank Fusion.

    Chunks are deduplicated by ``chunk_id``. The fused score replaces the
    original retrieval score on returned chunks.
    """
    if limit <= 0:
        return []

    fused_scores: dict[str, float] = {}
    best_chunks: dict[str, RetrievedChunk] = {}

    for weight, results in zip(weights, ranked_lists, strict=True):
        if weight <= 0.0 or not results:
            continue
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + weight / (k + rank)
            existing = best_chunks.get(chunk_id)
            if existing is None or chunk.score > existing.score:
                best_chunks[chunk_id] = chunk

    if not fused_scores:
        return []

    sorted_ids = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
    fused: list[RetrievedChunk] = []
    for chunk_id in sorted_ids[:limit]:
        chunk = best_chunks[chunk_id]
        fused.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                document_id=chunk.document_id,
                filename=chunk.filename,
                file_type=chunk.file_type,
                source=chunk.source,
                page_number=chunk.page_number,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                chunking_strategy=chunk.chunking_strategy,
                score=fused_scores[chunk_id],
            )
        )
    return fused
