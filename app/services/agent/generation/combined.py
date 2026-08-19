"""Merge multi-tool outputs into a unified retrieval context."""

from __future__ import annotations

from app.services.agent.generation.calculator import format_calculator_evidence
from app.services.agent.models import (
    CalculatorOutput,
    RAGRetrievalOutput,
    TavilySearchOutput,
    WebSearchResultItem,
)
from app.services.agent.tools.converters import output_to_chunk
from app.services.retrieval.service import RetrievedChunk


def merge_tool_outputs_to_chunks(
    *,
    rag_output: RAGRetrievalOutput | None,
    web_output: TavilySearchOutput | None,
    calculator_output: CalculatorOutput | None = None,
) -> list[RetrievedChunk]:
    """Combine RAG chunks, web results, and calculator evidence into one context."""
    chunks: list[RetrievedChunk] = []
    if rag_output is not None:
        chunks.extend(output_to_chunk(chunk) for chunk in rag_output.chunks)
    if web_output is not None:
        offset = len(chunks)
        for index, result in enumerate(web_output.results):
            chunks.append(_web_result_to_chunk(result, chunk_index=offset + index))
    if calculator_output is not None:
        chunks.append(_calculator_to_chunk(calculator_output, chunk_index=len(chunks)))
    return chunks


def _web_result_to_chunk(result: WebSearchResultItem, *, chunk_index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=result.url,
        text=result.content or result.title,
        document_id=result.url,
        filename=result.title,
        file_type="web",
        source=result.url,
        page_number=0,
        section=None,
        chunk_index=chunk_index,
        chunking_strategy="web",
        score=result.score if result.score is not None else 0.0,
    )


def _calculator_to_chunk(output: CalculatorOutput, *, chunk_index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"calculator:{chunk_index}",
        text=format_calculator_evidence(output),
        document_id="calculator",
        filename="Calculator Result",
        file_type="calculator",
        source="calculator",
        page_number=0,
        section=None,
        chunk_index=chunk_index,
        chunking_strategy="calculator",
        score=1.0,
    )
