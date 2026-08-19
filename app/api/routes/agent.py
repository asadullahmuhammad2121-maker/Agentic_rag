"""Agent query routes."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Query, status

from app.api.deps import AgentRunStoreDep, AgentServiceDep
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.schemas.agent import (
    AgentActionResponse,
    AgentObservationResponse,
    AgentQueryRequest,
    AgentQueryResponse,
    AgentStepResponse,
)
from app.schemas.agent_runs import (
    AgentRunDetailResponse,
    AgentRunListResponse,
    AgentRunSummaryResponse,
)
from app.schemas.query import CitationResponse
from app.services.agent.models import AgentActionType, AgentRunResult, AgentStep
from app.services.agent.runs.store import AgentRunDetail, AgentRunStore, AgentRunSummary

logger = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/query",
    response_model=AgentQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask the agent to answer using registered tools",
)
def agent_query(
    body: AgentQueryRequest,
    agent_service: AgentServiceDep,
    run_store: AgentRunStoreDep,
) -> AgentQueryResponse:
    """Run Phase 3E agent: route → retrieve/search → generate → answer."""
    retrieval_filters = body.build_retrieval_filters()
    run_id = run_store.create_run_id()
    started_at = datetime.now(UTC)
    started_perf = time.perf_counter()
    logger.info(
        "agent_query_request_received",
        extra={
            "operation": "agent_query",
            "run_id": run_id,
            "query_length": len(body.query),
            "top_k": body.top_k,
            "has_filters": retrieval_filters is not None,
        },
    )
    try:
        result = agent_service.run(
            body.query,
            top_k=body.top_k,
            filters=retrieval_filters,
        )
        response = _to_response(result)
        _persist_success(
            run_store,
            run_id=run_id,
            query=body.query.strip(),
            started_at=started_at,
            started_perf=started_perf,
            response=response,
        )
        return response
    except AppError as exc:
        _persist_failure(
            run_store,
            run_id=run_id,
            query=body.query.strip(),
            started_at=started_at,
            started_perf=started_perf,
            error_message=exc.message,
            error_code=exc.code,
        )
        raise


@router.get(
    "/runs",
    response_model=AgentRunListResponse,
    status_code=status.HTTP_200_OK,
    summary="List persisted agent runs",
)
def list_agent_runs(
    run_store: AgentRunStoreDep,
    search: str | None = Query(default=None, max_length=500),
    status_filter: Literal["success", "failure"] | None = Query(default=None, alias="status"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> AgentRunListResponse:
    """Return paginated agent run history."""
    page = run_store.list_runs(
        search=search,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    return AgentRunListResponse(
        runs=[_summary_to_response(item) for item in page.runs],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/runs/{run_id}",
    response_model=AgentRunDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a persisted agent run by ID",
)
def get_agent_run(
    run_id: str,
    run_store: AgentRunStoreDep,
) -> AgentRunDetailResponse:
    """Return one stored agent run including steps and citations."""
    record = run_store.get_run(run_id)
    if record is None:
        raise AppError(
            "Agent run not found",
            code="agent_run_not_found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"run_id": run_id},
        )
    return _detail_to_response(record)


def _persist_success(
    run_store: AgentRunStore,
    *,
    run_id: str,
    query: str,
    started_at: datetime,
    started_perf: float,
    response: AgentQueryResponse,
) -> None:
    try:
        run_store.save_success(
            run_id=run_id,
            query=query,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            duration_ms=_duration_ms(started_perf),
            response_payload=response.model_dump(mode="json"),
        )
    except Exception:
        logger.exception(
            "agent_run_persist_failed",
            extra={"operation": "agent_query", "run_id": run_id, "status": "success"},
        )


def _persist_failure(
    run_store: AgentRunStore,
    *,
    run_id: str,
    query: str,
    started_at: datetime,
    started_perf: float,
    error_message: str,
    error_code: str,
) -> None:
    try:
        run_store.save_failure(
            run_id=run_id,
            query=query,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            duration_ms=_duration_ms(started_perf),
            error_message=error_message,
            error_code=error_code,
        )
    except Exception:
        logger.exception(
            "agent_run_persist_failed",
            extra={"operation": "agent_query", "run_id": run_id, "status": "failure"},
        )


def _duration_ms(started_perf: float) -> int:
    return int((time.perf_counter() - started_perf) * 1000)


def _to_response(result: AgentRunResult) -> AgentQueryResponse:
    metadata = dict(result.metadata)
    metadata["citation_count"] = len(result.citations)
    return AgentQueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                document_id=citation.document_id,
                filename=citation.filename,
                file_type=citation.file_type,
                source=citation.source,
                page_number=citation.page_number,
                section=citation.section,
                chunk_index=citation.chunk_index,
                chunk_id=citation.chunk_id,
                score=citation.score,
                label=citation.label,
            )
            for citation in result.citations
        ],
        tool_used=result.tool_used,
        steps=[_to_step_response(step) for step in result.steps],
        metadata=metadata,
    )


def _to_step_response(step: AgentStep) -> AgentStepResponse:
    observation = None
    if step.observation is not None:
        metadata = step.observation.metadata
        expression = metadata.get("expression")
        result = metadata.get("result")
        observation = AgentObservationResponse(
            tool_name=step.observation.tool_name,
            success=step.observation.success,
            citation_count=len(step.observation.citations),
            expression=expression if isinstance(expression, str) else None,
            result=float(result) if isinstance(result, (int, float)) else None,
        )
    if step.action.type is AgentActionType.FINISH:
        action_type = "finish"
    elif step.action.type is AgentActionType.EXECUTE_PLAN:
        action_type = "execute_plan"
    elif step.action.type is AgentActionType.CALL_TOOLS:
        action_type = "call_tools"
    else:
        action_type = "call_tool"
    return AgentStepResponse(
        action=AgentActionResponse(
            type=action_type,
            tool_name=step.action.tool_name,
            tool_names=list(step.action.tool_names),
            reasoning=step.action.reasoning,
        ),
        observation=observation,
    )


def _summary_to_response(record: AgentRunSummary) -> AgentRunSummaryResponse:
    return AgentRunSummaryResponse(
        run_id=record.run_id,
        query=record.query,
        status=record.status,
        started_at=record.started_at,
        completed_at=record.completed_at,
        duration_ms=record.duration_ms,
        tool_used=record.tool_used,
        step_count=record.step_count,
        citation_count=record.citation_count,
        error_message=record.error_message,
        error_code=record.error_code,
    )


def _detail_to_response(record: AgentRunDetail) -> AgentRunDetailResponse:
    return AgentRunDetailResponse(
        run_id=record.run_id,
        query=record.query,
        status=record.status,
        started_at=record.started_at,
        completed_at=record.completed_at,
        duration_ms=record.duration_ms,
        tool_used=record.tool_used,
        step_count=record.step_count,
        citation_count=record.citation_count,
        error_message=record.error_message,
        error_code=record.error_code,
        answer=record.answer,
        citations=[CitationResponse.model_validate(item) for item in (record.citations or [])],
        steps=[AgentStepResponse.model_validate(item) for item in (record.steps or [])],
        metadata=dict(record.metadata or {}),
    )
