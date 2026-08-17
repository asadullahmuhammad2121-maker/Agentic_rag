"""Agent query routes."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import AgentServiceDep
from app.core.logging import get_logger
from app.schemas.agent import (
    AgentActionResponse,
    AgentObservationResponse,
    AgentQueryRequest,
    AgentQueryResponse,
    AgentStepResponse,
)
from app.schemas.query import CitationResponse
from app.services.agent.models import AgentActionType, AgentRunResult, AgentStep

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
) -> AgentQueryResponse:
    """Run Phase 3E agent: route → retrieve/search → generate → answer."""
    retrieval_filters = body.build_retrieval_filters()
    logger.info(
        "agent_query_request_received",
        extra={
            "operation": "agent_query",
            "query_length": len(body.query),
            "top_k": body.top_k,
            "has_filters": retrieval_filters is not None,
        },
    )
    result = agent_service.run(
        body.query,
        top_k=body.top_k,
        filters=retrieval_filters,
    )
    return _to_response(result)


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
        observation = AgentObservationResponse(
            tool_name=step.observation.tool_name,
            success=step.observation.success,
            citation_count=len(step.observation.citations),
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
