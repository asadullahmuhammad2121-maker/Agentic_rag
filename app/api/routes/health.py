"""Health check routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Response, status

from app.api.deps import KeywordSearchDep, SettingsDep, VectorStoreDep
from app.core.logging import get_logger
from app.schemas.health import ComponentHealth, HealthResponse

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


def _component_statuses(
    vector_store: VectorStoreDep,
    keyword_search: KeywordSearchDep,
    settings: SettingsDep,
) -> tuple[list[ComponentHealth], bool, bool]:
    qdrant_ok = vector_store.health_check()
    keyword_ok = keyword_search.health_check()
    keyword_chunks = keyword_search.chunk_count if keyword_ok else 0
    keyword_detail: str | None = None
    if not keyword_ok:
        keyword_detail = "Keyword index path is unavailable"
    elif keyword_chunks == 0:
        keyword_detail = "Index empty — upload documents to populate BM25"
    components = [
        ComponentHealth(
            name="qdrant",
            status="ok" if qdrant_ok else "unavailable",
            detail=None if qdrant_ok else "Qdrant is unreachable",
        ),
        ComponentHealth(
            name="keyword_index",
            status="ok" if keyword_ok else "unavailable",
            detail=keyword_detail,
            metadata={
                "chunk_count": keyword_chunks,
                "hybrid_search_enabled": settings.hybrid_search_enabled,
                "index_path": settings.keyword_index_path,
            },
        ),
    ]
    return components, qdrant_ok, keyword_ok


@router.get(
    "/live",
    summary="Liveness probe",
    status_code=status.HTTP_200_OK,
)
def liveness_probe() -> dict[str, str]:
    """Return 200 when the process is running."""
    return {"status": "alive"}


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application health check",
)
def health_check(
    settings: SettingsDep,
    vector_store: VectorStoreDep,
    keyword_search: KeywordSearchDep,
) -> HealthResponse:
    """Return application health and dependency connectivity status."""
    components, qdrant_ok, keyword_ok = _component_statuses(
        vector_store,
        keyword_search,
        settings,
    )
    if qdrant_ok and keyword_ok:
        overall: Literal["ok", "degraded", "unavailable"] = "ok"
    elif qdrant_ok or keyword_ok:
        overall = "degraded"
    else:
        overall = "unavailable"

    logger.info(
        "health_check_completed",
        extra={
            "operation": "health_check",
            "overall_status": overall,
            "qdrant_ok": qdrant_ok,
            "keyword_ok": keyword_ok,
        },
    )

    return HealthResponse(
        status=overall,
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        components=components,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
)
def readiness_probe(
    settings: SettingsDep,
    vector_store: VectorStoreDep,
    response: Response,
    keyword_search: KeywordSearchDep,
) -> HealthResponse:
    """Return 200 only when required dependencies are reachable."""
    components, qdrant_ok, keyword_ok = _component_statuses(
        vector_store,
        keyword_search,
        settings,
    )
    ready = qdrant_ok and keyword_ok
    overall: Literal["ok", "degraded", "unavailable"] = "ok" if ready else "unavailable"
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    logger.info(
        "readiness_check_completed",
        extra={
            "operation": "readiness_check",
            "ready": ready,
            "qdrant_ok": qdrant_ok,
            "keyword_ok": keyword_ok,
        },
    )

    return HealthResponse(
        status=overall,
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        components=components,
    )
