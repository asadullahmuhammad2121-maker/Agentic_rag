"""Health check route."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, status

from app.api.deps import SettingsDep, VectorStoreDep
from app.core.logging import get_logger
from app.schemas.health import ComponentHealth, HealthResponse

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application health check",
)
def health_check(settings: SettingsDep, vector_store: VectorStoreDep) -> HealthResponse:
    """Return application health and Qdrant connectivity status."""
    qdrant_ok = vector_store.health_check()
    components = [
        ComponentHealth(
            name="qdrant",
            status="ok" if qdrant_ok else "unavailable",
            detail=None if qdrant_ok else "Qdrant is unreachable",
        ),
    ]

    overall: Literal["ok", "degraded", "unavailable"] = "ok" if qdrant_ok else "degraded"

    logger.info(
        "health_check_completed",
        extra={
            "operation": "health_check",
            "overall_status": overall,
            "qdrant_ok": qdrant_ok,
        },
    )

    return HealthResponse(
        status=overall,
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        components=components,
    )
