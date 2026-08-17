"""Public settings routes."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import SettingsDep, ToolRegistryDep
from app.core.logging import get_logger
from app.schemas.settings import PublicSettingsResponse
from app.services.settings.public import build_public_settings

logger = get_logger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get(
    "",
    response_model=PublicSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get user-safe read-only application settings",
)
def get_public_settings(
    settings: SettingsDep,
    tools: ToolRegistryDep,
) -> PublicSettingsResponse:
    """Return sanitized configuration without secrets or raw environment variables."""
    logger.info(
        "public_settings_requested",
        extra={"operation": "get_public_settings", "environment": settings.app_env},
    )
    return build_public_settings(settings, tools)
