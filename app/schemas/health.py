"""Health check response schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ComponentHealth(BaseModel):
    """Health status of a single dependency."""

    name: str
    status: Literal["ok", "degraded", "unavailable"]
    detail: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Application health payload."""

    status: Literal["ok", "degraded", "unavailable"] = Field(
        description="Overall application health",
    )
    app: str
    version: str
    environment: str
    components: list[ComponentHealth] = Field(default_factory=list)
