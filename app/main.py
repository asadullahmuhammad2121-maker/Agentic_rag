"""FastAPI application factory and ASGI entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestTimeoutMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(level=settings.log_level, service_name=settings.app_name)
    logger.info(
        "application_startup",
        extra={
            "operation": "startup",
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
            "uvicorn_workers": settings.uvicorn_workers,
        },
    )

    from app.api.deps import get_vector_store

    try:
        vector_store = get_vector_store()
        qdrant_ok = vector_store.health_check()
        logger.info(
            "startup_dependency_check",
            extra={
                "operation": "startup",
                "dependency": "qdrant",
                "ok": qdrant_ok,
            },
        )
    except Exception as exc:
        logger.warning(
            "startup_dependency_check_failed",
            extra={
                "operation": "startup",
                "dependency": "qdrant",
                "error_type": type(exc).__name__,
            },
        )

    yield

    logger.info("application_shutdown", extra={"operation": "shutdown"})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    resolved_settings = settings or get_settings()
    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if not resolved_settings.is_production else None,
        redoc_url="/redoc" if not resolved_settings.is_production else None,
    )
    application.add_middleware(RequestTimeoutMiddleware, settings=resolved_settings)
    register_exception_handlers(application)
    application.include_router(api_router)
    return application


app = create_app()
