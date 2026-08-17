"""HTTP middleware for production hardening."""

from __future__ import annotations

import asyncio

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Abort requests that exceed the configured timeout."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self._timeout_seconds = settings.request_timeout_seconds

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "request_timeout",
                extra={
                    "operation": "http_request",
                    "path": request.url.path,
                    "method": request.method,
                    "timeout_seconds": self._timeout_seconds,
                },
            )
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timed out"},
            )
