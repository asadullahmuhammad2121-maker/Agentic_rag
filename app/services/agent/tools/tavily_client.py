"""Tavily API client wrapper — isolated from agent orchestration."""

from __future__ import annotations

import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class TavilyClientProtocol(ABC):
    """Minimal Tavily client interface for testing and swapping providers."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_depth: str,
    ) -> dict[str, Any]:
        """Execute a Tavily search and return the raw response payload."""


class TavilySearchClient(TavilyClientProtocol):
    """Production Tavily client backed by the official Python SDK."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        api_key = settings.tavily_api_key.get_secret_value().strip()
        if not api_key:
            raise ProviderError(
                "Tavily API key is not configured",
                provider="tavily",
                details={"reason": "missing_api_key"},
            )
        try:
            if client is None:
                from tavily import TavilyClient

                client = TavilyClient(api_key=api_key)
            self._client = client
        except Exception as exc:
            logger.error(
                "tavily_client_init_failed",
                extra={
                    "operation": "init",
                    "provider": "tavily",
                    "error_type": type(exc).__name__,
                },
            )
            raise ProviderError(
                "Failed to initialize Tavily client",
                provider="tavily",
                details={"reason": "client_init_failed"},
            ) from exc

        self._timeout_seconds = settings.tavily_timeout_seconds
        logger.info(
            "tavily_client_initialized",
            extra={"operation": "init", "provider": "tavily"},
        )

    def search(
        self,
        query: str,
        *,
        max_results: int,
        search_depth: str,
    ) -> dict[str, Any]:
        logger.info(
            "tavily_search_started",
            extra={
                "operation": "search",
                "provider": "tavily",
                "query_length": len(query),
                "max_results": max_results,
                "search_depth": search_depth,
            },
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._client.search,
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=False,
                )
                response = future.result(timeout=self._timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            logger.error(
                "tavily_search_timeout",
                extra={
                    "operation": "search",
                    "provider": "tavily",
                    "timeout_seconds": self._timeout_seconds,
                },
            )
            raise ProviderError(
                "Tavily request timed out",
                provider="tavily",
                details={"reason": "timeout"},
            ) from exc
        except TimeoutError as exc:
            logger.error(
                "tavily_search_timeout",
                extra={
                    "operation": "search",
                    "provider": "tavily",
                    "error_type": type(exc).__name__,
                },
            )
            raise ProviderError(
                "Tavily request timed out",
                provider="tavily",
                details={"reason": "timeout"},
            ) from exc
        except Exception as exc:
            error_type = type(exc).__name__
            logger.error(
                "tavily_search_failed",
                extra={
                    "operation": "search",
                    "provider": "tavily",
                    "error_type": error_type,
                },
            )
            if _looks_like_network_error(exc):
                raise ProviderError(
                    "Unable to reach Tavily",
                    provider="tavily",
                    details={"reason": "connection_error", "error_type": error_type},
                ) from exc
            raise ProviderError(
                "Tavily search request failed",
                provider="tavily",
                details={"reason": "api_error", "error_type": error_type},
            ) from exc

        if not isinstance(response, dict):
            raise ProviderError(
                "Tavily returned an unexpected response",
                provider="tavily",
                details={"reason": "invalid_response"},
            )

        logger.info(
            "tavily_search_completed",
            extra={
                "operation": "search",
                "provider": "tavily",
                "result_count": len(response.get("results", [])),
            },
        )
        return response


def _looks_like_network_error(exc: Exception) -> bool:
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    markers = ("connection", "network", "timeout", "connect")
    return any(marker in name or marker in message for marker in markers)
