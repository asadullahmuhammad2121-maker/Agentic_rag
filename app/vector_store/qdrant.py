"""Qdrant vector store implementation."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import Settings
from app.core.exceptions import QdrantConnectionError, VectorStoreError
from app.core.logging import get_logger
from app.utils.ids import normalize_point_id
from app.vector_store.base import SearchResult, VectorRecord, VectorStore
from app.vector_store.filters import PayloadFilter

logger = get_logger(__name__)

_DISTANCE_MAP: dict[str, qmodels.Distance] = {
    "Cosine": qmodels.Distance.COSINE,
    "Euclid": qmodels.Distance.EUCLID,
    "Dot": qmodels.Distance.DOT,
    "Manhattan": qmodels.Distance.MANHATTAN,
}


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store with a provider-independent interface."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self._settings = settings
        self._default_collection = settings.qdrant_collection_name
        try:
            self._client = client or QdrantClient(
                url=settings.qdrant_url,
                timeout=int(settings.qdrant_timeout_seconds),
                check_compatibility=False,
            )
        except Exception as exc:
            logger.error(
                "qdrant_client_init_failed",
                extra={
                    "operation": "init",
                    "error_type": type(exc).__name__,
                },
            )
            raise QdrantConnectionError() from exc

        logger.info(
            "qdrant_vector_store_initialized",
            extra={
                "operation": "init",
                "qdrant_url": settings.qdrant_url,
                "collection": self._default_collection,
            },
        )

    @property
    def client(self) -> QdrantClient:
        return self._client

    def health_check(self) -> bool:
        """Ping Qdrant and confirm the service responds."""
        try:
            self._client.get_collections()
            logger.debug(
                "qdrant_health_check_ok",
                extra={"operation": "health_check"},
            )
            return True
        except Exception as exc:
            logger.warning(
                "qdrant_health_check_failed",
                extra={
                    "operation": "health_check",
                    "error_type": type(exc).__name__,
                },
            )
            return False

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        *,
        distance: str = "Cosine",
    ) -> None:
        distance_enum = _DISTANCE_MAP.get(distance)
        if distance_enum is None:
            raise VectorStoreError(
                f"Unsupported distance metric: {distance}",
                details={"supported": sorted(_DISTANCE_MAP)},
            )

        try:
            exists = self._client.collection_exists(collection_name=collection_name)
            if exists:
                logger.info(
                    "qdrant_collection_exists",
                    extra={
                        "operation": "create_collection",
                        "collection": collection_name,
                    },
                )
                return

            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=distance_enum,
                ),
            )
            logger.info(
                "qdrant_collection_created",
                extra={
                    "operation": "create_collection",
                    "collection": collection_name,
                    "vector_size": vector_size,
                    "distance": distance,
                },
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                "Failed to create collection",
                details={"collection": collection_name},
            ) from exc
        except Exception as exc:
            if _is_connection_error(exc):
                raise QdrantConnectionError() from exc
            raise VectorStoreError(
                "Failed to create collection",
                details={"collection": collection_name},
            ) from exc

    def delete_collection(self, collection_name: str) -> None:
        try:
            self._client.delete_collection(collection_name=collection_name)
            logger.info(
                "qdrant_collection_deleted",
                extra={"operation": "delete_collection", "collection": collection_name},
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                "Failed to delete collection",
                details={"collection": collection_name},
            ) from exc
        except Exception as exc:
            if _is_connection_error(exc):
                raise QdrantConnectionError() from exc
            raise VectorStoreError(
                "Failed to delete collection",
                details={"collection": collection_name},
            ) from exc

    def add_vectors(
        self,
        collection_name: str,
        records: list[VectorRecord],
    ) -> None:
        if not records:
            return

        points = [
            qmodels.PointStruct(
                id=normalize_point_id(record.id),
                vector=record.vector,
                payload=record.payload,
            )
            for record in records
        ]

        try:
            self._client.upsert(collection_name=collection_name, points=points)
            logger.info(
                "qdrant_vectors_upserted",
                extra={
                    "operation": "add_vectors",
                    "collection": collection_name,
                    "count": len(points),
                },
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                "Failed to upsert vectors",
                details={"collection": collection_name, "count": len(points)},
            ) from exc
        except Exception as exc:
            if _is_connection_error(exc):
                raise QdrantConnectionError() from exc
            raise VectorStoreError(
                "Failed to upsert vectors",
                details={"collection": collection_name},
            ) from exc

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: PayloadFilter | None = None,
    ) -> list[SearchResult]:
        query_filter = build_qdrant_filter(filters)
        try:
            response = self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
            results = [
                SearchResult(
                    id=str(hit.id),
                    score=float(hit.score),
                    payload=dict(hit.payload or {}),
                )
                for hit in response.points
            ]
            logger.debug(
                "qdrant_search_completed",
                extra={
                    "operation": "search",
                    "collection": collection_name,
                    "limit": limit,
                    "result_count": len(results),
                    "has_filters": bool(filters),
                },
            )
            return results
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                "Vector search failed",
                details={"collection": collection_name},
            ) from exc
        except Exception as exc:
            if _is_connection_error(exc):
                raise QdrantConnectionError() from exc
            raise VectorStoreError(
                "Vector search failed",
                details={"collection": collection_name},
            ) from exc

    def delete(
        self,
        collection_name: str,
        ids: list[str | UUID],
    ) -> None:
        if not ids:
            return

        point_ids: list[Any] = [normalize_point_id(point_id) for point_id in ids]
        try:
            self._client.delete(
                collection_name=collection_name,
                points_selector=qmodels.PointIdsList(points=point_ids),
            )
            logger.info(
                "qdrant_vectors_deleted",
                extra={
                    "operation": "delete",
                    "collection": collection_name,
                    "count": len(point_ids),
                },
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                "Failed to delete vectors",
                details={"collection": collection_name},
            ) from exc
        except Exception as exc:
            if _is_connection_error(exc):
                raise QdrantConnectionError() from exc
            raise VectorStoreError(
                "Failed to delete vectors",
                details={"collection": collection_name},
            ) from exc

    def find_by_payload(
        self,
        collection_name: str,
        conditions: dict[str, Any],
        *,
        limit: int = 10,
    ) -> list[SearchResult]:
        if not conditions:
            return []

        query_filter = build_qdrant_filter(PayloadFilter.from_legacy_dict(conditions))
        if query_filter is None:
            return []

        try:
            points, _next = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = [
                SearchResult(
                    id=str(point.id),
                    score=1.0,
                    payload=dict(point.payload or {}),
                )
                for point in points
            ]
            logger.debug(
                "qdrant_payload_lookup_completed",
                extra={
                    "operation": "find_by_payload",
                    "collection": collection_name,
                    "result_count": len(results),
                    "condition_keys": sorted(conditions),
                },
            )
            return results
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                "Payload lookup failed",
                details={"collection": collection_name},
            ) from exc
        except Exception as exc:
            if _is_connection_error(exc):
                raise QdrantConnectionError() from exc
            # Collection may not exist yet — treat as no matches for duplicate checks.
            message = str(exc).lower()
            if "not found" in message or "doesn't exist" in message:
                return []
            raise VectorStoreError(
                "Payload lookup failed",
                details={"collection": collection_name},
            ) from exc

    def ensure_payload_index(
        self,
        collection_name: str,
        field_name: str,
        *,
        field_schema: str = "keyword",
    ) -> None:
        schema_map: dict[str, qmodels.PayloadSchemaType] = {
            "keyword": qmodels.PayloadSchemaType.KEYWORD,
            "integer": qmodels.PayloadSchemaType.INTEGER,
            "float": qmodels.PayloadSchemaType.FLOAT,
            "text": qmodels.PayloadSchemaType.TEXT,
        }
        schema = schema_map.get(field_schema)
        if schema is None:
            raise VectorStoreError(
                f"Unsupported payload field schema: {field_schema}",
                details={"supported": sorted(schema_map)},
            )

        try:
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
            )
            logger.info(
                "qdrant_payload_index_ensured",
                extra={
                    "operation": "ensure_payload_index",
                    "collection": collection_name,
                    "field_name": field_name,
                    "field_schema": field_schema,
                },
            )
        except UnexpectedResponse as exc:
            # Index already exists is acceptable.
            if "already exists" in str(exc).lower():
                return
            raise VectorStoreError(
                "Failed to create payload index",
                details={"collection": collection_name, "field_name": field_name},
            ) from exc
        except Exception as exc:
            if _is_connection_error(exc):
                raise QdrantConnectionError() from exc
            if "already exists" in str(exc).lower():
                return
            raise VectorStoreError(
                "Failed to create payload index",
                details={"collection": collection_name, "field_name": field_name},
            ) from exc


def build_qdrant_filter(payload_filter: PayloadFilter | None) -> qmodels.Filter | None:
    """Build a Qdrant payload filter from a structured filter specification."""
    if payload_filter is None or payload_filter.is_empty():
        return None

    must_conditions: list[qmodels.Condition] = []
    for key, value in payload_filter.exact.items():
        must_conditions.append(
            qmodels.FieldCondition(
                key=key,
                match=qmodels.MatchValue(value=value),
            )
        )
    for key, values in payload_filter.any_of.items():
        must_conditions.append(
            qmodels.FieldCondition(
                key=key,
                match=qmodels.MatchAny(any=list(values)),
            )
        )
    return qmodels.Filter(must=must_conditions)


def _is_connection_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    connection_markers = (
        "connection",
        "connect",
        "timeout",
        "timed out",
        "refused",
        "unreachable",
        "name or service not known",
    )
    return any(marker in name or marker in message for marker in connection_markers)
