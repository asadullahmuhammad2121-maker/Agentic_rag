"""Document ingestion orchestration service."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from app.core.config import Settings
from app.core.exceptions import (
    AppError,
    DocumentIngestionError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidDocumentError,
)
from app.core.logging import get_logger
from app.services.chunking.base import TextChunk
from app.services.chunking.service import ChunkingService
from app.services.embeddings.base import EmbeddingService
from app.services.ingestion.base import NormalizedDocument
from app.services.ingestion.format_detection import detect_file_type
from app.services.ingestion.metadata import build_chunk_payload
from app.services.ingestion.parser_registry import DocumentParserRegistry
from app.services.retrieval.keyword.base import KeywordSearch
from app.utils.checksum import sha256_digest
from app.utils.filenames import sanitize_upload_filename
from app.utils.ids import new_document_id
from app.vector_store.base import VectorRecord, VectorStore

logger = get_logger(__name__)

PAYLOAD_INDEX_FIELDS: Final[tuple[str, ...]] = (
    "checksum",
    "document_id",
    "chunk_checksum",
    "file_type",
    "filename",
    "section",
    "chunk_id",
    "chunking_strategy",
)

_checksum_locks_guard = threading.Lock()
_checksum_locks: dict[str, threading.Lock] = {}


@dataclass(slots=True, frozen=True)
class IngestedDocument:
    """Result of a successful document ingestion."""

    document_id: str
    filename: str
    content_type: str
    file_type: str
    file_size: int
    checksum: str
    source: str
    page_count: int
    pages_stored: int
    chunks_stored: int


@dataclass(slots=True, frozen=True)
class DeletedDocument:
    """Result of deleting an ingested document."""

    document_id: str
    chunks_deleted: int
    checksum: str | None
    filename: str | None


@dataclass(slots=True, frozen=True)
class ListedDocument:
    """Aggregated document metadata derived from stored vector payloads."""

    document_id: str
    filename: str
    content_type: str
    file_type: str
    file_size: int
    checksum: str
    source: str
    page_count: int
    pages_stored: int
    chunks_stored: int
    ingested_at: str | None


class DocumentIngestionService:
    """Validate, parse, chunk, embed, and persist uploaded documents."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        *,
        parser_registry: DocumentParserRegistry | None = None,
        chunking_service: ChunkingService | None = None,
        keyword_search: KeywordSearch | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._keyword_search = keyword_search
        self._parser_registry = parser_registry or DocumentParserRegistry()
        self._chunking = chunking_service or ChunkingService(
            settings,
            embedding_service=embedding_service,
        )

    def ingest_pdf(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None,
    ) -> IngestedDocument:
        """Backward-compatible PDF ingestion entry point."""
        return self.ingest_document(
            filename=filename,
            content=content,
            content_type=content_type,
        )

    def ingest_document(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None,
    ) -> IngestedDocument:
        """
        Ingest a single uploaded document of any supported format.

        Flow:
        detect format → parse → chunk → embed → store vectors + metadata in Qdrant.
        """
        normalized_name = sanitize_upload_filename(filename)
        file_type, normalized_type = detect_file_type(
            filename=normalized_name,
            content_type=content_type,
        )
        self._validate_size(content)

        checksum = sha256_digest(content)
        checksum_lock = _checksum_lock_for(checksum)
        with checksum_lock:
            return self._ingest_document_locked(
                filename=normalized_name,
                content=content,
                content_type=normalized_type,
                file_type=file_type,
                checksum=checksum,
            )

    def ingest_documents(
        self,
        uploads: list[tuple[str, bytes, str | None]],
    ) -> list[IngestedDocument]:
        """Ingest multiple independent documents sequentially."""
        results: list[IngestedDocument] = []
        for filename, content, content_type in uploads:
            results.append(
                self.ingest_document(
                    filename=filename,
                    content=content,
                    content_type=content_type,
                )
            )
        return results

    def delete_document(self, document_id: str) -> DeletedDocument:
        """Delete all stored chunks for a document and clear deduplication state."""
        normalized_id = document_id.strip()
        if not normalized_id:
            raise InvalidDocumentError(
                "Document ID must not be empty",
                details={"reason": "empty_document_id"},
            )

        collection = self._settings.qdrant_collection_name
        existing = self._vector_store.find_by_payload(
            collection,
            {"document_id": normalized_id},
            limit=1,
        )
        if not existing:
            raise DocumentNotFoundError(
                details={"document_id": normalized_id},
            )

        checksum = existing[0].payload.get("checksum")
        filename = existing[0].payload.get("filename")
        chunk_count = self._vector_store.count_by_payload(
            collection,
            {"document_id": normalized_id},
        )

        logger.info(
            "document_deletion_started",
            extra={
                "operation": "delete_document",
                "document_id": normalized_id,
                "chunk_count": chunk_count,
                "checksum": checksum,
            },
        )

        self._vector_store.delete_by_payload(collection, {"document_id": normalized_id})

        keyword_removed = 0
        if self._keyword_search is not None:
            keyword_removed = self._keyword_search.remove_document(normalized_id)

        logger.info(
            "document_deletion_completed",
            extra={
                "operation": "delete_document",
                "document_id": normalized_id,
                "chunks_deleted": chunk_count,
                "keyword_chunks_removed": keyword_removed,
                "checksum": checksum,
            },
        )
        return DeletedDocument(
            document_id=normalized_id,
            chunks_deleted=chunk_count,
            checksum=str(checksum) if checksum is not None else None,
            filename=str(filename) if filename is not None else None,
        )

    def list_documents(self) -> list[ListedDocument]:
        """Return unique ingested documents aggregated from vector-store payloads."""
        collection = self._settings.qdrant_collection_name
        payloads = self._vector_store.scroll_payloads(collection)
        documents = _aggregate_listed_documents(payloads)
        logger.info(
            "document_list_completed",
            extra={
                "operation": "list_documents",
                "document_count": len(documents),
            },
        )
        return documents

    def get_document(self, document_id: str) -> ListedDocument:
        """Return aggregated metadata for a single ingested document."""
        normalized_id = document_id.strip()
        if not normalized_id:
            raise InvalidDocumentError(
                "Document ID must not be empty",
                details={"reason": "empty_document_id"},
            )

        collection = self._settings.qdrant_collection_name
        hits = self._vector_store.find_by_payload(
            collection,
            {"document_id": normalized_id},
            limit=1,
        )
        if not hits:
            raise DocumentNotFoundError(details={"document_id": normalized_id})

        chunk_count = self._vector_store.count_by_payload(
            collection,
            {"document_id": normalized_id},
        )
        return _listed_document_from_payload(
            hits[0].payload,
            chunks_stored=chunk_count,
        )

    def _ingest_document_locked(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str,
        file_type: str,
        checksum: str,
    ) -> IngestedDocument:
        document_id = new_document_id()

        logger.info(
            "document_ingestion_started",
            extra={
                "operation": "ingest_document",
                "document_id": document_id,
                "document_filename": filename,
                "file_type": file_type,
                "content_type": content_type,
                "file_size": len(content),
                "checksum": checksum,
            },
        )

        self._ensure_collection()
        self._reject_duplicate(checksum=checksum, filename=filename)

        parser = self._parser_registry.get_parser(file_type)
        normalized = parser.parse(content, filename=filename)
        chunks = self._chunking.chunk_sections(
            normalized.sections,
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            source=normalized.source,
        )
        if not chunks:
            raise InvalidDocumentError(
                "Document contains no extractable text",
                details={
                    "reason": "empty_text",
                    "file_type": file_type,
                    "page_count": normalized.page_count,
                },
            )

        records = self._build_records(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            file_type=file_type,
            source=normalized.source,
            file_size=len(content),
            checksum=checksum,
            normalized=normalized,
            chunks=chunks,
        )

        chunk_ids = [str(record.id) for record in records]
        vectors_stored = False
        try:
            self._vector_store.add_vectors(
                self._settings.qdrant_collection_name,
                records,
            )
            vectors_stored = True

            if self._keyword_search is not None:
                try:
                    self._keyword_search.index_records(records)
                except Exception as exc:
                    logger.warning(
                        "keyword_index_update_failed",
                        extra={
                            "operation": "ingest_document",
                            "document_id": document_id,
                            "error_type": type(exc).__name__,
                        },
                    )

            pages_with_text = len({chunk.page_number for chunk in chunks})
            result = IngestedDocument(
                document_id=document_id,
                filename=filename,
                content_type=content_type,
                file_type=file_type,
                file_size=len(content),
                checksum=checksum,
                source=normalized.source,
                page_count=normalized.page_count,
                pages_stored=max(pages_with_text, 1),
                chunks_stored=len(records),
            )
        except Exception as exc:
            if vectors_stored:
                self._rollback_ingestion(document_id=document_id, chunk_ids=chunk_ids)
                raise
            if isinstance(exc, AppError):
                raise
            logger.error(
                "document_ingestion_store_failed",
                extra={
                    "operation": "ingest_document",
                    "document_id": document_id,
                    "error_type": type(exc).__name__,
                },
            )
            raise DocumentIngestionError(
                "Failed to store document in vector store",
                details={"document_id": document_id},
            ) from exc

        logger.info(
            "document_ingestion_completed",
            extra={
                "operation": "ingest_document",
                "document_id": result.document_id,
                "document_filename": result.filename,
                "file_type": result.file_type,
                "file_size": result.file_size,
                "checksum": result.checksum,
                "page_count": result.page_count,
                "pages_stored": result.pages_stored,
                "chunks_stored": result.chunks_stored,
            },
        )
        return result

    def _validate_size(self, content: bytes) -> None:
        if not content:
            raise InvalidDocumentError(
                "Document file is empty",
                details={"reason": "empty_file"},
            )
        max_bytes = self._settings.max_upload_file_size_bytes
        if len(content) > max_bytes:
            raise InvalidDocumentError(
                "File exceeds maximum upload size",
                details={
                    "reason": "file_too_large",
                    "file_size": len(content),
                    "max_upload_file_size_bytes": max_bytes,
                },
            )

    def _ensure_collection(self) -> None:
        collection = self._settings.qdrant_collection_name
        self._vector_store.create_collection(
            collection,
            vector_size=self._settings.embedding_dimension,
        )
        for field_name in PAYLOAD_INDEX_FIELDS:
            self._vector_store.ensure_payload_index(collection, field_name, field_schema="keyword")

    def _reject_duplicate(self, *, checksum: str, filename: str) -> None:
        existing = self._vector_store.find_by_payload(
            self._settings.qdrant_collection_name,
            {"checksum": checksum},
            limit=1,
        )
        if not existing:
            return

        existing_doc_id = existing[0].payload.get("document_id")
        logger.warning(
            "duplicate_document_rejected",
            extra={
                "operation": "ingest_document",
                "checksum": checksum,
                "document_filename": filename,
                "existing_document_id": existing_doc_id,
            },
        )
        raise DuplicateDocumentError(
            details={
                "checksum": checksum,
                "existing_document_id": existing_doc_id,
                "filename": filename,
            },
        )

    def _rollback_ingestion(self, *, document_id: str, chunk_ids: list[str]) -> None:
        collection = self._settings.qdrant_collection_name
        try:
            if chunk_ids:
                self._vector_store.delete(collection, list(chunk_ids))
            else:
                self._vector_store.delete_by_payload(collection, {"document_id": document_id})
        except Exception as exc:
            logger.error(
                "document_ingestion_rollback_failed",
                extra={
                    "operation": "rollback_ingestion",
                    "document_id": document_id,
                    "chunk_count": len(chunk_ids),
                    "error_type": type(exc).__name__,
                },
            )
            return

        if self._keyword_search is not None:
            try:
                self._keyword_search.remove_document(document_id)
            except Exception as exc:
                logger.warning(
                    "document_ingestion_keyword_rollback_failed",
                    extra={
                        "operation": "rollback_ingestion",
                        "document_id": document_id,
                        "error_type": type(exc).__name__,
                    },
                )

        logger.info(
            "document_ingestion_rolled_back",
            extra={
                "operation": "rollback_ingestion",
                "document_id": document_id,
                "chunk_count": len(chunk_ids),
            },
        )

    def _build_records(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str,
        file_type: str,
        source: str,
        file_size: int,
        checksum: str,
        normalized: NormalizedDocument,
        chunks: list[TextChunk],
    ) -> list[VectorRecord]:
        ingested_at = datetime.now(UTC).isoformat()
        texts = [chunk.text for chunk in chunks]

        try:
            vectors = self._embedding_service.embed_documents(texts)
        except AppError:
            raise
        except Exception as exc:
            logger.error(
                "document_embedding_failed",
                extra={
                    "operation": "ingest_document",
                    "document_id": document_id,
                    "error_type": type(exc).__name__,
                    "chunk_count": len(chunks),
                },
            )
            raise DocumentIngestionError(
                "Failed to generate embeddings for document chunks",
                details={"document_id": document_id},
            ) from exc

        if len(vectors) != len(chunks):
            raise DocumentIngestionError(
                "Embedding provider returned unexpected vector count",
                details={
                    "document_id": document_id,
                    "chunk_count": len(chunks),
                    "vector_count": len(vectors),
                },
            )

        records: list[VectorRecord] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk_checksum = sha256_digest(chunk.text.encode("utf-8"))
            payload = build_chunk_payload(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                source=source,
                page_number=chunk.page_number,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                chunking_strategy=self._chunking.strategy_name,
                extra={
                    "content_type": content_type,
                    "file_size": file_size,
                    "checksum": checksum,
                    "chunk_checksum": chunk_checksum,
                    "page_count": normalized.page_count,
                    "section_count": normalized.section_count,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "ingested_at": ingested_at,
                    "embedding_status": "ready",
                    "embedding_model": self._embedding_service.model_name,
                    "embedding_provider": self._embedding_service.provider_name,
                },
            )
            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )
        return records


def _checksum_lock_for(checksum: str) -> threading.Lock:
    with _checksum_locks_guard:
        lock = _checksum_locks.get(checksum)
        if lock is None:
            lock = threading.Lock()
            _checksum_locks[checksum] = lock
        return lock


def _aggregate_listed_documents(payloads: list[dict[str, object]]) -> list[ListedDocument]:
    template_by_id: dict[str, dict[str, object]] = {}
    chunk_counts: dict[str, int] = {}
    page_numbers: dict[str, set[int]] = {}

    for payload in payloads:
        document_id = str(payload.get("document_id", "")).strip()
        if not document_id:
            continue

        chunk_counts[document_id] = chunk_counts.get(document_id, 0) + 1
        page_number = payload.get("page_number")
        if isinstance(page_number, int):
            page_numbers.setdefault(document_id, set()).add(page_number)
        template_by_id.setdefault(document_id, payload)

    documents = [
        _listed_document_from_payload(
            template_by_id[document_id],
            chunks_stored=chunk_counts[document_id],
            pages_stored=len(page_numbers.get(document_id, set())),
        )
        for document_id in template_by_id
    ]
    return sorted(
        documents,
        key=lambda document: document.ingested_at or "",
        reverse=True,
    )


def _listed_document_from_payload(
    payload: dict[str, object],
    *,
    chunks_stored: int,
    pages_stored: int | None = None,
) -> ListedDocument:
    document_id = str(payload.get("document_id", "")).strip()
    page_count_raw = payload.get("page_count")
    page_count = int(page_count_raw) if isinstance(page_count_raw, int) else 1
    file_size_raw = payload.get("file_size")
    file_size = int(file_size_raw) if isinstance(file_size_raw, int) else 0
    resolved_pages_stored = (
        page_count if pages_stored is None or pages_stored < 1 else pages_stored
    )
    ingested_at = payload.get("ingested_at")

    return ListedDocument(
        document_id=document_id,
        filename=str(payload.get("filename", "")),
        content_type=str(payload.get("content_type", "application/octet-stream")),
        file_type=str(payload.get("file_type", "")),
        file_size=file_size,
        checksum=str(payload.get("checksum", "")),
        source=str(payload.get("source", "")),
        page_count=max(page_count, 1),
        pages_stored=max(resolved_pages_stored, 1),
        chunks_stored=max(chunks_stored, 1),
        ingested_at=str(ingested_at) if ingested_at is not None else None,
    )
