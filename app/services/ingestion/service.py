"""Document ingestion orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from app.core.config import Settings
from app.core.exceptions import (
    AppError,
    DocumentIngestionError,
    DuplicateDocumentError,
    InvalidDocumentError,
)
from app.core.logging import get_logger
from app.services.chunking.service import ChunkingService, TextChunk
from app.services.embeddings.base import EmbeddingService
from app.services.ingestion.pdf_extractor import ExtractedDocument, PdfTextExtractor
from app.utils.checksum import sha256_digest
from app.utils.ids import new_document_id, new_point_id
from app.vector_store.base import VectorRecord, VectorStore

logger = get_logger(__name__)

ALLOWED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "application/pdf",
        "application/x-pdf",
    }
)
ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset({".pdf"})


@dataclass(slots=True, frozen=True)
class IngestedDocument:
    """Result of a successful PDF ingestion."""

    document_id: str
    filename: str
    content_type: str
    file_size: int
    checksum: str
    page_count: int
    pages_stored: int
    chunks_stored: int


class DocumentIngestionService:
    """Validate, extract, chunk, embed, and persist PDF documents."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        *,
        extractor: PdfTextExtractor | None = None,
        chunking_service: ChunkingService | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._extractor = extractor or PdfTextExtractor()
        self._chunking = chunking_service or ChunkingService(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def ingest_pdf(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None,
    ) -> IngestedDocument:
        """
        Ingest a PDF file.

        Phase 1C flow:
        extract pages → chunk → embed (Hugging Face) → store vectors + metadata in Qdrant.
        Duplicate documents (same checksum) are rejected to avoid regenerating embeddings.
        """
        normalized_name = self._validate_filename(filename)
        normalized_type = self._validate_content_type(content_type, normalized_name)
        self._validate_size(content)

        checksum = sha256_digest(content)
        document_id = new_document_id()

        logger.info(
            "document_ingestion_started",
            extra={
                "operation": "ingest_pdf",
                "document_id": document_id,
                "document_filename": normalized_name,
                "content_type": normalized_type,
                "file_size": len(content),
                "checksum": checksum,
            },
        )

        self._ensure_collection()
        self._reject_duplicate(checksum=checksum, filename=normalized_name)

        extracted = self._extractor.extract(content)
        chunks = self._chunking.chunk_pages(
            extracted.pages,
            document_id=document_id,
            filename=normalized_name,
        )
        if not chunks:
            raise InvalidDocumentError(
                "PDF contains no extractable text",
                details={"reason": "empty_text", "page_count": extracted.page_count},
            )

        records = self._build_records(
            document_id=document_id,
            filename=normalized_name,
            content_type=normalized_type,
            file_size=len(content),
            checksum=checksum,
            extracted=extracted,
            chunks=chunks,
        )

        try:
            self._vector_store.add_vectors(
                self._settings.qdrant_collection_name,
                records,
            )
        except Exception as exc:
            if isinstance(exc, AppError):
                raise
            logger.error(
                "document_ingestion_store_failed",
                extra={
                    "operation": "ingest_pdf",
                    "document_id": document_id,
                    "error_type": type(exc).__name__,
                },
            )
            raise DocumentIngestionError(
                "Failed to store document in vector store",
                details={"document_id": document_id},
            ) from exc

        pages_with_text = len({chunk.page_number for chunk in chunks})
        result = IngestedDocument(
            document_id=document_id,
            filename=normalized_name,
            content_type=normalized_type,
            file_size=len(content),
            checksum=checksum,
            page_count=extracted.page_count,
            pages_stored=pages_with_text,
            chunks_stored=len(records),
        )
        logger.info(
            "document_ingestion_completed",
            extra={
                "operation": "ingest_pdf",
                "document_id": result.document_id,
                "document_filename": result.filename,
                "file_size": result.file_size,
                "checksum": result.checksum,
                "page_count": result.page_count,
                "pages_stored": result.pages_stored,
                "chunks_stored": result.chunks_stored,
            },
        )
        return result

    def _validate_filename(self, filename: str) -> str:
        name = (filename or "").strip()
        if not name:
            raise InvalidDocumentError(
                "Filename is required",
                details={"reason": "missing_filename"},
            )
        lower = name.lower()
        if not any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise InvalidDocumentError(
                "Only PDF files are supported",
                details={"reason": "invalid_extension", "filename": name},
            )
        return name

    def _validate_content_type(self, content_type: str | None, filename: str) -> str:
        if content_type is None or content_type == "" or content_type == "application/octet-stream":
            return "application/pdf"
        media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
        if media_type not in ALLOWED_CONTENT_TYPES:
            raise InvalidDocumentError(
                "Only PDF files are supported",
                details={
                    "reason": "invalid_content_type",
                    "content_type": media_type,
                    "filename": filename,
                },
            )
        return media_type

    def _validate_size(self, content: bytes) -> None:
        if not content:
            raise InvalidDocumentError(
                "PDF file is empty",
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
        self._vector_store.ensure_payload_index(collection, "checksum", field_schema="keyword")
        self._vector_store.ensure_payload_index(collection, "document_id", field_schema="keyword")
        self._vector_store.ensure_payload_index(
            collection,
            "chunk_checksum",
            field_schema="keyword",
        )

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
                "operation": "ingest_pdf",
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

    def _build_records(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str,
        file_size: int,
        checksum: str,
        extracted: ExtractedDocument,
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
                    "operation": "ingest_pdf",
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
            payload = {
                "document_id": document_id,
                "filename": filename,
                "content_type": content_type,
                "file_size": file_size,
                "checksum": checksum,
                "chunk_checksum": chunk_checksum,
                "page_number": chunk.page_number,
                "page_count": extracted.page_count,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "ingested_at": ingested_at,
                "embedding_status": "ready",
                "embedding_model": self._embedding_service.model_name,
                "embedding_provider": self._embedding_service.provider_name,
            }
            records.append(
                VectorRecord(
                    id=new_point_id(),
                    vector=vector,
                    payload=payload,
                )
            )
        return records
