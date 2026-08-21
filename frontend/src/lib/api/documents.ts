import { ApiError, getApiBaseUrl } from "@/lib/api/client";
import type {
  DocumentListResponse,
  DocumentSummary,
  DocumentUploadResponse,
} from "@/lib/types/documents";

async function parseApiError(response: Response, fallbackCode: string): Promise<ApiError> {
  let payload: {
    error?: string;
    message?: string;
    details?: Record<string, unknown>;
  } = {};
  try {
    payload = (await response.json()) as typeof payload;
  } catch {
    // ignore parse errors
  }
  return new ApiError(
    payload.message ?? `Request failed with status ${response.status}`,
    response.status,
    payload.error ?? fallbackCode,
    payload.details ?? {},
  );
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const url = `${getApiBaseUrl()}/documents`;
  const response = await fetch(url, { method: "GET", cache: "no-store" });

  if (!response.ok) {
    throw await parseApiError(response, "list_failed");
  }

  const body = (await response.json()) as DocumentListResponse;
  return body.documents;
}

export async function getDocument(documentId: string): Promise<DocumentSummary> {
  const url = `${getApiBaseUrl()}/documents/${encodeURIComponent(documentId)}`;
  const response = await fetch(url, { method: "GET", cache: "no-store" });

  if (!response.ok) {
    throw await parseApiError(response, "document_not_found");
  }

  return (await response.json()) as DocumentSummary;
}

export async function uploadDocuments(files: File[]): Promise<DocumentUploadResponse> {
  if (files.length === 0) {
    throw new ApiError("At least one file is required", 400, "missing_file");
  }

  const formData = new FormData();
  if (files.length === 1) {
    formData.append("file", files[0]);
  } else {
    for (const file of files) {
      formData.append("files", file);
    }
  }

  const url = `${getApiBaseUrl()}/documents/upload`;
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw await parseApiError(response, "upload_failed");
  }

  return (await response.json()) as DocumentUploadResponse;
}

export async function deleteDocument(documentId: string): Promise<void> {
  const url = `${getApiBaseUrl()}/documents/${encodeURIComponent(documentId)}`;
  const response = await fetch(url, { method: "DELETE" });

  if (!response.ok) {
    throw await parseApiError(response, "delete_failed");
  }
}

export const documentQueryKeys = {
  all: ["documents"] as const,
  detail: (documentId: string) => ["documents", documentId] as const,
};
