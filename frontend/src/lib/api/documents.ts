import { ApiError, getApiBaseUrl } from "@/lib/api/client";
import type { DocumentUploadResponse } from "@/lib/types/documents";

async function parseUploadError(response: Response): Promise<ApiError> {
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
    payload.message ?? `Upload failed with status ${response.status}`,
    response.status,
    payload.error ?? "upload_failed",
    payload.details ?? {},
  );
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
    throw await parseUploadError(response);
  }

  return (await response.json()) as DocumentUploadResponse;
}

export const documentQueryKeys = {
  all: ["documents"] as const,
  detail: (documentId: string) => ["documents", documentId] as const,
};
