/** Supported extensions aligned with backend `EXTENSION_TO_FILE_TYPE`. */
export const SUPPORTED_DOCUMENT_EXTENSIONS = [
  ".pdf",
  ".docx",
  ".txt",
  ".md",
  ".markdown",
  ".csv",
  ".json",
] as const;

export const SUPPORTED_DOCUMENT_ACCEPT = SUPPORTED_DOCUMENT_EXTENSIONS.join(",");

/** Default backend limit (`max_upload_file_size_mb=25`). */
export const MAX_UPLOAD_FILE_SIZE_BYTES = 25 * 1024 * 1024;

/** Default backend limit (`max_batch_upload_files=20`). */
export const MAX_BATCH_UPLOAD_FILES = 20;

export const DOCUMENTS_STORAGE_KEY = "agentic-rag:documents";
