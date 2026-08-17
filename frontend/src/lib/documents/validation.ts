import {
  MAX_BATCH_UPLOAD_FILES,
  MAX_UPLOAD_FILE_SIZE_BYTES,
  SUPPORTED_DOCUMENT_EXTENSIONS,
} from "@/lib/constants/documents";

export interface FileValidationIssue {
  filename: string;
  message: string;
}

function getExtension(filename: string): string {
  const dot = filename.lastIndexOf(".");
  if (dot === -1) return "";
  return filename.slice(dot).toLowerCase();
}

export function validateUploadFiles(files: File[]): FileValidationIssue[] {
  const issues: FileValidationIssue[] = [];

  if (files.length === 0) {
    issues.push({ filename: "", message: "Select at least one file to upload." });
    return issues;
  }

  if (files.length > MAX_BATCH_UPLOAD_FILES) {
    issues.push({
      filename: "",
      message: `You can upload up to ${MAX_BATCH_UPLOAD_FILES} files at once.`,
    });
  }

  for (const file of files) {
    const extension = getExtension(file.name);
    if (
      !SUPPORTED_DOCUMENT_EXTENSIONS.includes(
        extension as (typeof SUPPORTED_DOCUMENT_EXTENSIONS)[number],
      )
    ) {
      issues.push({
        filename: file.name,
        message: `Unsupported file type (${extension || "no extension"}). Supported: ${SUPPORTED_DOCUMENT_EXTENSIONS.join(", ")}`,
      });
    }

    if (file.size > MAX_UPLOAD_FILE_SIZE_BYTES) {
      issues.push({
        filename: file.name,
        message: `File exceeds the ${Math.round(MAX_UPLOAD_FILE_SIZE_BYTES / (1024 * 1024))} MB limit.`,
      });
    }
  }

  return issues;
}
