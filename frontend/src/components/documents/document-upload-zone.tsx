"use client";

import { useCallback, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, FileUp, Loader2, Upload } from "lucide-react";
import { SUPPORTED_DOCUMENT_ACCEPT } from "@/lib/constants/documents";
import { validateUploadFiles } from "@/lib/documents/validation";
import { useUploadDocuments } from "@/lib/hooks/use-documents";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export function DocumentUploadZone() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const uploadMutation = useUploadDocuments();

  const handleFiles = useCallback(
    (fileList: FileList | File[]) => {
      const files = Array.from(fileList);
      const issues = validateUploadFiles(files);
      if (issues.length > 0) {
        setValidationErrors(issues.map((issue) =>
          issue.filename ? `${issue.filename}: ${issue.message}` : issue.message,
        ));
        return;
      }
      setValidationErrors([]);
      uploadMutation.mutate(files);
    },
    [uploadMutation],
  );

  const onDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    if (uploadMutation.isPending) return;
    handleFiles(event.dataTransfer.files);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Upload className="h-4 w-4" />
          Upload documents
        </CardTitle>
        <CardDescription>
          Drag and drop or browse. Supported: PDF, DOCX, TXT, Markdown, CSV, JSON (max 25 MB each,
          up to 20 files per batch).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload documents drop zone"
          aria-disabled={uploadMutation.isPending}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            setIsDragging(false);
          }}
          onDrop={onDrop}
          onClick={() => {
            if (!uploadMutation.isPending) inputRef.current?.click();
          }}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors",
            isDragging
              ? "border-slate-900 bg-slate-100"
              : "border-slate-200 bg-slate-50 hover:border-slate-400 hover:bg-white",
            uploadMutation.isPending && "pointer-events-none opacity-60",
          )}
        >
          {uploadMutation.isPending ? (
            <>
              <Loader2 className="mb-3 h-10 w-10 animate-spin text-slate-500" />
              <p className="font-medium text-slate-700">Uploading and ingesting...</p>
              <p className="mt-1 text-sm text-slate-500">
                Parsing, chunking, and indexing may take a moment.
              </p>
            </>
          ) : (
            <>
              <FileUp className="mb-3 h-10 w-10 text-slate-400" />
              <p className="font-medium text-slate-700">Drop files here or click to browse</p>
              <p className="mt-1 text-sm text-slate-500">Multiple files supported</p>
            </>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          multiple
          accept={SUPPORTED_DOCUMENT_ACCEPT}
          className="sr-only"
          aria-hidden
          onChange={(event) => {
            if (event.target.files?.length) {
              handleFiles(event.target.files);
              event.target.value = "";
            }
          }}
        />

        <div className="flex justify-end">
          <Button
            type="button"
            variant="outline"
            disabled={uploadMutation.isPending}
            onClick={() => inputRef.current?.click()}
          >
            <Upload className="h-4 w-4" />
            Browse files
          </Button>
        </div>

        {validationErrors.length > 0 ? (
          <div
            role="alert"
            className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"
          >
            <div className="mb-2 flex items-center gap-2 font-medium">
              <AlertCircle className="h-4 w-4 shrink-0" />
              Validation issues
            </div>
            <ul className="list-inside list-disc space-y-1">
              {validationErrors.map((error) => (
                <li key={error}>{error}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {uploadMutation.isSuccess && !uploadMutation.isPending ? (
          <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">
            <CheckCircle2 className="h-4 w-4 shrink-0" />
            Last upload completed successfully.
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
