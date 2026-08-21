"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  documentQueryKeys,
  deleteDocument,
  getDocument,
  listDocuments,
  uploadDocuments,
} from "@/lib/api/documents";
import { ApiError } from "@/lib/api/client";
import { uploadResponseToDocuments } from "@/lib/types/documents";

export function useDocumentsList() {
  return useQuery({
    queryKey: documentQueryKeys.all,
    queryFn: listDocuments,
    staleTime: 30_000,
  });
}

export function useDocumentDetail(documentId: string) {
  return useQuery({
    queryKey: documentQueryKeys.detail(documentId),
    queryFn: () => getDocument(documentId),
    staleTime: 30_000,
    retry: false,
  });
}

export function useUploadDocuments() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: uploadDocuments,
    onSuccess: (response) => {
      const ingested = uploadResponseToDocuments(response);
      void queryClient.invalidateQueries({ queryKey: documentQueryKeys.all });
      for (const doc of ingested) {
        void queryClient.invalidateQueries({
          queryKey: documentQueryKeys.detail(doc.document_id),
        });
      }
      const count = ingested.length;
      toast.success(
        count === 1
          ? `"${ingested[0].filename}" ingested successfully`
          : `${count} documents ingested successfully`,
      );
    },
    onError: (error: Error) => {
      if (error instanceof ApiError && error.code === "duplicate_document") {
        const existingId =
          typeof error.details.existing_document_id === "string"
            ? error.details.existing_document_id
            : null;
        toast.error(
          existingId
            ? `${error.message} (existing ID: ${existingId})`
            : error.message,
        );
        return;
      }
      toast.error(error instanceof ApiError ? error.message : "Upload failed");
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteDocument,
    onSuccess: (_result, documentId) => {
      void queryClient.invalidateQueries({ queryKey: documentQueryKeys.all });
      void queryClient.removeQueries({ queryKey: documentQueryKeys.detail(documentId) });
      toast.success("Document deleted");
    },
    onError: (error: Error) => {
      toast.error(error instanceof ApiError ? error.message : "Could not delete document");
    },
  });
}
