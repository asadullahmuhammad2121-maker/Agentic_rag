"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { documentQueryKeys, uploadDocuments } from "@/lib/api/documents";
import { ApiError } from "@/lib/api/client";
import {
  getStoredDocument,
  listStoredDocuments,
  removeStoredDocument,
  upsertStoredDocuments,
} from "@/lib/documents/store";
import { uploadResponseToDocuments } from "@/lib/types/documents";

export function useDocumentsList() {
  return useQuery({
    queryKey: documentQueryKeys.all,
    queryFn: listStoredDocuments,
    staleTime: Infinity,
  });
}

export function useDocumentDetail(documentId: string) {
  return useQuery({
    queryKey: documentQueryKeys.detail(documentId),
    queryFn: () => {
      const document = getStoredDocument(documentId);
      if (!document) {
        throw new Error("Document not found in this browser session.");
      }
      return document;
    },
    staleTime: Infinity,
    retry: false,
  });
}

export function useUploadDocuments() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: uploadDocuments,
    onSuccess: (response) => {
      const ingested = uploadResponseToDocuments(response);
      upsertStoredDocuments(ingested);
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
    mutationFn: async (documentId: string) => {
      removeStoredDocument(documentId);
      return documentId;
    },
    onSuccess: (documentId) => {
      void queryClient.invalidateQueries({ queryKey: documentQueryKeys.all });
      void queryClient.removeQueries({ queryKey: documentQueryKeys.detail(documentId) });
      toast.success("Document removed from this browser list");
    },
    onError: () => {
      toast.error("Could not remove document");
    },
  });
}
