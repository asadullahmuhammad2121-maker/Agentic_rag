"use client";

import { Loader2 } from "lucide-react";
import type { StoredDocument } from "@/lib/types/documents";
import { useDeleteDocument } from "@/lib/hooks/use-documents";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface DeleteDocumentDialogProps {
  document: StoredDocument | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function DeleteDocumentDialog({
  document,
  open,
  onOpenChange,
}: DeleteDocumentDialogProps) {
  const deleteMutation = useDeleteDocument();

  const handleConfirm = () => {
    if (!document) return;
    deleteMutation.mutate(document.document_id, {
      onSuccess: () => onOpenChange(false),
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete document?</DialogTitle>
          <DialogDescription className="space-y-2 pt-1">
            <span className="block">
              This permanently deletes <strong>{document?.filename}</strong> from the knowledge base,
              including all indexed chunks and vectors.
            </span>
            <span className="block">
              You can upload the same file again afterward; it will be ingested as a new document.
            </span>
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={deleteMutation.isPending}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleConfirm}
            disabled={deleteMutation.isPending || !document}
          >
            {deleteMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Deleting...
              </>
            ) : (
              "Delete document"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
