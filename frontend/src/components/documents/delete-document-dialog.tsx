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
          <DialogTitle>Remove document from list?</DialogTitle>
          <DialogDescription className="space-y-2 pt-1">
            <span className="block">
              This removes <strong>{document?.filename}</strong> from your browser&apos;s document
              list only.
            </span>
            <span className="block text-amber-700">
              The backend does not expose a delete API yet — vectors and chunks remain in the
              index until a server-side delete endpoint is added.
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
                Removing...
              </>
            ) : (
              "Remove from list"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
