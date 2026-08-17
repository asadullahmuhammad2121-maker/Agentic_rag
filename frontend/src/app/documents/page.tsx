import { AppShell } from "@/components/layout/app-shell";
import { DocumentsPageContent } from "@/components/documents/documents-page-content";

export default function DocumentsPage() {
  return (
    <AppShell
      title="Documents"
      description="Upload, browse, and inspect ingested documents"
    >
      <DocumentsPageContent />
    </AppShell>
  );
}
