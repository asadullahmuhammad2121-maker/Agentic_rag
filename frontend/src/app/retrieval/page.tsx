import { AppShell } from "@/components/layout/app-shell";
import { RetrievalExplorerContent } from "@/components/retrieval/retrieval-explorer-content";

export default function RetrievalPage() {
  return (
    <AppShell
      title="Retrieval Explorer"
      description="Inspect vector, BM25, hybrid fusion, and final retrieved chunks"
    >
      <RetrievalExplorerContent />
    </AppShell>
  );
}
