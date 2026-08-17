import { AppShell } from "@/components/layout/app-shell";
import { DocumentDetailsContent } from "@/components/documents/document-details-content";

interface DocumentDetailsPageProps {
  params: Promise<{ documentId: string }>;
}

export default async function DocumentDetailsPage({ params }: DocumentDetailsPageProps) {
  const { documentId } = await params;

  return (
    <AppShell title="Document details" description="Metadata and indexing summary">
      <DocumentDetailsContent documentId={documentId} />
    </AppShell>
  );
}
