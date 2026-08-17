import { AppShell } from "@/components/layout/app-shell";
import { AgentRunDetailContent } from "@/components/agent-runs/agent-run-detail-content";

interface AgentRunDetailPageProps {
  params: Promise<{ runId: string }>;
}

export default async function AgentRunDetailPage({ params }: AgentRunDetailPageProps) {
  const { runId } = await params;

  return (
    <AppShell title="Agent run details" description="Execution trace and final answer">
      <AgentRunDetailContent runId={runId} />
    </AppShell>
  );
}
