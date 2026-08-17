import { AppShell } from "@/components/layout/app-shell";
import { AgentRunsPageContent } from "@/components/agent-runs/agent-runs-page-content";

export default function AgentRunsPage() {
  return (
    <AppShell
      title="Agent Runs"
      description="Observability and history for agent executions"
    >
      <AgentRunsPageContent />
    </AppShell>
  );
}
