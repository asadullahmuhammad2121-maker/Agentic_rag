import { AppShell } from "@/components/layout/app-shell";
import { AgentChatPanel } from "@/components/agent-chat/agent-chat-panel";

export default function AgentChatPage() {
  return (
    <AppShell
      title="Agent Chat"
      description="Query the agentic RAG orchestrator with tool routing and citations"
    >
      <AgentChatPanel />
    </AppShell>
  );
}
