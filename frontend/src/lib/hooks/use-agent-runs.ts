"use client";

import { useQuery } from "@tanstack/react-query";
import { agentRunQueryKeys, fetchAgentRun, fetchAgentRuns } from "@/lib/api/agent-runs";
import type { AgentRunListParams } from "@/lib/types/agent-runs";

export function useAgentRuns(params: AgentRunListParams) {
  return useQuery({
    queryKey: agentRunQueryKeys.list(params),
    queryFn: () => fetchAgentRuns(params),
    placeholderData: (previous) => previous,
  });
}

export function useAgentRunDetail(runId: string) {
  return useQuery({
    queryKey: agentRunQueryKeys.detail(runId),
    queryFn: () => fetchAgentRun(runId),
    enabled: Boolean(runId),
  });
}
