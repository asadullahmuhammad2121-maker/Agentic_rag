import { apiRequest } from "@/lib/api/client";
import type {
  AgentRunDetail,
  AgentRunListParams,
  AgentRunListResponse,
} from "@/lib/types/agent-runs";

function buildQuery(params: AgentRunListParams): string {
  const searchParams = new URLSearchParams();
  if (params.search?.trim()) searchParams.set("search", params.search.trim());
  if (params.status) searchParams.set("status", params.status);
  if (params.limit !== undefined) searchParams.set("limit", String(params.limit));
  if (params.offset !== undefined) searchParams.set("offset", String(params.offset));
  const query = searchParams.toString();
  return query ? `?${query}` : "";
}

export async function fetchAgentRuns(
  params: AgentRunListParams = {},
): Promise<AgentRunListResponse> {
  return apiRequest<AgentRunListResponse>(`/agent/runs${buildQuery(params)}`);
}

export async function fetchAgentRun(runId: string): Promise<AgentRunDetail> {
  return apiRequest<AgentRunDetail>(`/agent/runs/${encodeURIComponent(runId)}`);
}

export const agentRunQueryKeys = {
  all: ["agent-runs"] as const,
  list: (params: AgentRunListParams) => ["agent-runs", "list", params] as const,
  detail: (runId: string) => ["agent-runs", "detail", runId] as const,
};
