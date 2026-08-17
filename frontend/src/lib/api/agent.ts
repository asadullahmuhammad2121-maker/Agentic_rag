import { apiRequest } from "@/lib/api/client";
import type { AgentQueryRequest, AgentQueryResponse } from "@/lib/types/agent";

export async function postAgentQuery(
  request: AgentQueryRequest,
): Promise<AgentQueryResponse> {
  return apiRequest<AgentQueryResponse>("/agent/query", {
    method: "POST",
    body: request,
  });
}
