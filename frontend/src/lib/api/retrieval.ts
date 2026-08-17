import { apiRequest } from "@/lib/api/client";
import type {
  RetrievalExploreRequest,
  RetrievalExploreResponse,
} from "@/lib/types/retrieval";

export async function postRetrievalExplore(
  request: RetrievalExploreRequest,
): Promise<RetrievalExploreResponse> {
  return apiRequest<RetrievalExploreResponse>("/retrieval/explore", {
    method: "POST",
    body: request,
  });
}

export const retrievalQueryKeys = {
  explore: (query: string, topK?: number) => ["retrieval", "explore", query, topK ?? null] as const,
};
