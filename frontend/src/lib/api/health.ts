import { apiRequest } from "@/lib/api/client";
import type { HealthResponse, LivenessResponse } from "@/lib/types/health";

export async function fetchHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health");
}

export async function fetchReady(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/ready");
}

export async function fetchLive(): Promise<LivenessResponse> {
  return apiRequest<LivenessResponse>("/live");
}
