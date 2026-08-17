export interface ComponentHealth {
  name: string;
  status: "ok" | "degraded" | "unavailable";
  detail: string | null;
  metadata?: Record<string, unknown>;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "unavailable";
  app: string;
  version: string;
  environment: string;
  components: ComponentHealth[];
}

export interface LivenessResponse {
  status: string;
}
