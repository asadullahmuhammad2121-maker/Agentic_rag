import { apiRequest } from "@/lib/api/client";
import type { PublicSettingsResponse } from "@/lib/types/settings";

export async function fetchPublicSettings(): Promise<PublicSettingsResponse> {
  return apiRequest<PublicSettingsResponse>("/settings");
}

export const settingsQueryKeys = {
  public: ["settings", "public"] as const,
};
