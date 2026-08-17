"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchHealth, fetchLive, fetchReady } from "@/lib/api/health";
import { fetchPublicSettings, settingsQueryKeys } from "@/lib/api/settings";

export function usePublicSettings() {
  return useQuery({
    queryKey: settingsQueryKeys.public,
    queryFn: fetchPublicSettings,
    staleTime: 60_000,
  });
}

export function useSystemStatus() {
  return useQuery({
    queryKey: ["settings", "system-status"],
    queryFn: async () => {
      const [health, ready, live] = await Promise.allSettled([
        fetchHealth(),
        fetchReady(),
        fetchLive(),
      ]);
      return {
        health: health.status === "fulfilled" ? health.value : null,
        healthError: health.status === "rejected",
        ready: ready.status === "fulfilled" ? ready.value : null,
        readyReachable: ready.status === "fulfilled",
        readyStatus: ready.status === "fulfilled" ? ready.value.status : null,
        live: live.status === "fulfilled" ? live.value : null,
        liveReachable: live.status === "fulfilled",
      };
    },
    refetchInterval: 30_000,
  });
}
