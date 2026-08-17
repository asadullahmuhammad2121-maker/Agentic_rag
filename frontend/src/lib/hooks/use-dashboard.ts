"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchAgentRuns } from "@/lib/api/agent-runs";
import { fetchHealth, fetchLive, fetchReady } from "@/lib/api/health";
import { fetchPublicSettings } from "@/lib/api/settings";

export function useDashboardData() {
  const healthQuery = useQuery({
    queryKey: ["dashboard", "health"],
    queryFn: fetchHealth,
    refetchInterval: 30_000,
  });

  const readyQuery = useQuery({
    queryKey: ["dashboard", "ready"],
    queryFn: fetchReady,
    refetchInterval: 30_000,
    retry: false,
  });

  const liveQuery = useQuery({
    queryKey: ["dashboard", "live"],
    queryFn: fetchLive,
    refetchInterval: 30_000,
    retry: false,
  });

  const settingsQuery = useQuery({
    queryKey: ["dashboard", "settings"],
    queryFn: fetchPublicSettings,
    refetchInterval: 60_000,
    retry: false,
  });

  const agentRunsQuery = useQuery({
    queryKey: ["dashboard", "agent-runs"],
    queryFn: () => fetchAgentRuns({ limit: 5, offset: 0 }),
    refetchInterval: 60_000,
    retry: false,
  });

  const isLoading =
    healthQuery.isLoading ||
    readyQuery.isLoading ||
    liveQuery.isLoading ||
    settingsQuery.isLoading;

  const isError = healthQuery.isError && settingsQuery.isError;

  return {
    health: healthQuery.data,
    healthError: healthQuery.isError,
    ready: readyQuery.data,
    readyReachable: readyQuery.isSuccess,
    readyError: readyQuery.isError,
    live: liveQuery.data,
    liveReachable: liveQuery.isSuccess,
    settings: settingsQuery.data,
    settingsError: settingsQuery.isError,
    agentRuns: agentRunsQuery.data,
    agentRunsError: agentRunsQuery.isError,
    isLoading,
    isError,
    refetch: () => {
      void healthQuery.refetch();
      void readyQuery.refetch();
      void liveQuery.refetch();
      void settingsQuery.refetch();
      void agentRunsQuery.refetch();
    },
  };
}
