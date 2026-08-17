"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api/health";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface SystemHealthIndicatorProps {
  compact?: boolean;
}

function statusVariant(status: string): "success" | "warning" | "destructive" | "secondary" {
  switch (status) {
    case "ok":
      return "success";
    case "degraded":
      return "warning";
    case "unavailable":
      return "destructive";
    default:
      return "secondary";
  }
}

export function SystemHealthIndicator({ compact = false }: SystemHealthIndicatorProps) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 30_000,
  });

  if (isLoading) {
    return <Skeleton className={cn(compact ? "h-8 w-full" : "h-10 w-32")} />;
  }

  if (isError || !data) {
    return (
      <Badge variant="destructive" className={compact ? "w-full justify-center" : undefined}>
        System unreachable
      </Badge>
    );
  }

  return (
    <div className={cn("space-y-2", compact && "text-center")}>
      <div className="flex items-center gap-2">
        <Badge variant={statusVariant(data.status)}>{data.status.toUpperCase()}</Badge>
        {!compact && (
          <span className="text-xs text-slate-500">
            {data.app} v{data.version}
          </span>
        )}
      </div>
      {!compact && (
        <div className="flex flex-wrap gap-2">
          {data.components.map((component) => (
            <Badge key={component.name} variant={statusVariant(component.status)}>
              {component.name}: {component.status}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
