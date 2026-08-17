"use client";

import { MobileSidebarTrigger } from "@/components/layout/app-sidebar";
import { SystemHealthIndicator } from "@/components/layout/system-health-indicator";

interface AppHeaderProps {
  title: string;
  description?: string;
}

export function AppHeader({ title, description }: AppHeaderProps) {
  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/80">
      <div className="flex h-16 items-center justify-between gap-4 px-4 md:px-6">
        <div className="flex items-center gap-3">
          <MobileSidebarTrigger />
          <div>
            <h1 className="text-lg font-semibold text-slate-900">{title}</h1>
            {description ? (
              <p className="text-sm text-slate-500">{description}</p>
            ) : null}
          </div>
        </div>
        <div className="hidden md:block">
          <SystemHealthIndicator />
        </div>
      </div>
    </header>
  );
}
