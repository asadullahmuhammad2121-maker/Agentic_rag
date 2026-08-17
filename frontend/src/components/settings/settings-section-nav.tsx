"use client";

import { Bot, Cpu, Globe, Search, Settings2 } from "lucide-react";
import type { SettingsSectionId } from "@/lib/types/settings";
import { cn } from "@/lib/utils";

const sections: {
  id: SettingsSectionId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  { id: "general", label: "General", icon: Settings2 },
  { id: "rag", label: "RAG", icon: Cpu },
  { id: "agent", label: "Agent", icon: Bot },
  { id: "search", label: "Search", icon: Search },
  { id: "system", label: "System", icon: Globe },
];

interface SettingsSectionNavProps {
  active: SettingsSectionId;
  onChange: (section: SettingsSectionId) => void;
}

export function SettingsSectionNav({ active, onChange }: SettingsSectionNavProps) {
  return (
    <nav
      aria-label="Settings sections"
      className="flex flex-col gap-1 rounded-lg border border-slate-200 bg-white p-2 lg:sticky lg:top-6"
    >
      {sections.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          type="button"
          onClick={() => onChange(id)}
          className={cn(
            "flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm font-medium transition-colors",
            active === id
              ? "bg-slate-900 text-white"
              : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
          )}
        >
          <Icon className="h-4 w-4 shrink-0" />
          {label}
        </button>
      ))}
    </nav>
  );
}

export function SettingsSectionTabs({ active, onChange }: SettingsSectionNavProps) {
  return (
    <div className="flex gap-2 overflow-x-auto pb-1 lg:hidden">
      {sections.map(({ id, label }) => (
        <button
          key={id}
          type="button"
          onClick={() => onChange(id)}
          className={cn(
            "shrink-0 rounded-full border px-3 py-1.5 text-xs font-medium",
            active === id
              ? "border-slate-900 bg-slate-900 text-white"
              : "border-slate-200 bg-white text-slate-600",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
