"use client";

import { useState } from "react";
import {
  FileText,
  History,
  LayoutDashboard,
  MessageSquare,
  Search,
  Settings,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { navItems, type NavIcon } from "@/lib/constants/navigation";
import { Badge } from "@/components/ui/badge";

const iconMap: Record<NavIcon, React.ComponentType<{ className?: string }>> = {
  LayoutDashboard,
  MessageSquare,
  FileText,
  Search,
  History,
  Settings,
};

interface SidebarNavProps {
  onNavigate?: () => void;
}

export function SidebarNav({ onNavigate }: SidebarNavProps) {
  const pathname = usePathname();

  return (
    <nav className="flex flex-1 flex-col gap-1 px-3 py-4">
      {navItems.map((item) => {
        const Icon = iconMap[item.icon];
        const isActive =
          item.href === "/"
            ? pathname === "/"
            : pathname.startsWith(item.href);
        const disabled = "disabled" in item && item.disabled;

        if (disabled) {
          return (
            <div
              key={item.href}
              className="flex items-center justify-between rounded-lg px-3 py-2 text-sm text-slate-400"
            >
              <span className="flex items-center gap-3">
                <Icon className="h-4 w-4" />
                {item.label}
              </span>
              <Badge variant="secondary" className="text-[10px]">
                Soon
              </Badge>
            </div>
          );
        }

        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
              isActive
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
            )}
          >
            <Icon className="h-4 w-4" />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
