"use client";

import { Bot, Menu } from "lucide-react";
import { useState } from "react";
import { SidebarNav } from "@/components/layout/sidebar-nav";
import { SystemHealthIndicator } from "@/components/layout/system-health-indicator";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";

export function AppSidebar() {
  return (
    <aside className="hidden w-64 shrink-0 border-r border-slate-200 bg-white md:flex md:flex-col">
      <div className="flex h-16 items-center gap-2 border-b border-slate-200 px-6">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900 text-white">
          <Bot className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-900">Agentic RAG</p>
          <p className="text-xs text-slate-500">Intelligent retrieval</p>
        </div>
      </div>
      <SidebarNav />
      <div className="mt-auto border-t border-slate-200 p-4">
        <SystemHealthIndicator compact />
      </div>
    </aside>
  );
}

export function MobileSidebarTrigger() {
  const [open, setOpen] = useState(false);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" size="icon" className="md:hidden" aria-label="Open menu">
          <Menu className="h-4 w-4" />
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="p-0">
        <div className="flex h-16 items-center gap-2 border-b border-slate-200 px-6">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900 text-white">
            <Bot className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm font-semibold">Agentic RAG</p>
            <p className="text-xs text-slate-500">Intelligent retrieval</p>
          </div>
        </div>
        <SidebarNav onNavigate={() => setOpen(false)} />
      </SheetContent>
    </Sheet>
  );
}
