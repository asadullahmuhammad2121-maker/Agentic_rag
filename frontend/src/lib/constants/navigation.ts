export const navItems = [
  { href: "/", label: "Dashboard", icon: "LayoutDashboard" as const },
  { href: "/agent-chat", label: "Agent Chat", icon: "MessageSquare" as const },
  { href: "/documents", label: "Documents", icon: "FileText" as const },
  { href: "/retrieval", label: "Retrieval", icon: "Search" as const },
  { href: "/agent-runs", label: "Agent Runs", icon: "History" as const },
  { href: "/settings", label: "Settings", icon: "Settings" as const },
] as const;

export type NavIcon = (typeof navItems)[number]["icon"];
