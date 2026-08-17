import { AppShell } from "@/components/layout/app-shell";
import { DashboardContent } from "@/components/dashboard/dashboard-content";

export default function DashboardPage() {
  return (
    <AppShell
      title="Dashboard"
      description="System overview and operational health"
    >
      <DashboardContent />
    </AppShell>
  );
}
