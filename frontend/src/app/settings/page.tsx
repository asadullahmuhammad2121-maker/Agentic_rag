import { AppShell } from "@/components/layout/app-shell";
import { SettingsPageContent } from "@/components/settings/settings-page-content";

export default function SettingsPage() {
  return (
    <AppShell
      title="Settings"
      description="Read-only view of backend configuration and system status"
    >
      <SettingsPageContent />
    </AppShell>
  );
}
