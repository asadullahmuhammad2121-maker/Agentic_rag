import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface SettingsFieldProps {
  label: string;
  value: string | number | boolean;
  description?: string;
  readOnly?: boolean;
}

export function SettingsField({
  label,
  value,
  description,
  readOnly = true,
}: SettingsFieldProps) {
  const display =
    typeof value === "boolean" ? (value ? "Enabled" : "Disabled") : String(value);

  return (
    <div className="flex flex-col gap-1 border-b border-slate-100 py-3 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0 flex-1">
        <dt className="text-sm font-medium text-slate-900">{label}</dt>
        {description ? <p className="mt-0.5 text-xs text-slate-500">{description}</p> : null}
      </div>
      <dd className="flex items-center gap-2 sm:max-w-[45%] sm:justify-end">
        <span className="break-all text-sm text-slate-700">{display}</span>
        {readOnly ? (
          <Badge variant="secondary" className="shrink-0 text-[10px]">
            Read-only
          </Badge>
        ) : null}
      </dd>
    </div>
  );
}

interface SettingsSectionCardProps {
  title: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
}

export function SettingsSectionCard({
  title,
  description,
  children,
  className,
}: SettingsSectionCardProps) {
  return (
    <section className={cn("rounded-lg border border-slate-200 bg-white", className)}>
      <div className="border-b border-slate-100 px-4 py-4 sm:px-6">
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        {description ? <p className="mt-1 text-sm text-slate-500">{description}</p> : null}
      </div>
      <dl className="px-4 sm:px-6">{children}</dl>
    </section>
  );
}

interface StatusBadgeProps {
  active: boolean;
  activeLabel?: string;
  inactiveLabel?: string;
}

export function StatusBadge({
  active,
  activeLabel = "Active",
  inactiveLabel = "Inactive",
}: StatusBadgeProps) {
  return (
    <Badge variant={active ? "success" : "secondary"}>
      {active ? activeLabel : inactiveLabel}
    </Badge>
  );
}
