export function formatBoolean(value: boolean): string {
  return value ? "Enabled" : "Disabled";
}

export function formatOptionalNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "Not set";
  return String(value);
}

export function formatStatusLabel(status: string): string {
  switch (status) {
    case "ok":
      return "OK";
    case "degraded":
      return "Degraded";
    case "unavailable":
      return "Unavailable";
    case "alive":
      return "Alive";
    default:
      return status;
  }
}
