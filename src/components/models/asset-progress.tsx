import { cn } from "cn";
import { Progress } from "@/components/ui/progress";
import type { ActiveDownload } from "@/lib/api/asset-fetch-state";
import { formatBytes, formatPercent } from "@/lib/format";

/** "214 MB of 712 MB" and "30%" for a download, or what is happening before bytes arrive. */
export function describeDownload(
  download: ActiveDownload,
  canceling: boolean,
): { text: string; percent: string | null; value: number | null } {
  if (canceling) return { text: "Canceling", percent: null, value: null };
  if (download.phase === "preparing" || download.totalBytes <= 0) {
    const message = download.phase === "preparing" ? download.message : null;
    return { text: message ?? "Starting the download", percent: null, value: null };
  }
  return {
    text: `${formatBytes(download.doneBytes)} of ${formatBytes(download.totalBytes)}`,
    percent: formatPercent(download.doneBytes, download.totalBytes),
    value: Math.min(100, (download.doneBytes / download.totalBytes) * 100),
  };
}

/**
 * A model download in flight: bytes done of the total, the percentage, and the
 * 2px progress rule. Before the size is known (the environment is being
 * checked, the file list read) it says what is happening and the rule is
 * indeterminate. `rule` alone is the bare rule, for a strip that already
 * carries the figures in its sentence.
 */
export function AssetProgress({
  name,
  download,
  canceling = false,
  variant = "full",
  className,
}: {
  /** The asset's name, for the accessible label. */
  name: string;
  download: ActiveDownload;
  canceling?: boolean;
  variant?: "full" | "rule";
  className?: string;
}) {
  const { text, percent, value } = describeDownload(download, canceling);
  const rule = (
    <Progress
      value={value}
      aria-label={`Downloading ${name}`}
      aria-valuetext={percent ? `${text}, ${percent}` : text}
      className={variant === "rule" ? className : undefined}
    />
  );
  if (variant === "rule") return rule;
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <p className="tabular flex items-baseline justify-between gap-4 text-xs text-muted-foreground">
        <span className="min-w-0 truncate">{text}</span>
        {percent ? <span className="shrink-0 text-foreground">{percent}</span> : null}
      </p>
      {rule}
    </div>
  );
}
