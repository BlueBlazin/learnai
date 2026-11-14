import type { Metric } from "@/lib/api";

const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const shortDate = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" });
const fullDate = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

/** "0.4 ms", "12.3 ms", "184 ms", "1.84 s", "1 min 5 s". */
export function formatDuration(ms: number): string {
  if (ms < 100) return `${(Math.round(ms * 10) / 10).toFixed(1)} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return seconds === 0 ? `${minutes} min` : `${minutes} min ${seconds} s`;
}

export function formatNumber(value: number): string {
  return Number.isInteger(value) ? integer.format(value) : number.format(value);
}

/** A metric value with its unit: "36 tokens", "1.08x", "41.2 ms". */
export function formatMetricValue(value: number, unit?: string): string {
  const text = formatNumber(value);
  if (!unit) return text;
  if (unit === "x") return `${text}×`;
  if (unit === "%") return `${text}%`;
  return `${text} ${unit}`;
}

/** "model_tokens" becomes "Model tokens". */
export function metricLabel(name: string): string {
  const words = name.replace(/[_-]+/g, " ").trim();
  if (words === "mean ttft") return "Mean time to first token";
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * True when a metric misses its limit: above it for ordinary metrics, below it
 * for metrics where higher is better (throughput), whose limit is a minimum.
 */
export function metricExceeded(metric: Metric): boolean {
  if (metric.limit === undefined) return false;
  return metric.higherIsBetter ? metric.value < metric.limit : metric.value > metric.limit;
}

/** The words that introduce a metric's limit: which way round it is, and whether it was met. */
export function metricLimitPhrase(metric: Metric): string {
  const missed = metricExceeded(metric);
  if (metric.higherIsBetter) return missed ? "under the minimum of" : "at least";
  return missed ? "over the limit of" : "limit";
}

const sizeNumber = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const sizeNumberFine = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

/**
 * A byte count in decimal units, the way Hugging Face and macOS show sizes:
 * "0 kB", "412 kB", "7.2 MB", "712 MB", "3.90 GB".
 */
export function formatBytes(bytes: number): string {
  const value = Math.max(0, bytes);
  if (value >= 1e9) return `${sizeNumberFine.format(value / 1e9)} GB`;
  if (value >= 1e7) return `${integer.format(value / 1e6)} MB`;
  if (value >= 1e6) return `${sizeNumber.format(value / 1e6)} MB`;
  return `${integer.format(value / 1e3)} kB`;
}

/** A declared download size, which is approximate: "712 MB", "3.9 GB", "0.5 MB". */
export function formatSizeMb(megabytes: number): string {
  if (megabytes >= 1000) return `${sizeNumber.format(megabytes / 1000)} GB`;
  if (megabytes >= 10) return `${integer.format(megabytes)} MB`;
  if (megabytes >= 0.1) return `${sizeNumber.format(megabytes)} MB`;
  return "under 0.1 MB";
}

/** A whole-number percentage of `done` in `total`; never 100 before it is finished. */
export function formatPercent(done: number, total: number): string {
  if (total <= 0) return "0%";
  const ratio = Math.min(1, Math.max(0, done / total));
  return `${ratio < 1 ? Math.min(99, Math.floor(ratio * 100)) : 100}%`;
}

/** "just now", "12 min ago", "3 h ago", "yesterday", "4 days ago", then a date. */
export function formatRelativeTime(iso: string, now: Date = new Date()): string {
  const then = new Date(iso);
  const seconds = Math.max(0, (now.getTime() - then.getTime()) / 1000);
  if (seconds < 45) return "just now";
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))} min ago`;
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const days = Math.ceil((startOfToday - then.getTime()) / 86_400_000);
  if (days <= 0) return `${Math.round(seconds / 3600)} h ago`;
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  return then.getFullYear() === now.getFullYear()
    ? shortDate.format(then)
    : `${shortDate.format(then)}, ${then.getFullYear()}`;
}

export function formatDateTime(iso: string): string {
  return fullDate.format(new Date(iso));
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${formatNumber(count)} ${count === 1 ? singular : plural}`;
}
