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

export function metricExceeded(metric: Metric): boolean {
  return metric.limit !== undefined && metric.value > metric.limit;
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
