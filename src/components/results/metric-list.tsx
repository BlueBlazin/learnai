import { cn } from "cn";
import { TriangleAlert } from "lucide-react";
import type { Metric } from "@/lib/api";
import { formatMetricValue, metricExceeded, metricLabel, metricLimitPhrase } from "@/lib/format";

/**
 * Measured quantities with their limits: tokens processed, slowdown against the
 * reference, simulated latency, throughput. A limit reads the way round it
 * applies: "limit 36 tokens" is a ceiling, "at least 20 tok/s" a floor (metrics
 * where higher is better). A value that misses its limit is marked with an
 * icon and the words "over the limit of" or "under the minimum of", not by
 * color alone.
 */
export function MetricList({ metrics, className }: { metrics: Metric[]; className?: string }) {
  if (metrics.length === 0) return null;
  return (
    <dl className={cn("flex flex-wrap gap-x-8 gap-y-2", className)}>
      {metrics.map((metric) => {
        const over = metricExceeded(metric);
        return (
          <div key={metric.name} className="flex flex-col">
            <dt className="text-xs text-muted-foreground">{metricLabel(metric.name)}</dt>
            <dd className="tabular flex items-baseline gap-2 text-sm">
              <span className={cn("selectable font-medium", over ? "text-fail" : "text-foreground")}>
                {formatMetricValue(metric.value, metric.unit)}
              </span>
              {metric.limit !== undefined ? (
                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                  {over ? <TriangleAlert className="size-3 self-center text-fail" aria-hidden /> : null}
                  {metricLimitPhrase(metric)} {formatMetricValue(metric.limit, metric.unit)}
                </span>
              ) : null}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}
