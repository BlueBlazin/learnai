import { cn } from "cn";
import { Progress } from "@/components/ui/progress";
import { formatNumber } from "@/lib/format";

/**
 * Solved-of-total as a fraction and a hairline bar. The bar is deliberately
 * thin: it shows proportion at a glance without turning progress into a gauge.
 */
export function ProgressRule({
  value,
  total,
  label,
  showCount = true,
  className,
}: {
  value: number;
  total: number;
  /** Accessible name, for example "Attention progress". */
  label: string;
  showCount?: boolean;
  className?: string;
}) {
  return (
    <span className={cn("flex items-center gap-3", className)}>
      <Progress
        value={total === 0 ? 0 : (value / total) * 100}
        aria-label={label}
        aria-valuetext={`${value} of ${total} solved`}
        className="min-w-0 flex-1 gap-0"
      />
      {showCount ? (
        <span className="tabular shrink-0 text-xs text-muted-foreground">
          <span className="text-foreground">{formatNumber(value)}</span>
          {" / "}
          {formatNumber(total)}
        </span>
      ) : null}
    </span>
  );
}
