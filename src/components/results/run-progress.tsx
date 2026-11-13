import { LoaderCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { RunMode } from "@/lib/api";

/**
 * A run in flight: what the judge is doing, how many cases have finished, and
 * a way to stop it. Covers the preparing phase (no cases yet) and the running phase.
 */
export function RunProgress({
  mode,
  message,
  finished,
  total,
  canceling,
  onCancel,
}: {
  mode: RunMode | null;
  /** Shown while preparing, for example "Checking the Python environment". */
  message: string | null;
  finished: number;
  total: number;
  canceling: boolean;
  onCancel: () => void;
}) {
  const verb = mode === "submit" ? "Submitting" : "Running";
  const text = canceling
    ? "Canceling"
    : message
      ? message
      : total > 0
        ? `${verb}: ${finished} of ${total} cases finished`
        : `${verb}`;
  return (
    <div className="flex flex-col gap-3" role="status" aria-live="polite">
      <div className="flex items-center justify-between gap-4">
        <p className="flex min-w-0 items-center gap-2 text-sm">
          <LoaderCircle className="size-3.5 shrink-0 animate-spin text-primary" aria-hidden />
          <span className="truncate">{text}</span>
        </p>
        <Button variant="outline" size="xs" onClick={onCancel} disabled={canceling}>
          Cancel
        </Button>
      </div>
      <Progress value={total > 0 ? (finished / total) * 100 : null} aria-label="Run progress" />
    </div>
  );
}
