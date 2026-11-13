import { cn } from "cn";
import type { ProblemStatus } from "@/lib/api";
import { STATUS_LABEL } from "@/lib/labels";

const TONE: Record<ProblemStatus, string> = {
  todo: "text-faint",
  attempted: "text-warn",
  solved: "text-pass",
};

/**
 * Progress on a problem as a shape: an empty ring, a half-filled ring, or a
 * filled disc with a check. The shapes differ, so the hue is reinforcement only.
 */
export function StatusMark({
  status,
  showLabel = false,
  className,
}: {
  status: ProblemStatus;
  showLabel?: boolean;
  className?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-2 whitespace-nowrap", className)}>
      <svg viewBox="0 0 14 14" className={cn("size-3.5 shrink-0", TONE[status])} aria-hidden>
        {status === "solved" ? (
          <>
            <circle cx="7" cy="7" r="6.25" fill="currentColor" />
            <path
              d="M4.2 7.2 6.2 9.1 9.9 5"
              fill="none"
              className="stroke-paper"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </>
        ) : (
          <>
            <circle cx="7" cy="7" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
            {status === "attempted" ? <path d="M7 1.5a5.5 5.5 0 0 0 0 11Z" fill="currentColor" /> : null}
          </>
        )}
      </svg>
      <span className={showLabel ? undefined : "sr-only"}>{STATUS_LABEL[status]}</span>
    </span>
  );
}
