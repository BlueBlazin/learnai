import { cn } from "cn";
import {
  CircleCheck,
  CircleDashed,
  CircleX,
  LoaderCircle,
  type LucideIcon,
  TriangleAlert,
} from "lucide-react";
import type { ReactNode } from "react";

export type RowStatusTone = "pass" | "idle" | "warn" | "fail" | "busy";

const ICON: Record<RowStatusTone, LucideIcon> = {
  pass: CircleCheck,
  idle: CircleDashed,
  warn: TriangleAlert,
  fail: CircleX,
  busy: LoaderCircle,
};

const TONE: Record<RowStatusTone, string> = {
  pass: "text-pass",
  idle: "text-faint",
  warn: "text-warn",
  fail: "text-fail",
  busy: "animate-spin text-primary",
};

/**
 * Where a row of Settings stands, as a shape and a word: a check (in place), a
 * dashed ring (not there, and nothing is wrong), a triangle (needs the user), a
 * cross (failed), a spinner (under way). The word always says it; the shapes
 * differ, so hue only reinforces.
 */
export function RowStatus({
  tone,
  children,
  className,
}: {
  tone: RowStatusTone;
  /** The word, or a value such as a version. */
  children: ReactNode;
  className?: string;
}) {
  const Icon = ICON[tone];
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs whitespace-nowrap", className)}>
      <Icon className={cn("size-3.5 shrink-0", TONE[tone])} aria-hidden />
      <span
        className={cn("selectable tabular", tone === "pass" ? "text-foreground" : "text-muted-foreground")}
      >
        {children}
      </span>
    </span>
  );
}
