import { cn } from "cn";
import {
  Ban,
  CircleCheck,
  CircleX,
  type LucideIcon,
  OctagonAlert,
  PackageX,
  Timer,
  Unplug,
} from "lucide-react";
import type { Verdict } from "@/lib/api";
import { TONE_TEXT, VERDICT_LABEL, VERDICT_TONE } from "@/lib/labels";

const VERDICT_ICON: Record<Verdict, LucideIcon> = {
  accepted: CircleCheck,
  wrong_answer: CircleX,
  runtime_error: OctagonAlert,
  time_limit: Timer,
  import_error: PackageX,
  harness_error: Unplug,
  canceled: Ban,
};

/** A run's verdict as an icon and a word. Each verdict has its own icon, so color is never the only signal. */
export function VerdictMark({
  verdict,
  showLabel = true,
  size = "sm",
  className,
}: {
  verdict: Verdict;
  showLabel?: boolean;
  /** `lg` pairs with display type in a verdict header. */
  size?: "sm" | "lg";
  className?: string;
}) {
  const Icon = VERDICT_ICON[verdict];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 font-medium whitespace-nowrap",
        TONE_TEXT[VERDICT_TONE[verdict]],
        className,
      )}
    >
      <Icon className={cn("shrink-0", size === "lg" ? "size-4" : "size-3.5")} aria-hidden />
      <span className={showLabel ? undefined : "sr-only"}>{VERDICT_LABEL[verdict]}</span>
    </span>
  );
}
