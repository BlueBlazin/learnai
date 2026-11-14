import { cn } from "cn";
import { CircleCheck, CircleDashed, CircleDotDashed, type LucideIcon } from "lucide-react";
import type { AssetState } from "@/lib/api";
import { ASSET_STATE_LABEL } from "@/lib/labels";

const ICON: Record<AssetState, LucideIcon> = {
  present: CircleCheck,
  missing: CircleDashed,
  incomplete: CircleDotDashed,
};

const TONE: Record<AssetState, string> = {
  present: "text-pass",
  missing: "text-faint",
  incomplete: "text-warn",
};

/**
 * Whether a model is on disk, as a shape and a word: a check (downloaded), a
 * dashed ring (not downloaded), a dashed ring with a center (a download that
 * did not finish). The shapes differ, so the hue is reinforcement only.
 */
export function AssetStateMark({
  state,
  showLabel = true,
  className,
}: {
  state: AssetState;
  showLabel?: boolean;
  className?: string;
}) {
  const Icon = ICON[state];
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs whitespace-nowrap", className)}>
      <Icon className={cn("size-3.5 shrink-0", TONE[state])} aria-hidden />
      <span className={showLabel ? "text-muted-foreground" : "sr-only"}>{ASSET_STATE_LABEL[state]}</span>
    </span>
  );
}
