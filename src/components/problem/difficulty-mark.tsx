import { cn } from "cn";
import type { Difficulty } from "@/lib/api";
import { DIFFICULTY_LABEL, DIFFICULTY_LEVEL } from "@/lib/labels";

const BAR_HEIGHT = ["h-1.5", "h-2.5", "h-3.5"];

/**
 * Difficulty as a three-step mark plus the word. Difficulty is a magnitude, not
 * a status, so it is drawn in ink with no hue: readable without color vision
 * and never confused with pass or fail.
 */
export function DifficultyMark({
  difficulty,
  showLabel = true,
  className,
}: {
  difficulty: Difficulty;
  showLabel?: boolean;
  className?: string;
}) {
  const level = DIFFICULTY_LEVEL[difficulty];
  return (
    <span className={cn("inline-flex items-center gap-2 whitespace-nowrap", className)}>
      <span className="flex items-end gap-0.5" aria-hidden>
        {BAR_HEIGHT.map((height, index) => (
          <span
            key={height}
            className={cn("w-0.5 rounded-sm", height, index < level ? "bg-foreground" : "bg-input")}
          />
        ))}
      </span>
      <span className={showLabel ? undefined : "sr-only"}>{DIFFICULTY_LABEL[difficulty]}</span>
    </span>
  );
}
