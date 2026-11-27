import { cn } from "cn";
import type { Difficulty, DifficultyLevel } from "@/lib/api";
import { DIFFICULTY_FILL, DIFFICULTY_LABEL } from "@/lib/labels";

const BAR_HEIGHT = ["h-1.5", "h-2.5", "h-3.5"];

/**
 * Difficulty as three bars plus the word. The word and the color of the bars
 * (green, yellow, red) name the difficulty; how many bars are filled is the
 * level within it, so an easy problem at level 3 sits just under a medium one
 * at level 1. Without a `level` the mark stands for the whole difficulty (a
 * filter, a count) and every bar is filled.
 */
export function DifficultyMark({
  difficulty,
  level,
  showLabel = true,
  className,
}: {
  difficulty: Difficulty;
  level?: DifficultyLevel;
  showLabel?: boolean;
  className?: string;
}) {
  const label = DIFFICULTY_LABEL[difficulty];
  return (
    <span
      className={cn("inline-flex items-center gap-2 whitespace-nowrap", className)}
      title={level ? `${label}, level ${level} of 3` : undefined}
    >
      <span className="flex items-end gap-0.5" aria-hidden>
        {BAR_HEIGHT.map((height, index) => (
          <span
            key={height}
            className={cn(
              "w-0.5 rounded-sm",
              height,
              index < (level ?? 3) ? DIFFICULTY_FILL[difficulty] : "bg-input",
            )}
          />
        ))}
      </span>
      <span className={showLabel ? undefined : "sr-only"}>{label}</span>
      {level ? <span className="sr-only">, level {level} of 3</span> : null}
    </span>
  );
}
