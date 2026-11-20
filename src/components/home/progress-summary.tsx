import { DifficultyMark } from "@/components/problem/difficulty-mark";
import { ProgressRule } from "@/components/problem/progress-rule";
import { DIFFICULTIES, type Stats } from "@/lib/api";
import { DIFFICULTY_LABEL } from "@/lib/labels";

/** Solved and total problems by difficulty, as three plain rows. No rings, no percentages in display type. */
export function ProgressSummary({ stats }: { stats: Stats }) {
  return (
    <ul className="flex flex-col gap-2.5">
      {DIFFICULTIES.map((difficulty) => {
        const [solved, total] = stats.byDifficulty[difficulty];
        return (
          <li key={difficulty} className="flex items-center gap-4 text-xs">
            <DifficultyMark difficulty={difficulty} className="w-20" />
            <ProgressRule
              value={solved}
              total={total}
              label={`${DIFFICULTY_LABEL[difficulty]} problems solved`}
              className="flex-1"
            />
          </li>
        );
      })}
    </ul>
  );
}
