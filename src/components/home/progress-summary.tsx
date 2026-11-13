import { DifficultyMark } from "@/components/problem/difficulty-mark";
import { ProgressRule } from "@/components/problem/progress-rule";
import { DIFFICULTIES, type Stats } from "@/lib/api";
import { DIFFICULTY_LABEL } from "@/lib/labels";

/**
 * Overall progress, stated as a sentence and three plain rows. No rings, no
 * percentages in display type: the counts are the whole story.
 */
export function ProgressSummary({ stats }: { stats: Stats }) {
  const streak =
    stats.streakDays === 0
      ? "No streak at the moment; an accepted submission today starts one."
      : stats.streakDays === 1
        ? "You have solved something one day running."
        : `You have solved something ${stats.streakDays} days running.`;
  return (
    <div className="flex flex-col gap-5">
      <p className="selectable max-w-measure font-serif text-lg text-pretty">
        {stats.solved === 0 ? (
          <>None of the {stats.total} problems solved yet. </>
        ) : (
          <>
            <span className="font-semibold tabular">{stats.solved}</span> of {stats.total} problems solved.{" "}
          </>
        )}
        <span className="text-muted-foreground">{streak}</span>
      </p>
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
    </div>
  );
}
