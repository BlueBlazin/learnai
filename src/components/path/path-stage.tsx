import type { ProblemSummary, PathStage as Stage } from "@/lib/api";
import { pluralize } from "@/lib/format";
import { type PathEntry, romanNumeral } from "@/lib/paths";
import { sceneUrl } from "@/lib/scenes";
import { stageColor, stageMark } from "@/lib/stage-colors";
import { PathEntryList } from "./path-entry-list";
import { StageSwatch } from "./stage-swatch";

/** The id of a stage's section on the Path screen, for scrolling to it. */
export function stageSectionId(number: number): string {
  return `path-stage-${number}`;
}

/**
 * One stage of a path, set as a part in a table of contents: under an ink
 * rule, the part's number, title, and what it covers in the margin column, and
 * its problems beside them. A part with nothing solved states its length, not
 * a count of zero. A stage with a scene opens its margin column with the scene
 * as a plate and the place name under it; the stage color is a swatch before
 * the part number. The colors come from `stageColorVars` on an ancestor.
 */
export function PathStage({
  number,
  stage,
  entries,
  nextId,
  unmet,
}: {
  /** 1-based position of the stage in the path. */
  number: number;
  stage: Stage;
  entries: PathEntry[];
  nextId?: string | null;
  unmet?: Map<string, ProblemSummary[]>;
}) {
  const part = `Part ${romanNumeral(number)}`;
  const solved = entries.filter((entry) => entry.problem.status === "solved").length;
  const color = stageColor(stage);
  const scene = sceneUrl(stage.scene);
  const headingId = `path-part-${number}`;
  return (
    <section
      id={stageSectionId(number)}
      aria-labelledby={headingId}
      className="flex gap-14 border-t border-rule pt-3 pb-10"
    >
      <header className="w-60 shrink-0 pt-2">
        {scene ? (
          <figure className="mb-4">
            <img
              src={scene}
              alt=""
              draggable={false}
              className="aspect-3/2 w-full rounded-sm object-cover object-center"
            />
            {stage.place ? (
              <figcaption className="mt-1.5 text-xs text-muted-foreground">{stage.place}</figcaption>
            ) : null}
          </figure>
        ) : null}
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          {color ? <StageSwatch color={stageMark(number - 1)} /> : null}
          {part}
        </p>
        <h2 id={headingId} className="selectable mt-1 font-serif text-lg font-medium text-balance">
          {stage.title}
        </h2>
        {stage.description ? (
          <p className="selectable mt-2 text-sm text-pretty text-muted-foreground">{stage.description}</p>
        ) : null}
        <p className="tabular mt-3 text-xs text-muted-foreground">
          {solved === 0
            ? pluralize(entries.length, "problem")
            : solved === entries.length
              ? `All ${entries.length} solved`
              : `${solved} of ${entries.length} solved`}
        </p>
      </header>
      <PathEntryList entries={entries} nextId={nextId} unmet={unmet} label={`${part}: ${stage.title}`} />
    </section>
  );
}
