import { AppLink } from "@/components/app/app-link";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import { ModuleSwatch } from "@/components/curriculum/module-swatch";
import type { ProblemDetail, ProblemSummary } from "@/lib/api";
import type { Place } from "@/lib/curriculum";
import { REQUIRES_LABEL } from "@/lib/labels";
import { routes } from "@/lib/routes";
import { DifficultyMark } from "./difficulty-mark";
import { JudgeList } from "./judge-list";
import { PrerequisiteList } from "./prerequisite-list";
import { StatusMark } from "./status-mark";

/**
 * The title block of a statement, set like the head of a paper: the module
 * the problem belongs to (a link to the module's page) and whether it is one of
 * the module's core problems or a further one, its title, and a ruled strip of
 * facts.
 */
export function ProblemHeader({
  problem,
  place,
  prerequisites = [],
}: {
  problem: ProblemDetail;
  /** Where the problem sits in the curriculum; until it is known only the status shows. */
  place?: Place | null;
  /** The problems this one builds on, with their status, when known. */
  prerequisites?: ProblemSummary[];
}) {
  return (
    <header className="max-w-measure pb-7">
      <p className="flex min-h-4 flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
        {place ? (
          <>
            <span className="inline-flex min-w-0 items-center gap-2">
              <ModuleSwatch color={place.module.color} />
              <AppLink
                to={routes.module(place.module.areaId, place.module.track.id)}
                tone="quiet"
                className="text-muted-foreground"
              >
                {place.module.loose
                  ? place.module.track.title
                  : `Module ${place.module.numeral}, ${place.module.track.title}`}
              </AppLink>
            </span>
            <span className="tabular whitespace-nowrap">
              {place.entry.core ? `Core ${place.entry.position} of ${place.module.core.length}` : "Further"}
            </span>
          </>
        ) : null}
        {problem.status !== "todo" ? <StatusMark status={problem.status} showLabel /> : null}
      </p>
      <h1 className="selectable mt-2.5 font-serif text-2xl font-medium tracking-tight text-balance @2xl:text-3xl">
        {problem.title}
      </h1>
      <MetaList ruled className="mt-5">
        <MetaItem label="Difficulty">
          <DifficultyMark difficulty={problem.difficulty} />
        </MetaItem>
        <MetaItem label="Judged by">
          <JudgeList judges={problem.judges} explain />
        </MetaItem>
        <MetaItem label="Time limit">
          <span className="tabular">{problem.timeLimitS} s per case</span>
        </MetaItem>
        {problem.requires !== "cpu" ? (
          <MetaItem label="Requires">{REQUIRES_LABEL[problem.requires]}</MetaItem>
        ) : null}
        {prerequisites.length > 0 ? (
          <MetaItem label="Builds on">
            <PrerequisiteList prerequisites={prerequisites} />
          </MetaItem>
        ) : null}
      </MetaList>
    </header>
  );
}
