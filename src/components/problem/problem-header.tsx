import { Fragment } from "react";
import { AppLink } from "@/components/app/app-link";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import type { ProblemDetail, ProblemSummary, Track } from "@/lib/api";
import { REQUIRES_LABEL } from "@/lib/labels";
import { routes } from "@/lib/routes";
import { DifficultyMark } from "./difficulty-mark";
import { JudgeList } from "./judge-list";
import { StatusMark } from "./status-mark";

/**
 * The title block of a statement, set like the head of a paper: where the
 * problem sits in the catalog, its title, and a ruled strip of facts.
 */
export function ProblemHeader({
  problem,
  track,
  prerequisites = [],
}: {
  problem: ProblemDetail;
  track?: Track;
  /** Summaries of the problems this one builds on, when known. */
  prerequisites?: ProblemSummary[];
}) {
  return (
    <header className="max-w-measure pb-7">
      <p className="flex items-center gap-3 text-xs text-muted-foreground">
        <AppLink to={routes.library({ track: problem.track })} tone="quiet" className="text-muted-foreground">
          {track?.title ?? problem.track}
        </AppLink>
        <span className="tabular">No. {problem.order}</span>
        {problem.status !== "todo" ? <StatusMark status={problem.status} showLabel /> : null}
      </p>
      <h1 className="selectable mt-2.5 font-serif text-2xl font-medium tracking-tight text-balance @2xl:text-3xl">
        {problem.title}
      </h1>
      <MetaList className="mt-5 border-y border-border py-2.5">
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
            {prerequisites.map((prerequisite, index) => (
              <Fragment key={prerequisite.id}>
                {index > 0 ? ", " : null}
                <AppLink to={routes.problem(prerequisite.id)}>{prerequisite.title}</AppLink>
              </Fragment>
            ))}
          </MetaItem>
        ) : null}
      </MetaList>
    </header>
  );
}
