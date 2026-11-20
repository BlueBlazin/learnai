import { ArrowRight } from "lucide-react";
import type { ReactNode } from "react";
import { AppLink } from "@/components/app/app-link";
import { DifficultyMark } from "@/components/problem/difficulty-mark";
import { Button } from "@/components/ui/button";
import { romanNumeral } from "@/lib/paths";
import { routes } from "@/lib/routes";
import {
  type MockStage,
  type NextProblem,
  PATH_ID,
  PATH_TITLE,
  PROBLEM_COUNT,
  STAGES,
  stageColor,
} from "./mock-data";
import { StageSwatch } from "./stage-swatch";

/** The stage's scene as a wide banner across the column, and the line that says where this is. */
function Banner({ stage, children }: { stage: MockStage; children: ReactNode }) {
  return (
    <>
      <img
        src={stage.scene}
        alt=""
        draggable={false}
        className="block aspect-7/2 w-full rounded-md object-cover object-center"
      />
      <p className="mt-6 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        {children}
        <span className="ml-auto text-muted-foreground">{stage.place}</span>
      </p>
    </>
  );
}

/**
 * The opening of Home: the place on the route the learner has reached, as a
 * banner, and under it the next problem. Once the path is finished (`next` is
 * null) the banner is the summit and the text says the route is complete.
 */
export function Today({ next, onOpen }: { next: NextProblem | null; onOpen: (problemId: string) => void }) {
  if (!next) {
    const summit = STAGES[STAGES.length - 1];
    if (!summit) return null;
    return (
      <section aria-label="Today" className="pb-12">
        <Banner stage={summit}>
          <span className="text-muted-foreground">{PATH_TITLE}, finished</span>
        </Banner>
        <h1 className="selectable mt-2 max-w-measure font-serif text-3xl font-medium tracking-tight text-balance">
          The route is complete
        </h1>
        <p className="selectable mt-3 max-w-measure text-sm text-pretty text-muted-foreground">
          All {PROBLEM_COUNT} problems solved, through all {STAGES.length} stages from the trailhead to the
          summit. The library has more, and reviews keep the way back open.
        </p>
        <p className="mt-6 flex gap-5 text-sm">
          <AppLink to={routes.library()}>Browse the library</AppLink>
          <AppLink to={routes.paths(PATH_ID)}>Look back over the path</AppLink>
        </p>
      </section>
    );
  }

  const { stage } = next;
  return (
    <section aria-label="Today" className="pb-12">
      <Banner stage={stage}>
        <span className="text-muted-foreground">Up next</span>
        <span className="inline-flex items-center gap-2">
          <StageSwatch color={stageColor(stage)} />
          Part {romanNumeral(stage.number)}, {stage.title}
        </span>
      </Banner>
      <h1 className="selectable mt-2 max-w-measure font-serif text-3xl font-medium tracking-tight text-balance">
        <AppLink to={routes.problem(next.id)} tone="quiet">
          {next.title}
        </AppLink>
      </h1>
      <p className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-sm text-muted-foreground">
        <span>
          <AppLink to={routes.paths(PATH_ID)} tone="quiet" className="text-muted-foreground">
            {PATH_TITLE}
          </AppLink>
          , No. {next.position} of {PROBLEM_COUNT}
        </span>
        <DifficultyMark difficulty={next.difficulty} />
      </p>
      <div className="mt-6">
        <Button onClick={() => onOpen(next.id)}>
          Open problem
          <ArrowRight data-icon="inline-end" />
        </Button>
      </div>
    </section>
  );
}
