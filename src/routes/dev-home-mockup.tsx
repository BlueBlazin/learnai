import { createLazyRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Page, Section } from "@/components/app/page";
import { ActivityCalendar } from "@/components/mockup/activity-calendar";
import { FactStrip } from "@/components/mockup/fact-strip";
import {
  MOCK_STATES,
  type MockStateId,
  nextProblem,
  PATH_ID,
  PROBLEM_TITLES,
  STAGES,
  stageColorVars,
  stagesFinished,
} from "@/components/mockup/mock-data";
import { MockupNote, StateSwitcher } from "@/components/mockup/mockup-chrome";
import { RoutePlates } from "@/components/mockup/route-plates";
import { RouteProgress } from "@/components/mockup/route-progress";
import { Today } from "@/components/mockup/today";
import { SubmissionList } from "@/components/problem/submission-list";
import { useGo } from "@/hooks/use-go";
import { routes } from "@/lib/routes";

const STAGE_COLORS = stageColorVars();

/**
 * Development only: a proposal for a new Home, on static sample data. Nothing
 * here reads or writes the backend.
 */
function HomeMockupScreen() {
  const go = useGo();
  const [stateId, setStateId] = useState<MockStateId>("started");
  const state = MOCK_STATES.find((candidate) => candidate.id === stateId) ?? MOCK_STATES[0];
  if (!state) return null;
  const next = nextProblem(state.solved);
  const finished = stagesFinished(state.solved);

  return (
    <Page>
      <div style={STAGE_COLORS}>
        <StateSwitcher states={MOCK_STATES} value={state.id} onChange={setStateId} />

        <Today next={next} onOpen={(problemId) => go(routes.problem(problemId))} />

        <FactStrip state={state} />

        <Section
          title="The route"
          aside={
            finished === STAGES.length
              ? `All ${STAGES.length} stages finished`
              : finished === 0
                ? `The first of ${STAGES.length} stages is under way`
                : `${finished} of ${STAGES.length} stages finished`
          }
        >
          <div className="flex flex-col gap-8">
            <RouteProgress solved={state.solved} next={next} />
            <RoutePlates solved={state.solved} onOpen={() => go(routes.paths(PATH_ID))} />
          </div>
        </Section>

        <div className="grid grid-cols-2 gap-x-14">
          <Section title="Activity">
            <ActivityCalendar weeks={state.activity} />
          </Section>
          <Section title="Recent submissions">
            <SubmissionList
              submissions={state.submissions}
              titles={PROBLEM_TITLES}
              onOpen={(submission) => go(routes.problem(submission.problemId, { tab: "submissions" }))}
            />
          </Section>
        </div>

        <MockupNote>
          The progress sentence and segmented bar under "The route" would also head the Path screen, in place
          of its plain rule.
        </MockupNote>
      </div>
    </Page>
  );
}

export const Route = createLazyRoute("/dev/home-mockup")({ component: HomeMockupScreen });
