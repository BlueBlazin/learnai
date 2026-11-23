import { AppLink } from "@/components/app/app-link";
import { InlineButton } from "@/components/app/inline-button";
import { RuledList } from "@/components/app/ruled-list";
import { StatusMark } from "@/components/problem/status-mark";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import type { Skill } from "@/lib/api";
import { formatDateTime, formatRelativeTime } from "@/lib/format";
import { routes } from "@/lib/routes";
import type { SkillState } from "@/lib/skills";
import { SkillGraphic } from "./skill-graphic";

/** Where the skill stands, as one sentence. */
function Standing({ state }: { state: SkillState }) {
  const { earnedAt } = state.skill;
  if (earnedAt !== null) {
    return (
      <>
        Earned{" "}
        <time dateTime={earnedAt} title={formatDateTime(earnedAt)}>
          {formatRelativeTime(earnedAt)}
        </time>
        .
      </>
    );
  }
  if (state.waitingOn.length === 0) return `${state.solved} of ${state.total} problems solved.`;
  const names = state.waitingOn.map((skill) => skill.title).join(" and ");
  return `Every problem is solved. It is earned once ${names} ${state.waitingOn.length === 1 ? "is" : "are"}.`;
}

/**
 * One skill, opened from the tree: its graphic at full size, what it builds
 * on, and the problems that earn it, each a link to the problem.
 */
export function SkillDetail({
  state,
  states,
  onClose,
  onOpen,
}: {
  /** The skill shown; null while the dialog is closed. */
  state: SkillState | null;
  /** Every skill's state by id, for the skills this one builds on. */
  states: Map<string, SkillState>;
  onClose: () => void;
  /** Show another skill in its place: one of those it builds on. */
  onOpen: (skill: Skill) => void;
}) {
  return (
    <Dialog open={state !== null} onOpenChange={(open) => (open ? undefined : onClose())}>
      {state ? (
        <DialogContent className="grid-cols-[22rem_minmax(0,1fr)] gap-0 overflow-hidden p-0 sm:max-w-208">
          <div className="aspect-square self-start bg-muted">
            <SkillGraphic skill={state.skill} earned={state.earned} />
          </div>
          <div className="flex max-h-88 min-w-0 flex-col gap-5 overflow-y-auto px-6 pt-6 pb-5">
            <header>
              <DialogTitle className="font-serif text-xl leading-tight font-medium tracking-tight">
                {state.skill.title}
              </DialogTitle>
              <p className="mt-1 text-sm text-muted-foreground">({state.skill.name})</p>
              <DialogDescription className="mt-3 font-serif text-md text-pretty">
                {state.skill.summary}
              </DialogDescription>
              <p className="mt-3 flex items-center gap-2 text-sm">
                {state.earned ? <StatusMark status="solved" /> : null}
                <span>
                  <Standing state={state} />
                </span>
              </p>
            </header>
            {state.skill.requires.length > 0 ? (
              <section>
                <h3 className="pb-1.5 text-xs font-semibold">Builds on</h3>
                <RuledList>
                  {state.skill.requires.map((id) => {
                    const required = states.get(id);
                    if (!required) return null;
                    return (
                      <li key={id} className="flex items-center gap-2.5 py-1.5">
                        <StatusMark status={required.earned ? "solved" : "todo"} />
                        <InlineButton
                          className="text-foreground no-underline hover:text-primary hover:no-underline"
                          onClick={() => onOpen(required.skill)}
                        >
                          {required.skill.title}
                        </InlineButton>
                        <span className="min-w-0 truncate text-xs text-muted-foreground">
                          ({required.skill.name})
                        </span>
                        <span className="ml-auto text-xs text-muted-foreground">
                          {required.earned ? "Earned" : "Not yet"}
                        </span>
                      </li>
                    );
                  })}
                </RuledList>
              </section>
            ) : null}
            <section>
              <h3 className="flex items-baseline justify-between pb-1.5 text-xs font-semibold">
                Problems
                <span className="tabular font-normal text-muted-foreground">
                  {state.solved} of {state.total} solved
                </span>
              </h3>
              <RuledList>
                {state.problems.map((problem) => (
                  <li
                    key={problem.id}
                    className="relative flex items-center gap-2.5 py-1.5 hover:bg-accent/60"
                  >
                    <StatusMark status={problem.status} />
                    <AppLink
                      to={routes.problem(problem.id)}
                      tone="quiet"
                      stretched
                      className="min-w-0 shrink text-left"
                    >
                      {problem.title}
                    </AppLink>
                  </li>
                ))}
              </RuledList>
            </section>
          </div>
        </DialogContent>
      ) : null}
    </Dialog>
  );
}
