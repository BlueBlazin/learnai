import { cn } from "cn";
import type { ReactNode } from "react";
import { AppLink } from "@/components/app/app-link";
import { PixelIcon } from "@/components/app/pixel-icon";
import { routes } from "@/lib/routes";

const WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

/** Above this many problems a week the squares stop helping and the sentence stands alone. */
const MAX_GOAL_SQUARES = 10;

/** A row of small squares: the filled ones in `color`, the rest quiet. */
function Squares({
  cells,
  color,
  label,
}: {
  /** One entry per square: filled or not. */
  cells: boolean[];
  color: string;
  label: string;
}) {
  return (
    <span role="img" aria-label={label} className="flex gap-0.75">
      {cells.map((filled, index) => (
        <span
          // biome-ignore lint/suspicious/noArrayIndexKey: the squares are positions, not items.
          key={index}
          className={cn("size-2.5", !filled && "bg-input")}
          style={filled ? { backgroundColor: color } : undefined}
        />
      ))}
    </span>
  );
}

function Fact({ children }: { children: ReactNode }) {
  return <div className="flex flex-col gap-3 px-6 first:pl-0 last:pr-0">{children}</div>;
}

function FactLine({ children }: { children: ReactNode }) {
  return <p className="flex h-6 items-center gap-2 font-serif text-lg">{children}</p>;
}

/** The reviews fact: how many are due and where to start, or when the next one comes. */
export interface ReviewsFact {
  due: number;
  /** The review to start first, when any is due. */
  firstDueProblemId: string | null;
  /** The quiet sentence for when none is due, when something is scheduled. */
  upcoming: string | null;
}

/**
 * Up to three facts under one ink rule, divided by hairlines: the streak with
 * this week's days, the weekly goal, and the reviews that are due. Without a
 * `reviews` fact (they are off, or not loaded) the other two share the row.
 */
export function HabitStrip({
  streakDays,
  week,
  solvedThisWeek,
  goal,
  reviews,
}: {
  streakDays: number;
  /** Which days of the current Monday-to-Sunday week were active. */
  week: boolean[];
  solvedThisWeek: number;
  goal: number;
  reviews: ReviewsFact | null;
}) {
  const activeNames = week.flatMap((active, index) => (active ? [WEEKDAYS[index] ?? ""] : []));
  const goalMet = solvedThisWeek >= goal;
  return (
    <div className="border-t border-rule pt-5 pb-10">
      <div className={cn("grid divide-x divide-border", reviews ? "grid-cols-3" : "grid-cols-2")}>
        <Fact>
          <FactLine>
            <PixelIcon name="flame" className="-ml-0.5" />
            {streakDays === 0 ? (
              <span>No streak yet</span>
            ) : (
              <span>
                <span className="font-semibold tabular">{streakDays}</span>{" "}
                {streakDays === 1 ? "day" : "days"} running
              </span>
            )}
          </FactLine>
          <Squares
            cells={week}
            color="var(--flame)"
            label={
              activeNames.length > 0
                ? `Active this week: ${activeNames.join(", ")}`
                : "No active days this week"
            }
          />
          {streakDays === 0 ? (
            <p className="text-sm leading-none text-muted-foreground">
              An accepted submission today starts one.
            </p>
          ) : null}
        </Fact>
        <Fact>
          <FactLine>
            <span>
              <span className="font-semibold tabular">{solvedThisWeek}</span> of {goal} this week
            </span>
            {goalMet ? <PixelIcon name="star" label="Weekly goal met" /> : null}
          </FactLine>
          {goal <= MAX_GOAL_SQUARES ? (
            <Squares
              cells={Array.from({ length: goal }, (_, index) => index < solvedThisWeek)}
              color="var(--primary)"
              label={`${solvedThisWeek} of ${goal} problems toward this week's goal`}
            />
          ) : null}
        </Fact>
        {reviews ? (
          <Fact>
            <FactLine>
              {reviews.due > 0 ? (
                <span>
                  <span className="font-semibold tabular">{reviews.due}</span>{" "}
                  {reviews.due === 1 ? "review" : "reviews"} due
                </span>
              ) : (
                <span>No reviews due</span>
              )}
            </FactLine>
            {reviews.due > 0 && reviews.firstDueProblemId ? (
              <p className="text-sm leading-none">
                <AppLink to={routes.problem(reviews.firstDueProblemId, { review: true })}>Review now</AppLink>
              </p>
            ) : reviews.due === 0 && reviews.upcoming ? (
              <p className="selectable text-sm text-pretty text-muted-foreground">{reviews.upcoming}</p>
            ) : null}
          </Fact>
        ) : null}
      </div>
    </div>
  );
}
