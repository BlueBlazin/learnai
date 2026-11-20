import { cn } from "cn";
import type { ReactNode } from "react";
import { InlineButton } from "@/components/app/inline-button";
import { type ActivityWeeks, currentStreak, ICONS, type MockState } from "./mock-data";
import { PixelSprite } from "./pixel-sprite";

const WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

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

function Fact({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("flex flex-col gap-3 px-6 first:pl-0 last:pr-0", className)}>{children}</div>;
}

function FactLine({ children }: { children: ReactNode }) {
  return <p className="flex h-6 items-center gap-2 font-serif text-lg">{children}</p>;
}

/** This week's days so far, from the last week of the activity record. */
function weekDays(activity: ActivityWeeks): { done: boolean[]; names: string[] } {
  const week = (activity[activity.length - 1] ?? "").split("");
  return {
    done: week.map((day) => day !== "-" && Number(day) > 0),
    names: week.flatMap((day, index) => (day !== "-" && Number(day) > 0 ? [WEEKDAYS[index] ?? ""] : [])),
  };
}

/**
 * Three facts under one ink rule, divided by hairlines: the streak with this
 * week's days, the weekly goal, and the reviews that are due.
 */
export function FactStrip({ state }: { state: MockState }) {
  const streak = currentStreak(state.activity);
  const week = weekDays(state.activity);
  const goalMet = state.weekSolved >= state.weekGoal;
  return (
    <div className="border-t border-rule pt-5 pb-10">
      <div className="grid grid-cols-3 divide-x divide-border">
        <Fact>
          <FactLine>
            <PixelSprite src={ICONS.flame} scale={1} className="-ml-0.5" />
            <span>
              <span className="font-semibold tabular">{streak}</span> {streak === 1 ? "day" : "days"} running
            </span>
          </FactLine>
          <Squares
            cells={week.done}
            color="var(--flame)"
            label={
              week.names.length > 0
                ? `Active this week: ${week.names.join(", ")}`
                : "No active days this week"
            }
          />
        </Fact>
        <Fact>
          <FactLine>
            <span>
              <span className="font-semibold tabular">{state.weekSolved}</span> of {state.weekGoal} this week
            </span>
            {goalMet ? <PixelSprite src={ICONS.star} scale={1} label="Weekly goal met" /> : null}
          </FactLine>
          <Squares
            cells={Array.from({ length: state.weekGoal }, (_, index) => index < state.weekSolved)}
            color="var(--primary)"
            label={`${state.weekSolved} of ${state.weekGoal} problems toward this week's goal`}
          />
        </Fact>
        <Fact>
          <FactLine>
            {state.reviewsDue > 0 ? (
              <span>
                <span className="font-semibold tabular">{state.reviewsDue}</span>{" "}
                {state.reviewsDue === 1 ? "review" : "reviews"} due
              </span>
            ) : (
              <span>No reviews due</span>
            )}
          </FactLine>
          <p className="text-sm leading-none text-muted-foreground">
            {state.reviewsDue > 0 ? (
              <InlineButton>Review now</InlineButton>
            ) : (
              `The next one comes up ${state.nextReview}.`
            )}
          </p>
        </Fact>
      </div>
    </div>
  );
}
