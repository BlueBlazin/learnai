import { cn } from "cn";
import { useMemo } from "react";
import { StatusMark } from "@/components/problem/status-mark";
import { Button } from "@/components/ui/button";
import type { Skill } from "@/lib/api";
import { swatchColor } from "@/lib/module-colors";
import {
  CANVAS_WIDTH,
  canvasHeight,
  LABEL,
  rowTop,
  type SkillState,
  skillProgress,
  TILE,
  trails,
} from "@/lib/skills";
import { SkillGraphic } from "./skill-graphic";

/**
 * The skills as a tree, top to bottom: a tile per skill, and a trail from each
 * skill to every skill that builds on it. A trail not yet open is dotted and
 * quiet; once its skill is earned it is drawn solid in that skill's color.
 * Where two trails cross, the later one is drawn over a gap in the earlier.
 */
export function SkillTree({
  states,
  onOpen,
}: {
  /** Every skill's state by id, as `skillStates` builds it. */
  states: Map<string, SkillState>;
  onOpen: (skill: Skill) => void;
}) {
  const skills = useMemo(() => [...states.values()].map((state) => state.skill), [states]);
  const height = useMemo(() => canvasHeight(skills), [skills]);
  const all = useMemo(() => trails(skills), [skills]);
  const earned = (id: string) => states.get(id)?.earned ?? false;
  // Quiet trails first, so an open trail is never broken by one.
  const ordered = [
    ...all.filter((trail) => !earned(trail.from)),
    ...all.filter((trail) => earned(trail.from)),
  ];
  return (
    <div className="overflow-x-auto pb-2">
      <div className="relative mx-auto" style={{ width: CANVAS_WIDTH, height }}>
        <svg
          aria-hidden
          className="absolute inset-0"
          width={CANVAS_WIDTH}
          height={height}
          viewBox={`0 0 ${CANVAS_WIDTH} ${height}`}
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          {ordered.map((trail) => {
            const open = earned(trail.from);
            const color = open ? swatchColor(states.get(trail.from)?.skill.color ?? null) : "var(--faint)";
            return (
              <g key={trail.from}>
                {trail.paths.map((path) => (
                  <path
                    key={`gap-${path}`}
                    d={path}
                    stroke="var(--paper)"
                    strokeWidth={open ? 9 : 7}
                    strokeLinecap="butt"
                  />
                ))}
                {trail.paths.map((path) => (
                  <path
                    key={path}
                    d={path}
                    stroke={color}
                    strokeWidth={open ? 3 : 2.5}
                    strokeDasharray={open ? undefined : "0.1 7"}
                  />
                ))}
                <circle cx={trail.start.x} cy={trail.start.y} r={open ? 4.5 : 3.5} fill={color} />
              </g>
            );
          })}
        </svg>
        <ol>
          {[...states.values()].map((state) => {
            const { skill } = state;
            return (
              <li
                key={skill.id}
                className="absolute flex flex-col items-center"
                style={{ left: skill.x - TILE / 2 - 16, top: rowTop(skill.row), width: TILE + 32 }}
              >
                <Button
                  variant="ghost"
                  aria-label={`${skill.title} (${skill.name}): ${skillProgress(state)}`}
                  className="h-auto cursor-pointer overflow-hidden rounded-md border-0 bg-muted p-0"
                  style={{ width: TILE, height: TILE }}
                  onClick={() => onOpen(skill)}
                >
                  <SkillGraphic skill={skill} earned={state.earned} />
                </Button>
                <span
                  className="flex flex-col items-center bg-paper px-1 pt-2 text-center"
                  style={{ height: LABEL }}
                >
                  <span
                    className={cn(
                      "text-sm leading-tight whitespace-nowrap",
                      state.earned ? "font-semibold" : "text-muted-foreground",
                    )}
                  >
                    {skill.title}
                  </span>
                  <span className="mt-0.5 text-xs leading-tight whitespace-nowrap text-muted-foreground">
                    ({skill.name})
                  </span>
                  <span className="tabular mt-1 flex items-center gap-1.5 text-xs leading-tight text-muted-foreground">
                    {state.earned ? <StatusMark status="solved" /> : null}
                    {skillProgress(state)}
                  </span>
                </span>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
