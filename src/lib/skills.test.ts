import { describe, expect, it } from "vitest";
import type { ProblemStatus, ProblemSummary, Skill } from "@/lib/api";
import {
  arrivalPorts,
  CANVAS_WIDTH,
  canvasHeight,
  earnedCount,
  LABEL,
  rowTop,
  skillProgress,
  skillStates,
  TILE,
  trails,
} from "./skills";

const problem = (id: string, status: ProblemStatus = "todo"): ProblemSummary => ({
  id,
  title: id.toUpperCase(),
  track: "t",
  order: 10,
  difficulty: "easy",
  frameworks: ["torch"],
  judges: ["numeric"],
  tags: [],
  prerequisites: [],
  requires: "cpu",
  status,
  solvedFrameworks: status === "solved" ? ["torch"] : [],
  assets: [],
});

const skill = (id: string, row: number, x: number, over: Partial<Skill> = {}): Skill => ({
  id,
  title: `Title ${id}`,
  name: `Name ${id}`,
  summary: "",
  requires: [],
  problems: [`${id}-1`],
  row,
  x,
  lanes: [],
  earnedAt: null,
  ...over,
});

const EARNED = "2026-11-20T10:00:00.000Z";

/** Every point a path passes through or bends at, in order. */
const points = (path: string): [number, number][] =>
  [...path.matchAll(/(-?[\d.]+) (-?[\d.]+)/g)].map((match) => [Number(match[1]), Number(match[2])]);

/** The y of a path's horizontal run; null for a straight line. */
const levelOf = (path: string): number | null => {
  const turn = /Q-?[\d.]+ (-?[\d.]+)/.exec(path);
  return turn ? Number(turn[1]) : null;
};

const trailFrom = (skills: Skill[], id: string) => {
  const trail = trails(skills).find((candidate) => candidate.from === id);
  if (!trail) throw new Error(`no trail from ${id}`);
  return trail;
};

describe("skillStates", () => {
  const skills = [
    skill("a", 0, 300, { problems: ["a-1", "a-2"], earnedAt: EARNED }),
    skill("b", 1, 300, { requires: ["a"], problems: ["b-1", "b-2", "b-3"] }),
    skill("c", 1, 600, { problems: ["c-1", "c-2"] }),
    skill("d", 2, 300, { requires: ["a", "b", "c"], problems: ["d-1"] }),
  ];
  const problems = [
    problem("a-1", "solved"),
    problem("a-2", "solved"),
    problem("b-1", "solved"),
    problem("b-2", "attempted"),
    problem("b-3"),
    problem("c-1"),
    problem("c-2"),
    problem("d-1", "solved"),
  ];
  const states = skillStates(skills, problems);
  const state = (id: string) => {
    const found = states.get(id);
    if (!found) throw new Error(id);
    return found;
  };

  it("keeps the order of the skills and counts each one's solved problems", () => {
    expect([...states.keys()]).toEqual(["a", "b", "c", "d"]);
    expect([...states.values()].map((s) => [s.solved, s.total])).toEqual([
      [2, 2],
      [1, 3],
      [0, 2],
      [1, 1],
    ]);
    expect(state("b").problems.map((p) => p.id)).toEqual(["b-1", "b-2", "b-3"]);
  });

  it("takes earned from the skill, and counts the earned skills", () => {
    expect([...states.values()].map((s) => s.earned)).toEqual([true, false, false, false]);
    expect(earnedCount(states)).toBe(1);
  });

  it("lists the unearned required skills only once every problem is solved", () => {
    expect(state("d").waitingOn.map((s) => s.id)).toEqual(["b", "c"]);
    // Under way: what it builds on is not what holds it back yet.
    expect(state("b").waitingOn).toEqual([]);
    expect(state("a").waitingOn).toEqual([]);
  });

  it("says where a skill stands in a few words", () => {
    expect(skillProgress(state("a"))).toBe("Earned");
    expect(skillProgress(state("b"))).toBe("1 of 3 solved");
    expect(skillProgress(state("c"))).toBe("2 problems");
    expect(skillProgress(state("d"))).toBe("Needs Title b, Title c");
  });

  it("counts a problem the catalog does not serve as unsolved, and ignores an unknown requirement", () => {
    const [only] = [
      ...skillStates(
        [skill("a", 0, 300, { problems: ["a-1", "gone"], requires: ["nowhere"] })],
        problems,
      ).values(),
    ];
    expect(only).toMatchObject({ solved: 1, total: 2, earned: false, waitingOn: [] });
    expect(only?.problems.map((p) => p.id)).toEqual(["a-1"]);
    expect(only ? skillProgress(only) : "").toBe("1 of 2 solved");
  });

  it("counts one problem in the singular", () => {
    const [only] = [...skillStates([skill("a", 0, 300)], []).values()];
    expect(only ? skillProgress(only) : "").toBe("1 problem");
  });
});

describe("the tree's layout", () => {
  it("sizes the canvas by the rows present", () => {
    expect(canvasHeight([])).toBe(0);
    expect(canvasHeight([skill("a", 0, 500)])).toBe(TILE + LABEL);
    expect(canvasHeight([skill("a", 0, 500), skill("b", 3, 500)])).toBe(rowTop(3) + TILE + LABEL);
    expect(CANVAS_WIDTH).toBe(1000);
  });

  it("draws a trail to a skill straight below as one straight line", () => {
    const skills = [skill("a", 0, 400), skill("b", 1, 400, { requires: ["a"] })];
    const trail = trailFrom(skills, "a");
    expect(trail.paths).toEqual([`M400 ${TILE + LABEL} L400 ${rowTop(1)}`]);
    expect(trail.start).toEqual({ x: 400, y: TILE + LABEL });
  });

  it("turns once, in the band between the rows, toward a skill to the side", () => {
    const skills = [skill("a", 0, 200), skill("b", 1, 600, { requires: ["a"] })];
    const [path] = trailFrom(skills, "a").paths;
    const level = levelOf(path ?? "");
    expect(level).toBeGreaterThan(TILE + LABEL);
    expect(level).toBeLessThan(rowTop(1));
    const through = points(path ?? "");
    expect(through[0]).toEqual([200, TILE + LABEL]);
    expect(through.at(-1)).toEqual([600, rowTop(1)]);
  });

  it("only ever runs down", () => {
    const skills = [
      skill("a", 0, 300, { lanes: [[1, 520]] }),
      skill("b", 0, 760),
      skill("c", 1, 300, { requires: ["a"] }),
      skill("d", 1, 760, { requires: ["b"], lanes: [[2, 800]] }),
      skill("e", 2, 240, { requires: ["c"] }),
      skill("f", 2, 540, { requires: ["a", "c"] }),
      skill("g", 3, 540, { requires: ["e", "f", "d"] }),
    ];
    const all = trails(skills);
    expect(all.map((trail) => trail.from).sort()).toEqual(["a", "b", "c", "d", "e", "f"]);
    for (const trail of all) {
      for (const path of trail.paths) {
        const ys = points(path).map(([, y]) => y);
        expect(ys).toEqual([...ys].sort((a, b) => a - b));
      }
    }
  });

  it("brings the trails of one tile in side by side, ordered by where they come from", () => {
    const skills = [
      skill("left", 0, 100),
      skill("right", 0, 900),
      skill("middle", 0, 520),
      skill("t", 1, 500, { requires: ["right", "left", "middle"] }),
    ];
    const ports = arrivalPorts(skills);
    const [left, middle, right] = ["left", "middle", "right"].map((id) => ports.get(`${id}>t`) ?? Number.NaN);
    expect(left).toBeLessThan(middle as number);
    expect(middle).toBeLessThan(right as number);
    expect((middle as number) - (left as number)).toBe((right as number) - (middle as number));
    // The group slides so that the trail from nearly straight above has no kink.
    expect(middle).toBe(520);
    expect(trailFrom(skills, "middle").paths).toEqual([`M520 ${TILE + LABEL} L520 ${rowTop(1)}`]);
    // Each trail ends at its own port.
    expect(points(trailFrom(skills, "left").paths[0] ?? "").at(-1)).toEqual([left, rowTop(1)]);
    expect(points(trailFrom(skills, "right").paths[0] ?? "").at(-1)).toEqual([right, rowTop(1)]);
  });

  it("centers the arrivals on the tile when none comes from above it", () => {
    const skills = [skill("l", 0, 100), skill("r", 0, 900), skill("t", 1, 500, { requires: ["l", "r"] })];
    const ports = arrivalPorts(skills);
    expect(((ports.get("l>t") ?? 0) + (ports.get("r>t") ?? 0)) / 2).toBe(500);
    expect(ports.get("l>t")).toBeLessThan(ports.get("r>t") ?? 0);
  });

  it("passes a row in the skill's lane, and at the skill's own x when it has no lane there", () => {
    const withLane = [
      skill("a", 0, 300, { lanes: [[1, 520]] }),
      skill("mid", 1, 300),
      skill("b", 2, 520, { requires: ["a"] }),
    ];
    const laned = trailFrom(withLane, "a").paths;
    expect(laned).toHaveLength(2);
    expect(points(laned[0] ?? "").at(-1)).toEqual([520, rowTop(1)]);
    // The lane's stretch runs from the top of its row, past the tiles beside it.
    expect(laned[1]).toBe(`M520 ${rowTop(1)} L520 ${rowTop(2)}`);

    const without = withLane.map((s) => ({ ...s, lanes: [] }));
    expect(trailFrom(without, "a").paths).toEqual([
      `M300 ${TILE + LABEL} L300 ${rowTop(1)}`,
      expect.stringMatching(new RegExp(`^M300 ${rowTop(1)} .* L520 ${rowTop(2)}$`)),
    ]);
  });

  it("draws no trail for a requirement that is not on a higher row, or that is unknown", () => {
    const skills = [
      skill("a", 1, 300),
      skill("b", 1, 600, { requires: ["a"] }),
      skill("c", 0, 600, { requires: ["a", "nowhere"] }),
      skill("d", 2, 600, { requires: ["b"] }),
    ];
    expect(trails(skills).map((trail) => trail.from)).toEqual(["b"]);
    expect([...arrivalPorts(skills).keys()]).toEqual(["b>d"]);
  });

  it("stacks two turns in one band so that they do not cross when they need not", () => {
    // a runs right to 500 and b, which starts inside a's reach, runs right to 700. With a's turn
    // above b's, a's descent cuts b and b's start cuts a; with b's turn above, nothing crosses.
    const skills = [
      skill("a", 0, 200),
      skill("b", 0, 400),
      skill("c", 1, 500, { requires: ["a"] }),
      skill("d", 1, 700, { requires: ["b"] }),
    ];
    const levelA = levelOf(trailFrom(skills, "a").paths[0] ?? "");
    const levelB = levelOf(trailFrom(skills, "b").paths[0] ?? "");
    expect(levelA).not.toBeNull();
    expect(levelB).not.toBeNull();
    expect(levelB as number).toBeLessThan(levelA as number);
    // The mirror image stacks the other way round.
    const mirrored = skills.map((s) => ({ ...s, x: CANVAS_WIDTH - s.x }));
    expect(levelOf(trailFrom(mirrored, "b").paths[0] ?? "") as number).toBeLessThan(
      levelOf(trailFrom(mirrored, "a").paths[0] ?? "") as number,
    );
  });

  it("lets turns that do not overlap share a level", () => {
    const skills = [
      skill("a", 0, 100),
      skill("b", 0, 600),
      skill("c", 1, 300, { requires: ["a"] }),
      skill("d", 1, 800, { requires: ["b"] }),
    ];
    expect(levelOf(trailFrom(skills, "a").paths[0] ?? "")).toBe(
      levelOf(trailFrom(skills, "b").paths[0] ?? ""),
    );
  });
});
