import { describe, expect, it } from "vitest";
import type { Area, ProblemStatus, ProblemSummary, Submission, Track } from "@/lib/api";
import { neighbors, pickContinue } from "./continue";
import {
  allSolved,
  areaOrder,
  areaRoute,
  buildCurriculum,
  coreComplete,
  coresComplete,
  curriculumOrder,
  findArea,
  findModule,
  IMPLICIT_AREA_ID,
  moduleCounts,
  nextCore,
  nextInArea,
  placeOf,
  siblings,
} from "./curriculum";
import { unmetOutOfOrder, unmetPrerequisites } from "./prerequisites";

const problem = (
  id: string,
  track: string,
  order: number,
  status: ProblemStatus = "todo",
  over: Partial<ProblemSummary> = {},
): ProblemSummary => ({
  id,
  title: id.toUpperCase(),
  track,
  order,
  difficulty: "easy",
  frameworks: ["torch"],
  judges: ["numeric"],
  tags: [],
  prerequisites: [],
  requires: "cpu",
  status,
  solvedFrameworks: status === "solved" ? ["torch"] : [],
  assets: [],
  ...over,
});

const track = (id: string, order: number, core: string[], over: Partial<Track> = {}): Track => ({
  id,
  title: id.toUpperCase(),
  description: "",
  order,
  problemCount: 0,
  solvedCount: 0,
  core,
  ...over,
});

// Three modules. `x` has a core out of track order and one unknown id; `y` has no core;
// `z` belongs to no area.
const TRACKS: Track[] = [
  track("y", 2, []),
  track("x", 1, ["x3", "gone", "x1"], { scene: "no-such-scene", color: "#2a71cc" }),
  track("z", 3, ["z1"], { color: "blue" }),
];
const AREAS: Area[] = [
  { id: "one", title: "One", description: "The first area.", modules: ["x", "missing", "y"] },
];

const catalog = (statuses: Record<string, ProblemStatus> = {}): ProblemSummary[] =>
  [
    problem("x1", "x", 10),
    problem("x2", "x", 20),
    problem("x3", "x", 30),
    problem("x4", "x", 40),
    problem("y2", "y", 20),
    problem("y1", "y", 10),
    problem("z1", "z", 10),
    problem("z2", "z", 20),
  ].map((p) => ({ ...p, status: statuses[p.id] ?? p.status }));

const build = (statuses: Record<string, ProblemStatus> = {}, areas: Area[] = AREAS) =>
  buildCurriculum(areas, TRACKS, catalog(statuses));
const ids = (list: { problem: ProblemSummary }[]) => list.map((entry) => entry.problem.id);

const submission = (problemId: string, id = 1): Submission => ({
  id,
  problemId,
  framework: "torch",
  mode: "submit",
  verdict: "wrong_answer",
  passed: 0,
  total: 1,
  skipped: 0,
  durationMs: 1,
  target: "local",
  createdAt: "2026-11-14T10:00:00.000Z",
  review: false,
});

describe("buildCurriculum", () => {
  it("builds an area's modules in the area's order, numbered from one, skipping unknown tracks", () => {
    const { areas } = build();
    expect(areas).toHaveLength(1);
    const [area] = areas;
    expect(area).toMatchObject({ id: "one", title: "One", description: "The first area." });
    expect(area?.modules.map((m) => [m.track.id, m.number, m.numeral, m.areaId, m.loose])).toEqual([
      ["x", 1, "I", "one", false],
      ["y", 2, "II", "one", false],
    ]);
  });

  it("splits a module into core, in the order `core` gives, and further, in track order", () => {
    const [x, y] = build().areas[0]?.modules ?? [];
    expect(x?.core.map((e) => [e.problem.id, e.position, e.core])).toEqual([
      ["x3", 1, true],
      ["x1", 2, true],
    ]);
    expect(x?.further.map((e) => [e.problem.id, e.position, e.core])).toEqual([
      ["x2", 3, false],
      ["x4", 4, false],
    ]);
    expect(ids(y?.core ?? [])).toEqual([]);
    expect(ids(y?.further ?? [])).toEqual(["y1", "y2"]);
  });

  it("carries the color when it is a hex, and no scene for a slug that names none", () => {
    const { areas, loose } = build();
    expect(areas[0]?.modules[0]).toMatchObject({ color: "#2a71cc", scene: null });
    expect(areas[0]?.modules[1]?.color).toBeNull();
    expect(loose[0]?.color).toBeNull();
  });

  it("counts core solved and all solved per module and per area", () => {
    const { areas } = build({ x3: "solved", x2: "solved", x1: "attempted", y1: "solved" });
    const [area] = areas;
    expect(area).toMatchObject({ problemCount: 6, coreCount: 2, coreSolved: 1 });
    expect(area?.modules[0]).toMatchObject({ coreSolved: 1, solved: 2, total: 4 });
    expect(area?.modules[1]).toMatchObject({ coreSolved: 0, solved: 1, total: 2 });
  });

  it("groups the tracks no area names as loose modules, shown through the first area", () => {
    const { loose } = build();
    expect(loose.map((m) => [m.track.id, m.number, m.areaId, m.loose])).toEqual([["z", 1, "one", true]]);
    expect(ids(loose[0]?.core ?? [])).toEqual(["z1"]);
  });

  it("makes one implicit area of every track, in track order, when the content defines no areas", () => {
    for (const areas of [[], [{ id: "empty", title: "E", description: "", modules: ["missing"] }]]) {
      const curriculum = build({}, areas);
      expect(curriculum.loose).toEqual([]);
      expect(curriculum.areas.map((a) => a.id)).toEqual([IMPLICIT_AREA_ID]);
      expect(curriculum.areas[0]?.modules.map((m) => m.track.id)).toEqual(["x", "y", "z"]);
      expect(curriculum.areas[0]?.coreCount).toBe(3);
    }
    expect(buildCurriculum([], [], [])).toEqual({ areas: [], loose: [] });
  });

  it("leaves out an area with no known modules and lists a module once per area", () => {
    const { areas, loose } = build({}, [
      { id: "empty", title: "E", description: "", modules: ["missing"] },
      { id: "two", title: "Two", description: "", modules: ["y", "y", "z"] },
    ]);
    expect(areas.map((a) => a.id)).toEqual(["two"]);
    expect(areas[0]?.modules.map((m) => m.track.id)).toEqual(["y", "z"]);
    expect(loose.map((m) => m.track.id)).toEqual(["x"]);
  });
});

describe("module marks", () => {
  it("knows a complete core from a module with every problem solved", () => {
    const modules = (statuses: Record<string, ProblemStatus>) => build(statuses).areas[0]?.modules ?? [];
    const [x, y] = modules({ x1: "solved", x3: "solved", y1: "solved", y2: "solved" });
    expect(x && coreComplete(x)).toBe(true);
    expect(x && allSolved(x)).toBe(false);
    // A module with no core never has a complete core, but can be all solved.
    expect(y && coreComplete(y)).toBe(false);
    expect(y && allSolved(y)).toBe(true);
    expect(coresComplete([x, y].flatMap((m) => m ?? []))).toEqual({ complete: 1, withCore: 1 });
    const [started] = modules({ x1: "solved" });
    expect(started && coreComplete(started)).toBe(false);
  });

  it("writes a module's size, leaving out a kind it has none of", () => {
    const { areas } = build();
    expect(areas[0]?.modules.map(moduleCounts)).toEqual(["2 core, 2 further", "2 further"]);
    const [empty] = buildCurriculum([], [track("e", 1, [])], []).areas[0]?.modules ?? [];
    expect(empty && moduleCounts(empty)).toBe("No problems yet");
  });
});

describe("the route", () => {
  const twoAreas: Area[] = [
    { id: "one", title: "One", description: "", modules: ["y", "x"] },
    { id: "two", title: "Two", description: "", modules: ["z"] },
  ];

  it("is the core problems in area order, then module order, numbered along the route", () => {
    const [one] = build({}, [{ id: "one", title: "One", description: "", modules: ["x", "y", "z"] }]).areas;
    expect(one && areaRoute(one).map((e) => [e.problem.id, e.module.track.id, e.position])).toEqual([
      ["x3", "x", 1],
      ["x1", "x", 2],
      ["z1", "z", 3],
    ]);
  });

  it("finds the next unsolved core problem, counting an attempted one as unsolved", () => {
    const at = (statuses: Record<string, ProblemStatus>) => {
      const [area] = build(statuses, twoAreas).areas;
      return area ? (nextCore(area)?.problem.id ?? null) : undefined;
    };
    expect(at({})).toBe("x3");
    expect(at({ x3: "solved" })).toBe("x1");
    expect(at({ x3: "solved", x1: "attempted" })).toBe("x1");
    expect(at({ x3: "solved", x1: "solved" })).toBeNull();
  });

  it("moves on to further problems in module order once the core is finished", () => {
    const at = (statuses: Record<string, ProblemStatus>) => {
      const [area] = build(statuses, twoAreas).areas;
      const next = area ? nextInArea(area) : null;
      return next ? [next.problem.id, next.module.track.id] : null;
    };
    expect(at({ y1: "solved" })).toEqual(["x3", "x"]);
    const core = { x3: "solved", x1: "solved" } as const;
    expect(at(core)).toEqual(["y1", "y"]);
    expect(at({ ...core, y1: "solved", y2: "solved" })).toEqual(["x2", "x"]);
    expect(at({ ...core, y1: "solved", y2: "solved", x2: "solved", x4: "solved" })).toBeNull();
  });

  it("reads an area module by module, core before further, and the whole curriculum area by area", () => {
    const curriculum = build({}, twoAreas);
    const [one] = curriculum.areas;
    expect(one && areaOrder(one).map((p) => p.id)).toEqual(["y1", "y2", "x3", "x1", "x2", "x4"]);
    expect(curriculumOrder(curriculum).map((p) => p.id)).toEqual([
      "y1",
      "y2",
      "x3",
      "x1",
      "x2",
      "x4",
      "z1",
      "z2",
    ]);
    // Loose modules come after every area.
    expect(curriculumOrder(build()).map((p) => p.id)).toEqual([
      "x3",
      "x1",
      "x2",
      "x4",
      "y1",
      "y2",
      "z1",
      "z2",
    ]);
  });
});

describe("finding things", () => {
  it("finds an area, its modules, and the loose modules through the first area", () => {
    const curriculum = build({}, [
      { id: "one", title: "One", description: "", modules: ["x"] },
      { id: "two", title: "Two", description: "", modules: ["y"] },
    ]);
    expect(findArea(curriculum, "two")?.title).toBe("Two");
    expect(findArea(curriculum, "nope")).toBeNull();
    expect(findModule(curriculum, "one", "x")?.track.id).toBe("x");
    expect(findModule(curriculum, "one", "y")).toBeNull();
    expect(findModule(curriculum, "nope", "x")).toBeNull();
    expect(findModule(curriculum, "one", "z")?.loose).toBe(true);
    expect(findModule(curriculum, "two", "z")).toBeNull();
    const z = findModule(curriculum, "one", "z");
    const x = findModule(curriculum, "one", "x");
    expect(z && siblings(curriculum, z).map((m) => m.track.id)).toEqual(["z"]);
    expect(x && siblings(curriculum, x).map((m) => m.track.id)).toEqual(["x"]);
  });

  it("says where a problem sits, and its place on the route when it is core", () => {
    const curriculum = build();
    expect(placeOf(curriculum, "x1")).toMatchObject({
      area: { id: "one" },
      module: { numeral: "I" },
      entry: { position: 2, core: true },
      routePosition: 2,
    });
    expect(placeOf(curriculum, "y2")).toMatchObject({
      area: { id: "one" },
      module: { numeral: "II" },
      entry: { position: 2, core: false },
      routePosition: null,
    });
    expect(placeOf(curriculum, "z1")).toMatchObject({
      area: null,
      module: { loose: true },
      routePosition: null,
    });
    expect(placeOf(curriculum, "nope")).toBeNull();
  });
});

describe("pickContinue", () => {
  const pick = (statuses: Record<string, ProblemStatus> = {}, submissions: Submission[] = []) =>
    pickContinue(catalog(statuses), TRACKS, submissions, build(statuses));

  it("starts a new user at the first core problem of the first area, not the first in the catalog", () => {
    const target = pick();
    expect(target?.problem.id).toBe("x3");
    expect(target?.place).toMatchObject({ area: { id: "one" }, routePosition: 1 });
    expect(target?.lastSubmission).toBeUndefined();
  });

  it("moves to the next unsolved core problem after a solve, whatever else was attempted", () => {
    const target = pick({ x3: "solved", y1: "attempted" });
    expect(target?.problem.id).toBe("x1");
    expect(target?.place?.routePosition).toBe(2);
  });

  it("resumes unfinished submitted work first, and says where it sits", () => {
    const further = pick({ x4: "attempted" }, [submission("x3"), submission("x4", 2)]);
    expect(further?.problem.id).toBe("x3");
    const solvedFirst = pick({ x3: "solved", x4: "attempted" }, [submission("x3"), submission("x4", 2)]);
    expect(solvedFirst?.problem.id).toBe("x4");
    expect(solvedFirst?.lastSubmission?.id).toBe(2);
    expect(solvedFirst?.place).toMatchObject({ module: { numeral: "I" }, routePosition: null });
    const loose = pick({ z2: "attempted" }, [submission("z2")]);
    expect(loose?.place).toMatchObject({ area: null, module: { loose: true } });
  });

  it("goes on to further problems in module order, then catalog order, then nothing", () => {
    const core = { x3: "solved", x1: "solved" } as const;
    expect(pick(core)?.problem.id).toBe("x2");
    expect(pick({ ...core, x2: "solved", x4: "solved" })?.problem.id).toBe("y1");
    const area = { ...core, x2: "solved", x4: "solved", y1: "solved", y2: "solved" } as const;
    // The area is finished: what is left is the loose module, in catalog order.
    expect(pick(area)?.problem.id).toBe("z1");
    expect(pick({ ...area, z1: "solved", z2: "solved" })).toBeNull();
  });
});

describe("neighbors", () => {
  it("follows the reading order: a module's core, its further problems, then the next module", () => {
    const curriculum = build();
    expect(neighbors(curriculum, "x3")).toMatchObject({ previous: null, next: { id: "x1" } });
    expect(neighbors(curriculum, "x1")).toMatchObject({ previous: { id: "x3" }, next: { id: "x2" } });
    expect(neighbors(curriculum, "x4")).toMatchObject({ previous: { id: "x2" }, next: { id: "y1" } });
    expect(neighbors(curriculum, "z2")).toMatchObject({ previous: { id: "z1" }, next: null });
    expect(neighbors(curriculum, "missing")).toEqual({ previous: null, next: null });
  });
});

describe("unmetPrerequisites", () => {
  it("lists unsolved prerequisites of unsolved problems only", () => {
    const problems = [
      problem("a", "t", 1, "solved"),
      problem("b", "t", 2, "attempted"),
      problem("c", "t", 3, "todo", { prerequisites: ["a"] }),
      problem("d", "t", 4, "todo", { prerequisites: ["a", "b", "missing"] }),
      problem("e", "t", 5, "solved", { prerequisites: ["b"] }),
      problem("f", "t", 6, "attempted", { prerequisites: ["c", "b"] }),
    ];
    const unmet = unmetPrerequisites(problems);
    expect([...unmet.keys()]).toEqual(["d", "f"]);
    expect(unmet.get("d")?.map((p) => p.id)).toEqual(["b"]);
    expect(unmet.get("f")?.map((p) => p.id)).toEqual(["c", "b"]);
  });
});

describe("unmetOutOfOrder", () => {
  it("keeps only prerequisites the reading order does not already put first", () => {
    const problems = [
      problem("x1", "x", 10, "todo", { prerequisites: ["x3"] }),
      problem("x2", "x", 20),
      problem("x3", "x", 30, "todo", { prerequisites: ["x2", "z1"] }),
      problem("x4", "x", 40),
      problem("z1", "z", 10),
    ];
    const [area] = buildCurriculum(AREAS, TRACKS, problems).areas;
    // Reading order: x3, x1, x2, x4. `x3` comes before `x1`; `x2` after `x3`; `z1` is outside the area.
    const remaining = unmetOutOfOrder(unmetPrerequisites(problems), area ? areaOrder(area) : []);
    expect([...remaining.keys()]).toEqual(["x3"]);
    expect(remaining.get("x3")?.map((p) => p.id)).toEqual(["x2", "z1"]);
  });
});
