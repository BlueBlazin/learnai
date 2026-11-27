import { describe, expect, it } from "vitest";
import type { ProblemSummary, Track } from "@/lib/api";
import {
  activeFilterCount,
  coreProblemIds,
  EMPTY_FILTERS,
  filterProblems,
  groupByTrack,
  nextUnsolved,
  parseFilters,
  serializeFilters,
} from "./filters";

const problem = (over: Partial<ProblemSummary> & { id: string }): ProblemSummary => ({
  title: over.id,
  track: "a",
  order: 10,
  difficulty: "easy",
  level: 1,
  frameworks: ["torch"],
  judges: ["numeric"],
  tags: [],
  prerequisites: [],
  requires: "cpu",
  status: "todo",
  solvedFrameworks: [],
  assets: [],
  ...over,
});

const track = (id: string, order: number): Track => ({
  id,
  title: id.toUpperCase(),
  description: "",
  order,
  problemCount: 0,
  solvedCount: 0,
  core: [],
});

const problems = [
  problem({ id: "softmax", title: "Stable Softmax", tags: ["numerics"], status: "solved" }),
  problem({
    id: "top-p",
    title: "Nucleus Sampling",
    track: "b",
    difficulty: "medium",
    judges: ["statistical"],
    tags: ["sampling", "top-p"],
    status: "attempted",
    order: 20,
  }),
  problem({
    id: "batching",
    title: "Continuous Batching",
    track: "b",
    difficulty: "hard",
    frameworks: ["python"],
    judges: ["simclock"],
    order: 10,
  }),
  problem({
    id: "rope",
    title: "Rotary Embeddings",
    difficulty: "medium",
    frameworks: ["torch", "mlx"],
    judges: ["numeric", "speed"],
    order: 20,
  }),
];
const tracks = [track("b", 2), track("a", 1)];
const ids = (list: ProblemSummary[]) => list.map((p) => p.id);

describe("filterProblems", () => {
  it("returns everything for empty filters", () => {
    expect(filterProblems(problems, EMPTY_FILTERS)).toHaveLength(4);
  });

  it("filters by each dimension", () => {
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, track: "b" }))).toEqual(["top-p", "batching"]);
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, difficulty: ["medium", "hard"] }))).toEqual([
      "top-p",
      "batching",
      "rope",
    ]);
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, status: ["solved"] }))).toEqual(["softmax"]);
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, framework: "mlx" }))).toEqual(["rope"]);
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, judge: "speed" }))).toEqual(["rope"]);
  });

  it("combines filters with AND", () => {
    expect(
      ids(
        filterProblems(problems, { ...EMPTY_FILTERS, track: "b", difficulty: ["hard"], framework: "python" }),
      ),
    ).toEqual(["batching"]);
    expect(filterProblems(problems, { ...EMPTY_FILTERS, track: "a", judge: "simclock" })).toEqual([]);
  });

  it("matches every search term against title, id, and tags, ignoring case", () => {
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, q: "SAMPLING" }))).toEqual(["top-p"]);
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, q: "numerics" }))).toEqual(["softmax"]);
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, q: "  rotary   emb " }))).toEqual(["rope"]);
    expect(filterProblems(problems, { ...EMPTY_FILTERS, q: "rotary sampling" })).toEqual([]);
  });
});

describe("the core filter", () => {
  const core = coreProblemIds([
    { ...track("a", 1), core: ["rope"] },
    { ...track("b", 2), core: ["batching", "gone"] },
  ]);

  it("collects every module's core ids", () => {
    expect([...core]).toEqual(["rope", "batching", "gone"]);
  });

  it("keeps core problems only, further problems only, or both", () => {
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, core: "core" }, core))).toEqual([
      "batching",
      "rope",
    ]);
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, core: "further" }, core))).toEqual([
      "softmax",
      "top-p",
    ]);
    expect(filterProblems(problems, EMPTY_FILTERS, core)).toHaveLength(problems.length);
    expect(ids(filterProblems(problems, { ...EMPTY_FILTERS, core: "core", track: "a" }, core))).toEqual([
      "rope",
    ]);
  });

  it("round-trips through the URL, counts as a filter, and drops unknown values", () => {
    const filters = { ...EMPTY_FILTERS, core: "further" as const };
    expect(serializeFilters(filters)).toEqual({ core: "further" });
    expect(parseFilters({ core: "further" })).toEqual(filters);
    expect(parseFilters({ core: "all" }).core).toBeNull();
    expect(activeFilterCount(filters)).toBe(1);
  });
});

describe("filter URL round trip", () => {
  it("drops unknown values and keeps known ones in canonical order", () => {
    const parsed = parseFilters({
      q: "x",
      difficulty: "hard,bogus,easy",
      status: "solved",
      framework: "tensorflow",
      judge: "cost",
      track: "",
    });
    expect(parsed).toEqual({
      q: "x",
      track: null,
      difficulty: ["easy", "hard"],
      status: ["solved"],
      framework: null,
      judge: "cost",
      core: null,
    });
    expect(serializeFilters(parsed)).toEqual({
      q: "x",
      difficulty: "easy,hard",
      status: "solved",
      judge: "cost",
    });
    expect(parseFilters(serializeFilters(parsed))).toEqual(parsed);
  });

  it("serializes empty filters to nothing", () => {
    expect(serializeFilters(EMPTY_FILTERS)).toEqual({});
    expect(activeFilterCount(EMPTY_FILTERS)).toBe(0);
    expect(activeFilterCount({ ...EMPTY_FILTERS, q: " ", difficulty: ["easy"], judge: "cost" })).toBe(2);
  });
});

describe("catalog order", () => {
  it("groups by track order, then problem order, dropping empty tracks", () => {
    const groups = groupByTrack(problems, [...tracks, track("empty", 3)]);
    expect(groups.map((g) => g.track.id)).toEqual(["a", "b"]);
    expect(ids(groups[1]?.problems ?? [])).toEqual(["batching", "top-p"]);
  });

  it("finds the first unsolved problem", () => {
    expect(nextUnsolved(problems, tracks)?.id).toBe("rope");
  });
});
