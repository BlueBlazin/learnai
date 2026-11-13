import type { Difficulty, Framework, JudgeKind, ProblemStatus, ProblemSummary, Track } from "@/lib/api";
import { DIFFICULTIES, FRAMEWORKS, JUDGES, STATUSES } from "@/lib/api";

export interface LibraryFilters {
  q: string;
  track: string | null;
  difficulty: Difficulty[];
  status: ProblemStatus[];
  framework: Framework | null;
  judge: JudgeKind | null;
}

export const EMPTY_FILTERS: LibraryFilters = {
  q: "",
  track: null,
  difficulty: [],
  status: [],
  framework: null,
  judge: null,
};

function oneOf<T extends string>(value: unknown, allowed: readonly T[]): T | null {
  return typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : null;
}

function listOf<T extends string>(value: unknown, allowed: readonly T[]): T[] {
  const parts = Array.isArray(value) ? value : typeof value === "string" ? value.split(",") : [];
  return allowed.filter((item) => parts.includes(item));
}

/** Parses URL search parameters into filters, dropping anything unrecognized. */
export function parseFilters(search: Record<string, unknown>): LibraryFilters {
  return {
    q: typeof search.q === "string" ? search.q : "",
    track: typeof search.track === "string" && search.track ? search.track : null,
    difficulty: listOf(search.difficulty, DIFFICULTIES),
    status: listOf(search.status, STATUSES),
    framework: oneOf(search.framework, FRAMEWORKS),
    judge: oneOf(search.judge, JUDGES),
  };
}

/** The inverse of `parseFilters`: only non-default values appear in the URL. */
export function serializeFilters(filters: LibraryFilters): Record<string, string> {
  const search: Record<string, string> = {};
  if (filters.q) search.q = filters.q;
  if (filters.track) search.track = filters.track;
  if (filters.difficulty.length) search.difficulty = filters.difficulty.join(",");
  if (filters.status.length) search.status = filters.status.join(",");
  if (filters.framework) search.framework = filters.framework;
  if (filters.judge) search.judge = filters.judge;
  return search;
}

export function activeFilterCount(filters: LibraryFilters): number {
  return (
    (filters.q.trim() ? 1 : 0) +
    (filters.track ? 1 : 0) +
    (filters.difficulty.length ? 1 : 0) +
    (filters.status.length ? 1 : 0) +
    (filters.framework ? 1 : 0) +
    (filters.judge ? 1 : 0)
  );
}

function matchesText(problem: ProblemSummary, terms: string[]): boolean {
  if (terms.length === 0) return true;
  const haystack = `${problem.title} ${problem.id} ${problem.tags.join(" ")}`.toLowerCase();
  return terms.every((term) => haystack.includes(term));
}

/** Applies every filter. Text search requires all whitespace-separated terms to match title, id, or tags. */
export function filterProblems(problems: ProblemSummary[], filters: LibraryFilters): ProblemSummary[] {
  const terms = filters.q.toLowerCase().split(/\s+/).filter(Boolean);
  return problems.filter(
    (problem) =>
      (filters.track === null || problem.track === filters.track) &&
      (filters.difficulty.length === 0 || filters.difficulty.includes(problem.difficulty)) &&
      (filters.status.length === 0 || filters.status.includes(problem.status)) &&
      (filters.framework === null || problem.frameworks.includes(filters.framework)) &&
      (filters.judge === null || problem.judges.includes(filters.judge)) &&
      matchesText(problem, terms),
  );
}

export interface TrackGroup {
  track: Track;
  problems: ProblemSummary[];
}

/** Groups problems under their tracks in catalog order; tracks with no matches are dropped. */
export function groupByTrack(problems: ProblemSummary[], tracks: Track[]): TrackGroup[] {
  const byTrack = new Map<string, ProblemSummary[]>();
  for (const problem of problems) {
    const list = byTrack.get(problem.track);
    if (list) list.push(problem);
    else byTrack.set(problem.track, [problem]);
  }
  return [...tracks]
    .sort((a, b) => a.order - b.order)
    .flatMap((track) => {
      const list = byTrack.get(track.id);
      if (!list) return [];
      return [{ track, problems: [...list].sort((a, b) => a.order - b.order) }];
    });
}

/** The next problem to work on: the first unsolved one in catalog order. */
export function nextUnsolved(problems: ProblemSummary[], tracks: Track[]): ProblemSummary | null {
  for (const group of groupByTrack(problems, tracks)) {
    const candidate = group.problems.find((problem) => problem.status !== "solved");
    if (candidate) return candidate;
  }
  return null;
}
