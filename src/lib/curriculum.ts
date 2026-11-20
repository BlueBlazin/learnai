import type { Area, ProblemSummary, Track } from "@/lib/api";
import { romanNumeral } from "@/lib/format";
import { moduleColor } from "@/lib/module-colors";
import { sceneUrl } from "@/lib/scenes";

/**
 * The catalog as the learner studies it: areas (subjects) made of modules
 * (tracks), each module with its core problems, in teaching order, and its
 * further problems, the rest. The recommended route through an area is its
 * core problems, module by module.
 */

/** A problem at its place in a module. */
export interface ModuleEntry {
  problem: ProblemSummary;
  /** 1-based position in the module: the core first, the further problems continuing the count. */
  position: number;
  core: boolean;
}

export interface CurriculumModule {
  track: Track;
  /**
   * The area whose screens show the module. For a module outside every area
   * (`loose`) it is the first area, whose screen lists it under "More modules".
   */
  areaId: string;
  /** In no area's list of modules. */
  loose: boolean;
  /** 1-based position in its area (or among the loose modules). */
  number: number;
  /** "III": modules are numbered like the parts of a book. */
  numeral: string;
  /** The scene's image, or null when the module has none. */
  scene: string | null;
  /** `#rrggbb`, or null when the module has none (it falls back to the accent). */
  color: string | null;
  /** The core problems, in the order the track's `core` gives them. */
  core: ModuleEntry[];
  /** The rest of the module, in track order. */
  further: ModuleEntry[];
  coreSolved: number;
  /** Solved problems of the whole module, core and further. */
  solved: number;
  /** Problems in the whole module. */
  total: number;
}

export interface CurriculumArea {
  id: string;
  title: string;
  description: string;
  modules: CurriculumModule[];
  problemCount: number;
  coreCount: number;
  coreSolved: number;
}

export interface Curriculum {
  /** The areas in the order the content defines them. Never empty while there are tracks. */
  areas: CurriculumArea[];
  /** Modules that belong to no area, in track order. */
  loose: CurriculumModule[];
}

/** A core problem at its place on an area's route. */
export interface RouteEntry {
  problem: ProblemSummary;
  module: CurriculumModule;
  /** 1-based position along the route, across modules. */
  position: number;
}

/** Where a problem sits in the curriculum. */
export interface Place {
  /** Null for a problem of a loose module. */
  area: CurriculumArea | null;
  module: CurriculumModule;
  entry: ModuleEntry;
  /** For a core problem of an area: its 1-based position on the area's route. */
  routePosition: number | null;
}

/** The id and title of the area that stands in when the content defines none. */
export const IMPLICIT_AREA_ID = "all";
const IMPLICIT_AREA_TITLE = "Curriculum";

const isSolved = (problem: ProblemSummary) => problem.status === "solved";

function buildModule(
  track: Track,
  own: ProblemSummary[],
  place: { areaId: string; loose: boolean; number: number },
): CurriculumModule {
  const byId = new Map(own.map((problem) => [problem.id, problem]));
  const coreProblems: ProblemSummary[] = [];
  const inCore = new Set<string>();
  // An id the module does not hold (the catalog changed between two fetches) or lists twice is left out.
  for (const id of track.core) {
    const problem = byId.get(id);
    if (!problem || inCore.has(id)) continue;
    inCore.add(id);
    coreProblems.push(problem);
  }
  const furtherProblems = own.filter((problem) => !inCore.has(problem.id)).sort((a, b) => a.order - b.order);
  const core = coreProblems.map(
    (problem, index): ModuleEntry => ({ problem, position: index + 1, core: true }),
  );
  const further = furtherProblems.map(
    (problem, index): ModuleEntry => ({ problem, position: core.length + index + 1, core: false }),
  );
  const coreSolved = coreProblems.filter(isSolved).length;
  return {
    track,
    ...place,
    numeral: romanNumeral(place.number),
    scene: sceneUrl(track.scene),
    color: moduleColor(track),
    core,
    further,
    coreSolved,
    solved: coreSolved + furtherProblems.filter(isSolved).length,
    total: core.length + further.length,
  };
}

function buildArea(area: Omit<CurriculumArea, "problemCount" | "coreCount" | "coreSolved">): CurriculumArea {
  const sum = (count: (module: CurriculumModule) => number) =>
    area.modules.reduce((total, module) => total + count(module), 0);
  return {
    ...area,
    problemCount: sum((module) => module.total),
    coreCount: sum((module) => module.core.length),
    coreSolved: sum((module) => module.coreSolved),
  };
}

/**
 * The curriculum from what the backend serves. An area's modules are its track
 * ids in order (an id with no track is left out, and an area left with no
 * modules is not listed). Tracks no area names are the loose modules. When the
 * content defines no areas at all, every track forms one implicit area.
 */
export function buildCurriculum(areas: Area[], tracks: Track[], problems: ProblemSummary[]): Curriculum {
  const ordered = [...tracks].sort((a, b) => a.order - b.order);
  const byTrack = new Map<string, ProblemSummary[]>();
  for (const problem of problems) {
    const list = byTrack.get(problem.track);
    if (list) list.push(problem);
    else byTrack.set(problem.track, [problem]);
  }
  const trackById = new Map(ordered.map((track) => [track.id, track]));
  const modulesOf = (list: Track[], areaId: string, loose: boolean) =>
    list.map((track, index) =>
      buildModule(track, byTrack.get(track.id) ?? [], { areaId, loose, number: index + 1 }),
    );

  const built = areas.flatMap((area) => {
    const own = [...new Set(area.modules)].flatMap((id) => trackById.get(id) ?? []);
    if (own.length === 0) return [];
    return [
      buildArea({
        id: area.id,
        title: area.title,
        description: area.description,
        modules: modulesOf(own, area.id, false),
      }),
    ];
  });
  if (built.length === 0) {
    if (ordered.length === 0) return { areas: [], loose: [] };
    return {
      areas: [
        buildArea({
          id: IMPLICIT_AREA_ID,
          title: IMPLICIT_AREA_TITLE,
          description: "",
          modules: modulesOf(ordered, IMPLICIT_AREA_ID, false),
        }),
      ],
      loose: [],
    };
  }
  const placed = new Set(built.flatMap((area) => area.modules.map((module) => module.track.id)));
  const first = built[0] as CurriculumArea;
  return {
    areas: built,
    loose: modulesOf(
      ordered.filter((track) => !placed.has(track.id)),
      first.id,
      true,
    ),
  };
}

/** A module's problems in reading order: the core, then the further problems. */
export function moduleEntries(module: CurriculumModule): ModuleEntry[] {
  return [...module.core, ...module.further];
}

/** Whether the module has a core and all of it is solved. */
export function coreComplete(module: CurriculumModule): boolean {
  return module.core.length > 0 && module.coreSolved === module.core.length;
}

/** Whether the module has problems and every one of them, core and further, is solved. */
export function allSolved(module: CurriculumModule): boolean {
  return module.total > 0 && module.solved === module.total;
}

/** How many of the modules have their core complete, and how many have a core at all. */
export function coresComplete(modules: CurriculumModule[]): { complete: number; withCore: number } {
  return {
    complete: modules.filter(coreComplete).length,
    withCore: modules.filter((module) => module.core.length > 0).length,
  };
}

/** "6 core, 7 further", leaving out a kind the module has none of. */
export function moduleCounts(module: CurriculumModule): string {
  const parts = [
    module.core.length > 0 ? `${module.core.length} core` : null,
    module.further.length > 0 ? `${module.further.length} further` : null,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(", ") : "No problems yet";
}

/** The area's recommended route: its core problems, module by module. */
export function areaRoute(area: CurriculumArea): RouteEntry[] {
  const route: RouteEntry[] = [];
  for (const module of area.modules) {
    for (const entry of module.core) {
      route.push({ problem: entry.problem, module, position: route.length + 1 });
    }
  }
  return route;
}

/** The first core problem on the area's route that is not solved. Null when the core is finished. */
export function nextCore(area: CurriculumArea): RouteEntry | null {
  return areaRoute(area).find((entry) => !isSolved(entry.problem)) ?? null;
}

/**
 * What to study next in the area: the next unsolved core problem, and once the
 * core is finished the next unsolved further problem in module order. Null
 * when every problem of the area is solved.
 */
export function nextInArea(
  area: CurriculumArea,
): { problem: ProblemSummary; module: CurriculumModule } | null {
  const core = nextCore(area);
  if (core) return { problem: core.problem, module: core.module };
  for (const module of area.modules) {
    const entry = module.further.find((candidate) => !isSolved(candidate.problem));
    if (entry) return { problem: entry.problem, module };
  }
  return null;
}

/** The area's problems in reading order: each module's core, then its further problems, then the next module. */
export function areaOrder(area: CurriculumArea): ProblemSummary[] {
  return area.modules.flatMap((module) => moduleEntries(module).map((entry) => entry.problem));
}

/**
 * Every problem in reading order: the areas in turn, then the loose modules.
 * A problem of a module that two areas share is listed where it comes first.
 */
export function curriculumOrder(curriculum: Curriculum): ProblemSummary[] {
  const seen = new Set<string>();
  const order: ProblemSummary[] = [];
  const modules = [...curriculum.areas.flatMap((area) => area.modules), ...curriculum.loose];
  for (const module of modules) {
    for (const { problem } of moduleEntries(module)) {
      if (seen.has(problem.id)) continue;
      seen.add(problem.id);
      order.push(problem);
    }
  }
  return order;
}

/** The area with this id. */
export function findArea(curriculum: Curriculum, areaId: string): CurriculumArea | null {
  return curriculum.areas.find((area) => area.id === areaId) ?? null;
}

/**
 * The module with this track id as the area's screens show it: one of the
 * area's own, or a loose module, which is reached through the first area.
 */
export function findModule(
  curriculum: Curriculum,
  areaId: string,
  moduleId: string,
): CurriculumModule | null {
  const area = findArea(curriculum, areaId);
  if (!area) return null;
  return (
    area.modules.find((module) => module.track.id === moduleId) ??
    (area === curriculum.areas[0] ? curriculum.loose.find((module) => module.track.id === moduleId) : null) ??
    null
  );
}

/** The modules a module is numbered among: its area's, or the loose ones. */
export function siblings(curriculum: Curriculum, module: CurriculumModule): CurriculumModule[] {
  return module.loose ? curriculum.loose : (findArea(curriculum, module.areaId)?.modules ?? []);
}

/** Where the problem sits: in the first area that holds it, otherwise in a loose module. */
export function placeOf(curriculum: Curriculum, problemId: string): Place | null {
  for (const area of curriculum.areas) {
    for (const module of area.modules) {
      const entry = moduleEntries(module).find((candidate) => candidate.problem.id === problemId);
      if (!entry) continue;
      const routePosition = entry.core
        ? (areaRoute(area).find((candidate) => candidate.problem.id === problemId)?.position ?? null)
        : null;
      return { area, module, entry, routePosition };
    }
  }
  for (const module of curriculum.loose) {
    const entry = moduleEntries(module).find((candidate) => candidate.problem.id === problemId);
    if (entry) return { area: null, module, entry, routePosition: null };
  }
  return null;
}
