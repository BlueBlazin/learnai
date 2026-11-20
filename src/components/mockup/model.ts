import type { CSSProperties } from "react";
import type { Path, ProblemSummary, Track } from "@/lib/api";
import { type PathEntry, romanNumeral } from "@/lib/paths";
import { sceneUrl } from "@/lib/scenes";
import { markColor } from "@/lib/stage-colors";

/**
 * The proposed structure, derived from today's data for the mockup only: an
 * area holds modules (today's tracks), and a module's problems are core (on
 * the first study path) or further (the rest of the track). Nothing here is
 * stored; the real data model will carry scene, color, and core itself.
 */

/** The three views of the mockup. */
export type MockupView = "areas" | "area" | "module";
export const MOCKUP_VIEWS: readonly MockupView[] = ["areas", "area", "module"];

/** The module the Module view shows when none is named. */
export const DEFAULT_MODULE = "transformer";

/** Working title and sentence for the one area that exists. The name is the owner's to choose. */
export const AREA_TITLE = "Inference engineering";
export const AREA_DESCRIPTION =
  "Everything between trained weights and a served answer: tokenizing, the forward pass, sampling, caching, quantization, and serving.";

/** Areas that do not exist yet, shown to see how the top level scales. */
export const PLANNED_AREAS: readonly { title: string; description: string }[] = [
  {
    title: "Model architectures",
    description:
      "The building blocks of modern networks written from scratch: attention variants, normalization, convolution, and diffusion.",
  },
  {
    title: "Mathematics core",
    description:
      "The linear algebra, probability, and optimization the other areas lean on, as problems to work instead of chapters to read.",
  },
  {
    title: "Interpretability",
    description:
      "Reading what a trained model has learned: probes, attribution, activation patching, and sparse autoencoders.",
  },
];

/** Scene and color per track id. A track that is not listed gets no scene and the accent color. */
const MODULE_ART: Record<string, { scene: string; color: string }> = {
  foundations: { scene: "stage-01-trailhead", color: "#ed9b11" },
  tokenization: { scene: "stage-02-crossing", color: "#e5691d" },
  transformer: { scene: "stage-03-forest", color: "#2a71cc" },
  sampling: { scene: "stage-04-meadow", color: "#db5881" },
  "kv-cache": { scene: "stage-05-lake", color: "#18978e" },
  constrained: { scene: "stage-06-terraces", color: "#8fc30b" },
  quantization: { scene: "stage-07-canyon", color: "#834ccd" },
  speculative: { scene: "stage-08-ridge", color: "#f4c60a" },
  sizing: { scene: "stage-09-tower", color: "#c2332f" },
  mlx: { scene: "stage-11-glacier", color: "#22964c" },
  cuda: { scene: "module-trailblazer", color: "#a85c21" },
  serving: { scene: "stage-10-village", color: "#27add9" },
  systems: { scene: "stage-12-pass", color: "#c94db5" },
  frontier: { scene: "module-cape", color: "#5a5fc0" },
  capstones: { scene: "stage-13-summit", color: "#e6b125" },
};

const FALLBACK_COLOR = "var(--primary)";

export interface Module {
  track: Track;
  /** 0-based position in the area. */
  index: number;
  /** "III": modules are numbered like the parts of a book. */
  numeral: string;
  /** The scene's image, or null when the module has none. */
  scene: string | null;
  /** `#rrggbb`, or null when the module falls back to the accent. */
  color: string | null;
  /** The essential problems, in the order the study path gives them. */
  core: ProblemSummary[];
  /** The rest of the module, in track order. */
  further: ProblemSummary[];
  coreSolved: number;
  furtherSolved: number;
}

export interface Area {
  title: string;
  description: string;
  modules: Module[];
  problemCount: number;
  coreCount: number;
  coreSolved: number;
  /** The first unsolved core problem in module order, with its module. Null when every core problem is solved. */
  next: { problem: ProblemSummary; module: Module } | null;
}

const solvedIn = (problems: ProblemSummary[]) =>
  problems.filter((problem) => problem.status === "solved").length;

/** The area made of every track, with core taken from `path` (nothing is core when there is no path). */
export function buildArea(tracks: Track[], problems: ProblemSummary[], path: Path | null): Area {
  const routeOrder = new Map<string, number>();
  for (const id of path?.stages.flatMap((stage) => stage.problems) ?? []) {
    if (!routeOrder.has(id)) routeOrder.set(id, routeOrder.size);
  }
  const modules = [...tracks]
    .sort((a, b) => a.order - b.order)
    .map((track, index): Module => {
      const own = problems.filter((problem) => problem.track === track.id);
      const core = own
        .filter((problem) => routeOrder.has(problem.id))
        .sort((a, b) => (routeOrder.get(a.id) ?? 0) - (routeOrder.get(b.id) ?? 0));
      const further = own.filter((problem) => !routeOrder.has(problem.id)).sort((a, b) => a.order - b.order);
      const art = MODULE_ART[track.id];
      return {
        track,
        index,
        numeral: romanNumeral(index + 1),
        scene: sceneUrl(art?.scene),
        color: art?.color ?? null,
        core,
        further,
        coreSolved: solvedIn(core),
        furtherSolved: solvedIn(further),
      };
    });
  let next: Area["next"] = null;
  for (const module of modules) {
    const problem = module.core.find((candidate) => candidate.status !== "solved");
    if (problem) {
      next = { problem, module };
      break;
    }
  }
  return {
    title: AREA_TITLE,
    description: AREA_DESCRIPTION,
    modules,
    problemCount: modules.reduce((sum, module) => sum + module.core.length + module.further.length, 0),
    coreCount: modules.reduce((sum, module) => sum + module.core.length, 0),
    coreSolved: modules.reduce((sum, module) => sum + module.coreSolved, 0),
    next,
  };
}

/**
 * The same custom properties `stageColorVars` sets for a path's stages, for an
 * area's modules: `--stage-N` at full saturation for bar fills and
 * `--stage-N-mark` for pips and swatches, so `stageFill` and `stageMark` work
 * with a module's index.
 */
export function moduleColorVars(modules: Module[]): CSSProperties {
  const vars: Record<string, string> = {};
  for (const module of modules) {
    vars[`--stage-${module.index + 1}`] = module.color ?? FALLBACK_COLOR;
    vars[`--stage-${module.index + 1}-mark`] = module.color ? markColor(module.color) : FALLBACK_COLOR;
  }
  return vars as CSSProperties;
}

/**
 * A module's problems as the numbered lines `PathEntryList` draws: core first,
 * further continuing the count.
 */
export function moduleEntries(module: Module): { core: PathEntry[]; further: PathEntry[] } {
  const entry = (problem: ProblemSummary, offset: number): PathEntry => ({
    problem,
    position: offset + 1,
    stage: module.index,
  });
  return {
    core: module.core.map((problem, index) => entry(problem, index)),
    further: module.further.map((problem, index) => entry(problem, module.core.length + index)),
  };
}

/**
 * Unsolved prerequisites worth a remark on a module's page: those that do not
 * come earlier in the area's reading order (module by module, core before
 * further). The order of the page already says "do these first".
 */
export function unmetOutOfOrder(
  unmet: Map<string, ProblemSummary[]>,
  modules: Module[],
): Map<string, ProblemSummary[]> {
  const place = new Map<string, number>();
  for (const module of modules) {
    for (const problem of [...module.core, ...module.further]) place.set(problem.id, place.size);
  }
  const remaining = new Map<string, ProblemSummary[]>();
  for (const [id, prerequisites] of unmet) {
    const at = place.get(id);
    if (at === undefined) continue;
    const open = prerequisites.filter((prerequisite) => (place.get(prerequisite.id) ?? Infinity) > at);
    if (open.length > 0) remaining.set(id, open);
  }
  return remaining;
}

/** "5 core, 8 further", leaving out a kind the module has none of. */
export function moduleCounts(module: Module): string {
  const parts = [
    module.core.length > 0 ? `${module.core.length} core` : null,
    module.further.length > 0 ? `${module.further.length} further` : null,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(", ") : "No problems yet";
}
