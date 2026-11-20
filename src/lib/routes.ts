/**
 * Route targets. Every in-app link and navigation is built here so path
 * strings and search-parameter names exist in exactly one place.
 */

export type StatementTab = "statement" | "hints" | "editorial" | "submissions";
export const STATEMENT_TABS: readonly StatementTab[] = ["statement", "hints", "editorial", "submissions"];

export interface LibrarySearch {
  q?: string;
  track?: string;
  difficulty?: string;
  status?: string;
  framework?: string;
  judge?: string;
}

export interface ProblemSearch {
  tab?: StatementTab;
  /** A stored submission to open in the Submissions tab. */
  submission?: number;
  /** Review mode: the editor holds the review attempt and Run and Submit judge it. */
  review?: boolean;
}

export interface PathsSearch {
  /** The path to show; the first one when absent. */
  path?: string;
  /** A stage to scroll to, by its part number (1-based). */
  part?: number;
}

/** The categories of the Settings screen, in the order its list shows them. */
export type SettingsSection = "general" | "environment" | "models" | "targets" | "files";
export const SETTINGS_SECTIONS: readonly SettingsSection[] = [
  "general",
  "environment",
  "models",
  "targets",
  "files",
];

export interface SettingsSearch {
  /** The category to show; General when absent. */
  section?: SettingsSection;
}

/** The category a `section` search value names; anything unknown is General. */
export function parseSettingsSection(value: unknown): SettingsSection {
  return SETTINGS_SECTIONS.find((candidate) => candidate === value) ?? "general";
}

/** Development only: the areas and modules mockup. */
export interface AreasMockupSearch {
  /** The view to show; the list of areas when absent. */
  view?: "area" | "module";
  /** The module (a track id) the Module view shows. */
  module?: string;
}

export type RouteTarget =
  | { to: "/" }
  | { to: "/paths"; search: PathsSearch }
  | { to: "/library"; search: LibrarySearch }
  | { to: "/settings"; search: SettingsSearch }
  | { to: "/dev/components" }
  | { to: "/dev/areas-mockup"; search: AreasMockupSearch }
  | { to: "/problems/$problemId"; params: { problemId: string }; search: ProblemSearch };

export const routes = {
  home: (): RouteTarget => ({ to: "/" }),
  /** The study path with this id (the first when no id is given), scrolled to a part when one is given. */
  paths: (path?: string, part?: number): RouteTarget => ({
    to: "/paths",
    search: { ...(path ? { path } : {}), ...(part ? { part } : {}) },
  }),
  library: (search: LibrarySearch = {}): RouteTarget => ({ to: "/library", search }),
  /** Settings, opened at a category; General (the default) stays out of the URL. */
  settings: (section: SettingsSection = "general"): RouteTarget => ({
    to: "/settings",
    search: section === "general" ? {} : { section },
  }),
  devComponents: (): RouteTarget => ({ to: "/dev/components" }),
  /** The mockup at its list of areas, at the area, or at one module of it. */
  devAreasMockup: (view?: "area" | "module", module?: string): RouteTarget => ({
    to: "/dev/areas-mockup",
    search: { ...(view ? { view } : {}), ...(view === "module" && module ? { module } : {}) },
  }),
  problem: (problemId: string, search: ProblemSearch = {}): RouteTarget => ({
    to: "/problems/$problemId",
    params: { problemId },
    search,
  }),
};
