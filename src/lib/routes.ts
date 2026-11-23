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
  core?: string;
}

export interface ProblemSearch {
  tab?: StatementTab;
  /** A stored submission to open in the Submissions tab. */
  submission?: number;
  /** Review mode: the editor holds the review attempt and Run and Submit judge it. */
  review?: boolean;
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

export interface AccomplishmentsSearch {
  /** The id of the skill whose detail is open. */
  skill?: string;
}

/** The figure gallery's search: one module (a track id), or all when absent. */
export interface FiguresSearch {
  track?: string;
}

export type RouteTarget =
  | { to: "/" }
  | { to: "/curriculum" }
  | { to: "/curriculum/$area"; params: { area: string } }
  | { to: "/curriculum/$area/$module"; params: { area: string; module: string } }
  | { to: "/accomplishments"; search: AccomplishmentsSearch }
  | { to: "/library"; search: LibrarySearch }
  | { to: "/settings"; search: SettingsSearch }
  | { to: "/dev/components" }
  | { to: "/dev/figures"; search: FiguresSearch }
  | { to: "/problems/$problemId"; params: { problemId: string }; search: ProblemSearch };

export const routes = {
  home: (): RouteTarget => ({ to: "/" }),
  /** The Curriculum: the one area's screen, or the list of areas when there are several. */
  curriculum: (): RouteTarget => ({ to: "/curriculum" }),
  /** The screen of the area with this id. */
  area: (area: string): RouteTarget => ({ to: "/curriculum/$area", params: { area } }),
  /** The page of a module (a track id) as the area with this id shows it. */
  module: (area: string, module: string): RouteTarget => ({
    to: "/curriculum/$area/$module",
    params: { area, module },
  }),
  /** Accomplishments: the tree of skills, with the detail of the skill with this id open when one is given. */
  accomplishments: (skill?: string): RouteTarget => ({
    to: "/accomplishments",
    search: skill ? { skill } : {},
  }),
  library: (search: LibrarySearch = {}): RouteTarget => ({ to: "/library", search }),
  /** Settings, opened at a category; General (the default) stays out of the URL. */
  settings: (section: SettingsSection = "general"): RouteTarget => ({
    to: "/settings",
    search: section === "general" ? {} : { section },
  }),
  devComponents: (): RouteTarget => ({ to: "/dev/components" }),
  /** The dev-only figure gallery, for every module or one. */
  devFigures: (search: FiguresSearch = {}): RouteTarget => ({ to: "/dev/figures", search }),
  problem: (problemId: string, search: ProblemSearch = {}): RouteTarget => ({
    to: "/problems/$problemId",
    params: { problemId },
    search,
  }),
};
