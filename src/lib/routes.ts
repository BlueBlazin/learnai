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
}

export type RouteTarget =
  | { to: "/" }
  | { to: "/paths"; search: PathsSearch }
  | { to: "/library"; search: LibrarySearch }
  | { to: "/settings" }
  | { to: "/dev/components" }
  | { to: "/problems/$problemId"; params: { problemId: string }; search: ProblemSearch };

export const routes = {
  home: (): RouteTarget => ({ to: "/" }),
  /** The study path with this id, or the first path when no id is given. */
  paths: (path?: string): RouteTarget => ({ to: "/paths", search: path ? { path } : {} }),
  library: (search: LibrarySearch = {}): RouteTarget => ({ to: "/library", search }),
  settings: (): RouteTarget => ({ to: "/settings" }),
  devComponents: (): RouteTarget => ({ to: "/dev/components" }),
  problem: (problemId: string, search: ProblemSearch = {}): RouteTarget => ({
    to: "/problems/$problemId",
    params: { problemId },
    search,
  }),
};
