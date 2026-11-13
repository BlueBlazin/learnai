import {
  createHashHistory,
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
} from "@tanstack/react-router";
import { ErrorState } from "@/components/app/states";
import { parseFilters, serializeFilters } from "@/lib/filters";
import { type LibrarySearch, type ProblemSearch, STATEMENT_TABS, type StatementTab } from "@/lib/routes";
import { HomeScreen } from "@/routes/home";
import { LibraryScreen } from "@/routes/library";
import { ProblemScreen } from "@/routes/problem";
import { RootLayout } from "@/routes/root";
import { SettingsScreen } from "@/routes/settings";

const rootRoute = createRootRoute({
  component: RootLayout,
  notFoundComponent: () => (
    <div className="h-full bg-paper px-gutter pt-10">
      <ErrorState title="There is no such screen" error="The address does not match any part of Whetstone." />
    </div>
  ),
});

const homeRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: HomeScreen });

const libraryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/library",
  validateSearch: (search: Record<string, unknown>): LibrarySearch => serializeFilters(parseFilters(search)),
  component: function LibraryRoute() {
    return <LibraryScreen search={libraryRoute.useSearch()} />;
  },
});

const problemRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/problems/$problemId",
  validateSearch: (search: Record<string, unknown>): ProblemSearch => {
    const tab = STATEMENT_TABS.find((candidate) => candidate === search.tab) as StatementTab | undefined;
    const submission = Number(search.submission);
    return {
      ...(tab && tab !== "statement" ? { tab } : {}),
      ...(Number.isInteger(submission) && submission > 0 ? { submission } : {}),
    };
  },
  component: function ProblemRoute() {
    const { problemId } = problemRoute.useParams();
    // Keyed by problem so editor, run, and scroll state never leak between problems.
    return <ProblemScreen key={problemId} problemId={problemId} search={problemRoute.useSearch()} />;
  },
});

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: SettingsScreen,
});

/**
 * Development only: every composed component in its states. The whole route,
 * including its import, is dropped from production builds.
 */
const devComponentsRoute = import.meta.env.DEV
  ? createRoute({
      getParentRoute: () => rootRoute,
      path: "/dev/components",
      component: lazyRouteComponent(() => import("@/routes/dev-components"), "DevComponentsScreen"),
    })
  : null;

const routeTree = rootRoute.addChildren([
  homeRoute,
  libraryRoute,
  problemRoute,
  settingsRoute,
  ...(devComponentsRoute ? [devComponentsRoute] : []),
]);

export const router = createRouter({
  routeTree,
  // Hash routing works under any origin Tauri serves the app from.
  history: createHashHistory(),
  defaultPreload: false,
  defaultErrorComponent: ({ error, reset }) => (
    <div className="h-full bg-paper px-gutter pt-10">
      <ErrorState title="This screen stopped working" error={error} onRetry={reset} />
    </div>
  ),
  scrollRestoration: false,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
