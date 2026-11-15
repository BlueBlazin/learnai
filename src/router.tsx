import { createHashHistory, createRootRoute, createRoute, createRouter } from "@tanstack/react-router";
import { ScreenMessage } from "@/components/app/page";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { parseFilters, serializeFilters } from "@/lib/filters";
import {
  type LibrarySearch,
  type PathsSearch,
  type ProblemSearch,
  STATEMENT_TABS,
  type StatementTab,
} from "@/lib/routes";
import { HomeScreen } from "@/routes/home";
import { RootLayout } from "@/routes/root";

function ScreenLoading() {
  return (
    <ScreenMessage>
      <LoadingLines lines={8} className="max-w-measure" />
    </ScreenMessage>
  );
}

const rootRoute = createRootRoute({
  component: RootLayout,
  notFoundComponent: () => (
    <ScreenMessage>
      <ErrorState title="There is no such screen" error="The address does not match any part of Whetstone." />
    </ScreenMessage>
  ),
});

// Home is the first screen and stays in the main bundle. Every other screen is
// its own chunk: the route's critical half (path, search validation) is defined
// here, and `.lazy()` loads the half that renders (a `createLazyRoute` exported
// from the screen's module) while the router is still on the previous screen.
const homeRoute = createRoute({ getParentRoute: () => rootRoute, path: "/", component: HomeScreen });

const libraryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/library",
  validateSearch: (search: Record<string, unknown>): LibrarySearch => serializeFilters(parseFilters(search)),
}).lazy(() => import("@/routes/library").then((module) => module.Route));

const pathsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/paths",
  validateSearch: (search: Record<string, unknown>): PathsSearch =>
    typeof search.path === "string" && search.path ? { path: search.path } : {},
}).lazy(() => import("@/routes/paths").then((module) => module.Route));

const problemRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/problems/$problemId",
  validateSearch: (search: Record<string, unknown>): ProblemSearch => {
    const tab = STATEMENT_TABS.find((candidate) => candidate === search.tab) as StatementTab | undefined;
    const submission = Number(search.submission);
    return {
      ...(tab && tab !== "statement" ? { tab } : {}),
      ...(Number.isInteger(submission) && submission > 0 ? { submission } : {}),
      ...(search.review === true || search.review === "true" || search.review === 1 ? { review: true } : {}),
    };
  },
}).lazy(() => import("@/routes/problem").then((module) => module.Route));

const settingsRoute = createRoute({ getParentRoute: () => rootRoute, path: "/settings" }).lazy(() =>
  import("@/routes/settings").then((module) => module.Route),
);

/**
 * Development only: every composed component in its states. The whole route,
 * including its import, is dropped from production builds.
 */
const devComponentsRoute = import.meta.env.DEV
  ? createRoute({ getParentRoute: () => rootRoute, path: "/dev/components" }).lazy(() =>
      import("@/routes/dev-components").then((module) => module.Route),
    )
  : null;

const routeTree = rootRoute.addChildren([
  homeRoute,
  pathsRoute,
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
  defaultPendingComponent: ScreenLoading,
  defaultErrorComponent: ({ error, reset }) => (
    <ScreenMessage>
      <ErrorState title="This screen stopped working" error={error} onRetry={reset} />
    </ScreenMessage>
  ),
  scrollRestoration: false,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
