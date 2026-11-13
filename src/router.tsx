import {
  createHashHistory,
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
} from "@tanstack/react-router";
import { lazy, type ReactNode, Suspense } from "react";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { parseFilters, serializeFilters } from "@/lib/filters";
import { type LibrarySearch, type ProblemSearch, STATEMENT_TABS, type StatementTab } from "@/lib/routes";
import { HomeScreen } from "@/routes/home";
import { RootLayout } from "@/routes/root";

// Home is the first screen and stays in the main bundle; every other screen is
// its own chunk, loaded the first time it is opened.
const LibraryScreen = lazy(() => import("@/routes/library").then((m) => ({ default: m.LibraryScreen })));
const ProblemScreen = lazy(() => import("@/routes/problem").then((m) => ({ default: m.ProblemScreen })));
const SettingsScreen = lazy(() => import("@/routes/settings").then((m) => ({ default: m.SettingsScreen })));

function ScreenLoading() {
  return (
    <div className="h-full bg-paper px-gutter pt-10">
      <LoadingLines lines={8} className="max-w-measure" />
    </div>
  );
}

/** Shows the loading lines while a screen's chunk arrives. */
function Screen({ children }: { children: ReactNode }) {
  return <Suspense fallback={<ScreenLoading />}>{children}</Suspense>;
}

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
    return (
      <Screen>
        <LibraryScreen search={libraryRoute.useSearch()} />
      </Screen>
    );
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
    return (
      <Screen>
        <ProblemScreen key={problemId} problemId={problemId} search={problemRoute.useSearch()} />
      </Screen>
    );
  },
});

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: function SettingsRoute() {
    return (
      <Screen>
        <SettingsScreen />
      </Screen>
    );
  },
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
  defaultPendingComponent: ScreenLoading,
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
