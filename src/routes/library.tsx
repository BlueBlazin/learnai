import { createLazyRoute, getRouteApi } from "@tanstack/react-router";
import { useEffect, useMemo, useRef } from "react";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { LibraryFilterBar } from "@/components/library/library-filter-bar";
import { TrackList } from "@/components/library/track-list";
import { TrackSection } from "@/components/library/track-section";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useGo } from "@/hooks/use-go";
import { useScrollMemory } from "@/hooks/use-scroll-memory";
import { useProblems, useTracks } from "@/lib/api/queries";
import {
  activeFilterCount,
  EMPTY_FILTERS,
  filterProblems,
  groupByTrack,
  type LibraryFilters,
  parseFilters,
  serializeFilters,
} from "@/lib/filters";
import { pluralize } from "@/lib/format";
import { unmetPrerequisites } from "@/lib/paths";
import { routes } from "@/lib/routes";

const route = getRouteApi("/library");

function LibraryScreen() {
  const search = route.useSearch();
  const go = useGo();
  const tracks = useTracks();
  const problems = useProblems();
  const searchRef = useRef<HTMLInputElement>(null);
  const viewport = useRef<HTMLDivElement>(null);

  // The URL is the source of truth for filters, so Back restores them.
  const filters = useMemo(() => parseFilters({ ...search }), [search]);
  const setFilters = (next: LibraryFilters) => go(routes.library(serializeFilters(next)), { replace: true });

  const matches = useMemo(() => filterProblems(problems.data ?? [], filters), [problems.data, filters]);
  const groups = useMemo(() => groupByTrack(matches, tracks.data ?? []), [matches, tracks.data]);
  // Computed over the whole catalog: a prerequisite hidden by a filter still counts.
  const unmet = useMemo(() => unmetPrerequisites(problems.data ?? []), [problems.data]);
  const narrowed = activeFilterCount({ ...filters, track: null }) > 0;

  // "/" focuses search, as in most catalogs.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (event.key !== "/" || target?.closest("input, textarea, [contenteditable='true']")) return;
      event.preventDefault();
      searchRef.current?.focus();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // Coming back from a problem returns to the same place in the list...
  useScrollMemory(viewport, "library", Boolean(problems.data && tracks.data));
  // ...while choosing another track starts from the top of it.
  const shownTrack = useRef(filters.track);
  useEffect(() => {
    if (shownTrack.current === filters.track) return;
    shownTrack.current = filters.track;
    viewport.current?.querySelector("[data-slot=scroll-area-viewport]")?.scrollTo({ top: 0 });
  }, [filters.track]);

  const error = tracks.error ?? problems.error;
  const total = problems.data?.length ?? 0;

  return (
    <div className="flex h-full">
      <aside className="w-60 shrink-0 border-r bg-background">
        <ScrollArea className="h-full">
          <div className="p-3">
            {tracks.data ? (
              <TrackList
                tracks={tracks.data}
                selected={filters.track}
                onSelect={(track) => setFilters({ ...filters, track })}
              />
            ) : (
              <LoadingLines lines={8} className="px-2" />
            )}
          </div>
        </ScrollArea>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col bg-paper">
        <div className="border-b px-gutter pt-3 pb-2">
          <LibraryFilterBar
            filters={filters}
            onChange={setFilters}
            searchRef={searchRef}
            summary={
              !problems.data
                ? "Loading the catalog"
                : matches.length === total
                  ? pluralize(total, "problem")
                  : `${matches.length} of ${pluralize(total, "problem")}`
            }
          />
        </div>

        <ScrollArea className="min-h-0 flex-1" ref={viewport}>
          <div className="px-gutter pb-16">
            {error ? (
              <ErrorState
                title="The catalog could not be loaded"
                error={error}
                onRetry={() => {
                  void tracks.refetch();
                  void problems.refetch();
                }}
              />
            ) : !problems.data || !tracks.data ? (
              <LoadingLines lines={10} className="pt-6" />
            ) : groups.length === 0 ? (
              <EmptyState
                title="No problems match these filters"
                className="pt-10"
                action={
                  <Button variant="outline" size="sm" onClick={() => setFilters(EMPTY_FILTERS)}>
                    Clear all filters
                  </Button>
                }
              >
                Loosen a filter, or search for a different title or tag.
              </EmptyState>
            ) : (
              groups.map((group) => (
                <TrackSection
                  key={group.track.id}
                  track={group.track}
                  problems={group.problems}
                  filtered={narrowed}
                  unmet={unmet}
                />
              ))
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}

export const Route = createLazyRoute("/library")({ component: LibraryScreen });
