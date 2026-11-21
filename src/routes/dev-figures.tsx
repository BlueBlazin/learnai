import { createLazyRoute, getRouteApi } from "@tanstack/react-router";
import { useMemo } from "react";
import { Page, PageHeader, ScreenMessage, Section } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { GalleryProblem } from "@/components/figures/gallery-problem";
import { FilterSelect } from "@/components/library/filter-select";
import { useGo } from "@/hooks/use-go";
import { useProblems, useTracks } from "@/lib/api/queries";
import { groupGallery, useStatementFigures } from "@/lib/figure-gallery";
import { pluralize } from "@/lib/format";
import { routes } from "@/lib/routes";

const route = getRouteApi("/dev/figures");

/**
 * Development only: the figures of every statement, for review. Each is drawn
 * by the same components a statement uses.
 */
function DevFiguresScreen() {
  const { track } = route.useSearch();
  const go = useGo();
  const statements = useStatementFigures();
  const problems = useProblems();
  const tracks = useTracks();

  const groups = useMemo(
    () => groupGallery(statements.data ?? [], problems.data ?? [], tracks.data ?? []),
    [statements.data, problems.data, tracks.data],
  );
  const shown = track ? groups.filter((group) => group.id === track) : groups;
  const count = shown.reduce((sum, group) => sum + group.problems.length, 0);

  if (statements.error) {
    return (
      <ScreenMessage>
        <ErrorState title="The statements could not be read" error={statements.error} />
      </ScreenMessage>
    );
  }

  return (
    <Page>
      <PageHeader
        title="Figures"
        description="The figures of every statement, drawn the way a statement draws them. For reviewing; not part of the app."
        actions={
          <FilterSelect
            label="Module"
            anyLabel="All modules"
            value={track ?? null}
            options={groups.map((group) => ({ value: group.id, label: group.title }))}
            onChange={(next) => go(routes.devFigures(next ? { track: next } : {}), { replace: true })}
          />
        }
      />
      {!statements.data ? (
        <LoadingLines lines={6} className="max-w-measure" />
      ) : count === 0 ? (
        <EmptyState title="No figures">No problem in this view has a figure in its statement.</EmptyState>
      ) : (
        shown.map((group) => (
          <Section key={group.id} title={group.title} aside={pluralize(group.problems.length, "problem")}>
            {group.problems.map((problem) => (
              <GalleryProblem key={problem.id} problem={problem} />
            ))}
          </Section>
        ))
      )}
    </Page>
  );
}

export const Route = createLazyRoute("/dev/figures")({ component: DevFiguresScreen });
