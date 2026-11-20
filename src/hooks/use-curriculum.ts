import { useMemo } from "react";
import { useAreas, useProblems, useTracks } from "@/lib/api/queries";
import { buildCurriculum, type Curriculum } from "@/lib/curriculum";

/**
 * The curriculum built from the areas, tracks, and problems the backend
 * serves. Undefined until all three have loaded. `error` is the first failure
 * and `retry` asks again for all three.
 */
export function useCurriculum(): { curriculum: Curriculum | undefined; error: unknown; retry: () => void } {
  const areas = useAreas();
  const tracks = useTracks();
  const problems = useProblems();
  const curriculum = useMemo(
    () =>
      areas.data && tracks.data && problems.data
        ? buildCurriculum(areas.data, tracks.data, problems.data)
        : undefined,
    [areas.data, tracks.data, problems.data],
  );
  return {
    curriculum,
    error: areas.error ?? tracks.error ?? problems.error,
    retry: () => {
      for (const query of [areas, tracks, problems]) void query.refetch();
    },
  };
}
