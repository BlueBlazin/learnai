import { useMemo } from "react";
import { useProblems, useSkills } from "@/lib/api/queries";
import { type SkillState, skillStates } from "@/lib/skills";

/**
 * Every skill's state by id, built from the skills and problems the backend
 * serves. Undefined until both have loaded. `error` is the first failure and
 * `retry` asks again for both.
 */
export function useSkillStates(): {
  states: Map<string, SkillState> | undefined;
  error: unknown;
  retry: () => void;
} {
  const skills = useSkills();
  const problems = useProblems();
  const states = useMemo(
    () => (skills.data && problems.data ? skillStates(skills.data, problems.data) : undefined),
    [skills.data, problems.data],
  );
  return {
    states,
    error: skills.error ?? problems.error,
    retry: () => {
      for (const query of [skills, problems]) void query.refetch();
    },
  };
}
