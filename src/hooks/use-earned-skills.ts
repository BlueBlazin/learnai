import { useCallback, useEffect, useReducer, useRef } from "react";
import { useGo } from "@/hooks/use-go";
import { backend, type Skill } from "@/lib/api";
import type { RunState } from "@/lib/api/run-state";
import { earnedReducer, earningSubmission } from "@/lib/earned-skills";
import { routes } from "@/lib/routes";

/**
 * The skills a workspace's accepted submits earned, one popup at a time. After
 * an accepted, stored, non-review submit ends it asks the backend once which
 * skills that submission earned and queues them in the order returned. The
 * result is the props of `SkillEarned`: the skill to show, closing it shows the
 * next, and viewing it goes to Accomplishments and drops the rest.
 */
export function useEarnedSkills(run: RunState): {
  skill: Skill | null;
  onClose: () => void;
  onView: (skill: Skill) => void;
} {
  const go = useGo();
  const [queue, dispatch] = useReducer(earnedReducer, []);
  const submissionId = earningSubmission(run);
  /** The submission the backend was already asked about: once per submit. */
  const asked = useRef<number | null>(null);
  useEffect(() => {
    if (submissionId === null || asked.current === submissionId) return;
    asked.current = submissionId;
    backend.skillsEarnedBy(submissionId).then(
      (skills) => dispatch({ type: "earned", skills }),
      // The popup is an extra: without an answer the results stand as they are.
      () => {},
    );
  }, [submissionId]);

  const onClose = useCallback(() => dispatch({ type: "next" }), []);
  const onView = useCallback(
    (skill: Skill) => {
      dispatch({ type: "clear" });
      go(routes.accomplishments(skill.id));
    },
    [go],
  );
  return { skill: queue[0] ?? null, onClose, onView };
}
