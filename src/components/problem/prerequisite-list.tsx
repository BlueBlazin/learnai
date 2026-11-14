import { AppLink } from "@/components/app/app-link";
import type { ProblemSummary } from "@/lib/api";
import { routes } from "@/lib/routes";
import { StatusMark } from "./status-mark";

/**
 * The problems this one builds on, each with its status mark and a link to it.
 * Prerequisites are advice, never a gate: the list reports where the learner
 * stands and leaves the choice to them.
 */
export function PrerequisiteList({ prerequisites }: { prerequisites: ProblemSummary[] }) {
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-1">
      {prerequisites.map((prerequisite) => (
        <li key={prerequisite.id} className="flex items-center gap-1.5">
          <StatusMark status={prerequisite.status} />
          <AppLink to={routes.problem(prerequisite.id)}>{prerequisite.title}</AppLink>
        </li>
      ))}
    </ul>
  );
}
