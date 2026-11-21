import { AppLink } from "@/components/app/app-link";
import { ProblemFigures } from "@/components/prose/problem-figures";
import { Prose } from "@/components/prose/prose";
import type { GalleryProblem as GalleryEntry } from "@/lib/figure-gallery";
import { routes } from "@/lib/routes";

/**
 * One problem in the figure gallery: its title and id, then the figures of its
 * statement drawn the way the statement draws them (same renderer, so numbering,
 * captions, steps, interactive figures, and the text cap all apply).
 */
export function GalleryProblem({ problem }: { problem: GalleryEntry }) {
  return (
    <article className="pb-8">
      <h3 className="flex items-baseline gap-3 pb-2 text-md font-medium">
        <AppLink to={routes.problem(problem.id)} tone="quiet">
          {problem.title}
        </AppLink>
        <code className="selectable font-mono text-xs font-normal text-muted-foreground">{problem.id}</code>
      </h3>
      <ProblemFigures problemId={problem.id}>
        <Prose>{problem.figures.join("\n\n")}</Prose>
      </ProblemFigures>
    </article>
  );
}
