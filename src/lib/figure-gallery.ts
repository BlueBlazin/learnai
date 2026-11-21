import { useQuery } from "@tanstack/react-query";
import type { ProblemSummary, Track } from "@/lib/api";

/**
 * The data behind the dev-only figure gallery (`/dev/figures`).
 *
 * The figure paragraphs come from each statement's Markdown source, read at
 * build time from `problems/`, so the gallery lists every problem with a figure
 * with one request for none; the drawings themselves go through the backend like
 * in a statement. This file is only imported by the dev route.
 */

/** A fenced code block, whose lines are never figures. */
const FENCE = /^(```|~~~)[^\n]*\n[\s\S]*?^\1[^\n]*$/gm;
const FIGURE_LINE = /^!\[.*\]\(\S+[^)]*\)\s*$/;

/** The lines of a Markdown text that are only an image: `![caption](target)`. */
export function figureParagraphs(markdown: string): string[] {
  return markdown
    .replace(FENCE, "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => FIGURE_LINE.test(line));
}

export interface GalleryProblem {
  track: string;
  id: string;
  title: string;
  /** The figure paragraphs of the statement, in order. */
  figures: string[];
}

export interface GalleryGroup {
  id: string;
  title: string;
  problems: GalleryProblem[];
}

/** A statement found under `problems/`, before titles are known. */
export interface StatementFigures {
  track: string;
  id: string;
  figures: string[];
}

/** The problems that have figures, grouped by module in catalog order. */
export function groupGallery(
  found: StatementFigures[],
  summaries: ProblemSummary[],
  tracks: Track[],
): GalleryGroup[] {
  const summary = new Map(summaries.map((problem) => [problem.id, problem]));
  const trackOf = new Map(tracks.map((track) => [track.id, track]));
  const groups = new Map<string, GalleryProblem[]>();
  for (const item of found) {
    if (item.figures.length === 0) continue;
    const list = groups.get(item.track) ?? [];
    list.push({ ...item, title: summary.get(item.id)?.title ?? item.id });
    groups.set(item.track, list);
  }
  const order = (id: string) => trackOf.get(id)?.order ?? Number.MAX_SAFE_INTEGER;
  return [...groups.entries()]
    .sort(([a], [b]) => order(a) - order(b) || a.localeCompare(b))
    .map(([id, problems]) => ({
      id,
      title: trackOf.get(id)?.title ?? id,
      problems: problems.sort(
        (a, b) =>
          (summary.get(a.id)?.order ?? Number.MAX_SAFE_INTEGER) -
            (summary.get(b.id)?.order ?? Number.MAX_SAFE_INTEGER) || a.id.localeCompare(b.id),
      ),
    }));
}

const STATEMENTS = import.meta.glob<string>("../../problems/*/*/statement.md", {
  query: "?raw",
  import: "default",
});

/** Every statement under `problems/` with the figure paragraphs it has. */
export function useStatementFigures() {
  return useQuery({
    queryKey: ["dev", "statement-figures"],
    staleTime: Number.POSITIVE_INFINITY,
    queryFn: async (): Promise<StatementFigures[]> =>
      Promise.all(
        Object.entries(STATEMENTS).map(async ([path, load]) => {
          const [, track = "", id = ""] = /problems\/([^/]+)\/([^/]+)\/statement\.md$/.exec(path) ?? [];
          return { track, id, figures: figureParagraphs(await load()) };
        }),
      ),
  });
}
