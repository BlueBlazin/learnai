import type { CatalogIssue } from "@/lib/api";
import { CatalogIssueList } from "./catalog-issue-list";
import { PathField } from "./path-field";

/** Where the content lives on disk, how solutions are stored, and any catalog entries that failed to load. */
export function ContentPanel({ root, issues }: { root: string; issues: CatalogIssue[] }) {
  return (
    <div className="flex max-w-measure flex-col gap-2">
      <PathField path={root} label="Content root" />
      <p className="text-xs text-muted-foreground">
        Solutions are plain files under <code className="selectable font-mono">workspace/</code>. Edit them in
        any editor; Learn AI re-reads the open file when its window regains focus.
      </p>
      <CatalogIssueList issues={issues} root={root} />
    </div>
  );
}
