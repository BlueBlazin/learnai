import { ChevronRight } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { CatalogIssue } from "@/lib/api";
import { pluralize } from "@/lib/format";

/**
 * Problems and tracks the catalog skipped because they are malformed or still
 * being written. Closed by default: this matters to authors, not to solving.
 * Renders nothing when the catalog is clean.
 */
export function CatalogIssueList({ issues, root }: { issues: CatalogIssue[]; root?: string }) {
  if (issues.length === 0) return null;
  const prefix = root ? `${root.replace(/\/+$/, "")}/` : null;
  const relative = (path: string) => (prefix && path.startsWith(prefix) ? path.slice(prefix.length) : path);
  return (
    <Collapsible>
      <CollapsibleTrigger className="group/issues flex items-center gap-1.5 rounded-sm text-xs text-muted-foreground outline-none hover:text-foreground focus-visible:text-foreground">
        <ChevronRight
          className="size-3 shrink-0 text-faint transition-transform group-data-panel-open/issues:rotate-90"
          aria-hidden
        />
        {pluralize(issues.length, "catalog entry", "catalog entries")} could not be loaded and{" "}
        {issues.length === 1 ? "is" : "are"} hidden
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ul className="flex flex-col gap-2 pt-2 pl-4.5 text-xs">
          {issues.map((issue) => (
            <li key={`${issue.path}\n${issue.message}`} className="flex flex-col gap-0.5">
              <code className="selectable font-mono break-all text-foreground">{relative(issue.path)}</code>
              <span className="selectable text-muted-foreground">{issue.message}</span>
            </li>
          ))}
        </ul>
      </CollapsibleContent>
    </Collapsible>
  );
}
