import type { CatalogIssue } from "@/lib/api";

/**
 * The problems and tracks the catalog skipped because they are malformed or
 * still being written: each file, relative to the content root, and why.
 */
export function CatalogIssueList({ issues, root }: { issues: CatalogIssue[]; root?: string }) {
  const prefix = root ? `${root.replace(/\/+$/, "")}/` : null;
  const relative = (path: string) => (prefix && path.startsWith(prefix) ? path.slice(prefix.length) : path);
  return (
    <ul className="flex flex-col gap-2 text-xs" aria-label="Catalog entries that could not be loaded">
      {issues.map((issue) => (
        <li key={`${issue.path}\n${issue.message}`} className="flex flex-col gap-0.5">
          <code className="selectable font-mono break-all text-foreground">{relative(issue.path)}</code>
          <span className="selectable text-muted-foreground">{issue.message}</span>
        </li>
      ))}
    </ul>
  );
}
