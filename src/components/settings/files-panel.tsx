import type { CatalogIssue } from "@/lib/api";
import { pluralize } from "@/lib/format";
import { CatalogIssueList } from "./catalog-issue-list";
import { PathRow } from "./path-row";
import { RowStatus } from "./row-status";
import { SettingList, SettingRow } from "./setting-row";

/**
 * Where everything lives on disk: the content root, that solutions are plain
 * files, and whether the catalog loaded whole. Entries it skipped are listed
 * behind the Catalog row.
 */
export function FilesPanel({
  root,
  issues,
  problemCount,
}: {
  root: string;
  issues: CatalogIssue[];
  /** How many problems the catalog loaded, when that is known. */
  problemCount?: number;
}) {
  const loaded = problemCount === undefined ? null : pluralize(problemCount, "problem");
  const skipped = `${pluralize(issues.length, "entry", "entries")} could not be loaded and ${issues.length === 1 ? "is" : "are"} hidden.`;
  return (
    <SettingList aria-label="Files">
      <PathRow label="Content root" path={root} />
      <SettingRow
        label="Solutions"
        help={
          <>
            Plain files under <code className="font-mono">workspace/</code>, editable in any editor.
          </>
        }
      />
      {issues.length === 0 ? (
        <SettingRow
          label="Catalog"
          help={loaded ? `${loaded} loaded without issues.` : "Loaded without issues."}
        />
      ) : (
        <SettingRow
          label="Catalog"
          help={loaded ? `${loaded} loaded. ${skipped}` : skipped}
          status={<RowStatus tone="warn">{pluralize(issues.length, "issue")}</RowStatus>}
          details={<CatalogIssueList issues={issues} root={root} />}
        />
      )}
    </SettingList>
  );
}
