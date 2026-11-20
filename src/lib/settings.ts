import type { CatalogIssue, EnvStatus, ModelAsset, TargetInfo } from "@/lib/api";
import { type AssetDownload, isInFlight } from "@/lib/api/asset-fetch-state";
import { environmentProblem } from "@/lib/environment";
import type { SettingsSection } from "@/lib/routes";

/** What the Settings screen knows when it decides which categories to mark. Absent: still loading. */
export interface SettingsFacts {
  env?: EnvStatus;
  /** True when an install started from Settings in this session (a framework, or the language server) failed. */
  installFailed?: boolean;
  assets?: ModelAsset[];
  /** Downloads of this session, by asset name. */
  downloads?: Record<string, AssetDownload>;
  targets?: TargetInfo[];
  issues?: CatalogIssue[];
}

/**
 * The categories of Settings that need the user: an environment that cannot run
 * a solution or an install that failed, a model whose download failed or is
 * incomplete, a run target whose last check failed, catalog entries that could
 * not be loaded. What is unknown (still loading) is not marked, and neither is
 * anything optional: a framework or a model that was simply never installed.
 */
export function settingsAttention({
  env,
  installFailed = false,
  assets = [],
  downloads = {},
  targets = [],
  issues = [],
}: SettingsFacts): ReadonlySet<SettingsSection> {
  const marked = new Set<SettingsSection>();
  if (installFailed || environmentProblem(env, null) !== null) marked.add("environment");
  const stalled = (asset: ModelAsset) => {
    const download = downloads[asset.name];
    if (download?.phase === "failed") return true;
    // A download under way, or just finished, is being dealt with.
    return asset.state === "incomplete" && !isInFlight(download) && download?.phase !== "finished";
  };
  if (assets.some(stalled)) marked.add("models");
  if (targets.some((target) => target.lastCheck?.ok === false)) marked.add("targets");
  if (issues.length > 0) marked.add("files");
  return marked;
}
