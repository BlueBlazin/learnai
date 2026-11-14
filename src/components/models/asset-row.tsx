import { AppLink } from "@/components/app/app-link";
import { CopyButton } from "@/components/app/copy-button";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import { Notice } from "@/components/app/notice";
import type { ModelAsset } from "@/lib/api";
import { type AssetDownload, isInFlight } from "@/lib/api/asset-fetch-state";
import { formatBytes, formatSizeMb } from "@/lib/format";
import { routes } from "@/lib/routes";
import { AssetDownloadButton, RemoveAssetButton } from "./asset-actions";
import { AssetProgress } from "./asset-progress";
import { AssetStateMark } from "./asset-state-mark";

/** What is on disk against what a download brings: "712 MB", "about 712 MB", "214 MB of about 712 MB". */
function sizeText(asset: ModelAsset): string {
  if (asset.state === "present") return formatBytes(asset.bytesOnDisk);
  const declared = `about ${formatSizeMb(asset.sizeMb)}`;
  return asset.bytesOnDisk > 0 ? `${formatBytes(asset.bytesOnDisk)} of ${declared}` : declared;
}

/**
 * One model in Settings: its name and state, where it comes from, how large it
 * is, which problems use it, and where it is on disk; with Download, Cancel
 * (and live progress), or Remove. A failed download says why in one line and
 * offers another try.
 */
export function AssetRow({
  asset,
  download,
  busy,
  canceling = false,
  removing = false,
  titles,
  onDownload,
  onCancel,
  onRemove,
}: {
  asset: ModelAsset;
  /** This asset's download, when one is running or has just ended. */
  download?: AssetDownload;
  /** True while any download is running. */
  busy: boolean;
  canceling?: boolean;
  /** True while this asset is being removed. */
  removing?: boolean;
  /** Problem titles by id, for the "Used by" links; ids are shown where a title is unknown. */
  titles?: Map<string, string>;
  onDownload: () => void;
  onCancel: () => void;
  onRemove: () => void;
}) {
  const inFlight = isInFlight(download);
  // Between the judge reporting an asset complete and the listing being read again.
  const state = download?.phase === "finished" ? "present" : asset.state;
  const onDisk = state !== "missing" || asset.bytesOnDisk > 0;
  return (
    <li className="flex flex-col gap-2.5 py-4">
      <div className="flex items-start justify-between gap-6">
        <div className="flex min-w-0 flex-col gap-1.5">
          <p className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="selectable text-sm font-medium">{asset.name}</span>
            {inFlight ? null : <AssetStateMark state={state} />}
          </p>
          <MetaList>
            <MetaItem label="From">
              <span className="selectable font-mono">{asset.repo}</span>
            </MetaItem>
            <MetaItem label="Size">
              <span className="selectable tabular">{sizeText(asset)}</span>
            </MetaItem>
            {asset.problems.length > 0 ? (
              <MetaItem label="Used by">
                <ul className="flex flex-wrap gap-x-4 gap-y-1">
                  {asset.problems.map((id) => (
                    <li key={id}>
                      <AppLink to={routes.problem(id)}>{titles?.get(id) ?? id}</AppLink>
                    </li>
                  ))}
                </ul>
              </MetaItem>
            ) : null}
          </MetaList>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <AssetDownloadButton
            state={state}
            download={download}
            busy={busy || removing}
            canceling={canceling}
            onDownload={onDownload}
            onCancel={onCancel}
          />
          {state === "present" && !inFlight ? (
            <RemoveAssetButton
              name={asset.name}
              size={formatBytes(asset.bytesOnDisk)}
              disabled={busy || removing}
              onRemove={onRemove}
            />
          ) : null}
        </div>
      </div>

      {isInFlight(download) ? (
        <AssetProgress name={asset.name} download={download} canceling={canceling} />
      ) : null}
      {download?.phase === "failed" ? <Notice tone="fail">{download.message}</Notice> : null}
      {state === "incomplete" && !inFlight && download?.phase !== "failed" ? (
        <p className="selectable text-xs text-muted-foreground">
          {asset.reason
            ? `The download did not finish (${asset.reason}). Resuming keeps the files that are complete.`
            : "The download did not finish. Resuming keeps the files that are complete."}
        </p>
      ) : null}
      {onDisk && !inFlight ? (
        <p className="flex items-center gap-1 text-xs text-muted-foreground">
          <code className="selectable min-w-0 truncate font-mono" title={asset.path}>
            {asset.path}
          </code>
          <CopyButton text={asset.path} label="Copy path" />
        </p>
      ) : null}
    </li>
  );
}
