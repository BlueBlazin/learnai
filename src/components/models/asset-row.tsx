import { AppLink } from "@/components/app/app-link";
import { CopyButton } from "@/components/app/copy-button";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import { Notice } from "@/components/app/notice";
import { RowStatus } from "@/components/settings/row-status";
import { SettingRow } from "@/components/settings/setting-row";
import type { ModelAsset } from "@/lib/api";
import { type AssetDownload, isInFlight } from "@/lib/api/asset-fetch-state";
import { formatBytes, formatSizeMb, pluralize } from "@/lib/format";
import { routes } from "@/lib/routes";
import { AssetDownloadButton, RemoveAssetButton } from "./asset-actions";
import { AssetProgress } from "./asset-progress";
import { AssetStateMark } from "./asset-state-mark";

/** What is on disk against what a download brings: "712 MB", "About 712 MB", "214 MB of about 712 MB". */
function sizeText(asset: ModelAsset): string {
  if (asset.state === "present") return formatBytes(asset.bytesOnDisk);
  const declared = formatSizeMb(asset.sizeMb);
  return asset.bytesOnDisk > 0
    ? `${formatBytes(asset.bytesOnDisk)} of about ${declared}`
    : `About ${declared}`;
}

/**
 * One model in Settings, on one line: its name, its size and how many problems
 * use it, whether it is on disk, and Download, Cancel, or Remove. Progress
 * runs under the line while it downloads; a download that failed or stopped
 * short says why there. Behind the name: where it comes from, where it is on
 * disk, and the problems that use it.
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
  const failed = download?.phase === "failed";
  const users = asset.problems.length;
  return (
    <SettingRow
      label={asset.name}
      help={users > 0 ? `${sizeText(asset)}, used by ${pluralize(users, "problem")}` : sizeText(asset)}
      status={
        inFlight ? (
          <RowStatus tone="busy">{canceling ? "Canceling" : "Downloading"}</RowStatus>
        ) : (
          <AssetStateMark state={state} />
        )
      }
      body={
        inFlight ? (
          <AssetProgress name={asset.name} download={download} canceling={canceling} />
        ) : failed ? (
          <Notice tone="fail">{download.message}</Notice>
        ) : state === "incomplete" ? (
          <p className="selectable text-xs text-muted-foreground">
            {asset.reason
              ? `The download did not finish (${asset.reason}). Resuming keeps the files that are complete.`
              : "The download did not finish. Resuming keeps the files that are complete."}
          </p>
        ) : null
      }
      details={
        <MetaList layout="rows">
          <MetaItem label="From">
            <span className="selectable font-mono">{asset.repo}</span>
          </MetaItem>
          <MetaItem label="On disk">
            <span className="flex items-center gap-1">
              <code className="selectable min-w-0 truncate font-mono" title={asset.path}>
                {asset.path}
              </code>
              <CopyButton text={asset.path} label="Copy path" />
            </span>
          </MetaItem>
          {users > 0 ? (
            <MetaItem label="Used by">
              <ul className="flex flex-col gap-1">
                {asset.problems.map((id) => (
                  <li key={id}>
                    <AppLink to={routes.problem(id)}>{titles?.get(id) ?? id}</AppLink>
                  </li>
                ))}
              </ul>
            </MetaItem>
          ) : null}
        </MetaList>
      }
    >
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
    </SettingRow>
  );
}
