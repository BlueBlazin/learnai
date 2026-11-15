import { Notice } from "@/components/app/notice";
import type { AssetState } from "@/lib/api";
import { type AssetDownload, isInFlight } from "@/lib/api/asset-fetch-state";
import type { AssetFetchController } from "@/lib/api/use-asset-fetch";
import { formatSizeMb } from "@/lib/format";
import { AssetDownloadButton } from "./asset-actions";
import { AssetProgress, describeDownload } from "./asset-progress";

/** What a notice needs to know about a model: enough to name it, size it, and say where it stands. */
export interface NoticeAsset {
  name: string;
  sizeMb: number;
  /** Undefined while unknown; treated as not downloaded. */
  state?: AssetState;
}

/**
 * One line about a model a problem needs and does not have: which model, how
 * large, that the cases needing it are skipped until then, and the Download
 * action; while downloading, the bytes and percentage in the sentence and the
 * progress rule under it. Once the model is on disk the notice goes away, or,
 * given `whenPresent`, says so (after a run that skipped cases for it).
 */
export function AssetNotice({
  asset,
  download,
  busy,
  canceling = false,
  layout = "strip",
  whenPresent,
  host,
  onDownload,
  onCancel,
}: {
  asset: NoticeAsset;
  download?: AssetDownload;
  /** True while any download is running. */
  busy: boolean;
  canceling?: boolean;
  /** `strip` under the editor toolbar; `inline` as a line in the results. */
  layout?: "strip" | "inline";
  /** What to say once the model is downloaded; without it the notice disappears. */
  whenPresent?: string;
  /** The remote machine the model is needed on; absent for this computer. */
  host?: string;
  onDownload: () => void;
  onCancel: () => void;
}) {
  const name = <span className="font-medium">{asset.name}</span>;
  const present = asset.state === "present" || download?.phase === "finished";
  if (present) {
    return whenPresent ? (
      <Notice tone="pass" layout={layout}>
        {name} is downloaded{host ? ` on ${host}` : null}. {whenPresent}
      </Notice>
    ) : null;
  }

  const action = (
    <AssetDownloadButton
      state={asset.state}
      download={download}
      busy={busy}
      canceling={canceling}
      host={host}
      onDownload={onDownload}
      onCancel={onCancel}
    />
  );
  if (isInFlight(download)) {
    const { text, percent } = describeDownload(download, canceling);
    return (
      <Notice
        tone="info"
        layout={layout}
        action={action}
        footer={<AssetProgress name={asset.name} download={download} canceling={canceling} variant="rule" />}
      >
        Downloading {name}
        {host ? ` on ${host}` : null}
        <span className="tabular text-muted-foreground">
          {" "}
          {text}
          {percent ? `, ${percent}` : null}
        </span>
      </Notice>
    );
  }
  if (download?.phase === "failed") {
    return (
      <Notice tone="fail" layout={layout} action={action}>
        {name} could not be downloaded{host ? ` on ${host}` : null}: {download.message}
      </Notice>
    );
  }
  const size = formatSizeMb(asset.sizeMb);
  return (
    <Notice tone="info" layout={layout} action={action}>
      {whenPresent ? (
        // After a run that skipped cases for it: the run above has already said why.
        <>
          {asset.state === "incomplete" ? "Finish downloading" : "Download"} {name} ({size})
          {host ? ` on ${host}` : null} to run the skipped cases.
        </>
      ) : asset.state === "incomplete" ? (
        <>
          The download of {name} ({size}) did not finish. Cases that need it are skipped until it does.
        </>
      ) : (
        <>
          This problem uses the model {name} ({size}), which is not downloaded. Cases that need it are skipped
          until it is.
        </>
      )}
    </Notice>
  );
}

/** An `AssetNotice` for each of `assets`, wired to the shared download controller. */
export function AssetNoticeList({
  assets,
  fetch,
  layout,
  whenPresent,
  host,
}: {
  assets: NoticeAsset[];
  fetch: AssetFetchController;
  layout?: "strip" | "inline";
  whenPresent?: string;
  /** The remote machine the models are needed on; absent for this computer. */
  host?: string;
}) {
  return assets.map((asset) => (
    <AssetNotice
      key={asset.name}
      asset={asset}
      download={fetch.downloads[asset.name]}
      busy={fetch.busy}
      canceling={fetch.canceling}
      layout={layout}
      whenPresent={whenPresent}
      host={host}
      onDownload={() => fetch.download([asset.name])}
      onCancel={fetch.cancel}
    />
  ));
}
