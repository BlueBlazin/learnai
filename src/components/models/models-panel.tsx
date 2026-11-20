import { PathRow } from "@/components/settings/path-row";
import { SettingList, SettingRow } from "@/components/settings/setting-row";
import type { AssetList } from "@/lib/api";
import type { AssetFetchController } from "@/lib/api/use-asset-fetch";
import { AssetRow } from "./asset-row";

/**
 * The models that real-model problems run on, a row each, and last the folder
 * they are stored in. Nothing is downloaded until a Download button is clicked.
 */
export function ModelsPanel({
  listing,
  fetch,
  titles,
  removing,
  onRemove,
}: {
  listing: AssetList;
  fetch: AssetFetchController;
  /** Problem titles by id, for the "Used by" links. */
  titles?: Map<string, string>;
  /** The asset being removed, if any. */
  removing?: string | null;
  onRemove: (name: string) => void;
}) {
  return (
    <SettingList aria-label="Models">
      {listing.assets.length === 0 ? (
        <SettingRow label="No models" help="No problem in this catalog needs one." />
      ) : (
        listing.assets.map((asset) => (
          <AssetRow
            key={asset.name}
            asset={asset}
            download={fetch.downloads[asset.name]}
            busy={fetch.busy}
            canceling={fetch.canceling}
            removing={removing === asset.name}
            titles={titles}
            onDownload={() => fetch.download([asset.name])}
            onCancel={fetch.cancel}
            onRemove={() => onRemove(asset.name)}
          />
        ))
      )}
      <PathRow label="Stored in" path={listing.modelsDir} />
    </SettingList>
  );
}
