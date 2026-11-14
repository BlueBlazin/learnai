import { RuledList } from "@/components/app/ruled-list";
import { PathField } from "@/components/settings/path-field";
import type { AssetList } from "@/lib/api";
import type { AssetFetchController } from "@/lib/api/use-asset-fetch";
import { AssetRow } from "./asset-row";

/**
 * The models that real-model problems run on: every one the catalog declares,
 * whether it is on disk, and the directory they live in. Nothing is downloaded
 * until a Download button is clicked.
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
    <div className="flex flex-col gap-4">
      {listing.assets.length === 0 ? (
        <p className="max-w-measure text-sm text-muted-foreground">
          No problem in this catalog needs a model. When one does, it is listed here and downloads only when
          you ask.
        </p>
      ) : (
        <RuledList aria-label="Models">
          {listing.assets.map((asset) => (
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
          ))}
        </RuledList>
      )}
      <div className="flex max-w-measure flex-col gap-2">
        <PathField path={listing.modelsDir} label="Models directory" />
        <p className="text-xs text-muted-foreground">
          Models are stored here, outside version control. Each is pinned to one revision and verified after
          download; an interrupted download resumes where it stopped.
        </p>
      </div>
    </div>
  );
}
