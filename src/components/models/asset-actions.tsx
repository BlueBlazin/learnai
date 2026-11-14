import { Download } from "lucide-react";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { Button } from "@/components/ui/button";
import type { AssetState } from "@/lib/api";
import { type AssetDownload, isInFlight } from "@/lib/api/asset-fetch-state";

/**
 * The one button that moves a model toward being on disk: Download, Resume
 * download (some of it is there), Try again (the last attempt failed), or
 * Cancel while it is running. Nothing downloads until this is clicked. Renders
 * nothing for a model that is already downloaded.
 */
export function AssetDownloadButton({
  state,
  download,
  busy,
  canceling = false,
  onDownload,
  onCancel,
  size = "xs",
}: {
  /** Undefined while the model's state is not known; treated as not downloaded. */
  state: AssetState | undefined;
  download?: AssetDownload;
  /** True while any download is running: the core runs one at a time. */
  busy: boolean;
  canceling?: boolean;
  onDownload: () => void;
  onCancel: () => void;
  size?: "xs" | "sm";
}) {
  if (isInFlight(download)) {
    return (
      <Button variant="outline" size={size} onClick={onCancel} disabled={canceling}>
        {canceling ? "Canceling" : "Cancel"}
      </Button>
    );
  }
  if (state === "present" || download?.phase === "finished") return null;
  const label =
    download?.phase === "failed" ? "Try again" : state === "incomplete" ? "Resume download" : "Download";
  return (
    <Button
      variant="outline"
      size={size}
      onClick={onDownload}
      disabled={busy}
      title={busy ? "Another download is running" : undefined}
    >
      <Download data-icon="inline-start" />
      {label}
    </Button>
  );
}

/** Deletes a downloaded model, after a confirmation that states what is deleted and that it can come back. */
export function RemoveAssetButton({
  name,
  size,
  disabled = false,
  onRemove,
}: {
  name: string;
  /** What will be freed, already formatted: "712 MB". */
  size: string;
  disabled?: boolean;
  onRemove: () => void;
}) {
  return (
    <ConfirmDialog
      trigger={
        <Button variant="ghost" size="xs" className="text-muted-foreground" disabled={disabled}>
          Remove
        </Button>
      }
      title={`Remove ${name}?`}
      description={`This deletes ${size} from this computer. Cases that need the model are skipped until it is downloaded again, which you can do at any time.`}
      confirmLabel="Remove model"
      destructive
      onConfirm={onRemove}
    />
  );
}
