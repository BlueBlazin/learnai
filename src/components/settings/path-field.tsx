import { CopyButton } from "@/components/app/copy-button";

/** A filesystem path shown in full, selectable, with a copy button. */
export function PathField({ path, label }: { path: string; label: string }) {
  return (
    <div className="flex items-center gap-2 rounded-md bg-listing py-1 pr-1 pl-3">
      <span className="sr-only">{label}:</span>
      <code className="selectable min-w-0 flex-1 truncate font-mono text-xs" title={path}>
        {path}
      </code>
      <CopyButton text={path} label="Copy path" />
    </div>
  );
}
