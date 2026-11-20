import { CopyButton } from "@/components/app/copy-button";
import { SettingRow } from "./setting-row";

/** A place on disk, as a row: what it is, the path in full under it, and a copy button. */
export function PathRow({ label, path }: { label: string; path: string }) {
  return (
    <SettingRow
      label={label}
      help={
        <code className="block truncate font-mono" title={path}>
          {path}
        </code>
      }
    >
      <CopyButton text={path} label="Copy path" />
    </SettingRow>
  );
}
