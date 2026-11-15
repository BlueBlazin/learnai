import { CopyButton } from "@/components/app/copy-button";

/**
 * What to do about a failure: a command or a short instruction, set as a
 * listing so it can be selected or copied whole and pasted into a terminal.
 */
export function FixText({ children }: { children: string }) {
  return (
    <div className="flex items-start gap-2 rounded-md bg-listing py-1 pr-1 pl-3">
      <code className="selectable min-w-0 flex-1 py-1 font-mono text-xs break-words whitespace-pre-wrap">
        {children}
      </code>
      <CopyButton text={children} label="Copy" />
    </div>
  );
}
