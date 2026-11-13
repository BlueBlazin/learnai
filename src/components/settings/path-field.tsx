import { Check, Copy } from "lucide-react";
import { useState } from "react";
import { IconButton } from "@/components/app/icon-button";

/** A filesystem path shown in full, selectable, with a copy button. */
export function PathField({ path, label }: { path: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard.writeText(path).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div className="flex items-center gap-2 rounded-md bg-code py-1 pr-1 pl-3">
      <span className="sr-only">{label}:</span>
      <code className="selectable min-w-0 flex-1 truncate font-mono text-xs" title={path}>
        {path}
      </code>
      <IconButton label={copied ? "Copied" : "Copy path"} size="icon-xs" onClick={copy}>
        {copied ? <Check /> : <Copy />}
      </IconButton>
    </div>
  );
}
