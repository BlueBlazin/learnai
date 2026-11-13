import { Check, Copy } from "lucide-react";
import { useEffect, useState } from "react";
import { IconButton } from "./icon-button";

/** Copies `text` to the clipboard and confirms with a check for a moment. */
export function CopyButton({ text, label }: { text: string /** For example "Copy path". */; label: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timer);
  }, [copied]);
  return (
    <IconButton
      label={copied ? "Copied" : label}
      size="icon-xs"
      onClick={() => void navigator.clipboard.writeText(text).then(() => setCopied(true))}
    >
      {copied ? <Check /> : <Copy />}
    </IconButton>
  );
}
