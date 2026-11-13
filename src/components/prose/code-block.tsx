import { cn } from "cn";
import { Check, Copy } from "lucide-react";
import { memo, useMemo, useState } from "react";
import { IconButton } from "@/components/app/icon-button";
import { highlight } from "@/lib/highlight";

/**
 * A read-only code listing: statement examples, reference solutions, submission
 * snapshots. Highlighted with the same palette as the editor.
 */
export const CodeBlock = memo(function CodeBlock({
  code,
  language,
  lineNumbers = false,
  className,
}: {
  code: string;
  language?: string;
  lineNumbers?: boolean;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const source = code.replace(/\n$/, "");
  const lines = useMemo(() => highlight(source, language), [source, language]);

  const copy = () => {
    void navigator.clipboard.writeText(source).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <figure
      className={cn("code-block group/code relative rounded-md bg-code", className)}
      data-line-numbers={lineNumbers || undefined}
    >
      <pre className="selectable overflow-x-auto px-4 py-3 font-mono text-code font-normal not-italic">
        <code className="block w-fit min-w-full">
          {lines.map((line, index) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: lines of a static listing.
            <span key={index} className="code-line block">
              {line.map((token, i) => (
                <span
                  // biome-ignore lint/suspicious/noArrayIndexKey: tokens of a static listing.
                  key={i}
                  className={cn(token.italic && "italic", token.bold && "font-semibold")}
                  style={token.color ? { color: token.color } : undefined}
                >
                  {token.content}
                </span>
              ))}
              {line.length === 0 || (line.length === 1 && line[0]?.content === "") ? "\n" : null}
            </span>
          ))}
        </code>
      </pre>
      <div className="absolute top-1.5 right-1.5 opacity-0 transition-opacity group-hover/code:opacity-100 focus-within:opacity-100">
        <IconButton label={copied ? "Copied" : "Copy code"} size="icon-xs" onClick={copy}>
          {copied ? <Check /> : <Copy />}
        </IconButton>
      </div>
    </figure>
  );
});
