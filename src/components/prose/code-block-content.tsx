import { cn } from "cn";
import { memo, useMemo } from "react";
import { CopyButton } from "@/components/app/copy-button";
import { highlight } from "@/lib/highlight";
import { CodeFrame } from "./code-frame";

export interface CodeBlockProps {
  code: string;
  language?: string;
  lineNumbers?: boolean;
  className?: string;
}

/** The highlighted listing behind `CodeBlock`. Import `CodeBlock` instead; this module is heavy. */
export const CodeBlockContent = memo(function CodeBlockContent({
  code,
  language,
  lineNumbers = false,
  className,
}: CodeBlockProps) {
  const source = code.replace(/\n$/, "");
  const lines = useMemo(() => highlight(source, language), [source, language]);

  return (
    <CodeFrame
      className={className}
      lineNumbers={lineNumbers}
      actions={<CopyButton text={source} label="Copy code" />}
    >
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
    </CodeFrame>
  );
});
