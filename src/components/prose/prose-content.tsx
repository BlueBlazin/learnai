import { cn } from "cn";
import { type ComponentProps, isValidElement, memo, type ReactNode } from "react";
import Markdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Button } from "@/components/ui/button";
import { openExternal } from "@/lib/platform";
import { CodeBlockContent } from "./code-block-content";
import { remarkCallouts } from "./remark-callouts";

const CALLOUT_LABEL: Record<string, string> = { note: "Note", tip: "Tip", warning: "Warning" };

const remarkPlugins = [remarkGfm, remarkMath, remarkCallouts];
const rehypePlugins = [[rehypeKatex, { throwOnError: false, strict: false }]] as ComponentProps<
  typeof Markdown
>["rehypePlugins"];

function Pre({ children }: { children?: ReactNode }) {
  if (isValidElement<{ className?: string; children?: ReactNode }>(children)) {
    const language = /language-([\w-]+)/.exec(children.props.className ?? "")?.[1];
    return <CodeBlockContent code={String(children.props.children ?? "")} language={language} />;
  }
  return <pre>{children}</pre>;
}

const components: Components = {
  pre: Pre,
  table: ({ node: _node, ...props }) => (
    <div className="table-scroll">
      <table {...props} />
    </div>
  ),
  a: ({ node: _node, href, children }) => (
    <Button
      variant="link"
      className="type-inherit h-auto rounded-sm p-0 whitespace-normal"
      onClick={() => href && openExternal(href)}
    >
      {children}
    </Button>
  ),
  aside: ({ node: _node, children, ...props }) => {
    const kind = (props as { "data-kind"?: string })["data-kind"] ?? "note";
    return (
      <aside {...props}>
        <p className="callout-label">{CALLOUT_LABEL[kind] ?? "Note"}</p>
        {children}
      </aside>
    );
  },
  img: ({ node: _node, alt, title, ...props }) => (
    <figure>
      <img alt={alt ?? ""} {...props} />
      {title ? <figcaption>{title}</figcaption> : null}
    </figure>
  ),
};

export interface ProseProps {
  children: string;
  /** `reading` is the statement size; `compact` is for hints and panels. */
  size?: "reading" | "compact";
  className?: string;
}

/** The Markdown renderer behind `Prose`. Import `Prose` instead; this module is heavy. */
export const ProseContent = memo(function ProseContent({
  children,
  size = "reading",
  className,
}: ProseProps) {
  return (
    <div className={cn("prose selectable", className)} data-size={size}>
      <Markdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components}>
        {children}
      </Markdown>
    </div>
  );
});
