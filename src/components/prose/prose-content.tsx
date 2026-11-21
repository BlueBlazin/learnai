import { cn } from "cn";
import { type ComponentProps, isValidElement, memo, type ReactNode, useContext } from "react";
import Markdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { InlineButton } from "@/components/app/inline-button";
import { Notice } from "@/components/app/notice";
import { InteractiveFigure } from "@/components/figures/interactive-figure";
import { openExternal } from "@/lib/platform";
import { AssetFigureContext } from "./asset-figure-context";
import { CodeBlockContent } from "./code-block-content";
import { Figure } from "./figure";
import { ASSET_PREFIX, interactiveName, remarkProse } from "./remark-prose";

const CALLOUT_LABEL: Record<string, string> = { note: "Note", tip: "Tip", warning: "Warning" };

const remarkPlugins = [remarkGfm, remarkMath, remarkProse];
const captionRemarkPlugins = [remarkMath];
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

/** A figure caption: one line of Markdown (math, emphasis, code) with no paragraph around it. */
const captionComponents: Components = {
  p: ({ children }) => <>{children}</>,
  a: ({ children }) => <>{children}</>,
  img: () => null,
};

function Caption({ children }: { children: string }) {
  return (
    <Markdown
      remarkPlugins={captionRemarkPlugins}
      rehypePlugins={rehypePlugins}
      components={captionComponents}
    >
      {children}
    </Markdown>
  );
}

/**
 * A paragraph that holds only an image (see remark-prose.ts). A file under
 * `assets/` is drawn by whoever supplies the text's assets (`ProblemFigures`);
 * `interactive:<name>` is a figure built into the app; any other URL is an
 * ordinary image.
 */
function ProseFigure({
  src,
  number,
  alt,
  caption,
}: {
  src: string;
  number: number;
  alt: string;
  caption: string;
}) {
  const AssetFigure = useContext(AssetFigureContext);
  const captionNode = caption ? <Caption>{caption}</Caption> : null;
  const interactive = interactiveName(src);
  if (interactive !== null) {
    return <InteractiveFigure name={interactive} number={number} alt={alt} caption={captionNode} />;
  }
  if (!src.startsWith(ASSET_PREFIX)) {
    return (
      <Figure number={number} caption={captionNode}>
        <img src={src} alt={alt} />
      </Figure>
    );
  }
  if (!AssetFigure) {
    return (
      <Figure number={number} caption={captionNode}>
        <Notice>This figure belongs to a problem and cannot be shown here.</Notice>
      </Figure>
    );
  }
  return (
    <AssetFigure path={src.slice(ASSET_PREFIX.length)} number={number} alt={alt} caption={captionNode} />
  );
}

const components: Components = {
  pre: Pre,
  table: ({ node: _node, ...props }) => (
    <div className="table-scroll">
      <table {...props} />
    </div>
  ),
  a: ({ node: _node, href, children }) => (
    <InlineButton onClick={() => href && openExternal(href)}>{children}</InlineButton>
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
  figure: ({ node: _node, ...props }) => {
    const data = props as Record<`data-${"src" | "number" | "alt" | "caption"}`, string | number | undefined>;
    return (
      <ProseFigure
        src={String(data["data-src"] ?? "")}
        number={Number(data["data-number"] ?? 1)}
        alt={String(data["data-alt"] ?? "")}
        caption={String(data["data-caption"] ?? "")}
      />
    );
  },
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
