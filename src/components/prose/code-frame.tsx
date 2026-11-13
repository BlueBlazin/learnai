import { cn } from "cn";
import type { ReactNode } from "react";

/**
 * The tinted block a listing sits in. Long lines scroll sideways inside it and
 * are never clipped or wrapped. Shared by the highlighted listing and its
 * unhighlighted fallback so nothing moves when the colors arrive.
 */
export function CodeFrame({
  children,
  lineNumbers = false,
  actions,
  className,
}: {
  /** The lines of the listing. */
  children: ReactNode;
  lineNumbers?: boolean;
  /** Controls that appear over the top right corner on hover or focus. */
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <figure
      className={cn("code-block group/code relative rounded-md bg-listing", className)}
      data-line-numbers={lineNumbers || undefined}
    >
      <pre
        // biome-ignore lint/a11y/noNoninteractiveTabindex: scrollable regions must be keyboard reachable.
        tabIndex={0}
        className="selectable overflow-x-auto px-4 py-3 font-mono text-code font-normal text-foreground not-italic"
      >
        <code className="block w-fit min-w-full">{children}</code>
      </pre>
      {actions ? (
        <div className="absolute top-1.5 right-1.5 rounded-md bg-listing opacity-0 transition-opacity group-hover/code:opacity-100 focus-within:opacity-100">
          {actions}
        </div>
      ) : null}
    </figure>
  );
}
