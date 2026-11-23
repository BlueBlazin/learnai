import { cn } from "cn";
import type { ReactNode } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";

/**
 * A scrolling screen with one centered column: Home and Settings. `wide` gives
 * the column the room a wide figure needs (Accomplishments and its tree).
 */
export function Page({
  children,
  wide = false,
  className,
}: {
  children: ReactNode;
  wide?: boolean;
  className?: string;
}) {
  return (
    <ScrollArea className="h-full bg-paper">
      <div
        className={cn("mx-auto px-gutter pt-12 pb-20", wide ? "max-w-page-wide" : "max-w-page", className)}
      >
        {children}
      </div>
    </ScrollArea>
  );
}

/** A screen that has only a state to show: loading, failed, or nothing there. */
export function ScreenMessage({ children }: { children: ReactNode }) {
  return <div className="h-full bg-paper px-gutter pt-10">{children}</div>;
}

/** The title block at the top of a page: a serif title and an optional sentence under it. */
export function PageHeader({
  title,
  description,
  actions,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="flex items-end justify-between gap-6 pb-8">
      <div className="min-w-0">
        <h1 className="selectable font-serif text-2xl font-medium tracking-tight text-balance">{title}</h1>
        {description ? (
          <p className="selectable mt-2 max-w-measure font-serif text-md text-muted-foreground">
            {description}
          </p>
        ) : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </header>
  );
}

/** A titled section separated from the one above by a hairline, not a box. */
export function Section({
  title,
  aside,
  children,
  className,
}: {
  title: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("border-t border-rule pt-3 pb-10", className)}>
      <div className="flex items-baseline justify-between gap-4 pb-4">
        <h2 className="text-sm font-semibold">{title}</h2>
        {aside ? <div className="text-xs text-muted-foreground">{aside}</div> : null}
      </div>
      {children}
    </section>
  );
}
