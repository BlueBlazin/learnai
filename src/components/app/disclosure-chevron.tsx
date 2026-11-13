import { cn } from "cn";
import { ChevronRight } from "lucide-react";

/**
 * The chevron of a disclosure row. Put `group/disclosure` on the
 * `CollapsibleTrigger`; the chevron turns when its panel is open.
 */
export function DisclosureChevron({ className }: { className?: string }) {
  return (
    <ChevronRight
      className={cn(
        "size-3 shrink-0 text-faint transition-transform group-data-panel-open/disclosure:rotate-90",
        className,
      )}
      aria-hidden
    />
  );
}
