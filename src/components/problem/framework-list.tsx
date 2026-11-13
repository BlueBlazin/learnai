import { cn } from "cn";
import type { Framework } from "@/lib/api";
import { FRAMEWORK_LABEL } from "@/lib/labels";

/** The frameworks a problem supports. Ones with an accepted submission are set in ink; the rest are muted. */
export function FrameworkList({
  frameworks,
  solved = [],
  className,
}: {
  frameworks: Framework[];
  solved?: Framework[];
  className?: string;
}) {
  return (
    <span className={cn("inline-flex gap-2.5 whitespace-nowrap", className)}>
      {frameworks.map((framework) => {
        const done = solved.includes(framework);
        return (
          <span key={framework} className={done ? "font-medium text-foreground" : "text-muted-foreground"}>
            {FRAMEWORK_LABEL[framework]}
            {done ? <span className="sr-only"> (solved)</span> : null}
          </span>
        );
      })}
    </span>
  );
}
