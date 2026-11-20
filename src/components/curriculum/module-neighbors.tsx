import { cn } from "cn";
import { AppLink } from "@/components/app/app-link";
import type { CurriculumModule } from "@/lib/curriculum";
import { swatchColor } from "@/lib/module-colors";
import { routes } from "@/lib/routes";

function Neighbor({ module, direction }: { module: CurriculumModule; direction: "previous" | "next" }) {
  const next = direction === "next";
  return (
    <div className={cn("relative flex min-w-0 items-center gap-4", next && "flex-row-reverse text-right")}>
      {module.scene ? (
        <img
          src={module.scene}
          alt=""
          draggable={false}
          className="aspect-4/3 w-24 shrink-0 rounded-sm object-cover object-center"
        />
      ) : (
        <span
          aria-hidden
          className="aspect-4/3 w-24 shrink-0 rounded-sm"
          style={{ backgroundColor: module.color ?? swatchColor(null) }}
        />
      )}
      <span className="flex min-w-0 flex-col">
        <span className="tabular text-xs text-muted-foreground">
          {next ? "Next" : "Previous"}
          {module.loose ? " module" : `, Module ${module.numeral}`}
        </span>
        <AppLink
          to={routes.module(module.areaId, module.track.id)}
          tone="quiet"
          stretched
          className={cn("text-sm", next && "justify-end text-right")}
        >
          {module.track.title}
        </AppLink>
      </span>
    </div>
  );
}

/** The foot of a module's page: the modules before and after it, each with a small plate. */
export function ModuleNeighbors({
  previous,
  next,
}: {
  previous?: CurriculumModule;
  next?: CurriculumModule;
}) {
  if (!previous && !next) return null;
  return (
    <nav aria-label="Neighboring modules" className="flex items-start justify-between gap-8 border-t pt-6">
      {previous ? <Neighbor module={previous} direction="previous" /> : <span />}
      {next ? <Neighbor module={next} direction="next" /> : null}
    </nav>
  );
}
