import { type CurriculumModule, coresComplete } from "@/lib/curriculum";
import { moduleFill } from "@/lib/module-colors";

/**
 * Progress through an area's core as one bar of a segment per module. A
 * segment is as wide as its module has core problems; its track is the module
 * color mixed into the paper color, and the solved share is the color at full
 * saturation, a whole problem at a time. A module with no core problems has no
 * segment. The colors come from `moduleColorVars` on an ancestor.
 */
export function ModuleProgressBar({ title, modules }: { title: string; modules: CurriculumModule[] }) {
  const segments = modules.filter((module) => module.core.length > 0);
  const solved = segments.reduce((sum, module) => sum + module.coreSolved, 0);
  const total = segments.reduce((sum, module) => sum + module.core.length, 0);
  const { complete } = coresComplete(segments);
  return (
    <div
      role="progressbar"
      aria-label={`${title} progress`}
      aria-valuemin={0}
      aria-valuemax={total}
      aria-valuenow={solved}
      aria-valuetext={`${solved} of ${total} core problems solved, the core of ${complete} of ${segments.length} modules finished`}
      className="flex h-2.5 gap-0.75"
    >
      {segments.map((module) => {
        const color = moduleFill(module.number);
        return (
          <span
            key={module.track.id}
            className="h-full min-w-0 basis-0"
            style={{
              flexGrow: module.core.length,
              backgroundColor: `color-mix(in oklab, ${color} 30%, var(--paper))`,
            }}
          >
            <span
              className="block h-full"
              style={{
                width: `${(module.coreSolved / module.core.length) * 100}%`,
                backgroundColor: color,
              }}
            />
          </span>
        );
      })}
    </div>
  );
}
