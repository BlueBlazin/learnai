import { cn } from "cn";
import { stageFill } from "@/lib/stage-colors";
import type { Module } from "./model";

/**
 * An area's picture: one equal vertical slice of each module's scene, side by
 * side in module order, as a 7:2 banner. A module with no scene contributes a
 * flat slice of its color. Decorative: the text under it says what this is.
 */
export function AreaBanner({ modules, className }: { modules: Module[]; className?: string }) {
  return (
    <div aria-hidden className={cn("flex aspect-7/2 w-full overflow-hidden rounded-md", className)}>
      {modules.map((module) =>
        module.scene ? (
          <img
            key={module.track.id}
            src={module.scene}
            alt=""
            draggable={false}
            className="block h-full min-w-0 flex-1 basis-0 object-cover object-center"
          />
        ) : (
          <span
            key={module.track.id}
            className="h-full min-w-0 flex-1 basis-0"
            style={{ backgroundColor: stageFill(module.index) }}
          />
        ),
      )}
    </div>
  );
}
