import { cn } from "cn";
import flame from "@/assets/pixel/icon-flame.png";
import star from "@/assets/pixel/icon-star.png";

const ICONS = { flame, star };

/** Icons are drawn on a 24 by 24 grid. */
const GRID = 24;

/** A 24x24 pixel-art icon at a whole-number scale, never smoothed. Decorative unless it has a label. */
export function PixelIcon({
  name,
  scale = 1,
  label,
  className,
}: {
  name: keyof typeof ICONS;
  scale?: 1 | 2 | 3;
  /** Accessible name. Without one the icon is decorative. */
  label?: string;
  className?: string;
}) {
  const size = GRID * scale;
  return (
    <img
      src={ICONS[name]}
      alt={label ?? ""}
      width={size}
      height={size}
      draggable={false}
      className={cn("inline-block max-w-none shrink-0 align-bottom", className)}
      style={{ width: size, height: size, imageRendering: "pixelated" }}
    />
  );
}
