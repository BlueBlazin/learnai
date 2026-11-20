import { cn } from "cn";

/** Icons are drawn on a 24 by 24 grid. */
const GRID = 24;

/** A 24x24 pixel-art icon at a whole-number scale, never smoothed. Decorative unless it has a label. */
export function PixelSprite({
  src,
  scale = 1,
  label,
  className,
}: {
  src: string;
  scale?: 1 | 2 | 3;
  /** Accessible name. Without one the icon is decorative. */
  label?: string;
  className?: string;
}) {
  const size = GRID * scale;
  return (
    <img
      src={src}
      alt={label ?? ""}
      width={size}
      height={size}
      draggable={false}
      className={cn("inline-block max-w-none shrink-0 align-bottom", className)}
      style={{ width: size, height: size, imageRendering: "pixelated" }}
    />
  );
}
