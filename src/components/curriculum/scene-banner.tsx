import { cn } from "cn";

/**
 * A module's scene across the column, whole: the banner has the picture's own
 * proportions (every scene is 1584 by 672), so nothing is cropped. Decorative:
 * the text under it says where this is.
 */
export function SceneBanner({ src, className }: { src: string; className?: string }) {
  return (
    <img
      src={src}
      alt=""
      draggable={false}
      className={cn("block aspect-33/14 w-full rounded-md object-cover object-center", className)}
    />
  );
}
