import { cn } from "cn";

/** A stage's scene as a wide 7:2 banner across the column. Decorative: the text under it says where this is. */
export function SceneBanner({ src, className }: { src: string; className?: string }) {
  return (
    <img
      src={src}
      alt=""
      draggable={false}
      className={cn("block aspect-7/2 w-full rounded-md object-cover object-center", className)}
    />
  );
}
