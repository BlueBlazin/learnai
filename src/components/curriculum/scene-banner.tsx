import { cn } from "cn";

/** A module's scene as a wide 7:2 banner across the column. Decorative: the text under it says where this is. */
export function SceneBanner({
  src,
  focus,
  className,
}: {
  src: string;
  /** CSS `object-position` for a scene whose subject is off center (`sceneFocus`). */
  focus?: string;
  className?: string;
}) {
  return (
    <img
      src={src}
      alt=""
      draggable={false}
      style={focus ? { objectPosition: focus } : undefined}
      className={cn("block aspect-7/2 w-full rounded-md object-cover object-center", className)}
    />
  );
}
