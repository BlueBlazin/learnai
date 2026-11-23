import { cn } from "cn";
import type { Skill } from "@/lib/api";
import { skillArtUrl } from "@/lib/skill-art";

/**
 * A skill's graphic, filling its parent: in full color once earned, gray until
 * then. A skill with no bundled graphic shows a flat block of its color (the
 * accent when it has none).
 */
export function SkillGraphic({
  skill,
  earned,
  className,
}: {
  skill: Pick<Skill, "art" | "color">;
  earned: boolean;
  className?: string;
}) {
  const art = skillArtUrl(skill.art);
  return art ? (
    <img
      src={art}
      alt=""
      draggable={false}
      className={cn(
        "block size-full object-cover transition-[filter,opacity] duration-300",
        !earned && "opacity-55 contrast-85 grayscale",
        className,
      )}
    />
  ) : (
    <span
      aria-hidden
      className={cn("block size-full", !earned && "opacity-40 grayscale", className)}
      style={{ backgroundColor: skill.color ?? "var(--primary)" }}
    />
  );
}
