import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import type { Skill } from "@/lib/api";
import { Confetti } from "./confetti";
import { SkillGraphic } from "./skill-graphic";

/** How long the graphic stays gray before it takes its color, in ms. */
const REVEAL_DELAY_MS = 260;

/** A lighter tint of a `#rrggbb` color, for a second confetti ink. */
function tint(hex: string): string {
  const value = Number.parseInt(hex.slice(1), 16);
  const mix = (channel: number) => Math.round(channel + (255 - channel) * 0.5);
  const [r, g, b] = [(value >> 16) & 255, (value >> 8) & 255, value & 255].map(mix);
  return `rgb(${r} ${g} ${b})`;
}

/** The inside of the popup: mounted afresh each time one opens, so the reveal and the confetti replay. */
function Earned({ skill, onView }: { skill: Skill; onView: () => void }) {
  const graphicRef = useRef<HTMLDivElement>(null);
  const [revealed, setRevealed] = useState(false);
  const [origin, setOrigin] = useState<{ x: number; y: number } | null>(null);
  // A skill with no color of its own is thrown in the shared inks alone.
  const colors = useMemo(() => (skill.color ? [skill.color, tint(skill.color)] : []), [skill.color]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setRevealed(true);
      // The confetti is thrown from the middle of the graphic, as it takes its color.
      const box = graphicRef.current?.getBoundingClientRect();
      if (box) setOrigin({ x: box.left + box.width / 2, y: box.top + box.height / 2 });
    }, REVEAL_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <>
      <p className="flex items-center justify-center gap-3 text-md font-semibold">
        {/* Mirrored, so that both poppers fire toward the words. */}
        <span aria-hidden className="-scale-x-100 text-3xl leading-none">
          🎉
        </span>
        Skill earned
        <span aria-hidden className="text-3xl leading-none">
          🎉
        </span>
      </p>
      <div ref={graphicRef} className="mx-auto mt-5 size-64 overflow-hidden rounded-md bg-muted">
        <SkillGraphic skill={skill} earned={revealed} className="duration-700" />
      </div>
      <DialogTitle className="mt-5 font-serif text-2xl leading-tight font-medium tracking-tight text-balance">
        {skill.title}
      </DialogTitle>
      <p className="mt-1.5 text-md text-muted-foreground">({skill.name})</p>
      <DialogDescription className="sr-only">
        You earned the skill {skill.title} ({skill.name}).
      </DialogDescription>
      <div className="mt-6 flex flex-col items-center gap-2">
        <DialogClose render={<Button className="min-w-40" />}>Continue</DialogClose>
        <Button variant="link" size="sm" className="text-muted-foreground" onClick={onView}>
          View accomplishments
        </Button>
      </div>
      {origin ? <Confetti origin={origin} colors={colors} /> : null}
    </>
  );
}

/**
 * The moment a skill is earned: its graphic turning from gray to color, its
 * name, and a burst of confetti in its colors. A skill earned while this one
 * is showing replaces it and replays the moment.
 */
export function SkillEarned({
  skill,
  onClose,
  onView,
}: {
  /** The skill just earned; null while the popup is closed. */
  skill: Skill | null;
  onClose: () => void;
  /** Go to the skill on the Accomplishments screen. */
  onView: (skill: Skill) => void;
}) {
  return (
    <Dialog open={skill !== null} onOpenChange={(open) => (open ? undefined : onClose())}>
      {skill ? (
        <DialogContent className="block px-8 pt-7 pb-6 text-center sm:max-w-sm">
          <Earned key={skill.id} skill={skill} onView={() => onView(skill)} />
        </DialogContent>
      ) : null}
    </Dialog>
  );
}
