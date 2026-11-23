import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

/** The inks every print shares: paper and steel blue. */
const SHARED_INKS = ["#f1ede0", "#5f84a2"];

interface Piece {
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** Width and height of the cut paper, in px. */
  w: number;
  h: number;
  color: string;
  shape: "strip" | "square" | "disc";
  /** Rotation in the plane, and how fast it turns. */
  angle: number;
  spin: number;
  /** Tumble: the piece shows its edge and its face in turn. */
  phase: number;
  flutter: number;
}

const GRAVITY = 0.32;
const DRAG = 0.985;
const LIFETIME_MS = 3600;
const FADE_MS = 700;

function burst(count: number, x: number, y: number, colors: string[], power: number): Piece[] {
  return Array.from({ length: count }, () => {
    // Up and outward: a fan of 150 degrees around straight up.
    const direction = -Math.PI / 2 + (Math.random() - 0.5) * ((150 * Math.PI) / 180);
    const speed = power * (0.45 + Math.random() * 0.75);
    const roll = Math.random();
    const shape = roll < 0.55 ? "strip" : roll < 0.85 ? "square" : "disc";
    const size = 7 + Math.random() * 7;
    return {
      x,
      y,
      vx: Math.cos(direction) * speed,
      vy: Math.sin(direction) * speed,
      w: shape === "strip" ? size * 0.55 : size,
      h: shape === "strip" ? size * 1.7 : size,
      color: colors[Math.floor(Math.random() * colors.length)] ?? "#f1ede0",
      shape,
      angle: Math.random() * Math.PI * 2,
      spin: (Math.random() - 0.5) * 0.3,
      phase: Math.random() * Math.PI * 2,
      flutter: 0.12 + Math.random() * 0.18,
    };
  });
}

/**
 * One burst of cut-paper confetti over the whole window, thrown up from
 * `origin` (viewport px) and left to fall. Flat pieces in the given colors and
 * the two shared inks; nothing is drawn when the system asks for reduced
 * motion. It takes no clicks and removes itself from view when it is done.
 */
export function Confetti({ origin, colors }: { origin: { x: number; y: number }; colors: string[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const scale = window.devicePixelRatio || 1;
    const width = window.innerWidth;
    const height = window.innerHeight;
    canvas.width = width * scale;
    canvas.height = height * scale;
    context.scale(scale, scale);

    const palette = [...colors, ...colors, ...SHARED_INKS];
    let pieces = burst(110, origin.x, origin.y, palette, 19);
    // A second, smaller throw a moment later, so the burst has a tail.
    const second = window.setTimeout(() => {
      pieces = [...pieces, ...burst(60, origin.x, origin.y, palette, 14)];
    }, 180);

    const started = performance.now();
    let last = started;
    let frame = 0;
    const draw = (now: number) => {
      const elapsed = now - started;
      // Steps of a sixtieth of a second, whatever the display's rate.
      const step = Math.min(3, (now - last) / (1000 / 60));
      last = now;
      context.clearRect(0, 0, width, height);
      context.globalAlpha = Math.max(0, Math.min(1, (LIFETIME_MS - elapsed) / FADE_MS));
      for (const piece of pieces) {
        piece.vx *= DRAG ** step;
        piece.vy = piece.vy * DRAG ** step + GRAVITY * step;
        piece.x += piece.vx * step;
        piece.y += piece.vy * step;
        piece.angle += piece.spin * step;
        piece.phase += piece.flutter * step;
        context.save();
        context.translate(piece.x, piece.y);
        context.rotate(piece.angle);
        context.scale(1, Math.cos(piece.phase));
        context.fillStyle = piece.color;
        if (piece.shape === "disc") {
          context.beginPath();
          context.arc(0, 0, piece.w / 2, 0, Math.PI * 2);
          context.fill();
        } else {
          context.fillRect(-piece.w / 2, -piece.h / 2, piece.w, piece.h);
        }
        context.restore();
      }
      if (elapsed < LIFETIME_MS) frame = requestAnimationFrame(draw);
      else context.clearRect(0, 0, width, height);
    };
    frame = requestAnimationFrame(draw);

    return () => {
      window.clearTimeout(second);
      cancelAnimationFrame(frame);
    };
  }, [origin.x, origin.y, colors]);

  return createPortal(
    <canvas ref={canvasRef} aria-hidden className="pointer-events-none fixed inset-0 z-60 size-full" />,
    document.body,
  );
}
