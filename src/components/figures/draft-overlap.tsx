import { useState } from "react";
import { FigureDrawing } from "@/components/prose/figure";
import { FigureNumber, FigureNumbers, FigureSlider, FigureSliders } from "./controls";
import { acceptanceRate, bump, overlap, residualTotal, TARGET } from "./draft-math";
import type { InteractiveFigureProps } from "./registry";

/** The chart, in drawing units: a 480-unit drawing like an author's figure. */
const CHART = { width: 480, height: 212, left: 40, right: 474, base: 184, rise: 144, slot: 53, bar: 36 };
const TOP = 0.5;
const TICKS = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
/** The key above the chart: one swatch for each part of a bar. */
const KEY = [
  { part: "kept", name: "min(p, q)", x: 48 },
  { part: "residual", name: "r = max(0, p − q)", x: 150 },
  { part: "excess", name: "q above p", x: 296 },
] as const;

const yOf = (probability: number) => CHART.base - (probability / TOP) * CHART.rise;

/** A number with a true minus sign. */
const signed = (text: string) => text.replace("-", "−");

/**
 * `draft-overlap`: the target `p` and a draft `q` of
 * problems/speculative/speculative-sampling-step over eight tokens. The draft
 * is the target's bump with its logits multiplied by the sharpness and its
 * center moved by the shift. Each bar is split at the lower of `p` and `q`:
 * beneath it the overlap `min(p, q)`, the mass that is drafted and accepted;
 * above it the residual `r` where `p` is higher, or the part of `q` that is
 * rejected where `q` is higher. Overlap and residual stack to `p`, so the
 * acceptance rate and `R` add up to 1.
 */
export default function DraftOverlap({ label }: InteractiveFigureProps) {
  const [sharpness, setSharpness] = useState(1.5);
  const [shift, setShift] = useState(1);

  const p = TARGET;
  const q = bump(sharpness, shift);
  const shared = overlap(p, q);
  const rate = acceptanceRate(p, q);
  const rejected = residualTotal(p, q);

  return (
    <>
      <FigureDrawing label={label}>
        {/* The host is the image and carries the name; the readout below gives the numbers. */}
        <svg viewBox={`0 0 ${CHART.width} ${CHART.height}`} className="figure-bars" aria-hidden="true">
          {KEY.map(({ part, name, x }) => (
            <g key={part}>
              <rect className={part} x={x} y={8} width={16} height={10} />
              <text x={x + 22} y={13} dominantBaseline="central">
                {name}
              </text>
            </g>
          ))}
          {TICKS.map((tick) => (
            <g key={tick}>
              {tick > 0 ? (
                <line className="grid" x1={CHART.left} y1={yOf(tick)} x2={CHART.right} y2={yOf(tick)} />
              ) : null}
              <text
                className="tick"
                x={CHART.left - 6}
                y={yOf(tick)}
                textAnchor="end"
                dominantBaseline="central"
              >
                {tick.toFixed(1)}
              </text>
            </g>
          ))}
          {p.map((target, token) => {
            const x = CHART.left + 8 + token * CHART.slot;
            const draft = q[token] ?? 0;
            const low = yOf(shared[token] ?? 0);
            const high = yOf(Math.max(target, draft));
            return (
              // biome-ignore lint/suspicious/noArrayIndexKey: the tokens are fixed and never reorder.
              <g key={token}>
                {low - high > 0.5 ? (
                  <rect
                    className={target > draft ? "residual" : "excess"}
                    x={x}
                    y={high}
                    width={CHART.bar}
                    height={low - high}
                  />
                ) : null}
                <rect
                  className="kept"
                  x={x}
                  y={low}
                  width={CHART.bar}
                  height={Math.max(CHART.base - low, 1)}
                />
                <text
                  className="token"
                  x={x + CHART.bar / 2}
                  y={CHART.base + 15}
                  textAnchor="middle"
                  dominantBaseline="central"
                >
                  {token}
                </text>
              </g>
            );
          })}
          <line className="axis" x1={CHART.left} y1={CHART.base} x2={CHART.right} y2={CHART.base} />
        </svg>
      </FigureDrawing>
      <FigureSliders columns={2}>
        <FigureSlider
          name="sharpness of q"
          value={sharpness}
          shown={sharpness === 1 ? "1.00 (as p)" : sharpness.toFixed(2)}
          min={0.5}
          max={3}
          step={0.25}
          onChange={setSharpness}
        />
        <FigureSlider
          name="shift of q"
          value={shift}
          shown={shift === 0 ? "0.00 (as p)" : signed(shift.toFixed(2))}
          min={-2}
          max={2}
          step={0.25}
          onChange={setShift}
        />
      </FigureSliders>
      <FigureNumbers>
        <FigureNumber name="Acceptance rate">{rate.toFixed(3)}</FigureNumber>
        <FigureNumber name="R">{rejected.toFixed(3)}</FigureNumber>
        <FigureNumber name="Rate + R">{(rate + rejected).toFixed(3)}</FigureNumber>
      </FigureNumbers>
    </>
  );
}
