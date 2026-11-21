import { useState } from "react";
import { FigureDrawing } from "@/components/prose/figure";
import { FigureChoice, FigureNumber, FigureNumbers, FigureSlider, FigureSliders } from "./controls";
import {
  type GridMode,
  gridLevels,
  histogram,
  isClipped,
  makeGrid,
  quantError,
  sampleWeights,
} from "./quant-math";
import type { InteractiveFigureProps } from "./registry";

/** The weights the figure quantizes, and the axis they are drawn on. */
const SAMPLES = sampleWeights();
/** Wide enough for the symmetric grid, which reaches as far below zero as the largest weight is above it. */
const AXIS = { low: -2.75, high: 2.75, bins: 88, ticks: [-2, -1, 0, 1, 2] };
const COUNTS = histogram(SAMPLES, AXIS.low, AXIS.high, AXIS.bins);
const TALLEST = Math.max(...COUNTS);

const MODES: readonly { value: GridMode; label: string }[] = [
  { value: "symmetric", label: "symmetric" },
  { value: "affine", label: "affine" },
];

/** The chart, in drawing units: a 480-unit drawing like an author's figure. */
const CHART = { width: 480, height: 186, left: 12, right: 468, base: 150, rise: 106, comb: 10 };
const BIN = (CHART.right - CHART.left) / AXIS.bins;

const xOf = (value: number) =>
  CHART.left + ((value - AXIS.low) / (AXIS.high - AXIS.low)) * (CHART.right - CHART.left);

/** A value on the axis: two decimals, with a true minus sign. */
const signed = (value: number) => value.toFixed(2).replace("-", "−");

/** A mean squared error to five decimals; one too small to show that way is given as a bound. */
const squared = (error: number) => (error > 0 && error < 0.000005 ? "< 0.00001" : error.toFixed(5));

/**
 * `quant-grid`: a fixed sample of weights as a histogram, under the grid of
 * problems/quantization/clipping-calibration (symmetric) or of affine-int8
 * (affine) at a chosen width and clipping fraction. The comb beneath the axis
 * is the grid's levels; the dashed lines are the ends of the range it was
 * fitted to, and the bars in the tinted tails beyond them are clipped. The readout splits the mean squared error into the part
 * from those values and the part from rounding the rest.
 */
export default function QuantGrid({
  label,
  initialMode = "symmetric",
  initialBits = 4,
}: InteractiveFigureProps & { initialMode?: GridMode; initialBits?: number }) {
  const [bits, setBits] = useState(initialBits);
  const [clip, setClip] = useState(1);
  const [mode, setMode] = useState<GridMode>(initialMode);

  const grid = makeGrid(mode, SAMPLES, bits, clip);
  const levels = gridLevels(grid);
  const { low, high } = grid;
  const error = quantError(SAMPLES, grid);
  const below = SAMPLES.filter((x) => x < low).length;
  const above = error.clipped - below;
  // A bar is drawn as clipped when most of its values are.
  const clipped = histogram(
    SAMPLES.filter((x) => isClipped(x, grid)),
    AXIS.low,
    AXIS.high,
    AXIS.bins,
  );
  const spacing = xOf(grid.scale) - xOf(0);

  return (
    <>
      <FigureDrawing label={label}>
        {/* The host is the image and carries the name; the readout below gives the numbers. */}
        <svg viewBox={`0 0 ${CHART.width} ${CHART.height}`} className="figure-quant" aria-hidden="true">
          {COUNTS.map((count, bin) => {
            if (count === 0) return null;
            const height = Math.max((count / TALLEST) * CHART.rise, 3);
            return (
              <rect
                // biome-ignore lint/suspicious/noArrayIndexKey: the bins are fixed and never reorder.
                key={bin}
                className={(clipped[bin] ?? 0) * 2 > count ? "bar clipped" : "bar"}
                x={CHART.left + bin * BIN + 0.75}
                y={CHART.base - height}
                width={BIN - 1.5}
                height={height}
              />
            );
          })}
          <rect className="tail" x={CHART.left} y={8} width={xOf(low) - CHART.left} height={CHART.base - 8} />
          <rect
            className="tail"
            x={xOf(high)}
            y={8}
            width={CHART.right - xOf(high)}
            height={CHART.base - 8}
          />
          <g className="levels" strokeWidth={Math.min(1, spacing / 2)}>
            {levels.map((level) => (
              <line
                key={level}
                x1={xOf(level)}
                y1={CHART.base}
                x2={xOf(level)}
                y2={CHART.base + CHART.comb}
              />
            ))}
          </g>
          <line className="axis" x1={CHART.left} y1={CHART.base} x2={CHART.right} y2={CHART.base} />
          {AXIS.ticks.map((tick) => (
            <text
              key={tick}
              className="tick"
              x={xOf(tick)}
              y={CHART.base + CHART.comb + 14}
              textAnchor="middle"
            >
              {signed(tick).replace(".00", "")}
            </text>
          ))}
          {[
            { at: low, anchor: "end" as const, shift: -5, count: below },
            { at: high, anchor: "start" as const, shift: 5, count: above },
          ].map(({ at, anchor, shift, count }) => (
            <g key={anchor}>
              <line className="end" x1={xOf(at)} y1={8} x2={xOf(at)} y2={CHART.base} />
              <text className="value" x={xOf(at) + shift} y={18} textAnchor={anchor}>
                {signed(at)}
              </text>
              {count > 0 ? (
                <text className="note" x={xOf(at) + shift} y={33} textAnchor={anchor}>
                  {count} clipped
                </text>
              ) : null}
            </g>
          ))}
        </svg>
      </FigureDrawing>
      <FigureSliders columns={2}>
        <FigureSlider
          name="bits"
          value={bits}
          shown={`${bits} (${levels.length} levels)`}
          min={2}
          max={8}
          step={1}
          onChange={setBits}
        />
        <FigureSlider
          name="clipping fraction"
          value={clip}
          shown={clip >= 1 ? "1.00 (off)" : clip.toFixed(2)}
          min={0.1}
          max={1}
          step={0.05}
          onChange={setClip}
        />
      </FigureSliders>
      <FigureChoice name="Grid" label="Kind of grid" value={mode} options={MODES} onChange={setMode} />
      <FigureNumbers>
        <FigureNumber name="Clipping error">{squared(error.clipping)}</FigureNumber>
        <FigureNumber name="Rounding error">{squared(error.rounding)}</FigureNumber>
        <FigureNumber name="MSE">{squared(error.mse)}</FigureNumber>
      </FigureNumbers>
    </>
  );
}
