import { useState } from "react";
import { FigureDrawing } from "@/components/prose/figure";
import { FigureNumber, FigureNumbers, FigureSlider, FigureSliders } from "./controls";
import type { InteractiveFigureProps } from "./registry";
import { frequencies, ropeScore } from "./rope-math";

/**
 * The setting of the worked examples in problems/transformer/rope and
 * problems/kv-cache/cached-rope-offsets: D = 4 and base 100, so the two pairs
 * turn by 1 and 0.1 radian per position, and the query and the key are both
 * [1, 0, 1, 0] before they are rotated.
 */
const BASE = 100;
const QUERY = [1, 0, 1, 0] as const;
const KEY = [1, 0, 1, 0] as const;
const THETAS = frequencies(QUERY.length, BASE);
const SUBSCRIPTS = ["₀", "₁", "₂", "₃"] as const;

/** The two planes, in drawing units: a 480-unit drawing like an author's figure. */
const PLANE = { width: 480, height: 244, centers: [122, 350], middle: 138, radius: 70, reach: 102 };
const HEAD = { length: 9, half: 3.2 };
const LABEL_RADIUS = PLANE.radius + 13;
/** The smallest angle between the two tip labels, so they never touch. */
const LABEL_GAP = 0.36;
const ARC_RADIUS = 26;

/** `angle` brought into (-pi, pi]. */
const wrap = (angle: number) => angle - 2 * Math.PI * Math.ceil((angle - Math.PI) / (2 * Math.PI));

/** A number with a true minus sign. */
const signed = (text: string) => text.replace("-", "−");

/** The point at `angle` (counterclockwise from the first axis) and `radius` around a center; y runs down. */
const polar = (cx: number, angle: number, radius: number) => ({
  x: cx + radius * Math.cos(angle),
  y: PLANE.middle - radius * Math.sin(angle),
});

/** An arrow from the center of a plane to the circle, the head a small filled triangle. */
function Arrow({ cx, angle, className }: { cx: number; angle: number; className: string }) {
  const tip = polar(cx, angle, PLANE.radius);
  const neck = polar(cx, angle, PLANE.radius - HEAD.length);
  const across = { x: HEAD.half * Math.sin(angle), y: HEAD.half * Math.cos(angle) };
  return (
    <g className={className}>
      <line x1={cx} y1={PLANE.middle} x2={neck.x} y2={neck.y} />
      <polygon
        points={`${tip.x},${tip.y} ${neck.x + across.x},${neck.y + across.y} ${neck.x - across.x},${neck.y - across.y}`}
      />
    </g>
  );
}

/** One pair of dimensions: its plane, the rotated query and key in it, and the angle between them. */
function Plane({ pair, q, k }: { pair: number; q: readonly number[]; k: readonly number[] }) {
  const cx = PLANE.centers[pair] ?? 0;
  const first = SUBSCRIPTS[2 * pair];
  const second = SUBSCRIPTS[2 * pair + 1];
  const queryAngle = Math.atan2(q[2 * pair + 1] ?? 0, q[2 * pair] ?? 0);
  const keyAngle = Math.atan2(k[2 * pair + 1] ?? 0, k[2 * pair] ?? 0);
  // From the query to the key the short way round; counterclockwise when positive.
  const between = wrap(keyAngle - queryAngle);
  const spread = Math.max(0, LABEL_GAP - Math.abs(between)) / 2;
  const side = between >= 0 ? 1 : -1;
  const queryLabel = polar(cx, queryAngle - side * spread, LABEL_RADIUS);
  const keyLabel = polar(cx, keyAngle + side * spread, LABEL_RADIUS);
  const arcStart = polar(cx, queryAngle, ARC_RADIUS);
  const arcEnd = polar(cx, keyAngle, ARC_RADIUS);
  return (
    <g>
      <text className="title" x={cx} y={15} textAnchor="middle">
        Pair {pair}: θ{SUBSCRIPTS[pair]} = {THETAS[pair]?.toFixed(pair === 0 ? 0 : 1)}
      </text>
      <g className="guide">
        <line x1={cx - PLANE.reach} y1={PLANE.middle} x2={cx + PLANE.reach} y2={PLANE.middle} />
        <line x1={cx} y1={PLANE.middle - PLANE.reach} x2={cx} y2={PLANE.middle + PLANE.reach} />
        <circle className="unit" cx={cx} cy={PLANE.middle} r={PLANE.radius} />
      </g>
      <text className="axis-name" x={cx + PLANE.reach + 4} y={PLANE.middle} dominantBaseline="central">
        <tspan fontStyle="italic">x</tspan>
        {first}
      </text>
      <text className="axis-name" x={cx + 6} y={PLANE.middle - PLANE.reach + 4} dominantBaseline="central">
        <tspan fontStyle="italic">x</tspan>
        {second}
      </text>
      {Math.abs(between) > 0.05 ? (
        <path
          className="between"
          d={`M${arcStart.x} ${arcStart.y} A${ARC_RADIUS} ${ARC_RADIUS} 0 0 ${between > 0 ? 0 : 1} ${arcEnd.x} ${arcEnd.y}`}
        />
      ) : null}
      <Arrow cx={cx} angle={queryAngle} className="query" />
      <Arrow cx={cx} angle={keyAngle} className="key" />
      <text
        className="query-name"
        x={queryLabel.x}
        y={queryLabel.y}
        textAnchor="middle"
        dominantBaseline="central"
      >
        q
      </text>
      <text className="key-name" x={keyLabel.x} y={keyLabel.y} textAnchor="middle" dominantBaseline="central">
        k
      </text>
    </g>
  );
}

/**
 * `rope-shift`: a query at position m and a key at position n, both moved by
 * one shift and rotated by RoPE for where they land. Each plane is one pair of
 * dimensions with the rotated query (solid) and key (dashed) in it. The shift
 * turns both arrows of a plane by the same angle, so the angle between them
 * and the score stay as they are; the readout gives the score pair by pair.
 */
export default function RopeShift({ label }: InteractiveFigureProps) {
  const [query, setQuery] = useState(1);
  const [key, setKey] = useState(0);
  const [shift, setShift] = useState(0);

  const result = ropeScore(QUERY, KEY, query + shift, key + shift, BASE);

  return (
    <>
      <FigureDrawing label={label}>
        {/* The host is the image and carries the name; the readout below gives the numbers. */}
        <svg viewBox={`0 0 ${PLANE.width} ${PLANE.height}`} className="figure-planes" aria-hidden="true">
          {THETAS.map((theta, pair) => (
            <Plane key={theta} pair={pair} q={result.q} k={result.k} />
          ))}
        </svg>
      </FigureDrawing>
      <FigureSliders>
        <FigureSlider
          name="m (query)"
          value={query}
          shown={String(query)}
          min={0}
          max={8}
          step={1}
          onChange={setQuery}
        />
        <FigureSlider
          name="n (key)"
          value={key}
          shown={String(key)}
          min={0}
          max={8}
          step={1}
          onChange={setKey}
        />
        <FigureSlider
          name="shift"
          value={shift}
          shown={String(shift)}
          min={0}
          max={40}
          step={1}
          onChange={setShift}
        />
      </FigureSliders>
      <FigureNumbers>
        <FigureNumber name="Positions">
          {query + shift}, {key + shift}
        </FigureNumber>
        <FigureNumber name="n − m">{signed(String(key - query))}</FigureNumber>
        <FigureNumber name="Score">
          {result.pairs.map((part) => signed(part.toFixed(4))).join(" + ")} ={" "}
          {signed(result.score.toFixed(4))}
        </FigureNumber>
      </FigureNumbers>
    </>
  );
}
