import type { ProblemSummary, Skill } from "@/lib/api";
import { pluralize } from "@/lib/format";

/**
 * The skills as the Accomplishments screen shows them: where each one stands,
 * and the tree they form. A skill has a tile at its `row` and `x`, and a trail
 * runs down from it to every skill that builds on it.
 */

// ---- Where a skill stands ----

export interface SkillState {
  skill: Skill;
  /** The skill's problems that the catalog serves, in the skill's order. */
  problems: ProblemSummary[];
  solved: number;
  /** Problems the skill lists. */
  total: number;
  /** Every problem solved and every required skill earned, as the core decided it. */
  earned: boolean;
  /** The required skills still unearned once every problem is solved; empty before that. */
  waitingOn: Skill[];
}

/** The state of every skill, by skill id, in the order of `skills`. */
export function skillStates(skills: Skill[], problems: ProblemSummary[]): Map<string, SkillState> {
  const problemById = new Map(problems.map((problem) => [problem.id, problem]));
  const skillById = new Map(skills.map((skill) => [skill.id, skill]));
  const states = new Map<string, SkillState>();
  for (const skill of skills) {
    const own = skill.problems.flatMap((id) => problemById.get(id) ?? []);
    const solved = own.filter((problem) => problem.status === "solved").length;
    const total = skill.problems.length;
    const earned = skill.earnedAt !== null;
    const unearned = skill.requires
      .flatMap((id) => skillById.get(id) ?? [])
      .filter((required) => required.earnedAt === null);
    states.set(skill.id, {
      skill,
      problems: own,
      solved,
      total,
      earned,
      waitingOn: !earned && solved === total ? unearned : [],
    });
  }
  return states;
}

/** How many of the skills are earned. */
export function earnedCount(states: Map<string, SkillState>): number {
  let count = 0;
  for (const state of states.values()) if (state.earned) count += 1;
  return count;
}

/** How far along a skill is, in a few words: the line under its name on the tree. */
export function skillProgress(state: SkillState): string {
  if (state.earned) return "Earned";
  if (state.waitingOn.length > 0) return `Needs ${state.waitingOn.map((skill) => skill.title).join(", ")}`;
  if (state.solved === 0) return pluralize(state.total, "problem");
  return `${state.solved} of ${state.total} solved`;
}

// ---- The tree's layout ----

/** What the layout reads of a skill. */
export type PlacedSkill = Pick<Skill, "id" | "requires" | "row" | "x" | "lanes">;

/** The width every skill's `x` is measured on, in px. */
export const CANVAS_WIDTH = 1000;
/** The side of a skill's square tile. */
export const TILE = 124;
/** The label block under a tile. */
export const LABEL = 76;
/** The clear band between one row's labels and the next row's tiles, where the trails turn. */
const GAP = 84;
const PITCH = TILE + LABEL + GAP;

/** Distance between two trails arriving at the same tile. */
const PORT_SPACING = 14;
/** How far a tile's arrivals may slide so that a trail from straight above stays straight. */
const PORT_SLIDE = 22;
/** Vertical distance between two trails that turn in the same band. */
const LEVEL_SPACING = 20;
const CORNER = 12;
/** Bands with more turning trails than this keep them in left-to-right order: every order is too many to try. */
const MAX_ORDERED = 7;

/** The top of a row's tiles on the canvas. */
export const rowTop = (row: number) => row * PITCH;
const rowBottom = (row: number) => rowTop(row) + TILE + LABEL;

const rowCount = (skills: PlacedSkill[]) => skills.reduce((rows, skill) => Math.max(rows, skill.row + 1), 0);

/** The height of the canvas: down to the labels of the lowest row present. */
export function canvasHeight(skills: PlacedSkill[]): number {
  const rows = rowCount(skills);
  return rows === 0 ? 0 : rowBottom(rows - 1);
}

/** A skill's trails: everything that leaves it, as one set of paths that share a color. */
export interface Trail {
  /** The id of the skill the trail leaves. */
  from: string;
  /** SVG path data on the canvas, one per stretch. */
  paths: string[];
  /** Where the trail leaves its skill. */
  start: { x: number; y: number };
}

/** A skill with the skills below it that build on it. A requirement that is not on a higher row has no trail. */
interface Source {
  skill: PlacedSkill;
  targets: PlacedSkill[];
}

function sourcesOf(skills: PlacedSkill[]): Source[] {
  return skills.map((skill) => ({
    skill,
    targets: skills.filter((target) => target.row > skill.row && target.requires.includes(skill.id)),
  }));
}

/** The x a skill's trail has while it passes `row`: its own x on its row, its lane further down, or its x when it has no lane there. */
function passX(source: PlacedSkill, row: number): number {
  if (source.row === row) return source.x;
  return source.lanes.find(([laneRow]) => laneRow === row)?.[1] ?? source.x;
}

const portKey = (from: string, to: string) => `${from}>${to}`;

/**
 * Where each requirement arrives on the top edge of its skill's tile, by
 * "from>to": side by side, in the order the trails come from, left to right.
 */
export function arrivalPorts(skills: PlacedSkill[]): Map<string, number> {
  const ports = new Map<string, number>();
  for (const target of skills) {
    const incoming = skills
      .filter((source) => source.row < target.row && target.requires.includes(source.id))
      .map((source) => ({ source, from: passX(source, target.row - 1) }))
      .sort((a, b) => a.from - b.from);
    const xs = incoming.map((_, index) => target.x + (index - (incoming.length - 1) / 2) * PORT_SPACING);
    // Slide the group so that a trail coming from straight above has no kink.
    const straight = incoming.findIndex(
      (entry, index) => Math.abs(entry.from - (xs[index] ?? 0)) <= PORT_SLIDE,
    );
    const slide = straight >= 0 ? (incoming[straight]?.from ?? 0) - (xs[straight] ?? 0) : 0;
    incoming.forEach((entry, index) => {
      ports.set(portKey(entry.source.id, target.id), (xs[index] ?? target.x) + slide);
    });
  }
  return ports;
}

/** One source's lines within the band between row `gap` and the row below. */
interface Net {
  from: string;
  x0: number;
  /** Starts at the source's label, not from a lane above. */
  fromSource: boolean;
  exits: number[];
  /** The reach of the net's horizontal. */
  min: number;
  max: number;
}

function netsOfGap(gap: number, sources: Source[], ports: Map<string, number>): Net[] {
  const nets: Net[] = [];
  for (const { skill: source, targets } of sources) {
    if (source.row > gap || !targets.some((target) => target.row > gap)) continue;
    const x0 = passX(source, gap);
    const exits = targets
      .filter((target) => target.row === gap + 1)
      .map((target) => ports.get(portKey(source.id, target.id)) ?? target.x);
    if (targets.some((target) => target.row > gap + 1)) exits.push(passX(source, gap + 1));
    nets.push({
      from: source.id,
      x0,
      fromSource: source.row === gap,
      exits,
      min: Math.min(x0, ...exits),
      max: Math.max(x0, ...exits),
    });
  }
  return nets;
}

const turns = (net: Net) => net.max - net.min > 0.5;
const within = (x: number, net: Net) => x > net.min + 0.5 && x < net.max - 0.5;

/** How many times the verticals of `a` cut the horizontal of `b`, given the level each turns at. */
function cuts(a: Net, levelA: number, b: Net, levelB: number): number {
  if (!turns(b)) return 0;
  let count = 0;
  // A stretch that goes straight down spans the whole band.
  if (a.exits.some((exit) => Math.abs(exit - a.x0) <= 0.5)) {
    if (within(a.x0, b)) count += 1;
  } else if (within(a.x0, b) && levelB < levelA) count += 1;
  for (const exit of a.exits) {
    if (Math.abs(exit - a.x0) <= 0.5) continue;
    if (within(exit, b) && levelB > levelA) count += 1;
  }
  return count;
}

function permutations<T>(items: T[]): T[][] {
  if (items.length <= 1) return [items];
  return items.flatMap((item, index) =>
    permutations([...items.slice(0, index), ...items.slice(index + 1)]).map((rest) => [item, ...rest]),
  );
}

/**
 * The level each turning net turns at (0 is highest): the stacking with the
 * fewest crossings, then the fewest levels. Nets that do not overlap share one.
 */
function levelsOfGap(nets: Net[]): Map<Net, number> {
  const turning = nets.filter(turns);
  const orders =
    turning.length <= MAX_ORDERED ? permutations(turning) : [[...turning].sort((a, b) => a.x0 - b.x0)];
  let best: { cost: number; levels: Map<Net, number> } | null = null;
  for (const order of orders) {
    const levels = new Map<Net, number>();
    order.forEach((net, index) => {
      const above = order
        .slice(0, index)
        .filter((other) => other.min < net.max + 16 && net.min < other.max + 16)
        .map((other) => levels.get(other) ?? 0);
      levels.set(net, above.length > 0 ? Math.max(...above) + 1 : 0);
    });
    let crossings = 0;
    for (const a of nets) {
      for (const b of nets) {
        if (a !== b) crossings += cuts(a, levels.get(a) ?? 0, b, levels.get(b) ?? 0);
      }
    }
    const cost = crossings * 100 + Math.max(0, ...levels.values());
    if (!best || cost < best.cost) best = { cost, levels };
  }
  return best?.levels ?? new Map();
}

/** A vertical, a horizontal, and a vertical, with both corners rounded. */
function stretch(x0: number, top: number, level: number, x1: number, bottom: number): string {
  if (Math.abs(x1 - x0) <= 0.5) return `M${x0} ${top} L${x0} ${bottom}`;
  const radius = Math.min(CORNER, Math.abs(x1 - x0) / 2, level - top, bottom - level);
  const toward = x1 > x0 ? 1 : -1;
  return [
    `M${x0} ${top}`,
    `L${x0} ${level - radius}`,
    `Q${x0} ${level} ${x0 + toward * radius} ${level}`,
    `L${x1 - toward * radius} ${level}`,
    `Q${x1} ${level} ${x1} ${level + radius}`,
    `L${x1} ${bottom}`,
  ].join(" ");
}

/**
 * Every skill's trail to the skills that build on it. Trails run down, turn only
 * in the bands between rows, pass rows in their lanes, and arrive side by side.
 */
export function trails(skills: PlacedSkill[]): Trail[] {
  const sources = sourcesOf(skills);
  const ports = arrivalPorts(skills);
  const rows = rowCount(skills);
  const bySource = new Map<string, Trail>();
  for (let gap = 0; gap < rows - 1; gap++) {
    const nets = netsOfGap(gap, sources, ports);
    const levels = levelsOfGap(nets);
    const count = 1 + Math.max(0, ...levels.values());
    const middle = (rowBottom(gap) + rowTop(gap + 1)) / 2;
    for (const net of nets) {
      const level = middle + ((levels.get(net) ?? 0) - (count - 1) / 2) * LEVEL_SPACING;
      // A lane's stretch starts at the top of its row, so that it runs past the tiles beside it.
      const top = net.fromSource ? rowBottom(gap) : rowTop(gap);
      const trail = bySource.get(net.from) ?? {
        from: net.from,
        paths: [],
        start: { x: net.x0, y: top },
      };
      for (const exit of net.exits) {
        const path = stretch(net.x0, top, level, exit, rowTop(gap + 1));
        // Two stretches that coincide are one line.
        if (!trail.paths.includes(path)) trail.paths.push(path);
      }
      bySource.set(net.from, trail);
    }
  }
  return [...bySource.values()];
}
