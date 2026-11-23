import type { Skill } from "../types";

/** A fixture skill before validation and without its earned time: `problems` lists fixture problem ids. */
export type FixtureSkill = Omit<Skill, "earnedAt">;

/**
 * The fixture's skills: seven over the authored fixture problems, on four rows,
 * so that the seeded progress shows every state at once. Sure Footing and Word
 * by Word are earned; Keen Eye is under way and one accepted submit of
 * `rotary-embeddings` away; Luck of the Draw has its one problem solved and
 * waits on Keen Eye, so that submit earns both; the other three are untouched.
 * Sure Footing and Across the Stream each pass a row in a lane.
 */
export const FIXTURE_SKILLS: FixtureSkill[] = [
  {
    id: "stable-numerics",
    title: "Sure Footing",
    name: "Stable Numerics",
    summary: "The small numerical functions everything else stands on, kept stable at the extremes.",
    art: "trail-start",
    color: "#ed9b11",
    requires: [],
    problems: ["stable-softmax", "log-sum-exp"],
    row: 0,
    x: 330,
    lanes: [[1, 545]],
  },
  {
    id: "text-to-tokens",
    title: "Word by Word",
    name: "Text to Tokens",
    summary: "From a chat to the token ids a model reads, with a real tokenizer's merges.",
    art: "journal",
    color: "#e5691d",
    requires: [],
    problems: ["bpe-merge"],
    row: 0,
    x: 760,
    lanes: [],
  },
  {
    id: "attention",
    title: "Keen Eye",
    name: "Attention",
    summary: "Masks, heads, rotary positions, and grouped queries.",
    art: "spyglass",
    color: "#2a71cc",
    requires: ["stable-numerics"],
    problems: ["causal-mask", "rotary-embeddings"],
    row: 1,
    x: 330,
    lanes: [],
  },
  {
    id: "streaming-text",
    title: "Across the Stream",
    name: "Streaming Text",
    summary: "Tokens out as text, piece by piece, without breaking a character or missing a stop.",
    art: "stepping-stones",
    color: "#27add9",
    requires: ["text-to-tokens"],
    problems: ["sse-token-stream"],
    row: 1,
    x: 760,
    lanes: [[2, 760]],
  },
  {
    id: "cached-decoding",
    title: "Cache and Carry",
    name: "Cached Decoding",
    summary: "Generation that keeps its keys and values and never recomputes the prefix.",
    art: "cache",
    color: "#18978e",
    requires: ["attention"],
    problems: ["scaled-dot-product-attention", "kv-cache-decode"],
    row: 2,
    x: 240,
    lanes: [],
  },
  {
    id: "sampling",
    title: "Luck of the Draw",
    name: "Sampling",
    summary: "From logits to one token: the filters, the draw, and their order.",
    art: "fishing",
    color: "#db5881",
    requires: ["stable-numerics", "attention"],
    problems: ["temperature-top-k"],
    row: 2,
    x: 545,
    lanes: [],
  },
  {
    id: "real-model-generation",
    title: "The Summit",
    name: "Real Model Generation",
    summary: "A real model answering a chat turn through your own runtime.",
    art: "summit",
    color: "#e6b125",
    requires: ["cached-decoding", "sampling", "streaming-text"],
    problems: ["top-p-sampling", "real-model-logits", "continuous-batching"],
    row: 3,
    x: 545,
    lanes: [],
  },
];

/**
 * When each skill was earned, by skill id, with core's rule (`earned_times` in
 * `crates/core/src/skills.rs`): a skill is unearned while one of its problems
 * has no first accepted submit or a skill it requires is unearned; otherwise it
 * was earned at the latest of those submits and its required skills' earned
 * times. `firstAccepted` maps a problem id to the time of its first accepted
 * submit that is not a review attempt.
 */
export function earnedTimes(
  skills: FixtureSkill[],
  firstAccepted: Map<string, string>,
): Map<string, string | null> {
  const byId = new Map(skills.map((skill) => [skill.id, skill]));
  const earned = new Map<string, string | null>();
  const resolve = (skill: FixtureSkill): string | null => {
    const known = earned.get(skill.id);
    if (known !== undefined) return known;
    // Unearned while it is being resolved, so a cycle ends instead of recursing forever.
    earned.set(skill.id, null);
    const times: string[] = [];
    for (const problem of skill.problems) {
      const time = firstAccepted.get(problem);
      if (time === undefined) return null;
      times.push(time);
    }
    for (const id of skill.requires) {
      const required = byId.get(id);
      const time = required ? resolve(required) : null;
      if (time === null) return null;
      times.push(time);
    }
    const latest = times.reduce<string | null>(
      (best, time) => (best === null || Date.parse(time) >= Date.parse(best) ? time : best),
      null,
    );
    earned.set(skill.id, latest);
    return latest;
  };
  for (const skill of skills) resolve(skill);
  return earned;
}
