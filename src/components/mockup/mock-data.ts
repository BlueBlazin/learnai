/**
 * Static sample data for the Home mockup (`/#/dev/home-mockup`). Nothing here
 * talks to the backend. The stages, problem ids, titles, and difficulties are
 * copied from `paths/inference-75.toml` and the problem folders, so the mockup
 * shows the real path; only the progress, activity, and submissions are made
 * up. Each stage is a place on a route from a trailhead to a summit, with a
 * scene in `src/assets/scenes/`.
 */

import type { CSSProperties } from "react";
import iconFlame from "@/assets/pixel/icon-flame.png";
import iconStar from "@/assets/pixel/icon-star.png";
import palette from "@/assets/pixel/palette.json";
import scenes from "@/assets/scenes/scenes.json";
import type { Difficulty, Framework, Submission, Verdict } from "@/lib/api";

export const PATH_ID = "inference-75";
export const PATH_TITLE = "Inference 75";

export const ICONS = { flame: iconFlame, star: iconStar };

/* ---------- Color ---------- */

/**
 * A color for interface use (pips, bar fills, swatches): on the light theme
 * `shade` percent of black is mixed in so it holds against the paper, on the
 * dark theme it is used as is.
 */
function uiColor(hex: string, shade: number): string {
  return shade > 0 ? `light-dark(color-mix(in oklch, ${hex}, black ${shade}%), ${hex})` : hex;
}

/**
 * How much black each scene's accent needs on the light theme to reach about
 * 3:1 against the paper, by stage. The other accents clear it as they are; on
 * the dark theme every accent clears 3.2:1 undarkened.
 */
const LIGHT_SHADE: Record<number, number> = { 1: 12, 6: 14, 8: 22, 10: 8, 13: 16 };

/** The scene images by file name, bundled by Vite. */
const SCENE_FILES = import.meta.glob<string>("../../assets/scenes/*.webp", {
  eager: true,
  import: "default",
});

function sceneUrl(slug: string): string {
  return SCENE_FILES[`../../assets/scenes/${slug}.webp`] ?? "";
}

/* ---------- The path ---------- */

export interface MockStage {
  /** 1-based part number. */
  number: number;
  title: string;
  description: string;
  /** The stage's place on the route, from trailhead to summit ("The pine forest"). */
  place: string;
  /** The place as a 21:9 scene, built on one saturated accent ink. */
  scene: string;
  /** The scene's accent ink: the stage color. */
  accent: string;
  problems: string[];
  /** Problems on the path before this stage. */
  offset: number;
}

const STAGE_SOURCE: Pick<MockStage, "title" | "description" | "problems">[] = [
  {
    title: "Numerics and tensors",
    description: "Stable arithmetic and the indexing that every later problem assumes.",
    problems: [
      "stable-softmax",
      "rmsnorm",
      "attention-mask-builder",
      "last-token-gather",
      "sequence-log-likelihood",
    ],
  },
  {
    title: "Text to tokens and back",
    description: "Encode, decode, and template text exactly as a production tokenizer does.",
    problems: [
      "bpe-encode",
      "special-tokens-split",
      "streaming-detokenizer",
      "chat-template-render",
      "tokenizer-json-load",
    ],
  },
  {
    title: "The forward pass",
    description: "Build a current decoder-only model piece by piece, then match a real one.",
    problems: [
      "causal-self-attention",
      "rope",
      "swiglu-mlp",
      "grouped-query-attention",
      "decoder-forward",
      "padded-batch-invariance",
      "moe-router-forward",
      "real-model-logits",
    ],
  },
  {
    title: "Choosing the next token",
    description: "The samplers engines ship, composed in the right order, and when to stop.",
    problems: [
      "temperature-scaling",
      "top-k-filtering",
      "top-p-filtering",
      "min-p-filtering",
      "repetition-penalty",
      "categorical-sample",
      "sampler-chain",
      "stop-sequences",
    ],
  },
  {
    title: "The KV cache",
    description: "Compute each token once, then share, page, and index the memory.",
    problems: [
      "kv-cache-append",
      "cached-greedy-decode",
      "cached-rope-offsets",
      "preallocated-kv-cache",
      "batched-cached-decode",
      "prefix-cache-reuse",
      "paged-kv-allocator",
      "radix-prefix-cache",
    ],
  },
  {
    title: "Structured output",
    description: "Turn a regex or schema into per-step token masks and decode under them.",
    problems: [
      "token-mask-apply",
      "dfa-token-mask",
      "dfa-token-index",
      "constrained-greedy-decode",
      "json-schema-regex",
    ],
  },
  {
    title: "Quantization",
    description: "Shrink weights, compute on the codes, and measure what was lost.",
    problems: [
      "affine-int8",
      "symmetric-per-channel",
      "groupwise-4bit",
      "quantized-matmul",
      "lora-merge",
      "gptq-column-update",
    ],
  },
  {
    title: "Speculative decoding",
    description: "Draft cheaply and verify exactly, first for one token, then in a loop.",
    problems: [
      "greedy-speculative-verify",
      "speculative-sampling-step",
      "speculative-sampling-block",
      "speculative-decode-loop",
    ],
  },
  {
    title: "Sizing and measurement",
    description: "Predict memory and speed from a config, and measure latency correctly.",
    problems: [
      "param-count",
      "kv-cache-bytes",
      "memory-bandwidth-bound",
      "latency-metrics",
      "latency-percentiles",
      "goodput-slo",
    ],
  },
  {
    title: "Serving",
    description: "Batch, admit, schedule, cancel, and stream under load.",
    problems: [
      "dynamic-batcher",
      "sse-token-stream",
      "token-bucket-limiter",
      "admission-control",
      "continuous-batching-scheduler",
      "chunked-prefill-scheduling",
      "deadline-cancellation",
      "vtc-fair-scheduler",
      "openai-chat-endpoint",
      "preemption-recompute",
      "prefix-aware-router",
    ],
  },
  {
    title: "On the GPU",
    description: "Lazy evaluation, compilation, and static caches on MLX and CUDA.",
    problems: [
      "mlx-lazy-eval-placement",
      "mlx-compile-step",
      "mlx-streaming-generate",
      "cuda-sync-free-decode-loop",
      "cuda-static-kv-cache",
    ],
  },
  {
    title: "Around the model",
    description: "Place a model across devices and drive it from an application loop.",
    problems: ["device-map-solver", "tool-call-loop"],
  },
  {
    title: "Capstones",
    description: "Put it together: an engine and a complete runtime for a real model.",
    problems: ["capstone-continuous-batching-engine", "capstone-real-model-runtime"],
  },
];

export const STAGES: MockStage[] = STAGE_SOURCE.reduce<MockStage[]>((stages, stage, index) => {
  const previous = stages[index - 1];
  const offset = previous ? previous.offset + previous.problems.length : 0;
  const scene = scenes[index];
  stages.push({
    ...stage,
    number: index + 1,
    offset,
    place: scene?.place ?? "",
    scene: scene ? sceneUrl(scene.slug) : "",
    accent: scene?.accent ?? "",
  });
  return stages;
}, []);

export const PROBLEM_COUNT = STAGES.reduce((sum, stage) => sum + stage.problems.length, 0);

/** The CSS variable holding a stage's interface color; defined by `stageColorVars` on the mockup root. */
export function stageColor(stage: MockStage): string {
  return `var(--stage-${stage.number})`;
}

/** Custom properties for the mockup root: one interface color per stage, plus the flame's. */
export function stageColorVars(): CSSProperties {
  const vars: Record<string, string> = { "--flame": uiColor(palette.orange, 4) };
  for (const stage of STAGES) {
    vars[`--stage-${stage.number}`] = uiColor(stage.accent, LIGHT_SHADE[stage.number] ?? 0);
  }
  return vars as CSSProperties;
}

export const PROBLEMS: Record<string, { title: string; difficulty: Difficulty }> = {
  "stable-softmax": { title: "Numerically Stable Softmax", difficulty: "easy" },
  rmsnorm: { title: "RMSNorm", difficulty: "easy" },
  "attention-mask-builder": { title: "Causal and Padding Masks", difficulty: "easy" },
  "last-token-gather": { title: "Gathering the Last Real Token", difficulty: "easy" },
  "sequence-log-likelihood": { title: "Sequence Log-Likelihood and Perplexity", difficulty: "easy" },
  "bpe-encode": { title: "Byte-Level BPE Encoding", difficulty: "medium" },
  "special-tokens-split": { title: "Special Tokens Without Injection", difficulty: "medium" },
  "streaming-detokenizer": { title: "Streaming Detokenization", difficulty: "medium" },
  "chat-template-render": { title: "Rendering a Chat Template", difficulty: "easy" },
  "tokenizer-json-load": { title: "Running a Real Tokenizer's Merges", difficulty: "medium" },
  "causal-self-attention": { title: "Causal Self-Attention", difficulty: "medium" },
  rope: { title: "Rotary Position Embeddings (RoPE)", difficulty: "medium" },
  "swiglu-mlp": { title: "SwiGLU Feed-Forward", difficulty: "easy" },
  "grouped-query-attention": { title: "Grouped-Query Attention", difficulty: "easy" },
  "decoder-forward": { title: "Full Decoder Forward Pass", difficulty: "medium" },
  "padded-batch-invariance": { title: "Batching Without Changing the Answer", difficulty: "medium" },
  "moe-router-forward": { title: "Mixture-of-Experts Layer", difficulty: "medium" },
  "real-model-logits": { title: "Matching a Real Model's Logits", difficulty: "hard" },
  "temperature-scaling": { title: "Temperature", difficulty: "easy" },
  "top-k-filtering": { title: "Top-k Filtering", difficulty: "easy" },
  "top-p-filtering": { title: "Nucleus (Top-p) Filtering", difficulty: "medium" },
  "min-p-filtering": { title: "Min-p Filtering", difficulty: "easy" },
  "repetition-penalty": { title: "Vectorized Repetition Penalty", difficulty: "medium" },
  "categorical-sample": { title: "Inverse-CDF Sampling", difficulty: "easy" },
  "sampler-chain": { title: "A Sampler Chain in the Right Order", difficulty: "medium" },
  "stop-sequences": { title: "Stop Sequences Across Token Boundaries", difficulty: "medium" },
  "kv-cache-append": { title: "Attention with an Appended Cache", difficulty: "easy" },
  "cached-greedy-decode": { title: "Greedy Decoding with a KV Cache", difficulty: "medium" },
  "cached-rope-offsets": { title: "RoPE Positions Under a Cache", difficulty: "easy" },
  "preallocated-kv-cache": { title: "A Preallocated KV Cache", difficulty: "medium" },
  "batched-cached-decode": { title: "Batched Decoding with Early Finish", difficulty: "medium" },
  "prefix-cache-reuse": { title: "Reusing a Cached Prefix", difficulty: "medium" },
  "paged-kv-allocator": { title: "A Paged KV Block Allocator", difficulty: "easy" },
  "radix-prefix-cache": { title: "A Radix-Tree Prefix Cache", difficulty: "hard" },
  "token-mask-apply": { title: "Applying a Token Mask", difficulty: "easy" },
  "dfa-token-mask": { title: "Token Masks from a Character DFA", difficulty: "medium" },
  "dfa-token-index": { title: "Precomputing the Token Index", difficulty: "medium" },
  "constrained-greedy-decode": { title: "Constrained Decoding Loop", difficulty: "medium" },
  "json-schema-regex": { title: "JSON Schema to Regex", difficulty: "hard" },
  "affine-int8": { title: "Affine int8 Quantization", difficulty: "medium" },
  "symmetric-per-channel": { title: "Symmetric Per-Channel Weights", difficulty: "easy" },
  "groupwise-4bit": { title: "Group-Wise 4-bit Quantization", difficulty: "medium" },
  "quantized-matmul": { title: "Multiplying Without Dequantizing", difficulty: "medium" },
  "lora-merge": { title: "Merging a LoRA Adapter", difficulty: "easy" },
  "gptq-column-update": { title: "GPTQ Error Propagation", difficulty: "hard" },
  "greedy-speculative-verify": { title: "Verifying a Greedy Draft", difficulty: "easy" },
  "speculative-sampling-step": { title: "Speculative Sampling: One Verification Step", difficulty: "hard" },
  "speculative-sampling-block": { title: "Verifying a Block of Sampled Drafts", difficulty: "medium" },
  "speculative-decode-loop": { title: "A Speculative Decoding Loop", difficulty: "hard" },
  "param-count": { title: "Counting Parameters", difficulty: "easy" },
  "kv-cache-bytes": { title: "KV Cache Bytes per Token", difficulty: "easy" },
  "memory-bandwidth-bound": { title: "The Memory-Bandwidth Bound on Tokens per Second", difficulty: "easy" },
  "latency-metrics": { title: "TTFT, ITL, and TPOT", difficulty: "easy" },
  "latency-percentiles": { title: "Percentiles Done Right", difficulty: "easy" },
  "goodput-slo": { title: "Goodput Under an SLO", difficulty: "easy" },
  "dynamic-batcher": { title: "Dynamic Batching Policy", difficulty: "medium" },
  "sse-token-stream": { title: "Streaming Tokens over Server-Sent Events", difficulty: "medium" },
  "token-bucket-limiter": { title: "Token-Based Rate Limiting", difficulty: "easy" },
  "admission-control": { title: "Admission Control and Backpressure", difficulty: "easy" },
  "continuous-batching-scheduler": { title: "Continuous Batching", difficulty: "hard" },
  "chunked-prefill-scheduling": { title: "Chunked Prefill Under a Token Budget", difficulty: "medium" },
  "deadline-cancellation": { title: "Deadlines and Cancellation", difficulty: "medium" },
  "vtc-fair-scheduler": { title: "Fair Scheduling with Virtual Token Counters", difficulty: "medium" },
  "openai-chat-endpoint": { title: "An OpenAI-Compatible Chat Endpoint", difficulty: "medium" },
  "preemption-recompute": { title: "Preemption by Recompute", difficulty: "hard" },
  "prefix-aware-router": { title: "Prefix-Cache-Aware Routing", difficulty: "medium" },
  "mlx-lazy-eval-placement": { title: "Where to Evaluate", difficulty: "easy" },
  "mlx-compile-step": { title: "Compiling a Decode Step", difficulty: "medium" },
  "mlx-streaming-generate": { title: "Streaming Generation with Async Evaluation", difficulty: "medium" },
  "cuda-sync-free-decode-loop": { title: "A Decode Loop Without Host Syncs", difficulty: "medium" },
  "cuda-static-kv-cache": { title: "A Static KV Cache", difficulty: "medium" },
  "device-map-solver": { title: "Solving a Device Map", difficulty: "medium" },
  "tool-call-loop": { title: "A Tool-Calling Loop", difficulty: "medium" },
  "capstone-continuous-batching-engine": {
    title: "A Continuous-Batching Engine with a Paged KV Cache",
    difficulty: "hard",
  },
  "capstone-real-model-runtime": { title: "A Complete Runtime for a Real Model", difficulty: "hard" },
};

export const PROBLEM_TITLES = new Map(Object.entries(PROBLEMS).map(([id, problem]) => [id, problem.title]));

/* ---------- Progress derived from a solved count ---------- */

/** How many of a stage's problems are solved when the first `solved` problems of the path are. */
export function stageSolved(stage: MockStage, solved: number): number {
  return Math.min(stage.problems.length, Math.max(0, solved - stage.offset));
}

/** Stages with every problem solved. */
export function stagesFinished(solved: number): number {
  return STAGES.filter((stage) => stageSolved(stage, solved) === stage.problems.length).length;
}

export interface NextProblem {
  id: string;
  title: string;
  difficulty: Difficulty;
  /** 1-based place along the whole path. */
  position: number;
  stage: MockStage;
}

/** The first unsolved problem on the path, or null when the path is finished. */
export function nextProblem(solved: number): NextProblem | null {
  const stage = STAGES.find((candidate) => stageSolved(candidate, solved) < candidate.problems.length);
  const id = stage?.problems[solved - stage.offset];
  const problem = id ? PROBLEMS[id] : undefined;
  if (!stage || !id || !problem) return null;
  return { id, ...problem, position: solved + 1, stage };
}

/* ---------- Activity ---------- */

/**
 * Twenty-six weeks of activity, oldest first (the calendar shows as many of the
 * latest weeks as fit its column). Each week is seven characters,
 * Monday to Sunday: "0" to "4" is how much was done that day, "-" is a day
 * that has not happened yet (only at the end of the last week).
 */
export type ActivityWeeks = string[];

function days(weeks: ActivityWeeks): number[] {
  return weeks
    .join("")
    .split("")
    .filter((day) => day !== "-")
    .map(Number);
}

export function activeDays(weeks: ActivityWeeks): number {
  return days(weeks).filter((level) => level > 0).length;
}

export function longestStreak(weeks: ActivityWeeks): number {
  let longest = 0;
  let run = 0;
  for (const level of days(weeks)) {
    run = level > 0 ? run + 1 : 0;
    longest = Math.max(longest, run);
  }
  return longest;
}

/** The run of active days that ends today. */
export function currentStreak(weeks: ActivityWeeks): number {
  const all = days(weeks);
  let run = 0;
  for (let index = all.length - 1; index >= 0 && (all[index] ?? 0) > 0; index--) run++;
  return run;
}

/* ---------- Submissions ---------- */

let submissionId = 0;

function submission(
  problemId: string,
  verdict: Verdict,
  passed: number,
  total: number,
  minutesAgo: number,
  framework: Framework = "torch",
): Submission {
  submissionId += 1;
  return {
    id: submissionId,
    problemId,
    framework,
    mode: "submit",
    verdict,
    passed,
    total,
    skipped: 0,
    durationMs: 1840,
    target: "local",
    createdAt: new Date(Date.now() - minutesAgo * 60_000).toISOString(),
    review: false,
  };
}

const HOUR = 60;
const DAY = 24 * HOUR;

/* ---------- The three sample states ---------- */

export type MockStateId = "started" | "underway" | "finished";

export interface MockState {
  id: MockStateId;
  label: string;
  /** The first `solved` problems of the path are solved. */
  solved: number;
  /** Problems solved this week, against the weekly goal. */
  weekSolved: number;
  weekGoal: number;
  reviewsDue: number;
  /** When nothing is due: when the next review comes up, as the end of a sentence. */
  nextReview: string;
  activity: ActivityWeeks;
  submissions: Submission[];
}

const EMPTY_WEEK = "0000000";

export const MOCK_STATES: MockState[] = [
  {
    id: "started",
    label: "Just started",
    solved: 2,
    weekSolved: 2,
    weekGoal: 5,
    reviewsDue: 0,
    nextReview: "in 3 days",
    activity: [...Array<string>(25).fill(EMPTY_WEEK), "23-----"],
    submissions: [
      submission("rmsnorm", "accepted", 8, 8, 2 * HOUR),
      submission("rmsnorm", "wrong_answer", 6, 8, 2 * HOUR + 14),
      submission("stable-softmax", "accepted", 7, 7, DAY + 3 * HOUR, "jax"),
      submission("stable-softmax", "accepted", 7, 7, DAY + 4 * HOUR),
      submission("stable-softmax", "wrong_answer", 5, 7, DAY + 4 * HOUR + 20),
    ],
  },
  {
    id: "underway",
    label: "Under way",
    solved: 16,
    weekSolved: 3,
    weekGoal: 5,
    reviewsDue: 4,
    nextReview: "tomorrow",
    activity: [
      ...Array<string>(10).fill(EMPTY_WEEK),
      EMPTY_WEEK,
      EMPTY_WEEK,
      "0010000",
      EMPTY_WEEK,
      "0100200",
      EMPTY_WEEK,
      "0012000",
      "0000010",
      EMPTY_WEEK,
      EMPTY_WEEK,
      "0021000",
      EMPTY_WEEK,
      "0012132",
      "2413000",
      "0000213",
      "232----",
    ],
    submissions: [
      submission("padded-batch-invariance", "accepted", 9, 9, 50),
      submission("padded-batch-invariance", "wrong_answer", 7, 9, 72),
      submission("decoder-forward", "accepted", 10, 10, DAY + 2 * HOUR, "mlx"),
      submission("decoder-forward", "runtime_error", 3, 10, DAY + 3 * HOUR, "mlx"),
      submission("grouped-query-attention", "accepted", 8, 8, 2 * DAY + 5 * HOUR),
    ],
  },
  {
    id: "finished",
    label: "Finished",
    solved: PROBLEM_COUNT,
    weekSolved: 5,
    weekGoal: 5,
    reviewsDue: 0,
    nextReview: "in 2 days",
    activity: [
      "0010000",
      "0000200",
      "0100010",
      "0021000",
      "1000200",
      "0012000",
      "0200100",
      "0001200",
      "2100010",
      "0012001",
      "0120010",
      "2301200",
      "0012300",
      "1200021",
      "0231000",
      "1023100",
      "0002310",
      "2310020",
      "0123010",
      "3012200",
      "0200132",
      "1321000",
      "0023120",
      "2103210",
      "1232413",
      "23142--",
    ],
    submissions: [
      submission("capstone-real-model-runtime", "accepted", 14, 14, 3 * HOUR, "mlx"),
      submission("capstone-real-model-runtime", "time_limit", 11, 14, 5 * HOUR, "mlx"),
      submission("capstone-continuous-batching-engine", "accepted", 12, 12, DAY + 2 * HOUR, "python"),
      submission("capstone-continuous-batching-engine", "wrong_answer", 10, 12, DAY + 4 * HOUR, "python"),
      submission("tool-call-loop", "accepted", 9, 9, 2 * DAY + 3 * HOUR, "python"),
    ],
  },
];
