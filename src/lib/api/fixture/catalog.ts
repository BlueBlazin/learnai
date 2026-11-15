import type { Difficulty, Framework, JudgeKind, ProblemStatus, ProblemSummary, Requires } from "../types";
import { FIXTURE_TRACKS, type FixtureTrack } from "./tracks";

/** How a simulated run of non-starter code ends for a problem. */
export type Scenario =
  | "accepted"
  | "wrong_answer"
  | "cost_exceeded"
  | "too_slow"
  | "runtime_error"
  | "time_limit"
  | "import_error"
  | "harness_error";

export interface FixtureCase {
  name: string;
  public: boolean;
  /**
   * What the case cannot run without: the problem's model assets, or the
   * hardware the problem requires. Absent: it runs everywhere.
   */
  needs?: "asset" | "hardware";
  /** The stage the case belongs to; only capstones have stages. */
  stage?: string;
}

export type ProblemMeta = Omit<ProblemSummary, "status" | "solvedFrameworks">;

export interface FixtureProblem {
  meta: ProblemMeta;
  statement: string;
  hints: string[];
  editorial: string;
  entry: string[];
  timeLimitS: number;
  /** Statement figures by path relative to the problem's `assets/` directory: SVG source. */
  assets: Record<string, string>;
  cases: FixtureCase[];
  /** Code templates with `{{import}}` and `{{T}}` placeholders. */
  starterTemplate: string;
  referenceTemplate: string;
  /** A saved solution present in the seeded workspace, if any. */
  savedTemplate: string | null;
  /** Outcome of running the seeded (or any edited) solution. */
  scenario: Scenario;
  seedStatus: ProblemStatus;
  seedSolved: Framework[];
}

const content = import.meta.glob<string>("./content/*/*", {
  query: "?raw",
  import: "default",
  eager: true,
});

function file(id: string, name: string): string | null {
  return content[`./content/${id}/${name}`] ?? null;
}

/**
 * Statement figures. The fixtures have no drawings of their own: the rotary
 * problem borrows the figure of the real `rope` problem, so the figure
 * components can be worked on against the file authors actually ship.
 */
const ropeFigures = import.meta.glob<string>("../../../../problems/transformer/rope/assets/*.svg", {
  query: "?raw",
  import: "default",
  eager: true,
});

const FIGURES: Record<string, Record<string, string>> = {
  "rotary-embeddings": Object.fromEntries(
    Object.entries(ropeFigures).map(([path, svg]) => [path.slice(path.lastIndexOf("/") + 1), svg]),
  ),
};

function assetsOf(id: string): Record<string, string> {
  return FIGURES[id] ?? {};
}

const TENSOR_FRAMEWORKS: Framework[] = ["torch", "jax", "mlx"];

interface Authored {
  id: string;
  title: string;
  track: string;
  order: number;
  difficulty: Difficulty;
  frameworks: Framework[];
  judges: JudgeKind[];
  tags: string[];
  prerequisites?: string[];
  entry: string[];
  timeLimitS?: number;
  publicCases: string[];
  hiddenCases: number;
  scenario?: Scenario;
  status: ProblemStatus;
  solved?: Framework[];
  requires?: Requires;
  /** Names of the model assets the problem declares. */
  models?: string[];
  /** What each case needs, public cases first, then hidden; shorter than the case list means "nothing". */
  needs?: (FixtureCase["needs"] | null)[];
}

const AUTHORED: Authored[] = [
  {
    id: "stable-softmax",
    title: "Numerically Stable Softmax",
    track: "foundations",
    order: 10,
    difficulty: "easy",
    frameworks: TENSOR_FRAMEWORKS,
    judges: ["numeric"],
    tags: ["softmax", "numerics"],
    entry: ["softmax"],
    publicCases: ["small vector", "large magnitudes", "batch, last dim"],
    hiddenCases: 2,
    status: "solved",
    solved: ["torch", "jax"],
  },
  {
    id: "log-sum-exp",
    title: "Log-Sum-Exp Without Overflow",
    track: "foundations",
    order: 20,
    difficulty: "easy",
    frameworks: TENSOR_FRAMEWORKS,
    judges: ["numeric", "property"],
    tags: ["numerics", "log-space"],
    prerequisites: ["stable-softmax"],
    entry: ["logsumexp"],
    publicCases: ["two equal terms", "mixed magnitudes", "shift invariance"],
    hiddenCases: 4,
    status: "solved",
    solved: ["torch"],
  },
  {
    id: "bpe-merge",
    title: "Train a Byte-Pair Encoder",
    track: "tokenization",
    order: 10,
    difficulty: "medium",
    frameworks: ["python"],
    judges: ["property"],
    tags: ["bpe", "tokenizer"],
    entry: ["train_bpe"],
    publicCases: ["four-word corpus", "tie-breaking", "overlapping pairs"],
    hiddenCases: 5,
    status: "solved",
    solved: ["python"],
  },
  {
    id: "scaled-dot-product-attention",
    title: "Scaled Dot-Product Attention",
    track: "attention",
    order: 10,
    difficulty: "medium",
    frameworks: TENSOR_FRAMEWORKS,
    judges: ["numeric"],
    tags: ["attention", "masking"],
    prerequisites: ["stable-softmax"],
    entry: ["attention"],
    publicCases: ["single head, no mask", "single head, causal mask", "batched heads, d = 64"],
    hiddenCases: 5,
    status: "todo",
  },
  {
    id: "causal-mask",
    title: "Causal Mask for Padded Batches",
    track: "attention",
    order: 20,
    difficulty: "easy",
    frameworks: TENSOR_FRAMEWORKS,
    judges: ["property"],
    tags: ["masking", "padding"],
    entry: ["causal_mask"],
    publicCases: ["two sequences", "full-length batch", "length one"],
    hiddenCases: 3,
    status: "solved",
    solved: ["torch"],
  },
  {
    id: "rotary-embeddings",
    title: "Rotary Position Embeddings",
    track: "attention",
    order: 30,
    difficulty: "medium",
    frameworks: TENSOR_FRAMEWORKS,
    judges: ["numeric", "property", "speed"],
    tags: ["positions", "rope"],
    prerequisites: ["scaled-dot-product-attention"],
    entry: ["apply_rope"],
    publicCases: ["one pair, two positions", "norm preservation", "relative shift"],
    hiddenCases: 4,
    scenario: "too_slow",
    status: "attempted",
  },
  {
    id: "temperature-top-k",
    title: "Temperature and Top-k",
    track: "sampling",
    order: 10,
    difficulty: "easy",
    frameworks: TENSOR_FRAMEWORKS,
    judges: ["numeric"],
    tags: ["sampling", "top-k"],
    prerequisites: ["stable-softmax"],
    entry: ["top_k_probs"],
    publicCases: ["k = 2, unit temperature", "k = 2, low temperature", "k = V"],
    hiddenCases: 3,
    status: "solved",
    solved: ["torch"],
  },
  {
    id: "top-p-sampling",
    title: "Nucleus (Top-p) Sampling",
    track: "sampling",
    order: 20,
    difficulty: "medium",
    frameworks: TENSOR_FRAMEWORKS,
    judges: ["statistical"],
    tags: ["sampling", "top-p"],
    prerequisites: ["temperature-top-k"],
    entry: ["sample_top_p"],
    publicCases: ["p = 1 is plain sampling", "boundary uniforms", "four-token nucleus"],
    hiddenCases: 4,
    scenario: "wrong_answer",
    status: "attempted",
  },
  {
    id: "kv-cache-decode",
    title: "Greedy Decoding with a KV Cache",
    track: "kv-cache",
    order: 10,
    difficulty: "medium",
    frameworks: TENSOR_FRAMEWORKS,
    judges: ["cost", "numeric"],
    tags: ["kv-cache", "decoding"],
    prerequisites: ["scaled-dot-product-attention"],
    entry: ["greedy_decode"],
    publicCases: ["matches uncached output", "prompt 32, five tokens", "stops at end of sequence"],
    hiddenCases: 4,
    scenario: "cost_exceeded",
    status: "attempted",
  },
  {
    id: "int8-affine-quant",
    title: "Affine Int8 Quantization",
    track: "quantization",
    order: 10,
    difficulty: "medium",
    frameworks: TENSOR_FRAMEWORKS,
    judges: ["numeric", "property"],
    tags: ["quantization", "int8"],
    entry: ["quantize", "dequantize"],
    publicCases: ["symmetric range", "exact zero", "constant tensor"],
    hiddenCases: 5,
    status: "todo",
  },
  {
    id: "continuous-batching",
    title: "Continuous Batching Scheduler",
    track: "batching",
    order: 10,
    difficulty: "hard",
    frameworks: ["python"],
    judges: ["simclock"],
    tags: ["scheduling", "batching"],
    entry: ["Scheduler"],
    timeLimitS: 5,
    publicCases: ["three requests, two slots", "staggered arrivals", "trace of 1,000 requests"],
    hiddenCases: 3,
    scenario: "time_limit",
    status: "attempted",
  },
  {
    id: "sse-token-stream",
    title: "Streaming Tokens over Server-Sent Events",
    track: "serving",
    order: 10,
    difficulty: "medium",
    frameworks: ["python"],
    judges: ["service"],
    tags: ["http", "streaming", "asgi"],
    entry: ["create_app"],
    timeLimitS: 15,
    publicCases: ["event format", "incremental delivery", "validation errors"],
    hiddenCases: 3,
    status: "todo",
  },
  // Real-model and hardware problems: a model that is downloaded, one that is not,
  // one whose download stopped half way, and hardware this (fixture) Mac lacks.
  {
    id: "real-model-logits",
    title: "Next-Token Logits from a Real Checkpoint",
    track: "transformer",
    order: 10,
    difficulty: "medium",
    frameworks: TENSOR_FRAMEWORKS,
    judges: ["numeric", "cost"],
    tags: ["real-model", "rmsnorm", "unembedding"],
    entry: ["next_token_logits"],
    publicCases: ["random hidden states, d = 8", "the checkpoint: one prompt"],
    hiddenCases: 3,
    status: "todo",
    models: ["granite-4.0-350m"],
    needs: [null, "asset", null, "asset", "asset"],
  },
  {
    id: "lazy-safetensors-load",
    title: "Load Only the Tensors You Need",
    track: "memory",
    order: 10,
    difficulty: "medium",
    frameworks: ["python"],
    judges: ["cost"],
    tags: ["real-model", "safetensors", "io"],
    entry: ["load_tensors"],
    publicCases: ["a file written in the test", "two tensors of the checkpoint"],
    hiddenCases: 2,
    status: "todo",
    models: ["granite-4.0-1b"],
    needs: [null, "asset", null, "asset"],
  },
  {
    id: "mlx-8bit-linear",
    title: "An 8-bit Linear Layer Without Dequantizing",
    track: "mlx-gpu",
    order: 10,
    difficulty: "hard",
    frameworks: ["mlx"],
    judges: ["numeric", "cost"],
    tags: ["real-model", "quantization", "mlx"],
    entry: ["quantized_linear"],
    publicCases: [
      "one group, hand-written codes",
      "two rows, three groups",
      "layers.0.mlp.fc1 of the checkpoint",
    ],
    hiddenCases: 3,
    status: "attempted",
    requires: "mlx-gpu",
    models: ["granite-4.2-3b-mlx-8bit"],
    needs: [null, null, "asset", null, "asset", "asset"],
  },
  {
    id: "cuda-device-placement",
    title: "Build the Causal Bias on the Device",
    track: "distributed",
    order: 10,
    difficulty: "easy",
    frameworks: ["torch"],
    judges: ["numeric", "property"],
    tags: ["cuda", "devices"],
    entry: ["causal_bias"],
    publicCases: ["values for T = 4", "float32, no gradient"],
    hiddenCases: 3,
    status: "attempted",
    requires: "cuda",
    needs: [null, null, null, "hardware", "hardware"],
  },
];

/** Small deterministic hash so generated metadata is stable across reloads. */
export function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967296;
}

export function slug(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

const PYTHON_TRACKS = new Set(["tokenization", "batching", "serving", "structured", "distributed"]);

const TRACK_JUDGES: Record<string, JudgeKind[]> = {
  foundations: ["numeric", "property"],
  tokenization: ["property", "numeric"],
  attention: ["numeric", "property", "speed"],
  transformer: ["numeric", "cost"],
  sampling: ["statistical", "numeric", "property"],
  "kv-cache": ["cost", "numeric"],
  quantization: ["numeric", "property"],
  batching: ["simclock", "property"],
  serving: ["service", "simclock"],
  memory: ["cost", "property"],
  speculative: ["statistical", "cost"],
  structured: ["property", "speed"],
  "mlx-gpu": ["numeric", "speed"],
  distributed: ["simclock", "cost"],
  capstones: ["service", "speed", "cost"],
};

/** Share of generated problems seeded as solved, by track order: early tracks are further along. */
const SOLVED_SHARE = [0.75, 0.5, 0.35, 0.15, 0.3, 0.1, 0.08, 0.08, 0.05, 0, 0, 0, 0, 0, 0];

function fillerStatement(title: string, track: FixtureTrack, entry: string): string {
  return [
    `This is a generated placeholder for *${title}* in the fixture catalog. ${track.description}`,
    "",
    "## Interface",
    "",
    "```python",
    `def ${entry}(x):`,
    '    """Return the result described above."""',
    "```",
    "",
    "## Constraints",
    "",
    "- Inputs are float32 unless stated otherwise.",
    "- The reference runs in $O(n)$ time and $O(1)$ extra memory.",
  ].join("\n");
}

/**
 * A capstone's cases, in stages. Stages gate nothing: every case runs. The
 * second label already begins with "Stage", so it is shown as written.
 */
const CAPSTONE_CASES: FixtureCase[] = [
  { name: "round-trips ASCII", public: true, stage: "1: Tokenizer" },
  { name: "round-trips multi-byte text", public: true, stage: "1: Tokenizer" },
  { name: "Hidden case 1", public: false, stage: "1: Tokenizer" },
  { name: "greedy decode matches the reference", public: true, stage: "Stage 2: Decode loop" },
  { name: "Hidden case 2", public: false, stage: "Stage 2: Decode loop" },
  { name: "Hidden case 3", public: false, stage: "Stage 2: Decode loop" },
  { name: "Hidden case 4", public: false, stage: "3: Serving" },
  { name: "Hidden case 5", public: false, stage: "3: Serving" },
];

function fillerProblems(track: FixtureTrack, startOrder: number): FixtureProblem[] {
  const python = PYTHON_TRACKS.has(track.id) || track.id === "capstones";
  const frameworks: Framework[] = track.id === "mlx-gpu" ? ["mlx"] : python ? ["python"] : TENSOR_FRAMEWORKS;
  const requires: Requires = track.id === "mlx-gpu" ? "mlx-gpu" : "cpu";
  const judgePool = TRACK_JUDGES[track.id] ?? ["numeric"];
  const solvedShare = SOLVED_SHARE[track.order - 1] ?? 0;

  return track.filler.map((title, index) => {
    const id = slug(title);
    const h = hash(id);
    const position = (index + h * 0.6) / track.filler.length;
    const difficulty: Difficulty =
      track.id === "capstones" ? "hard" : position < 0.34 ? "easy" : position < 0.8 ? "medium" : "hard";
    const first = judgePool[Math.floor(h * judgePool.length)] ?? "numeric";
    const second = judgePool[Math.floor(hash(`${id}:2`) * judgePool.length)] ?? first;
    const judges = second !== first && hash(`${id}:j`) < 0.35 ? [first, second] : [first];
    const roll = hash(`${id}:status`);
    // The first capstone is under way, so the fixtures hold a stored submission with stages.
    const seedStatus: ProblemStatus =
      track.id === "capstones" && index === 0
        ? "attempted"
        : roll < solvedShare
          ? "solved"
          : roll < solvedShare + 0.07
            ? "attempted"
            : "todo";
    const staged = track.id === "capstones";
    const entry = id.replace(/-/g, "_").slice(0, 28);
    const lead = frameworks[0] ?? "torch";

    return {
      meta: {
        id,
        title,
        track: track.id,
        order: startOrder + (index + 1) * 10,
        difficulty,
        frameworks,
        judges,
        tags: [track.id],
        prerequisites: [],
        requires,
        assets: [],
      },
      statement: fillerStatement(title, track, entry),
      hints: [
        "Work a three-element example by hand before writing code.",
        "The reference is a single pass; look for the quantity you can carry forward.",
      ],
      editorial: `The reference for *${title}* is a single pass over the input. This editorial is a fixture placeholder.`,
      entry: [entry],
      timeLimitS: 10,
      assets: {},
      cases: staged
        ? CAPSTONE_CASES
        : [
            { name: "small input", public: true },
            { name: "typical input", public: true },
            { name: "edge case", public: true },
            ...Array.from({ length: 4 }, (_, i) => ({ name: `Hidden case ${i + 1}`, public: false })),
          ],
      starterTemplate: `{{import}}\n\n\ndef ${entry}(x):\n    """Return the result described in the statement."""\n    raise NotImplementedError\n`,
      referenceTemplate: `{{import}}\n\n\ndef ${entry}(x):\n    return x\n`,
      savedTemplate: seedStatus === "todo" ? null : `{{import}}\n\n\ndef ${entry}(x):\n    return x\n`,
      scenario: seedStatus === "attempted" ? "wrong_answer" : "accepted",
      seedStatus,
      seedSolved: seedStatus === "solved" ? [lead] : [],
    };
  });
}

function authoredProblem(spec: Authored): FixtureProblem {
  const saved = file(spec.id, "solution.py");
  return {
    meta: {
      id: spec.id,
      title: spec.title,
      track: spec.track,
      order: spec.order,
      difficulty: spec.difficulty,
      frameworks: spec.frameworks,
      judges: spec.judges,
      tags: spec.tags,
      prerequisites: spec.prerequisites ?? [],
      requires: spec.requires ?? "cpu",
      assets: spec.models ?? [],
    },
    statement: file(spec.id, "statement.md") ?? "",
    hints: (file(spec.id, "hints.md") ?? "")
      .split(/^---$/m)
      .map((hint) => hint.trim())
      .filter(Boolean),
    editorial: file(spec.id, "editorial.md") ?? "",
    entry: spec.entry,
    timeLimitS: spec.timeLimitS ?? 10,
    assets: assetsOf(spec.id),
    cases: [
      ...spec.publicCases.map((name) => ({ name, public: true })),
      ...Array.from({ length: spec.hiddenCases }, (_, i) => ({
        name: `Hidden case ${i + 1}`,
        public: false,
      })),
    ].map((fixtureCase, index): FixtureCase => {
      const needs = spec.needs?.[index];
      return needs ? { ...fixtureCase, needs } : fixtureCase;
    }),
    starterTemplate: file(spec.id, "starter.py") ?? "",
    referenceTemplate: file(spec.id, "reference.py") ?? "",
    savedTemplate: saved,
    scenario: spec.scenario ?? "accepted",
    seedStatus: spec.status,
    seedSolved: spec.solved ?? [],
  };
}

const IMPORTS: Record<Framework, { line: string; tensor: string; ns: string }> = {
  torch: { line: "import torch", tensor: "torch.Tensor", ns: "torch." },
  jax: { line: "import jax.numpy as jnp", tensor: "jnp.ndarray", ns: "jnp." },
  mlx: { line: "import mlx.core as mx", tensor: "mx.array", ns: "mx." },
  python: { line: "", tensor: "list[float]", ns: "" },
};

/** Renders a code template for one framework. Good enough for fixtures; not a transpiler. */
export function renderTemplate(template: string, framework: Framework): string {
  const spec = IMPORTS[framework];
  let code = template.replace(/\{\{import\}\}\n?/g, spec.line ? `${spec.line}\n` : "");
  code = code.replace(/\{\{T\}\}/g, spec.tensor);
  if (framework !== "torch" && spec.ns) code = code.replace(/\btorch\./g, spec.ns);
  return code.replace(/^\n+/, "");
}

export function buildCatalog(): { tracks: FixtureTrack[]; problems: FixtureProblem[] } {
  const problems: FixtureProblem[] = [];
  for (const track of FIXTURE_TRACKS) {
    const authored = AUTHORED.filter((spec) => spec.track === track.id).map(authoredProblem);
    const lastOrder = authored.reduce((max, problem) => Math.max(max, problem.meta.order), 0);
    problems.push(...authored, ...fillerProblems(track, lastOrder));
  }
  return { tracks: FIXTURE_TRACKS, problems };
}
