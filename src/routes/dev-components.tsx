import { createLazyRoute } from "@tanstack/react-router";
import { cn } from "cn";
import { Lock, Play } from "lucide-react";
import { type ReactNode, useState } from "react";
import { ActionButton } from "@/components/app/action-button";
import { AppLink } from "@/components/app/app-link";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { CopyButton } from "@/components/app/copy-button";
import { DisclosureChevron } from "@/components/app/disclosure-chevron";
import { IconButton } from "@/components/app/icon-button";
import { InlineButton } from "@/components/app/inline-button";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import { Notice } from "@/components/app/notice";
import { Page, PageHeader, ScreenMessage, Section } from "@/components/app/page";
import { RuledList } from "@/components/app/ruled-list";
import { SHORTCUTS, ShortcutHint } from "@/components/app/shortcut-hint";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { CodeEditor } from "@/components/editor/code-editor";
import { ContinueEntry } from "@/components/home/continue-entry";
import { GettingStarted } from "@/components/home/getting-started";
import { PathProgress } from "@/components/home/path-progress";
import { ProgressSummary } from "@/components/home/progress-summary";
import { TrackProgress } from "@/components/home/track-progress";
import { FilterSelect } from "@/components/library/filter-select";
import { LibraryFilterBar } from "@/components/library/library-filter-bar";
import { TrackList } from "@/components/library/track-list";
import { TrackSection } from "@/components/library/track-section";
import { AssetDownloadButton, RemoveAssetButton } from "@/components/models/asset-actions";
import { AssetNotice } from "@/components/models/asset-notice";
import { AssetProgress } from "@/components/models/asset-progress";
import { AssetRow } from "@/components/models/asset-row";
import { AssetStateMark } from "@/components/models/asset-state-mark";
import { ModelsPanel } from "@/components/models/models-panel";
import { PathEntryList } from "@/components/path/path-entry-list";
import { PathStage } from "@/components/path/path-stage";
import { PathSummary } from "@/components/path/path-summary";
import { DifficultyMark } from "@/components/problem/difficulty-mark";
import { EditorialPanel } from "@/components/problem/editorial-panel";
import { FrameworkList } from "@/components/problem/framework-list";
import { FrameworkSwitcher } from "@/components/problem/framework-switcher";
import { HintList } from "@/components/problem/hint-list";
import { JudgeList } from "@/components/problem/judge-list";
import { PrerequisiteList } from "@/components/problem/prerequisite-list";
import { PrerequisiteNote } from "@/components/problem/prerequisite-note";
import { ProblemHeader } from "@/components/problem/problem-header";
import { ProblemTable } from "@/components/problem/problem-table";
import { ProgressRule } from "@/components/problem/progress-rule";
import { StatusMark } from "@/components/problem/status-mark";
import { SubmissionList } from "@/components/problem/submission-list";
import { SubmissionView } from "@/components/problem/submission-view";
import { CodeBlock } from "@/components/prose/code-block";
import { Figure, SvgFigureBody } from "@/components/prose/figure";
import { Prose } from "@/components/prose/prose";
import { CaseList } from "@/components/results/case-list";
import { CaseMark } from "@/components/results/case-mark";
import { MetricList } from "@/components/results/metric-list";
import { OutputBlock } from "@/components/results/output-block";
import { ResultsPanel } from "@/components/results/results-panel";
import { RunProgress } from "@/components/results/run-progress";
import { VerdictHeader } from "@/components/results/verdict-header";
import { VerdictMark } from "@/components/results/verdict-mark";
import { ReviewDueNotice } from "@/components/review/review-due-notice";
import { ReviewHidden } from "@/components/review/review-hidden";
import { ReviewList } from "@/components/review/review-list";
import { ReviewOutcomeNote } from "@/components/review/review-outcome";
import { ReviewStrip } from "@/components/review/review-strip";
import { UpcomingReviews } from "@/components/review/upcoming-reviews";
import { CatalogIssueList } from "@/components/settings/catalog-issue-list";
import { ContentPanel } from "@/components/settings/content-panel";
import { EnvironmentPanel } from "@/components/settings/environment-panel";
import { InstallButton, UvGuideButton } from "@/components/settings/install-actions";
import { PathField } from "@/components/settings/path-field";
import { ReviewSetting } from "@/components/settings/review-setting";
import { SuggestionsPanel } from "@/components/settings/suggestions-panel";
import { ThemePicker } from "@/components/settings/theme-picker";
import { AddTargetForm } from "@/components/targets/add-target-form";
import { CheckStepList } from "@/components/targets/check-step-list";
import { CheckStepMark } from "@/components/targets/check-step-mark";
import { FixText } from "@/components/targets/fix-text";
import { TargetPicker } from "@/components/targets/target-picker";
import { TargetRow } from "@/components/targets/target-row";
import { TargetSummary } from "@/components/targets/target-summary";
import { TargetsPanel } from "@/components/targets/targets-panel";
import { Button } from "@/components/ui/button";
import { EnvironmentNotice } from "@/components/workbench/environment-notice";
import { HardwareNotice } from "@/components/workbench/hardware-notice";
import { PaneToolbar } from "@/components/workbench/pane-toolbar";
import { SaveIndicator } from "@/components/workbench/save-indicator";
import { SuggestionsStatus } from "@/components/workbench/suggestions-status";
import {
  type AssetList,
  type CaseResult,
  DIFFICULTIES,
  type EditorStatus,
  type Framework,
  type JudgeKind,
  type Path,
  type ProblemDetail,
  type ProblemSummary,
  type Review,
  STATUSES,
  type Submission,
  type SubmissionDetail,
  type TargetInfo,
  type Track,
  type Verdict,
} from "@/lib/api";
import type { ActiveDownload, AssetDownload } from "@/lib/api/asset-fetch-state";
import { FIXTURE_SSH_HOSTS, fixtureTargets } from "@/lib/api/fixture/targets";
import { initialRunState, type RunState } from "@/lib/api/run-state";
import { stepProgress, type TargetCheckState } from "@/lib/api/target-check-state";
import type { AssetFetchController } from "@/lib/api/use-asset-fetch";
import type { EditorInstallState } from "@/lib/api/use-editor-install";
import type { InstallState } from "@/lib/api/use-env-install";
import { EMPTY_FILTERS, type LibraryFilters } from "@/lib/filters";
import { nextOnPath, pathEntries, unmetPrerequisites } from "@/lib/paths";
import { dueReviews, reviewEntries } from "@/lib/reviews";
import { routes } from "@/lib/routes";
import type { ThemePreference } from "@/stores/theme";

/** Sample data. Kept inline so this page has no dependency on any backend. */
const summary = (over: Partial<ProblemSummary>): ProblemSummary => ({
  id: "scaled-dot-product-attention",
  title: "Scaled Dot-Product Attention",
  track: "attention",
  order: 10,
  difficulty: "medium",
  frameworks: ["torch", "jax", "mlx"],
  judges: ["numeric"],
  tags: ["attention"],
  prerequisites: [],
  requires: "cpu",
  status: "todo",
  solvedFrameworks: [],
  assets: [],
  ...over,
});

const PROBLEMS: ProblemSummary[] = [
  summary({
    id: "a",
    title: "Numerically Stable Softmax",
    difficulty: "easy",
    status: "solved",
    solvedFrameworks: ["torch", "jax"],
  }),
  summary({}),
  summary({
    id: "c",
    order: 30,
    title: "Rotary Position Embeddings",
    judges: ["numeric", "property", "speed"],
    status: "attempted",
  }),
  summary({
    id: "d",
    order: 40,
    title:
      "A Deliberately Long Problem Title That Has to Truncate Somewhere Before It Reaches the Difficulty Column",
    difficulty: "hard",
    frameworks: ["python"],
    judges: ["simclock", "service"],
  }),
];

/** A short path over the sample problems: one solved, one up next, one that builds on an unsolved problem. */
const PATH_PROBLEMS: ProblemSummary[] = [
  PROBLEMS[0] as ProblemSummary,
  PROBLEMS[1] as ProblemSummary,
  { ...(PROBLEMS[2] as ProblemSummary), prerequisites: ["scaled-dot-product-attention"] },
  PROBLEMS[3] as ProblemSummary,
];
const PATH: Path = {
  id: "inference-75",
  title: "Inference 75",
  description: "The shortest route through LLM inference, from one layer's arithmetic to serving many users.",
  stages: [
    {
      title: "Numerics and the forward pass",
      description: "The functions a decoder runs for every token, written from their formulas.",
      problems: ["a", "scaled-dot-product-attention", "c"],
    },
    {
      title: "Serving many requests",
      description: "The model as a service: batching under a latency budget and streaming tokens.",
      problems: ["d"],
    },
  ],
  problemCount: 4,
  solvedCount: 1,
};
const PATH_ENTRIES = pathEntries(PATH, PATH_PROBLEMS);
const PATH_NEXT = nextOnPath(PATH_ENTRIES);
const PATH_UNMET = unmetPrerequisites(PATH_PROBLEMS);

/** A small figure in the authoring conventions: currentColor ink, the palette through style, text at 13 units on a 480-unit drawing. */
const SAMPLE_FIGURE = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 96">
  <g fill="none" stroke-width="1" style="stroke: var(--figure-faint)">
    <line x1="12" y1="70" x2="468" y2="70"/>
  </g>
  <g fill="none" stroke-width="1.5" stroke-linecap="round">
    <path d="M12 70 C 100 70, 130 14, 240 14 S 380 70, 468 70" stroke="currentColor"/>
    <path d="M12 70 C 150 70, 180 40, 240 40 S 330 70, 468 70" style="stroke: var(--figure-accent)"/>
    <path d="M240 14 V 70" stroke-dasharray="2 4" style="stroke: var(--figure-straw)"/>
  </g>
  <g font-size="13" style="font-family: var(--figure-serif)">
    <text x="12" y="14">ink: currentColor</text>
    <text x="12" y="32" style="fill: var(--figure-accent)">accent</text>
    <text x="12" y="50" style="fill: var(--figure-straw)">straw</text>
    <text x="12" y="90" style="fill: var(--figure-muted)">muted, on a faint axis</text>
  </g>
  <script>window.figureScriptRan = true</script>
</svg>`;

const TRACKS: Track[] = [
  {
    id: "foundations",
    title: "Numerical Foundations",
    description: "Float32 arithmetic, stable reductions, and the normalizations every layer depends on.",
    order: 1,
    problemCount: 14,
    solvedCount: 11,
  },
  {
    id: "attention",
    title: "Attention",
    description: "Scaled dot-product attention, masks, and the shapes that go wrong.",
    order: 2,
    problemCount: 15,
    solvedCount: 4,
  },
  { id: "serving", title: "Serving", description: "", order: 3, problemCount: 14, solvedCount: 0 },
];

const DETAIL: ProblemDetail = {
  ...summary({ prerequisites: ["a"], judges: ["numeric", "property"] }),
  statement: "",
  hints: [
    "Write out the shapes of every intermediate before writing code.",
    "A boolean mask becomes additive with a `where`. Apply it **before** the softmax.",
    "`scores = q @ k.transpose(-2, -1) / sqrt(d)`, mask, softmax, then `weights @ v`.",
  ],
  editorial: null,
  editorialUnlocked: false,
  entry: ["attention"],
  timeLimitS: 10,
};

const SAMPLE_MARKDOWN = String.raw`Softmax turns logits $z \in \mathbb{R}^V$ into probabilities, and is invariant to a shift $c$:

$$
\operatorname{softmax}(z)_i = \frac{e^{z_i - c}}{\sum_{j=1}^{V} e^{z_j - c}}
$$

## Interface

Call it as ${"`softmax(logits, dim)`"}. It must not overflow.

### Example 1

${"```"}python
softmax(torch.tensor([1000.0, 1001.0, 1002.0]))
# tensor([0.0900, 0.2447, 0.6652])
${"```"}

| Case | Shape | Tolerance |
|---|---|---:|
| small vector | ${"`(3,)`"} | $10^{-6}$ |
| batch, last dim | ${"`(8, 512)`"} | $10^{-6}$ |

> [!NOTE]
> A row that is entirely $-\infty$ has no valid distribution.

> [!WARNING]
> Scale by the key dimension, not the value dimension.

> [!TIP]
> Build the angles as one outer product.

1. Subtract the maximum.
2. Exponentiate and normalize.

- Inputs are float32.
- $1 \le V \le 65{,}536$.

> An ordinary quotation, for comparison with the callouts.
`;

const EDITOR_STATUS: EditorStatus = {
  enabled: true,
  installed: true,
  installing: false,
  serverVersion: "0.0.104",
  state: null,
};
const IDLE_EDITOR_INSTALL: EditorInstallState = { status: "idle", log: [], message: null };

const SAMPLE_CODE = `import torch


def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return softmax along \`dim\`."""
    shifted = logits - logits.max(dim=dim, keepdim=True).values  # shift by the max
    weights = torch.exp(shifted)
    return weights / weights.sum(dim=dim, keepdim=True)
`;

const TRACEBACK = `Traceback (most recent call last):
  File "learnai/harness/runner.py", line 212, in run_case
    outcome = case.check(solution)
  File "workspace/scaled-dot-product-attention/torch.py", line 9, in attention
    weights = scores.softmax(dim=-1) @ v
RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x512 and 64x64)`;

const result = (id: string, over: Partial<CaseResult> = {}): CaseResult => ({
  id,
  status: "pass",
  durationMs: 12.3,
  message: null,
  detail: null,
  stdout: "",
  metrics: [],
  ...over,
});

const CASES = [
  {
    info: { id: "c1", name: "single head, no mask", public: true },
    state: "done" as const,
    result: result("c1"),
  },
  {
    info: { id: "c2", name: "single head, causal mask", public: true },
    state: "done" as const,
    result: result("c2", {
      status: "fail",
      message: "Max absolute error 1.2e-04 exceeds tolerance 1.0e-06",
      detail: "expected: tensor([0.0900, 0.2447, 0.6652])\n     got: tensor([0.0900, 0.2446, 0.6653])",
      stdout: "scores shape: torch.Size([1, 1, 2, 2])\n",
      metrics: [{ name: "model_tokens", value: 41, unit: "tokens", limit: 36 }],
    }),
  },
  {
    info: { id: "c3", name: "batched heads, d = 64", public: true },
    state: "done" as const,
    result: result("c3", {
      status: "error",
      message: "RuntimeError: shapes cannot be multiplied",
      detail: TRACEBACK,
      durationMs: 0.4,
    }),
  },
  {
    info: { id: "c4", name: "Hidden case 1", public: false },
    state: "done" as const,
    result: result("c4", { status: "timeout", message: "Exceeded the 10 s time limit", durationMs: 10000 }),
  },
  {
    info: { id: "c5", name: "Hidden case 2", public: false },
    state: "done" as const,
    result: result("c5", { status: "skipped", durationMs: 0 }),
  },
];

/** A submit on a machine without the model: two cases ran, two were skipped for it. */
const SKIPPED_FOR_ASSET = [
  CASES[0],
  {
    info: { id: "c2", name: "layers.0.mlp.fc1 of the checkpoint", public: true },
    state: "done" as const,
    result: result("c2", {
      status: "skipped",
      durationMs: 0,
      message:
        "skipped: model asset `granite-4.2-3b-mlx-8bit` (lmstudio-community/granite-4.2-3b-MLX-8bit, about 3.9 GB) is not downloaded; fetch it with `learnai assets fetch granite-4.2-3b-mlx-8bit`",
      skip: { kind: "asset", asset: "granite-4.2-3b-mlx-8bit" },
    }),
  },
  {
    info: { id: "c3", name: "Hidden case 1", public: false },
    state: "done" as const,
    result: result("c3", { durationMs: 3.1 }),
  },
  {
    info: { id: "c4", name: "Hidden case 2", public: false },
    state: "done" as const,
    result: result("c4", {
      status: "skipped",
      durationMs: 0,
      message:
        "skipped: model asset `granite-4.2-3b-mlx-8bit` (lmstudio-community/granite-4.2-3b-MLX-8bit, about 3.9 GB) is not downloaded; fetch it with `learnai assets fetch granite-4.2-3b-mlx-8bit`",
      skip: { kind: "asset", asset: "granite-4.2-3b-mlx-8bit" },
    }),
  },
].flatMap((c) => (c ? [c] : []));

/** A submit on a machine without the GPU the problem requires. */
const SKIPPED_FOR_HARDWARE = [
  CASES[0],
  {
    info: { id: "c2", name: "Hidden case 1", public: false },
    state: "done" as const,
    result: result("c2", { durationMs: 0.2 }),
  },
  {
    info: { id: "c3", name: "Hidden case 2", public: false, requires: "cuda" as const },
    state: "done" as const,
    result: result("c3", {
      status: "skipped",
      durationMs: 0,
      message: "skipped: needs an NVIDIA GPU (CUDA), which this machine does not have",
      skip: { kind: "hardware", requires: "cuda" },
    }),
  },
].flatMap((c) => (c ? [c] : []));

const RUNNING_CASES = [
  CASES[0],
  {
    info: { id: "c2", name: "single head, causal mask", public: true },
    state: "running" as const,
    result: null,
  },
  {
    info: { id: "c3", name: "batched heads, d = 64", public: true },
    state: "pending" as const,
    result: null,
  },
].flatMap((c) => (c ? [c] : []));

const run = (over: Partial<RunState>): RunState => ({
  ...initialRunState,
  mode: "submit",
  framework: "torch",
  ...over,
});
const ENV = { python: "3.15.2", torch: "2.15.0", device: "mps" };

const RUN_STATES: { name: string; state: RunState }[] = [
  { name: "Idle", state: initialRunState },
  {
    name: "Preparing",
    state: run({ phase: "preparing", preparingMessage: "Checking the Python environment" }),
  },
  { name: "Running", state: run({ phase: "running", cases: RUNNING_CASES, total: 3, passed: 1, env: ENV }) },
  {
    name: "Canceling",
    state: run({ phase: "running", cases: RUNNING_CASES, total: 3, passed: 1, canceling: true }),
  },
  {
    name: "Accepted, with metrics",
    state: run({
      phase: "done",
      verdict: "accepted",
      passed: 1,
      total: 1,
      durationMs: 84.2,
      env: ENV,
      submissionId: 12,
      cases: CASES.slice(0, 1),
      metrics: [
        { name: "model_tokens", value: 36, unit: "tokens", limit: 36 },
        { name: "slowdown", value: 1.08, unit: "x", limit: 1.5 },
      ],
    }),
  },
  {
    name: "Wrong answer, mixed cases",
    state: run({
      phase: "done",
      verdict: "wrong_answer",
      passed: 1,
      total: 5,
      durationMs: 10024.7,
      env: ENV,
      cases: CASES,
    }),
  },
  {
    name: "Not judged, model not downloaded (with the download offered)",
    state: run({
      phase: "done",
      verdict: "assets_missing",
      framework: "mlx",
      passed: 2,
      total: 4,
      skipped: 2,
      durationMs: 223.8,
      env: { python: "3.15.2", mlx: "0.32.4", device: "gpu" },
      submissionId: 14,
      requires: "mlx-gpu",
      requiresMet: true,
      cases: SKIPPED_FOR_ASSET,
      missingAssets: [
        {
          name: "granite-4.2-3b-mlx-8bit",
          repo: "lmstudio-community/granite-4.2-3b-MLX-8bit",
          sizeMb: 3900,
        },
      ],
      message:
        "2 of 2 cases that could run passed; 2 of 4 cases were skipped because a model asset is not downloaded: `granite-4.2-3b-mlx-8bit` (lmstudio-community/granite-4.2-3b-MLX-8bit, about 3.9 GB). Fetch with `learnai assets fetch granite-4.2-3b-mlx-8bit`, then run again.",
    }),
  },
  {
    name: "Not judged, hardware missing",
    state: run({
      phase: "done",
      verdict: "hardware_missing",
      passed: 2,
      total: 3,
      skipped: 1,
      durationMs: 820.3,
      env: { python: "3.15.2", torch: "2.15.0", device: "cpu" },
      submissionId: 15,
      requires: "cuda",
      requiresMet: false,
      missingHardware: "cuda",
      cases: SKIPPED_FOR_HARDWARE,
      message:
        "2 of 2 cases that could run passed; 1 of 3 cases were skipped because they need an NVIDIA GPU (CUDA), which this machine does not have. The solution can only be accepted on a machine that runs every case.",
    }),
  },
  {
    name: "Canceled",
    state: run({
      phase: "done",
      verdict: "canceled",
      passed: 1,
      total: 3,
      cases: RUNNING_CASES.map((c) => ({
        ...c,
        state: c.state === "running" ? ("pending" as const) : c.state,
      })),
    }),
  },
  {
    name: "Import error",
    state: run({
      phase: "done",
      verdict: "import_error",
      total: 3,
      env: ENV,
      logs: [{ stream: "stderr", text: "ModuleNotFoundError: No module named 'einops'" }],
    }),
  },
  {
    name: "Judge failure",
    state: run({
      phase: "done",
      verdict: "harness_error",
      logs: [
        { stream: "stderr", text: "error: failed to spawn `uv`: No such file or directory (os error 2)" },
      ],
    }),
  },
  {
    name: "Could not start",
    state: run({
      phase: "done",
      verdict: "harness_error",
      startError: "Cannot reach the devbridge at http://127.0.0.1:1421. Is it running?",
    }),
  },
];

const VERDICTS: Verdict[] = [
  "accepted",
  "wrong_answer",
  "runtime_error",
  "time_limit",
  "import_error",
  "harness_error",
  "canceled",
  "assets_missing",
  "hardware_missing",
];
const ALL_JUDGES: JudgeKind[] = [
  "numeric",
  "property",
  "statistical",
  "cost",
  "speed",
  "simclock",
  "service",
];

const SUBMISSION_VERDICTS: Verdict[] = [
  "accepted",
  "wrong_answer",
  "runtime_error",
  "time_limit",
  "assets_missing",
  "hardware_missing",
];
const SUBMISSIONS: Submission[] = SUBMISSION_VERDICTS.map((verdict, index) => ({
  id: index + 1,
  problemId: index % 2 === 0 ? "a" : "d",
  framework: (["torch", "jax", "mlx", "python", "mlx", "torch"] as Framework[])[index] ?? "torch",
  mode: "submit",
  verdict,
  passed: verdict === "accepted" ? 7 : index,
  total: 7,
  skipped: verdict === "assets_missing" ? 3 : verdict === "hardware_missing" ? 2 : 0,
  durationMs: [84.2, 31.5, 0.4, 10024, 223.8, 820.3][index] ?? 1,
  target: "local",
  createdAt: new Date(
    Date.now() - ([5, 190, 60 * 26, 60 * 24 * 9, 60 * 24 * 11, 60 * 24 * 12][index] ?? 5) * 60_000,
  ).toISOString(),
  // The newest one is a review attempt.
  review: index === 0,
}));

/** Review schedules: two due (one long overdue, on its last review), two still to come. */
const REVIEW_NOW = new Date("2026-11-15T10:00:00");
const review = (problemId: string, stage: number, dueInDays: number): Review => ({
  problemId,
  stage,
  intervalDays: [3, 7, 21, 60, 180][stage] ?? 180,
  dueAt: new Date(REVIEW_NOW.getTime() + dueInDays * 86_400_000 - 3_600_000).toISOString(),
  lastReviewedAt: stage === 0 ? null : "2026-11-01T09:00:00.000Z",
  due: dueInDays <= 0,
});
const REVIEWS: Review[] = [review("a", 4, -3), review("d", 1, 0), review("b", 0, 2), review("c", 2, 19)];

/** A capstone's cases in stages, part-way through a run: stage one done, stage two running. */
const stagedCase = (
  id: string,
  name: string,
  stage: string,
  state: "pending" | "running" | "done",
  over: Partial<CaseResult> = {},
  isPublic = true,
) => ({
  info: { id, name, public: isPublic, stage },
  state,
  result: state === "done" ? result(id, over) : null,
});
const STAGED_RUNNING = [
  stagedCase("s1", "round-trips ASCII", "1: Tokenizer", "done"),
  stagedCase("s2", "round-trips multi-byte text", "1: Tokenizer", "done"),
  stagedCase("s3", "Hidden case 1", "1: Tokenizer", "done", {}, false),
  stagedCase("s4", "greedy decode matches the reference", "Stage 2: Decode loop", "done"),
  stagedCase("s5", "Hidden case 2", "Stage 2: Decode loop", "running", {}, false),
  stagedCase("s6", "Hidden case 3", "Stage 2: Decode loop", "pending", {}, false),
  stagedCase("s7", "Hidden case 4", "3: Serving", "pending", {}, false),
  stagedCase("s8", "Hidden case 5", "3: Serving", "pending", {}, false),
];
/** The same capstone finished: a failure in stage two, a case skipped for hardware in stage three. */
const STAGED_ENDED = [
  ...STAGED_RUNNING.slice(0, 4),
  stagedCase(
    "s5",
    "Hidden case 2",
    "Stage 2: Decode loop",
    "done",
    { status: "fail", message: "Output does not match the reference" },
    false,
  ),
  stagedCase("s6", "Hidden case 3", "Stage 2: Decode loop", "done", {}, false),
  stagedCase("s7", "Hidden case 4", "3: Serving", "done", {}, false),
  stagedCase(
    "s8",
    "Hidden case 5",
    "3: Serving",
    "done",
    {
      status: "skipped",
      durationMs: 0,
      message: "skipped: needs an NVIDIA GPU (CUDA), which this machine does not have",
      skip: { kind: "hardware", requires: "cuda" },
    },
    false,
  ),
];

/** A stored submission that was not judged on every case: history keeps the skips, not the run's message. */
const NOT_JUDGED_SUBMISSION: SubmissionDetail = {
  ...(SUBMISSIONS[4] as Submission),
  passed: 2,
  total: 4,
  skipped: 2,
  code: SAMPLE_CODE,
  cases: SKIPPED_FOR_ASSET.map((c) => ({ ...c.result, name: c.info.name, public: c.info.public })),
  metrics: [],
};

/** A stored submission of a staged problem: the stages come back from history as they ran. */
const STAGED_SUBMISSION: SubmissionDetail = {
  ...(SUBMISSIONS[1] as Submission),
  passed: 6,
  total: 8,
  skipped: 1,
  code: SAMPLE_CODE,
  cases: STAGED_ENDED.flatMap((c) =>
    c.result ? [{ ...c.result, name: c.info.name, public: c.info.public, stage: c.info.stage }] : [],
  ),
  metrics: [],
};

const MODELS: AssetList = {
  modelsDir: "/Users/ada/Developer/learnai/.learnai/models",
  assets: [
    {
      name: "granite-4.0-1b",
      repo: "ibm-granite/granite-4.0-1b",
      revision: "5d0c3bbf0c1f4c7e9a2f3f0a8f0d6f1f8b1c2d3e",
      files: ["config.json", "tokenizer.json", "model.safetensors"],
      sizeMb: 1630,
      state: "incomplete",
      reason: "not downloaded yet: model.safetensors",
      path: "/Users/ada/Developer/learnai/.learnai/models/granite-4.0-1b",
      bytesOnDisk: 7_156_949,
      problems: ["c"],
    },
    {
      name: "granite-4.0-350m",
      repo: "ibm-granite/granite-4.0-350m",
      revision: "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2",
      files: ["config.json", "tokenizer.json", "model.safetensors"],
      sizeMb: 712,
      state: "present",
      reason: null,
      path: "/Users/ada/Developer/learnai/.learnai/models/granite-4.0-350m",
      bytesOnDisk: 711_966_212,
      problems: ["a", "scaled-dot-product-attention"],
    },
    {
      name: "granite-4.2-3b-mlx-8bit",
      repo: "lmstudio-community/granite-4.2-3b-MLX-8bit",
      revision: "7c630b526561db6b4641767c54595fe494176695",
      files: ["config.json", "tokenizer.json", "model.safetensors"],
      sizeMb: 3900,
      state: "missing",
      reason: null,
      path: "/Users/ada/Developer/learnai/.learnai/models/granite-4.2-3b-mlx-8bit",
      bytesOnDisk: 0,
      problems: ["d"],
    },
  ],
};
const [MODEL_INCOMPLETE, MODEL_PRESENT, MODEL_MISSING] = MODELS.assets as [
  AssetList["assets"][number],
  AssetList["assets"][number],
  AssetList["assets"][number],
];
const MODEL_TITLES = new Map(PROBLEMS.map((problem) => [problem.id, problem.title]));

const PREPARING: ActiveDownload = { phase: "preparing", message: "Checking the Python environment" };
const DOWNLOADING: ActiveDownload = {
  phase: "downloading",
  doneBytes: 1_184_302_080,
  totalBytes: 3_897_310_556,
  file: "model.safetensors",
};
const DOWNLOAD_FAILED: AssetDownload = {
  phase: "failed",
  message:
    "model.safetensors: the connection was reset; fetch again to resume from the bytes already on disk",
};
const NO_DOWNLOADS: AssetFetchController = {
  downloads: {},
  busy: false,
  canceling: false,
  download: () => {},
  cancel: () => {},
  dismiss: () => {},
};
const noop = () => {};

const SUBMISSION_DETAIL: SubmissionDetail = {
  ...(SUBMISSIONS[1] as Submission),
  code: SAMPLE_CODE,
  cases: CASES.map((c) => ({ ...c.result, name: c.info.name, public: c.info.public })),
  metrics: [],
};

const IDLE_INSTALL: InstallState = { framework: null, status: "idle", log: [], message: null };

const TARGETS = fixtureTargets(new Date());
const [TARGET_LOCAL, TARGET_HEALTHY, TARGET_FAILING] = TARGETS as [TargetInfo, TargetInfo, TargetInfo];
const TARGET_NEW: TargetInfo = {
  name: "ssh:lab-a100",
  kind: "ssh",
  label: "lab-a100",
  host: "lab-a100",
  remoteDir: "~/.learnai",
  lastCheck: null,
};
const CHECK_RUNNING: TargetCheckState = {
  running: true,
  steps: (TARGET_HEALTHY.lastCheck?.steps ?? []).slice(0, 3),
  log: [
    "$ /home/ada/.local/bin/uv sync --inexact --project /home/ada/.learnai/root/python",
    "Resolved 61 packages in 12ms",
  ],
  result: null,
  error: null,
};

function Specimen({ name, children }: { name: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2 pb-6">
      <p className="text-xs text-faint">{name}</p>
      <div>{children}</div>
    </div>
  );
}

/** Development only: every composed component, in every state it has. */
function DevComponentsScreen() {
  const [filters, setFilters] = useState<LibraryFilters>(EMPTY_FILTERS);
  const [track, setTrack] = useState<string | null>("attention");
  const [framework, setFramework] = useState<Framework>("torch");
  const [hints, setHints] = useState(1);
  const [theme, setTheme] = useState<ThemePreference>("system");
  const [code, setCode] = useState(SAMPLE_CODE);
  const [judge, setJudge] = useState<JudgeKind | null>(null);
  const [pickedTarget, setPickedTarget] = useState("local");
  const [reviewsOn, setReviewsOn] = useState(true);

  return (
    <Page>
      <PageHeader
        title="Components"
        description="Every composed component in its states. Screens are built only from these and the primitives in components/ui."
        actions={
          <Button variant="outline" size="sm">
            Header action
          </Button>
        }
      />

      <Section title="Indicators" aside="problem/, results/">
        <Specimen name="DifficultyMark">
          <div className="flex gap-8 text-xs">
            {DIFFICULTIES.map((d) => (
              <DifficultyMark key={d} difficulty={d} />
            ))}
            <DifficultyMark difficulty="hard" showLabel={false} />
          </div>
        </Specimen>
        <Specimen name="StatusMark">
          <div className="flex gap-8 text-xs">
            {STATUSES.map((s) => (
              <StatusMark key={s} status={s} showLabel />
            ))}
          </div>
        </Specimen>
        <Specimen name="VerdictMark">
          <div className="flex flex-wrap gap-x-8 gap-y-2 text-xs">
            {VERDICTS.map((v) => (
              <VerdictMark key={v} verdict={v} />
            ))}
          </div>
        </Specimen>
        <Specimen name="CaseMark">
          <div className="flex items-center gap-6">
            {(["pass", "fail", "error", "timeout", "skipped", "running", "pending", "not-run"] as const).map(
              (s) => (
                <CaseMark key={s} status={s} />
              ),
            )}
          </div>
        </Specimen>
        <Specimen name="JudgeList: plain, and with explanations">
          <div className="flex flex-col gap-2 text-xs">
            <JudgeList judges={ALL_JUDGES} />
            <JudgeList judges={ALL_JUDGES} explain />
          </div>
        </Specimen>
        <Specimen name="FrameworkList">
          <FrameworkList
            frameworks={["torch", "jax", "mlx", "python"]}
            solved={["torch"]}
            className="text-xs"
          />
        </Specimen>
        <Specimen name="ProgressRule">
          <div className="flex w-80 flex-col gap-3">
            <ProgressRule value={0} total={14} label="None" />
            <ProgressRule value={9} total={14} label="Some" />
            <ProgressRule value={14} total={14} label="All" />
          </div>
        </Specimen>
        <Specimen name="ShortcutHint, SaveIndicator">
          <div className="flex items-center gap-8">
            <ShortcutHint label="Run" keys={SHORTCUTS.run} />
            <ShortcutHint label="Submit" keys={SHORTCUTS.submit} />
            <ShortcutHint keys={SHORTCUTS.palette} />
            {(["saved", "unsaved", "saving", "error"] as const).map((s) => (
              <SaveIndicator key={s} state={s} />
            ))}
          </div>
        </Specimen>
      </Section>

      <Section title="Controls" aside="app/, problem/, library/, settings/">
        <Specimen name="FrameworkSwitcher: several, one not installed; single framework">
          <div className="flex items-center gap-8">
            <FrameworkSwitcher
              frameworks={["torch", "jax", "mlx"]}
              value={framework}
              onChange={setFramework}
              solved={["torch"]}
              env={[{ framework: "mlx", installed: false, version: null }]}
            />
            <FrameworkSwitcher frameworks={["python"]} value="python" onChange={() => {}} />
          </div>
        </Specimen>
        <Specimen name="ActionButton, IconButton, ConfirmDialog, AppLink">
          <div className="flex items-center gap-4">
            <ActionButton variant="outline" size="sm" hint="Run the public cases" shortcut={SHORTCUTS.run}>
              <Play data-icon="inline-start" />
              Run
            </ActionButton>
            <ActionButton size="sm" hint="Run every case and record the attempt" shortcut={SHORTCUTS.submit}>
              Submit
            </ActionButton>
            <ActionButton
              size="sm"
              hint="Unavailable while a run is in flight"
              shortcut={SHORTCUTS.submit}
              disabled
            >
              Submit
            </ActionButton>
            <IconButton label="Run the public cases" shortcut={SHORTCUTS.run}>
              <Play />
            </IconButton>
            <ConfirmDialog
              trigger={
                <Button variant="outline" size="sm">
                  Reset code
                </Button>
              }
              title="Reset to the starter code?"
              description="Your PyTorch solution for this problem will be replaced with the starter."
              confirmLabel="Reset code"
              destructive
              onConfirm={() => {}}
            />
            <AppLink to={routes.library()}>Text link</AppLink>
            <AppLink to={routes.library()} tone="quiet">
              Quiet link
            </AppLink>
          </div>
        </Specimen>
        <Specimen name="InlineButton (in a sentence), CopyButton, InstallButton (idle, installing), UvGuideButton">
          <div className="flex items-center gap-4">
            <p className="text-sm">
              A sentence with <InlineButton onClick={() => {}}>a button set as text</InlineButton> in it.
            </p>
            <CopyButton text="copied from the components page" label="Copy sample" />
            <InstallButton installing={false} onClick={() => {}}>
              Install JAX
            </InstallButton>
            <InstallButton installing disabled>
              Install JAX
            </InstallButton>
            <UvGuideButton />
          </div>
        </Specimen>
        <Specimen name="LibraryFilterBar, FilterSelect">
          <div className="flex flex-col gap-3">
            <LibraryFilterBar filters={filters} onChange={setFilters} summary="12 of 196 problems" />
            <FilterSelect
              label="Judge"
              anyLabel="Any judge"
              value={judge}
              onChange={setJudge}
              options={ALL_JUDGES.map((value) => ({ value, label: value }))}
            />
          </div>
        </Specimen>
        <Specimen name="TrackList">
          <div className="w-64">
            <TrackList tracks={TRACKS} selected={track} onSelect={setTrack} />
          </div>
        </Specimen>
        <Specimen name="TrackSection: all problems, and narrowed by a filter">
          <TrackSection track={TRACKS[1] as Track} problems={PROBLEMS} filtered={false} />
          <TrackSection track={TRACKS[0] as Track} problems={PROBLEMS.slice(0, 1)} filtered />
        </Specimen>
        <Specimen name="ThemePicker, PathField, ContentPanel">
          <div className="flex max-w-measure flex-col gap-3">
            <ThemePicker value={theme} onChange={setTheme} />
            <ContentPanel root="/Users/ada/Developer/learnai" issues={[]} />
            <PathField
              label="Content root"
              path="/Users/ada/Developer/learnai/a/very/long/path/that/keeps/going/and/going/until/it/truncates"
            />
          </div>
        </Specimen>
      </Section>

      <Section title="Prose and code" aside="prose/, editor/">
        <Specimen name="Prose: reading size">
          <Prose>{SAMPLE_MARKDOWN}</Prose>
        </Specimen>
        <Specimen name="Prose: compact">
          <Prose size="compact">{DETAIL.hints[1] ?? ""}</Prose>
        </Specimen>
        <Specimen name="Figure with SvgFigureBody: an author's SVG, sanitized and inlined (its script element is dropped); a figure whose file failed; not an SVG">
          <div className="flex max-w-measure flex-col gap-8">
            <Figure
              number={1}
              caption="Two curves and a guide, in the figure palette. The colors follow the theme because the SVG is part of the page."
            >
              <SvgFigureBody source={SAMPLE_FIGURE} label="Two curves and a guide in the figure palette" />
            </Figure>
            <Figure number={2} caption="The caption stands even when the drawing cannot be shown.">
              <Notice>The figure could not be loaded: missing file assets/attention-mask.svg</Notice>
            </Figure>
            <Figure number={3} caption="A file that does not parse as SVG.">
              <SvgFigureBody source="<svg><g></svg>" label="Broken" />
            </Figure>
          </div>
        </Specimen>
        <Specimen name="CodeBlock: plain, with line numbers, unknown language">
          <div className="flex max-w-measure flex-col gap-3">
            <CodeBlock code={SAMPLE_CODE} language="python" />
            <CodeBlock code={SAMPLE_CODE} language="python" lineNumbers />
            <CodeBlock code={'data: {"index": 0, "token": "The"}\n\ndata: [DONE]'} />
          </div>
        </Specimen>
        <Specimen name="CodeEditor: with suggestions, hover documentation, and argument hints (try torch.ex, then hover it)">
          <div className="h-56 max-w-measure border">
            <CodeEditor
              value={code}
              onChange={setCode}
              label="Sample editor"
              document={{ problemId: "component-gallery", framework: "torch" }}
            />
          </div>
        </Specimen>
        <Specimen name="SuggestionsStatus (editor toolbar): not installed, restarting, stopped, working (nothing); and in a narrow pane">
          <div className="flex flex-col gap-3">
            {(["max-w-none", "max-w-sm"] as const).map((width) => (
              <PaneToolbar key={width} className={cn("gap-8 border", width)}>
                <SuggestionsStatus
                  session={{ status: "notInstalled", message: "The language server is not installed." }}
                  onRetry={noop}
                />
                <SuggestionsStatus
                  session={{
                    status: "restarting",
                    attempt: 1,
                    maxAttempts: 4,
                    message: "Starting it again.",
                  }}
                  onRetry={noop}
                />
                <SuggestionsStatus
                  session={{
                    status: "failed",
                    message:
                      "The language server keeps stopping. Suggestions are off until you open another problem or restart Learn AI.",
                  }}
                  onRetry={noop}
                />
                <SuggestionsStatus session={{ status: "idle" }} onRetry={noop} />
              </PaneToolbar>
            ))}
          </div>
        </Specimen>
      </Section>

      <Section title="Problem" aside="problem/">
        <Specimen name="ProblemHeader">
          <div className="@container">
            <ProblemHeader problem={DETAIL} track={TRACKS[1]} prerequisites={PROBLEMS.slice(0, 1)} />
          </div>
        </Specimen>
        <Specimen name="ProblemTable: solved, not started, attempted, long title">
          <ProblemTable problems={PROBLEMS} label="Sample problems" />
        </Specimen>
        <Specimen name="ProblemTable with unmet prerequisites noted beside the title">
          <ProblemTable
            problems={PATH_PROBLEMS}
            label="Sample problems with prerequisites"
            unmet={PATH_UNMET}
          />
        </Specimen>
        <Specimen name="PrerequisiteList (problem header); PrerequisiteNote: one, several, none">
          <div className="flex flex-col gap-3 text-xs">
            <PrerequisiteList prerequisites={PROBLEMS.slice(0, 3)} />
            <div className="flex items-baseline gap-8">
              <PrerequisiteNote unmet={PROBLEMS.slice(1, 2)} />
              <PrerequisiteNote unmet={PROBLEMS.slice(1, 4)} />
              <PrerequisiteNote unmet={[]} />
            </div>
          </div>
        </Specimen>
        <Specimen name="HintList: partly revealed, none">
          <div className="flex flex-col gap-6">
            <HintList hints={DETAIL.hints} revealed={hints} onReveal={setHints} />
            <HintList hints={[]} revealed={0} onReveal={() => {}} />
          </div>
        </Specimen>
        <Specimen name="EditorialPanel: locked, unlocked, reference loading, reference failed">
          <div className="flex flex-col gap-8">
            <EditorialPanel unlocked={false} editorial={null} framework="torch" onReveal={() => {}} />
            <EditorialPanel
              unlocked
              editorial="Shift by the maximum; the largest exponent becomes $0$."
              reference={SAMPLE_CODE}
              framework="torch"
              onReveal={() => {}}
            />
            <EditorialPanel unlocked editorial="Loading the reference." framework="jax" onReveal={() => {}} />
            <EditorialPanel
              unlocked
              editorial="The reference failed."
              referenceError={new Error("reference/mlx.py is missing")}
              framework="mlx"
              onReveal={() => {}}
            />
          </div>
        </Specimen>
        <Specimen name="SubmissionList: per problem, and with problem titles; the newest is a review attempt">
          <div className="flex max-w-measure flex-col gap-6">
            <SubmissionList submissions={SUBMISSIONS} onOpen={() => {}} />
            <SubmissionList
              submissions={SUBMISSIONS}
              titles={new Map(PROBLEMS.map((p) => [p.id, p.title]))}
              onOpen={() => {}}
            />
          </div>
        </Specimen>
        <Specimen name="SubmissionView: loaded, not judged (skipped cases), a staged problem, during a review (code hidden), loading, failed">
          <div className="flex flex-col gap-8">
            <SubmissionView submission={SUBMISSION_DETAIL} onBack={() => {}} />
            <SubmissionView submission={NOT_JUDGED_SUBMISSION} onBack={() => {}} />
            <SubmissionView submission={STAGED_SUBMISSION} onBack={() => {}} />
            <SubmissionView
              submission={{ ...SUBMISSION_DETAIL, review: true }}
              codeHidden
              onBack={() => {}}
            />
            <SubmissionView onBack={() => {}} />
            <SubmissionView
              error={new Error("Unknown submission: 99")}
              onBack={() => {}}
              onRetry={() => {}}
            />
          </div>
        </Specimen>
      </Section>

      <Section title="Results" aside="results/">
        <Specimen name="VerdictHeader: each verdict">
          <div className="grid grid-cols-2 gap-x-10 gap-y-5">
            {VERDICTS.map((verdict) => (
              <VerdictHeader
                key={verdict}
                verdict={verdict}
                mode="submit"
                passed={verdict === "accepted" ? 7 : 2}
                total={7}
                skipped={verdict === "assets_missing" || verdict === "hardware_missing" ? 5 : 0}
                durationMs={84.2}
                env={ENV}
              />
            ))}
            <VerdictHeader verdict="accepted" mode="run" passed={3} total={3} durationMs={31} />
          </div>
        </Specimen>
        <Specimen name="RunProgress: preparing, running, canceling">
          <div className="flex max-w-measure flex-col gap-5">
            <RunProgress
              mode="run"
              message="Checking the Python environment"
              finished={0}
              total={0}
              canceling={false}
              onCancel={() => {}}
            />
            <RunProgress
              mode="submit"
              message={null}
              finished={3}
              total={7}
              canceling={false}
              onCancel={() => {}}
            />
            <RunProgress mode="submit" message={null} finished={3} total={7} canceling onCancel={() => {}} />
          </div>
        </Specimen>
        <Specimen name="MetricList: within and over the limit, no limit; higher is better, met and missed">
          <div className="flex flex-col gap-4">
            <MetricList
              metrics={[
                { name: "model_tokens", value: 36, unit: "tokens", limit: 36 },
                { name: "slowdown", value: 212.4, unit: "x", limit: 1.5 },
                { name: "mean_ttft", value: 41.2, unit: "ms" },
              ]}
            />
            <MetricList
              metrics={[
                { name: "throughput", value: 41.3, unit: "tok/s", limit: 20, higherIsBetter: true },
                { name: "cache_hit_rate", value: 62, unit: "%", limit: 90, higherIsBetter: true },
              ]}
            />
          </div>
        </Specimen>
        <Specimen name="CaseList: pass, fail with detail, error, timeout, skipped">
          <div className="max-w-measure">
            <CaseList cases={CASES} ended />
          </div>
        </Specimen>
        <Specimen name="CaseList: cases skipped for a model that is not downloaded, and for missing hardware (skipped, not failed; nothing opens by itself)">
          <div className="flex max-w-measure flex-col gap-5">
            <CaseList cases={SKIPPED_FOR_ASSET} ended />
            <CaseList cases={SKIPPED_FOR_HARDWARE} ended />
          </div>
        </Specimen>
        <Specimen name="CaseList with stages (a capstone): streaming, with the count of each stage following the run; ended, with a failure and a skipped case">
          <div className="flex max-w-measure flex-col gap-8">
            <CaseList cases={STAGED_RUNNING} ended={false} />
            <CaseList cases={STAGED_ENDED} ended />
          </div>
        </Specimen>
        <Specimen name="OutputBlock: neutral, failure, tall">
          <div className="flex max-w-measure flex-col gap-3">
            <OutputBlock label="Printed output">{"nucleus sizes: [2]\n"}</OutputBlock>
            <OutputBlock label="Judge output" tone="fail">
              {TRACEBACK}
            </OutputBlock>
            <OutputBlock label="Install log" size="tall">
              {Array.from({ length: 30 }, (_, i) => ` + package-${i}==1.0.${i}`).join("\n")}
            </OutputBlock>
          </div>
        </Specimen>
        {RUN_STATES.map(({ name, state }) => (
          <Specimen key={name} name={`ResultsPanel: ${name}`}>
            <div className="h-64 max-w-measure border">
              <ResultsPanel
                run={state}
                onCancel={() => {}}
                resolution={state.missingAssets.map((asset) => (
                  <AssetNotice
                    key={asset.name}
                    asset={asset}
                    busy={false}
                    layout="inline"
                    onDownload={noop}
                    onCancel={noop}
                  />
                ))}
                actions={
                  state.submissionId !== null ? (
                    <Button variant="outline" size="xs">
                      View submission
                    </Button>
                  ) : null
                }
              />
            </div>
          </Specimen>
        ))}
      </Section>

      <Section title="Review" aside="review/, settings/">
        <Specimen name="ReviewList (Home): overdue on the last review, due today">
          <div className="max-w-measure">
            <ReviewList
              entries={reviewEntries(dueReviews(REVIEWS), PROBLEMS, TRACKS)}
              now={REVIEW_NOW}
              onOpen={() => {}}
            />
          </div>
        </Specimen>
        <Specimen name="UpcomingReviews (Home, when nothing is due): several, one, none (renders nothing)">
          <div className="flex flex-col gap-2">
            <UpcomingReviews reviews={REVIEWS} now={REVIEW_NOW} />
            <UpcomingReviews reviews={REVIEWS.slice(2, 3)} now={REVIEW_NOW} />
            <UpcomingReviews reviews={[]} now={REVIEW_NOW} />
          </div>
        </Specimen>
        <Specimen name="ReviewStrip (under the editor toolbar): in progress, busy during a run, an action that failed">
          <div className="flex max-w-measure flex-col gap-4 border-x border-t">
            <ReviewStrip
              review={REVIEWS[1] as Review}
              onForgot={noop}
              onSnooze={noop}
              onStartOver={noop}
              onRetire={noop}
              onLeave={noop}
            />
            <ReviewStrip
              review={REVIEWS[0] as Review}
              busy
              onForgot={noop}
              onSnooze={noop}
              onStartOver={noop}
              onRetire={noop}
              onLeave={noop}
            />
            <ReviewStrip
              review={REVIEWS[1] as Review}
              error="state database error: database is locked"
              onForgot={noop}
              onSnooze={noop}
              onStartOver={noop}
              onRetire={noop}
              onLeave={noop}
            />
          </div>
        </Specimen>
        <Specimen name="ReviewDueNotice (normal workspace): strip, inline">
          <div className="flex max-w-measure flex-col gap-4">
            <div className="border-x border-t">
              <ReviewDueNotice review={REVIEWS[0] as Review} now={REVIEW_NOW} onStart={noop} />
            </div>
            <ReviewDueNotice review={REVIEWS[1] as Review} now={REVIEW_NOW} onStart={noop} layout="inline" />
          </div>
        </Specimen>
        <Specimen name="ReviewHidden: hints, editorial, submission code">
          <div className="flex flex-col gap-6">
            <ReviewHidden what="The hints are" />
            <ReviewHidden what="The editorial and the reference solution are" />
          </div>
        </Specimen>
        <Specimen name="ReviewOutcomeNote (under an accepted review verdict): next review scheduled, mastered">
          <div className="flex max-w-measure flex-col gap-3">
            <ReviewOutcomeNote
              now={REVIEW_NOW}
              outcome={{ kind: "advanced", label: "Review 2 of 5", next: review("d", 2, 21) }}
            />
            <ReviewOutcomeNote now={REVIEW_NOW} outcome={{ kind: "mastered", label: "Review 5 of 5" }} />
          </div>
        </Specimen>
        <Specimen name="ResultsPanel: an accepted review submit, with what it led to">
          <div className="h-64 max-w-measure border">
            <ResultsPanel
              run={run({
                phase: "done",
                review: true,
                verdict: "accepted",
                passed: 5,
                total: 5,
                durationMs: 84.2,
                env: ENV,
                submissionId: 31,
                cases: CASES.slice(0, 1),
              })}
              onCancel={noop}
              note={
                <ReviewOutcomeNote
                  now={REVIEW_NOW}
                  outcome={{ kind: "advanced", label: "Review 2 of 5", next: review("d", 2, 21) }}
                />
              }
            />
          </div>
        </Specimen>
        <Specimen name="SuggestionsPanel (Settings): starting; not installed, install running; stopped; off">
          <div className="flex flex-col gap-10">
            <SuggestionsPanel
              status={EDITOR_STATUS}
              session={{ status: "starting" }}
              onToggle={noop}
              install={IDLE_EDITOR_INSTALL}
              onInstall={noop}
              onRetry={noop}
            />
            <SuggestionsPanel
              status={{ ...EDITOR_STATUS, installed: false, serverVersion: null }}
              session={{ status: "notInstalled", message: "The language server is not installed." }}
              onToggle={noop}
              install={{
                status: "running",
                log: ["$ uv sync --project python --extra torch --extra editor", "Downloading ty (14.1MiB)"],
                message: null,
              }}
              onInstall={noop}
              onRetry={noop}
            />
            <SuggestionsPanel
              status={EDITOR_STATUS}
              session={{ status: "failed", message: "The language server keeps stopping. Try again." }}
              onToggle={noop}
              install={IDLE_EDITOR_INSTALL}
              onInstall={noop}
              onRetry={noop}
            />
            <SuggestionsPanel
              status={{ ...EDITOR_STATUS, enabled: false }}
              session={{ status: "off" }}
              onToggle={noop}
              install={IDLE_EDITOR_INSTALL}
              onInstall={noop}
              onRetry={noop}
            />
          </div>
        </Specimen>
        <Specimen name="ReviewSetting (Settings): on, off, saving">
          <div className="flex flex-col gap-6">
            <ReviewSetting enabled={reviewsOn} onChange={setReviewsOn} />
            <ReviewSetting enabled={false} onChange={noop} />
            <ReviewSetting enabled disabled onChange={noop} />
          </div>
        </Specimen>
      </Section>

      <Section title="Path" aside="path/, home/">
        <Specimen name="PathSummary: under way, not started, finished">
          <PathSummary path={PATH} next={PATH_NEXT} />
          <PathSummary path={{ ...PATH, solvedCount: 0 }} next={PATH_ENTRIES[0]?.[0] ?? null} />
          <PathSummary path={{ ...PATH, solvedCount: 4 }} next={null} />
        </Specimen>
        <Specimen name="PathStage: a part with the next problem and an unmet prerequisite; a part not begun">
          <PathStage
            number={1}
            stage={PATH.stages[0] as Path["stages"][number]}
            entries={PATH_ENTRIES[0] ?? []}
            nextId={PATH_NEXT?.problem.id}
            unmet={PATH_UNMET}
          />
          <PathStage
            number={2}
            stage={PATH.stages[1] as Path["stages"][number]}
            entries={PATH_ENTRIES[1] ?? []}
            nextId={PATH_NEXT?.problem.id}
            unmet={PATH_UNMET}
          />
        </Specimen>
        <Specimen name="PathEntryList on its own">
          <div className="flex max-w-measure">
            <PathEntryList entries={PATH_ENTRIES[0] ?? []} label="Sample entries" />
          </div>
        </Specimen>
        <Specimen name="PathProgress (Home): under way, finished">
          <div className="grid grid-cols-2 gap-10">
            <PathProgress path={PATH} next={PATH_NEXT} />
            <PathProgress path={{ ...PATH, solvedCount: 4 }} next={null} />
          </div>
        </Specimen>
      </Section>

      <Section title="Home and settings" aside="home/, settings/">
        <Specimen name="ContinueEntry: first run on a path, first run without one, resuming, and up next">
          <ContinueEntry
            problem={PROBLEMS[0] as ProblemSummary}
            track={TRACKS[1]}
            onPath={{ path: PATH, position: 1 }}
            firstRun
            onOpen={() => {}}
          />
          <ContinueEntry
            problem={PROBLEMS[1] as ProblemSummary}
            track={TRACKS[1]}
            firstRun
            onOpen={() => {}}
          />
          <ContinueEntry
            problem={PROBLEMS[2] as ProblemSummary}
            track={TRACKS[1]}
            lastSubmission={SUBMISSIONS[1]}
            onOpen={() => {}}
          />
          <ContinueEntry problem={PROBLEMS[1] as ProblemSummary} track={TRACKS[1]} onOpen={() => {}} />
        </Specimen>
        <Specimen name="ProgressSummary: in progress, and empty">
          <div className="grid grid-cols-2 gap-10">
            <ProgressSummary
              stats={{
                solved: 37,
                total: 189,
                byDifficulty: { easy: [22, 60], medium: [13, 92], hard: [2, 37] },
                streakDays: 6,
                reviewsDue: 2,
              }}
            />
            <ProgressSummary
              stats={{
                solved: 0,
                total: 189,
                byDifficulty: { easy: [0, 60], medium: [0, 92], hard: [0, 37] },
                streakDays: 0,
                reviewsDue: 0,
              }}
            />
          </div>
        </Specimen>
        <Specimen name="TrackProgress: progress, and contents (first run)">
          <div className="grid grid-cols-2 gap-10">
            <TrackProgress tracks={TRACKS} />
            <TrackProgress tracks={TRACKS} variant="contents" />
          </div>
        </Specimen>
        <Specimen name="GettingStarted (first run)">
          <div className="max-w-measure">
            <GettingStarted />
          </div>
        </Specimen>
        <Specimen name="EnvironmentPanel: one framework missing, install running; uv missing, install failed">
          <div className="flex flex-col gap-10">
            <EnvironmentPanel
              env={{
                uv: "0.13.8",
                python: "3.15.2",
                root: "/Users/ada/Developer/learnai",
                frameworks: [
                  { framework: "torch", installed: true, version: "2.15.0" },
                  { framework: "mlx", installed: false, version: null },
                ],
              }}
              install={{
                framework: "mlx",
                status: "running",
                log: ["$ uv sync --extra mlx", "Resolved 61 packages in 4ms"],
                message: null,
              }}
              onInstall={() => {}}
            />
            <EnvironmentPanel
              env={{
                uv: null,
                python: null,
                root: "/Users/ada/Developer/learnai",
                frameworks: [{ framework: "jax", installed: false, version: null }],
              }}
              install={{
                framework: "jax",
                status: "failed",
                log: ["error: failed to spawn `uv`"],
                message: "uv was not found on your PATH.",
              }}
              onInstall={() => {}}
            />
          </div>
        </Specimen>
      </Section>

      <Section title="Models and hardware" aside="models/, workbench/">
        <Specimen name="AssetStateMark: downloaded, not downloaded, incomplete; without the word">
          <div className="flex flex-wrap items-center gap-6">
            <AssetStateMark state="present" />
            <AssetStateMark state="missing" />
            <AssetStateMark state="incomplete" />
            <AssetStateMark state="incomplete" showLabel={false} />
          </div>
        </Specimen>
        <Specimen name="AssetProgress: preparing, downloading, canceling; the bare rule">
          <div className="flex max-w-measure flex-col gap-5">
            <AssetProgress name="granite-4.2-3b-mlx-8bit" download={PREPARING} />
            <AssetProgress name="granite-4.2-3b-mlx-8bit" download={DOWNLOADING} />
            <AssetProgress name="granite-4.2-3b-mlx-8bit" download={DOWNLOADING} canceling />
            <AssetProgress name="granite-4.2-3b-mlx-8bit" download={DOWNLOADING} variant="rule" />
          </div>
        </Specimen>
        <Specimen name="AssetDownloadButton: download, resume, try again, cancel, canceling, another download running; RemoveAssetButton (confirms)">
          <div className="flex flex-wrap items-center gap-3">
            <AssetDownloadButton state="missing" busy={false} onDownload={noop} onCancel={noop} />
            <AssetDownloadButton state="incomplete" busy={false} onDownload={noop} onCancel={noop} />
            <AssetDownloadButton
              state="missing"
              download={DOWNLOAD_FAILED}
              busy={false}
              onDownload={noop}
              onCancel={noop}
            />
            <AssetDownloadButton
              state="missing"
              download={DOWNLOADING}
              busy
              onDownload={noop}
              onCancel={noop}
            />
            <AssetDownloadButton
              state="missing"
              download={DOWNLOADING}
              busy
              canceling
              onDownload={noop}
              onCancel={noop}
            />
            <AssetDownloadButton state="missing" busy onDownload={noop} onCancel={noop} />
            <RemoveAssetButton name="granite-4.0-350m" size="712 MB" onRemove={noop} />
          </div>
        </Specimen>
        <Specimen name="AssetRow: downloaded, not downloaded, incomplete; preparing, downloading, canceling, failed, being removed">
          <RuledList>
            <AssetRow
              asset={MODEL_PRESENT}
              busy={false}
              titles={MODEL_TITLES}
              onDownload={noop}
              onCancel={noop}
              onRemove={noop}
            />
            <AssetRow
              asset={MODEL_MISSING}
              busy={false}
              titles={MODEL_TITLES}
              onDownload={noop}
              onCancel={noop}
              onRemove={noop}
            />
            <AssetRow
              asset={MODEL_INCOMPLETE}
              busy={false}
              titles={MODEL_TITLES}
              onDownload={noop}
              onCancel={noop}
              onRemove={noop}
            />
            <AssetRow
              asset={MODEL_MISSING}
              download={PREPARING}
              busy
              titles={MODEL_TITLES}
              onDownload={noop}
              onCancel={noop}
              onRemove={noop}
            />
            <AssetRow
              asset={MODEL_MISSING}
              download={DOWNLOADING}
              busy
              titles={MODEL_TITLES}
              onDownload={noop}
              onCancel={noop}
              onRemove={noop}
            />
            <AssetRow
              asset={MODEL_MISSING}
              download={DOWNLOADING}
              busy
              canceling
              titles={MODEL_TITLES}
              onDownload={noop}
              onCancel={noop}
              onRemove={noop}
            />
            <AssetRow
              asset={{ ...MODEL_INCOMPLETE, bytesOnDisk: 741_238_784 }}
              download={DOWNLOAD_FAILED}
              busy={false}
              titles={MODEL_TITLES}
              onDownload={noop}
              onCancel={noop}
              onRemove={noop}
            />
            <AssetRow
              asset={MODEL_PRESENT}
              busy={false}
              removing
              titles={MODEL_TITLES}
              onDownload={noop}
              onCancel={noop}
              onRemove={noop}
            />
          </RuledList>
        </Specimen>
        <Specimen name="ModelsPanel: three models, one downloading; a catalog that needs no model">
          <div className="flex flex-col gap-10">
            <ModelsPanel
              listing={MODELS}
              titles={MODEL_TITLES}
              fetch={{
                ...NO_DOWNLOADS,
                busy: true,
                downloads: { "granite-4.2-3b-mlx-8bit": DOWNLOADING },
              }}
              onRemove={noop}
            />
            <ModelsPanel listing={{ ...MODELS, assets: [] }} fetch={NO_DOWNLOADS} onRemove={noop} />
          </div>
        </Specimen>
        <Specimen name="AssetNotice (strip, under the editor toolbar): not downloaded, incomplete, preparing, downloading, canceling, failed">
          <div className="flex flex-col gap-3">
            <AssetNotice asset={MODEL_MISSING} busy={false} onDownload={noop} onCancel={noop} />
            <AssetNotice asset={MODEL_INCOMPLETE} busy={false} onDownload={noop} onCancel={noop} />
            <AssetNotice asset={MODEL_MISSING} download={PREPARING} busy onDownload={noop} onCancel={noop} />
            <AssetNotice
              asset={MODEL_MISSING}
              download={DOWNLOADING}
              busy
              onDownload={noop}
              onCancel={noop}
            />
            <AssetNotice
              asset={MODEL_MISSING}
              download={DOWNLOADING}
              busy
              canceling
              onDownload={noop}
              onCancel={noop}
            />
            <AssetNotice
              asset={MODEL_MISSING}
              download={DOWNLOAD_FAILED}
              busy={false}
              onDownload={noop}
              onCancel={noop}
            />
          </div>
        </Specimen>
        <Specimen name="AssetNotice (inline, in the results): the download offered, downloading, and downloaded after a run that skipped cases">
          <div className="flex max-w-measure flex-col gap-3">
            <AssetNotice
              asset={MODEL_MISSING}
              busy={false}
              layout="inline"
              onDownload={noop}
              onCancel={noop}
            />
            <AssetNotice
              asset={MODEL_MISSING}
              download={DOWNLOADING}
              busy
              layout="inline"
              onDownload={noop}
              onCancel={noop}
            />
            <AssetNotice
              asset={MODEL_PRESENT}
              busy={false}
              layout="inline"
              whenPresent="Submit again to judge the skipped cases."
              onDownload={noop}
              onCancel={noop}
            />
          </div>
        </Specimen>
        <Specimen name="HardwareNotice: no remote machine known; MLX; a remote machine has it; the chosen remote machine lacks it; inline">
          <div className="flex flex-col gap-3">
            <HardwareNotice requires="cuda" />
            <HardwareNotice requires="mlx-gpu" />
            <HardwareNotice requires="cuda" remote="gpu-box" onUseRemote={noop} />
            <HardwareNotice requires="mlx-gpu" machine="gpu-box" />
            <HardwareNotice requires="cuda" remote="gpu-box" onUseRemote={noop} layout="inline" />
          </div>
        </Specimen>
        <Specimen name="AssetNotice for a remote run: not downloaded there, downloading there, downloaded there">
          <div className="flex flex-col gap-3">
            <AssetNotice
              asset={{ name: MODEL_MISSING.name, sizeMb: MODEL_MISSING.sizeMb }}
              busy={false}
              layout="inline"
              whenPresent="Run again to judge the skipped cases."
              host="gpu-box"
              onDownload={noop}
              onCancel={noop}
            />
            <AssetNotice
              asset={{ name: MODEL_MISSING.name, sizeMb: MODEL_MISSING.sizeMb }}
              download={DOWNLOADING}
              busy
              layout="inline"
              whenPresent="Run again to judge the skipped cases."
              host="gpu-box"
              onDownload={noop}
              onCancel={noop}
            />
            <AssetNotice
              asset={{ name: MODEL_MISSING.name, sizeMb: MODEL_MISSING.sizeMb }}
              download={{ phase: "finished" }}
              busy={false}
              layout="inline"
              whenPresent="Run again to judge the skipped cases."
              host="gpu-box"
              onDownload={noop}
              onCancel={noop}
            />
          </div>
        </Specimen>
      </Section>

      <Section title="Run targets" aside="targets/, results/, problem/">
        <Specimen name="CheckStepMark: passed, failed, running, waiting, not reached">
          <div className="flex items-center gap-4">
            <CheckStepMark state="pass" />
            <CheckStepMark state="fail" />
            <CheckStepMark state="running" />
            <CheckStepMark state="pending" />
            <CheckStepMark state="skipped" />
          </div>
        </Specimen>
        <Specimen name="FixText: a command; an instruction">
          <div className="flex max-w-measure flex-col gap-2">
            <FixText>ssh studio 'curl -LsSf https://astral.sh/uv/install.sh | sh'</FixText>
            <FixText>
              Load your key into the SSH agent with `ssh-add`, confirm that `ssh gpu-box true` works in a
              terminal without a prompt, then check again.
            </FixText>
          </div>
        </Specimen>
        <Specimen name="CheckStepList: running; stopped at uv with a fix; this computer">
          <div className="flex flex-col gap-5">
            <CheckStepList steps={stepProgress("ssh", CHECK_RUNNING.steps, true)} />
            <CheckStepList steps={stepProgress("ssh", TARGET_FAILING.lastCheck?.steps ?? [], false)} />
            <CheckStepList steps={stepProgress("local", TARGET_LOCAL.lastCheck?.steps ?? [], false)} />
          </div>
        </Specimen>
        <Specimen name="TargetSummary: an NVIDIA machine; a Mac">
          <div className="flex flex-col gap-3">
            {TARGET_HEALTHY.lastCheck?.doctor ? (
              <TargetSummary doctor={TARGET_HEALTHY.lastCheck.doctor} />
            ) : null}
            {TARGET_LOCAL.lastCheck?.doctor ? <TargetSummary doctor={TARGET_LOCAL.lastCheck.doctor} /> : null}
          </div>
        </Specimen>
        <Specimen name="TargetRow: this computer, healthy, failing its check, never checked, checking, a check that could not start">
          <RuledList>
            <TargetRow target={TARGET_LOCAL} onCheck={noop} />
            <TargetRow target={TARGET_HEALTHY} onCheck={noop} onRemove={noop} />
            <TargetRow target={TARGET_FAILING} onCheck={noop} onRemove={noop} />
            <TargetRow target={TARGET_NEW} onCheck={noop} onRemove={noop} />
            <TargetRow target={TARGET_NEW} check={CHECK_RUNNING} onCheck={noop} onRemove={noop} />
            <TargetRow
              target={TARGET_NEW}
              check={{
                ...CHECK_RUNNING,
                running: false,
                steps: [],
                log: [],
                error: "unknown target `ssh:lab-a100`",
              }}
              onCheck={noop}
              onRemove={noop}
            />
          </RuledList>
        </Specimen>
        <Specimen name="AddTargetForm: hosts to choose from; every host already a target; no host in the configuration">
          <div className="flex flex-col gap-6">
            <AddTargetForm hosts={FIXTURE_SSH_HOSTS} taken={["gpu-box"]} onAdd={noop} />
            <AddTargetForm hosts={FIXTURE_SSH_HOSTS.slice(0, 1)} taken={["gpu-box"]} onAdd={noop} />
            <AddTargetForm hosts={[]} taken={[]} onAdd={noop} />
          </div>
        </Specimen>
        <Specimen name="TargetsPanel">
          <TargetsPanel
            targets={TARGETS}
            hosts={FIXTURE_SSH_HOSTS}
            checks={{}}
            onCheck={noop}
            onAdd={noop}
            onRemove={noop}
          />
        </Specimen>
        <Specimen name="TargetPicker: in a toolbar; disabled during a run; hidden with only this computer">
          <div className="flex flex-col gap-2">
            <PaneToolbar>
              <TargetPicker targets={TARGETS} value={pickedTarget} onChange={setPickedTarget} />
              <TargetPicker targets={TARGETS} value="ssh:gpu-box" disabled onChange={noop} />
              <TargetPicker targets={TARGETS.slice(0, 1)} value="local" onChange={noop} />
            </PaneToolbar>
          </div>
        </Specimen>
        <Specimen name="VerdictHeader and SubmissionList: a run and submissions judged on a remote machine">
          <div className="flex max-w-measure flex-col gap-4">
            <VerdictHeader
              verdict="accepted"
              mode="submit"
              passed={8}
              total={8}
              durationMs={2140}
              env={{ python: "3.15.2", torch: "2.15.0", device: "cuda" }}
              target="ssh:gpu-box"
            />
            <SubmissionList
              submissions={SUBMISSIONS.slice(0, 3).map((submission, index) => ({
                ...submission,
                target: index === 1 ? "local" : "ssh:gpu-box",
              }))}
              onOpen={noop}
            />
          </div>
        </Specimen>
      </Section>

      <Section title="First run and catalog health" aside="app/, workbench/, settings/">
        <Specimen name="Notice: inline warning with an action, information, success, failure, strip">
          <div className="flex max-w-measure flex-col gap-3">
            <Notice
              action={
                <Button variant="outline" size="xs">
                  Fix it
                </Button>
              }
            >
              Something needs attention before this can work.
            </Notice>
            <Notice tone="info">Nothing is wrong; this is worth knowing.</Notice>
            <Notice tone="pass">MLX installed.</Notice>
            <Notice tone="fail">The install failed. The log above has the details.</Notice>
            <Notice layout="strip">A strip sits under a toolbar and spans its pane.</Notice>
          </div>
        </Specimen>
        <Specimen name="EnvironmentNotice: uv missing, not set up, framework missing, installing, install failed; inline (Home)">
          <div className="flex flex-col gap-3">
            <EnvironmentNotice problem={{ kind: "uv-missing" }} install={IDLE_INSTALL} onInstall={() => {}} />
            <EnvironmentNotice problem={{ kind: "not-set-up" }} install={IDLE_INSTALL} onInstall={() => {}} />
            <EnvironmentNotice
              problem={{ kind: "framework-missing", framework: "jax" }}
              install={IDLE_INSTALL}
              onInstall={() => {}}
            />
            <EnvironmentNotice
              problem={{ kind: "framework-missing", framework: "mlx" }}
              install={{ framework: "mlx", status: "running", log: [], message: null }}
              onInstall={() => {}}
            />
            <EnvironmentNotice
              problem={{ kind: "framework-missing", framework: "jax" }}
              install={{ framework: "jax", status: "failed", log: [], message: "uv sync exited with code 2" }}
              onInstall={() => {}}
            />
            <EnvironmentNotice
              layout="inline"
              problem={{ kind: "not-set-up" }}
              install={IDLE_INSTALL}
              onInstall={() => {}}
            />
          </div>
        </Specimen>
        <Specimen name="CatalogIssueList: closed by default; nothing when the catalog is clean">
          <div className="max-w-measure">
            <CatalogIssueList
              root="/Users/ada/Developer/learnai"
              issues={[
                {
                  path: "/Users/ada/Developer/learnai/problems/sampling/min-p-draft/problem.toml",
                  message: "missing starter/torch.py",
                },
                {
                  path: "/Users/ada/Developer/learnai/problems/serving/track.toml",
                  message: "missing field `title`",
                },
              ]}
            />
            <CatalogIssueList issues={[]} />
          </div>
        </Specimen>
      </Section>

      <Section title="States and layout" aside="app/">
        <Specimen name="EmptyState: plain, with an action, with an icon">
          <EmptyState
            title="The editorial is locked"
            icon={<Lock className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />}
          >
            It opens when you solve the problem.
          </EmptyState>
          <EmptyState title="No submissions yet">
            Submit runs your code against every case and keeps a copy here.
          </EmptyState>
          <EmptyState
            title="No problems match these filters"
            action={
              <Button variant="outline" size="sm">
                Clear all filters
              </Button>
            }
          >
            Loosen a filter, or search for a different title or tag.
          </EmptyState>
        </Specimen>
        <Specimen name="ErrorState, with and without retry">
          <ErrorState
            title="The catalog could not be loaded"
            error={new Error("problems/attention/track.toml: missing field `order`")}
            onRetry={() => {}}
          />
          <ErrorState error="Unknown problem: nope" />
        </Specimen>
        <Specimen name="LoadingLines">
          <LoadingLines lines={4} className="max-w-measure" />
        </Specimen>
        <Specimen name="ScreenMessage: a whole screen that is loading, failed, or not there">
          <div className="h-40 border">
            <ScreenMessage>
              <ErrorState
                title="There is no such screen"
                error="The address does not match any part of Learn AI."
              />
            </ScreenMessage>
          </div>
        </Specimen>
        <Specimen name="RuledList: framed, and unframed and ordered; DisclosureChevron">
          <div className="grid max-w-measure grid-cols-2 gap-10 text-sm">
            <RuledList>
              <li className="py-2">First row</li>
              <li className="py-2">Second row</li>
            </RuledList>
            <RuledList ordered framed={false}>
              <li className="flex items-center gap-2 py-2">
                <DisclosureChevron />A row that opens
              </li>
              <li className="py-2">Second row</li>
            </RuledList>
          </div>
        </Specimen>
        <Specimen name="PaneToolbar">
          <div className="max-w-measure border">
            <PaneToolbar>
              <FrameworkSwitcher frameworks={["python"]} value="python" onChange={() => {}} />
              <SaveIndicator state="saved" />
            </PaneToolbar>
          </div>
        </Specimen>
        <Specimen name="MetaList: inline and rows">
          <div className="flex flex-col gap-5">
            <MetaList ruled>
              <MetaItem label="Difficulty">
                <DifficultyMark difficulty="medium" />
              </MetaItem>
              <MetaItem label="Time limit">10 s per case</MetaItem>
            </MetaList>
            <MetaList layout="rows">
              <MetaItem label="uv">0.13.8</MetaItem>
              <MetaItem label="Python">3.15.2</MetaItem>
            </MetaList>
          </div>
        </Specimen>
      </Section>
    </Page>
  );
}

export const Route = createLazyRoute("/dev/components")({ component: DevComponentsScreen });
