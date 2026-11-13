import { Play } from "lucide-react";
import { type ReactNode, useState } from "react";
import { ActionButton } from "@/components/app/action-button";
import { AppLink } from "@/components/app/app-link";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { IconButton } from "@/components/app/icon-button";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import { Notice } from "@/components/app/notice";
import { Page, PageHeader, Section } from "@/components/app/page";
import { SHORTCUTS, ShortcutHint } from "@/components/app/shortcut-hint";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { CodeEditor } from "@/components/editor/code-editor";
import { ContinueEntry } from "@/components/home/continue-entry";
import { ProgressSummary } from "@/components/home/progress-summary";
import { TrackProgress } from "@/components/home/track-progress";
import { FilterSelect } from "@/components/library/filter-select";
import { LibraryFilterBar } from "@/components/library/library-filters";
import { TrackList } from "@/components/library/track-list";
import { DifficultyMark } from "@/components/problem/difficulty-mark";
import { EditorialPanel } from "@/components/problem/editorial-panel";
import { FrameworkList } from "@/components/problem/framework-list";
import { FrameworkSwitcher } from "@/components/problem/framework-switcher";
import { HintList } from "@/components/problem/hint-list";
import { JudgeList } from "@/components/problem/judge-list";
import { ProblemHeader } from "@/components/problem/problem-header";
import { ProblemTable } from "@/components/problem/problem-table";
import { ProgressRule } from "@/components/problem/progress-rule";
import { StatusMark } from "@/components/problem/status-mark";
import { SubmissionList } from "@/components/problem/submission-list";
import { SubmissionView } from "@/components/problem/submission-view";
import { CodeBlock } from "@/components/prose/code-block";
import { Prose } from "@/components/prose/prose";
import { CaseList } from "@/components/results/case-row";
import { MetricList } from "@/components/results/metric-list";
import { OutputBlock } from "@/components/results/output-block";
import { ResultsPanel } from "@/components/results/results-panel";
import { RunProgress } from "@/components/results/run-progress";
import { VerdictHeader } from "@/components/results/verdict-header";
import { CaseMark, VerdictMark } from "@/components/results/verdict-mark";
import { CatalogIssueList } from "@/components/settings/catalog-issue-list";
import { EnvironmentPanel } from "@/components/settings/environment-panel";
import { PathField } from "@/components/settings/path-field";
import { ThemePicker } from "@/components/settings/theme-picker";
import { Button } from "@/components/ui/button";
import { EnvironmentNotice } from "@/components/workbench/environment-notice";
import { SaveIndicator } from "@/components/workbench/save-indicator";
import {
  type CaseResult,
  DIFFICULTIES,
  type Framework,
  type JudgeKind,
  type ProblemDetail,
  type ProblemSummary,
  STATUSES,
  type Submission,
  type SubmissionDetail,
  type Track,
  type Verdict,
} from "@/lib/api";
import { initialRunState, type RunState } from "@/lib/api/run-state";
import type { InstallState } from "@/lib/api/use-env-install";
import { EMPTY_FILTERS, type LibraryFilters } from "@/lib/filters";
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

const TRACKS: Track[] = [
  {
    id: "foundations",
    title: "Numerical Foundations",
    description: "",
    order: 1,
    problemCount: 14,
    solvedCount: 11,
  },
  { id: "attention", title: "Attention", description: "", order: 2, problemCount: 15, solvedCount: 4 },
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

const SAMPLE_CODE = `import torch


def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return softmax along \`dim\`."""
    shifted = logits - logits.max(dim=dim, keepdim=True).values  # shift by the max
    weights = torch.exp(shifted)
    return weights / weights.sum(dim=dim, keepdim=True)
`;

const TRACEBACK = `Traceback (most recent call last):
  File "whetstone/harness/runner.py", line 212, in run_case
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

const SUBMISSIONS: Submission[] = VERDICTS.slice(0, 4).map((verdict, index) => ({
  id: index + 1,
  problemId: index % 2 === 0 ? "a" : "d",
  framework: (["torch", "jax", "mlx", "python"] as Framework[])[index] ?? "torch",
  mode: "submit",
  verdict,
  passed: verdict === "accepted" ? 7 : index,
  total: 7,
  durationMs: [84.2, 31.5, 0.4, 10024][index] ?? 1,
  target: "local",
  createdAt: new Date(Date.now() - ([5, 190, 60 * 26, 60 * 24 * 9][index] ?? 5) * 60_000).toISOString(),
}));

const SUBMISSION_DETAIL: SubmissionDetail = {
  ...(SUBMISSIONS[1] as Submission),
  code: SAMPLE_CODE,
  cases: CASES.map((c) => ({ ...c.result, name: c.info.name, public: c.info.public })),
  metrics: [],
};

const IDLE_INSTALL: InstallState = { framework: null, status: "idle", log: [], message: null };

function Specimen({ name, children }: { name: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2 pb-6">
      <p className="text-xs text-faint">{name}</p>
      <div>{children}</div>
    </div>
  );
}

/** Development only: every composed component, in every state it has. */
export function DevComponentsScreen() {
  const [filters, setFilters] = useState<LibraryFilters>(EMPTY_FILTERS);
  const [track, setTrack] = useState<string | null>("attention");
  const [framework, setFramework] = useState<Framework>("torch");
  const [hints, setHints] = useState(1);
  const [theme, setTheme] = useState<ThemePreference>("system");
  const [code, setCode] = useState(SAMPLE_CODE);
  const [judge, setJudge] = useState<JudgeKind | null>(null);

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
        <Specimen name="LibraryFilterBar, FilterSelect">
          <div className="flex flex-col gap-3">
            <LibraryFilterBar filters={filters} onChange={setFilters} />
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
        <Specimen name="ThemePicker, PathField">
          <div className="flex max-w-measure flex-col gap-3">
            <ThemePicker value={theme} onChange={setTheme} />
            <PathField
              label="Content root"
              path="/Users/ada/Developer/whetstone/a/very/long/path/that/keeps/going/and/going/until/it/truncates"
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
        <Specimen name="CodeBlock: plain, with line numbers, unknown language">
          <div className="flex max-w-measure flex-col gap-3">
            <CodeBlock code={SAMPLE_CODE} language="python" />
            <CodeBlock code={SAMPLE_CODE} language="python" lineNumbers />
            <CodeBlock code={'data: {"index": 0, "token": "The"}\n\ndata: [DONE]'} />
          </div>
        </Specimen>
        <Specimen name="CodeEditor">
          <div className="h-56 max-w-measure border">
            <CodeEditor value={code} onChange={setCode} label="Sample editor" />
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
        <Specimen name="SubmissionList: per problem, and with problem titles">
          <div className="flex max-w-measure flex-col gap-6">
            <SubmissionList submissions={SUBMISSIONS} onOpen={() => {}} />
            <SubmissionList
              submissions={SUBMISSIONS}
              titles={new Map(PROBLEMS.map((p) => [p.id, p.title]))}
              onOpen={() => {}}
            />
          </div>
        </Specimen>
        <Specimen name="SubmissionView: loaded, loading, failed">
          <div className="flex flex-col gap-8">
            <SubmissionView submission={SUBMISSION_DETAIL} onBack={() => {}} />
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
        <Specimen name="MetricList: within and over the limit, no limit">
          <MetricList
            metrics={[
              { name: "model_tokens", value: 36, unit: "tokens", limit: 36 },
              { name: "slowdown", value: 212.4, unit: "x", limit: 1.5 },
              { name: "mean_ttft", value: 41.2, unit: "ms" },
            ]}
          />
        </Specimen>
        <Specimen name="CaseList: pass, fail with detail, error, timeout, skipped">
          <div className="max-w-measure">
            <CaseList cases={CASES} ended />
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

      <Section title="Home and settings" aside="home/, settings/">
        <Specimen name="ContinueEntry: resuming, and up next">
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
              }}
            />
            <ProgressSummary
              stats={{
                solved: 0,
                total: 189,
                byDifficulty: { easy: [0, 60], medium: [0, 92], hard: [0, 37] },
                streakDays: 0,
              }}
            />
          </div>
        </Specimen>
        <Specimen name="TrackProgress">
          <div className="max-w-measure">
            <TrackProgress tracks={TRACKS} />
          </div>
        </Specimen>
        <Specimen name="EnvironmentPanel: one framework missing, install running; uv missing, install failed">
          <div className="flex flex-col gap-10">
            <EnvironmentPanel
              env={{
                uv: "0.13.8",
                python: "3.15.2",
                root: "/Users/ada/Developer/whetstone",
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
                root: "/Users/ada/Developer/whetstone",
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

      <Section title="First run and catalog health" aside="app/, workbench/, settings/">
        <Specimen name="Notice: inline warning with an action, inline information, strip">
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
            <Notice layout="strip">A strip sits under a toolbar and spans its pane.</Notice>
          </div>
        </Specimen>
        <Specimen name="EnvironmentNotice: uv missing, not set up, framework missing, installing, install failed">
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
          </div>
        </Specimen>
        <Specimen name="CatalogIssueList: closed by default; nothing when the catalog is clean">
          <div className="max-w-measure">
            <CatalogIssueList
              root="/Users/ada/Developer/whetstone"
              issues={[
                {
                  path: "/Users/ada/Developer/whetstone/problems/sampling/min-p-draft/problem.toml",
                  message: "missing starter/torch.py",
                },
                {
                  path: "/Users/ada/Developer/whetstone/problems/serving/track.toml",
                  message: "missing field `title`",
                },
              ]}
            />
            <CatalogIssueList issues={[]} />
          </div>
        </Specimen>
      </Section>

      <Section title="States and layout" aside="app/">
        <Specimen name="EmptyState, with and without an action">
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
        <Specimen name="MetaList: inline and rows">
          <div className="flex flex-col gap-5">
            <MetaList className="border-y border-border py-2.5">
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
