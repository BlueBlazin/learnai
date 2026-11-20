import type { Backend, StartRunArgs } from "../backend";
import type {
  AssetEvent,
  AssetList,
  CatalogIssue,
  Difficulty,
  EditorDocs,
  EditorEvent,
  EditorStatus,
  EnvStatus,
  Framework,
  InstallEvent,
  MissingAsset,
  ModelAsset,
  Path,
  ProblemDetail,
  ProblemStatus,
  ProblemSummary,
  Review,
  ReviewAction,
  RunEvent,
  Settings,
  SolutionFile,
  SshHost,
  Stats,
  Submission,
  SubmissionDetail,
  TargetCheckEvent,
  TargetInfo,
  Track,
} from "../types";
import { FIXTURE_HARDWARE, type FixtureAsset, fixtureAssets } from "./assets";
import { buildCatalog, type FixtureProblem, hash, renderTemplate } from "./catalog";
import { FixtureEditor } from "./editor";
import { FIXTURE_PATHS } from "./paths";
import { buildPlan, type RunContext, simulateRun, verdictOf } from "./runs";
import { FIXTURE_SSH_HOSTS, fixtureCheckScript, fixtureTargets } from "./targets";

export interface FixtureOptions {
  /** Multiplier on every simulated delay. 0 makes the backend instantaneous (tests). */
  speed?: number;
  now?: () => Date;
  /** Whether the language server starts out installed (default true). */
  editorInstalled?: boolean;
}

const ROOT = "/Users/ada/Developer/learnai";
const MODELS_DIR = `${ROOT}/.learnai/models`;
/** A simulated download takes about this long whatever its size, in steps of this many. */
const FETCH_STEPS = 48;
const FETCH_STEP_MS = 180;
const DAY_MS = 86_400_000;
/** The review intervals of the core (section 4.2). */
const REVIEW_INTERVALS_DAYS = [3, 7, 21, 60, 180] as const;

interface ReviewState {
  stage: number;
  /** Milliseconds since the epoch. */
  dueAt: number;
  lastReviewedAt: number | null;
  retired: boolean;
}

/**
 * Reviews seeded as due, so Home has a list to show: the stage, and how many
 * days ago each fell due. The last one is on its final review.
 */
const SEEDED_DUE: Record<string, { stage: number; overdueDays: number }> = {
  "stable-softmax": { stage: 1, overdueDays: 3 },
  "causal-mask": { stage: 2, overdueDays: 1 },
  "log-sum-exp": { stage: 0, overdueDays: 0 },
  "bpe-merge": { stage: 4, overdueDays: 0 },
};

interface ProblemState {
  status: ProblemStatus;
  solved: Set<Framework>;
  editorialRevealed: boolean;
}

/**
 * In-memory stand-in for learnai-core, for component work and tests. It keeps
 * the contract's rules (only submits are stored, solved means an accepted
 * submit, attempted means an edited file or any submit) so the UI's cache
 * invalidation is exercised for real.
 */
export class FixtureBackend implements Backend {
  readonly kind = "fixture" as const;

  private readonly speed: number;
  private readonly now: () => Date;
  private readonly catalog = buildCatalog();
  private readonly problems = new Map<string, FixtureProblem>();
  private readonly state = new Map<string, ProblemState>();
  private readonly solutions = new Map<string, string>();
  /** Review attempts, by problem and framework; absent means the starter. */
  private readonly reviewSolutions = new Map<string, string>();
  private readonly reviews = new Map<string, ReviewState>();
  private reviewsEnabled = true;
  private readonly submissions: SubmissionDetail[] = [];
  private readonly canceled = new Set<string>();
  private readonly installed = new Map<Framework, string | null>([
    ["torch", "2.15.0"],
    ["jax", "0.12.0.1"],
    ["mlx", null],
  ]);
  private readonly models = new Map<string, FixtureAsset>(
    fixtureAssets().map((asset) => [asset.name, asset]),
  );
  /** Models on each remote machine, by target name; nothing is downloaded there at first. */
  private readonly remoteModels = new Map<string, Map<string, FixtureAsset>>();
  private targets: TargetInfo[] = [];
  private readonly checking = new Set<string>();
  /** The fetch in flight; the core runs one at a time. */
  private fetching: { jobId: string; canceled: boolean } | null = null;
  private nextSubmissionId = 1;
  private nextRunId = 1;
  private nextJobId = 1;
  private readonly editor: FixtureEditor;

  constructor(options: FixtureOptions = {}) {
    this.speed = options.speed ?? 1;
    this.editor = new FixtureEditor({ sleep: this.sleep, installed: options.editorInstalled });
    this.now = options.now ?? (() => new Date());
    this.targets = fixtureTargets(this.now());
    for (const problem of this.catalog.problems) {
      this.problems.set(problem.meta.id, problem);
      this.state.set(problem.meta.id, {
        status: problem.seedStatus,
        solved: new Set(problem.seedSolved),
        editorialRevealed: false,
      });
      if (problem.savedTemplate) {
        const frameworks = problem.seedSolved.length
          ? problem.seedSolved
          : problem.meta.frameworks.slice(0, 1);
        for (const framework of frameworks) {
          const template =
            problem.seedStatus === "solved" ? problem.referenceTemplate : problem.savedTemplate;
          this.solutions.set(key(problem.meta.id, framework), renderTemplate(template, framework));
        }
      }
    }
    this.seedSubmissions();
    this.seedReviews();
  }

  /**
   * A schedule for every seeded solve. A few are due (see `SEEDED_DUE`); the
   * rest look like reviews that were kept up with, due over the coming weeks.
   */
  private seedReviews(): void {
    const today = this.now().getTime();
    for (const problem of this.catalog.problems) {
      const id = problem.meta.id;
      if (this.stateOf(id).status !== "solved") continue;
      const due = SEEDED_DUE[id];
      const stage = due?.stage ?? Math.floor(hash(`${id}:review-stage`) * 3);
      const dueAt = due
        ? today - due.overdueDays * DAY_MS - 3_600_000
        : today + (1 + Math.floor(hash(`${id}:review-due`) * 30)) * DAY_MS;
      this.reviews.set(id, {
        stage,
        dueAt,
        lastReviewedAt: stage === 0 ? null : dueAt - (REVIEW_INTERVALS_DAYS[stage] ?? 3) * DAY_MS,
        retired: false,
      });
    }
  }

  private reviewOf(id: string, state: ReviewState): Review {
    return {
      problemId: id,
      stage: state.stage,
      intervalDays: REVIEW_INTERVALS_DAYS[state.stage] ?? 180,
      dueAt: new Date(state.dueAt).toISOString(),
      lastReviewedAt: state.lastReviewedAt === null ? null : new Date(state.lastReviewedAt).toISOString(),
      due: state.dueAt <= this.now().getTime(),
    };
  }

  private activeReviews(): Review[] {
    if (!this.reviewsEnabled) return [];
    return [...this.reviews]
      .filter(([, state]) => !state.retired)
      .sort(([a, x], [b, y]) => x.dueAt - y.dueAt || a.localeCompare(b))
      .map(([id, state]) => this.reviewOf(id, state));
  }

  /** The active schedule of a problem, with the core's errors when there is none. */
  private activeReview(id: string): ReviewState {
    this.problem(id);
    if (!this.reviewsEnabled) throw new Error("reviews are turned off in Settings");
    const state = this.reviews.get(id);
    if (!state || state.retired) throw new Error(`no review is scheduled for \`${id}\``);
    return state;
  }

  /** Applies a stored accepted submit to the schedule, by the core's rules. */
  private recordAccepted(id: string, review: boolean): void {
    const now = this.now().getTime();
    const state = this.reviews.get(id);
    if (review) {
      if (!state || state.retired) return;
      state.lastReviewedAt = now;
      if (state.stage + 1 >= REVIEW_INTERVALS_DAYS.length) {
        state.retired = true;
      } else {
        state.stage += 1;
        state.dueAt = now + (REVIEW_INTERVALS_DAYS[state.stage] ?? 180) * DAY_MS;
      }
      // The review is resolved: its next attempt starts from the starter.
      this.clearReviewSolutions(id);
    } else if (!state && this.reviewsEnabled) {
      this.reviews.set(id, {
        stage: 0,
        dueAt: now + REVIEW_INTERVALS_DAYS[0] * DAY_MS,
        lastReviewedAt: null,
        retired: false,
      });
    }
  }

  private clearReviewSolutions(id: string): void {
    for (const stored of [...this.reviewSolutions.keys()]) {
      if (stored.startsWith(`${id}/`)) this.reviewSolutions.delete(stored);
    }
  }

  private sleep = (ms: number): Promise<void> =>
    this.speed === 0 ? Promise.resolve() : new Promise((resolve) => setTimeout(resolve, ms * this.speed));

  private async respond<T>(value: () => T): Promise<T> {
    await this.sleep(40);
    return value();
  }

  private problem(id: string): FixtureProblem {
    const problem = this.problems.get(id);
    if (!problem) throw new Error(`Unknown problem: ${id}`);
    return problem;
  }

  private stateOf(id: string): ProblemState {
    const state = this.state.get(id);
    if (!state) throw new Error(`Unknown problem: ${id}`);
    return state;
  }

  private assertFramework(problem: FixtureProblem, framework: Framework): void {
    if (!problem.meta.frameworks.includes(framework)) {
      throw new Error(`${problem.meta.id} does not support ${framework}`);
    }
  }

  /** What a run of `problem` would find on this machine right now. */
  private runContext(problem: FixtureProblem, target: TargetInfo): RunContext {
    const models = this.modelsOn(target.name);
    const missingAssets: MissingAsset[] = problem.meta.assets.flatMap((name) => {
      const asset = models.get(name);
      return asset && asset.state !== "present"
        ? [{ name: asset.name, repo: asset.repo, sizeMb: asset.sizeMb }]
        : [];
    });
    // A remote machine's hardware is what a check of it reports.
    const hardware = target.kind === "local" ? FIXTURE_HARDWARE : fixtureCheckScript(target).doctor?.hardware;
    return { missingAssets, hardware: hardware ?? ["cpu"] };
  }

  private target(name: string): TargetInfo {
    const target = this.targets.find((candidate) => candidate.name === name);
    if (!target) throw new Error(`unknown target \`${name}\``);
    return target;
  }

  /** The models on a target: the local disk, or a remote machine's (empty until fetched there). */
  private modelsOn(target: string): Map<string, FixtureAsset> {
    if (target === "local") return this.models;
    let models = this.remoteModels.get(target);
    if (!models) {
      models = new Map(
        fixtureAssets().map((asset) => [
          asset.name,
          { ...asset, state: "missing" as const, reason: null, bytesOnDisk: 0, failsOnce: false },
        ]),
      );
      this.remoteModels.set(target, models);
    }
    return models;
  }

  private summary(problem: FixtureProblem): ProblemSummary {
    const state = this.stateOf(problem.meta.id);
    return { ...problem.meta, status: state.status, solvedFrameworks: [...state.solved] };
  }

  private detail(problem: FixtureProblem): ProblemDetail {
    const state = this.stateOf(problem.meta.id);
    const unlocked = state.editorialRevealed || state.status === "solved";
    return {
      ...this.summary(problem),
      statement: problem.statement,
      hints: problem.hints,
      editorial: unlocked ? problem.editorial : null,
      editorialUnlocked: unlocked,
      entry: problem.entry,
      timeLimitS: problem.timeLimitS,
    };
  }

  /** Past submissions consistent with each problem's seeded status, spread over recent days. */
  private seedSubmissions(): void {
    const today = this.now().getTime();
    const seeded: { problem: FixtureProblem; framework: Framework; accepted: boolean; at: number }[] = [];
    for (const problem of this.catalog.problems) {
      const id = problem.meta.id;
      const state = this.stateOf(id);
      const lead = problem.meta.frameworks[0] ?? "torch";
      if (state.status === "solved") {
        // Most recent work first: the last six days are all active (the streak).
        const age = Math.floor(hash(`${id}:age`) * 24);
        const at = today - age * DAY_MS - hash(`${id}:t`) * 6 * 3_600_000 - 3_600_000;
        if (hash(`${id}:retry`) < 0.4) {
          seeded.push({ problem, framework: lead, accepted: false, at: at - 900_000 });
        }
        for (const framework of state.solved) {
          seeded.push({ problem, framework, accepted: true, at: at + (framework === lead ? 0 : 1_200_000) });
        }
      } else if (state.status === "attempted" && problem.savedTemplate) {
        const at =
          today - Math.floor(hash(`${id}:age`) * 5) * DAY_MS - hash(`${id}:t`) * 5 * 3_600_000 - 1_800_000;
        seeded.push({ problem, framework: lead, accepted: false, at });
      }
    }
    // Guarantee activity on each of the last six days so the streak is honest.
    const solved = seeded.filter((s) => s.accepted);
    for (let day = 0; day < 6 && day < solved.length; day++) {
      const entry = solved[day];
      if (entry) entry.at = today - day * DAY_MS - 5_400_000 - day * 600_000;
    }
    // Earlier practice, ten or twelve weeks back: a solved problem revisited on a quiet day. The
    // last six days stay the only ones in the streak; these all fall before them.
    for (const entry of solved) {
      const id = entry.problem.meta.id;
      if (hash(`${id}:old`) < 0.35) continue;
      const age = 26 + Math.floor(hash(`${id}:old-age`) * 58);
      const at = today - age * DAY_MS - hash(`${id}:old-t`) * 8 * 3_600_000;
      seeded.push({ problem: entry.problem, framework: entry.framework, accepted: true, at });
    }
    seeded.sort((a, b) => a.at - b.at);
    for (const { problem, framework, accepted, at } of seeded) {
      const code = renderTemplate(
        accepted ? problem.referenceTemplate : (problem.savedTemplate ?? problem.starterTemplate),
        framework,
      );
      const plan = buildPlan(
        accepted ? { ...problem, scenario: "accepted" } : failing(problem),
        framework,
        "submit",
        code,
        this.runContext(problem, this.target("local")),
      );
      const cases = plan.cases.map((c) => ({
        ...c.result,
        name: c.info.name,
        public: c.info.public,
        ...(c.info.stage ? { stage: c.info.stage } : {}),
      }));
      const verdict = plan.abort ?? verdictOf(cases, plan.requiresMet);
      this.submissions.push({
        id: this.nextSubmissionId++,
        problemId: problem.meta.id,
        framework,
        mode: "submit",
        verdict,
        passed: cases.filter((c) => c.status === "pass").length,
        total: cases.length,
        skipped: cases.filter((c) => c.status === "skipped").length,
        durationMs: Math.round(cases.reduce((sum, c) => sum + c.durationMs, 0) * 10) / 10,
        target: "local",
        createdAt: new Date(at).toISOString(),
        review: false,
        code,
        cases,
        metrics: verdict === "accepted" ? plan.metrics : [],
      });
    }
  }

  // Catalog -----------------------------------------------------------------

  listTracks = (): Promise<Track[]> =>
    this.respond(() =>
      this.catalog.tracks.map((track) => {
        const problems = this.catalog.problems.filter((p) => p.meta.track === track.id);
        return {
          id: track.id,
          title: track.title,
          description: track.description,
          order: track.order,
          problemCount: problems.length,
          solvedCount: problems.filter((p) => this.stateOf(p.meta.id).status === "solved").length,
        };
      }),
    );

  listProblems = (): Promise<ProblemSummary[]> =>
    this.respond(() => this.catalog.problems.map((problem) => this.summary(problem)));

  getProblem = (id: string): Promise<ProblemDetail> => this.respond(() => this.detail(this.problem(id)));

  // Solutions ---------------------------------------------------------------

  private solutionFile(id: string, framework: Framework): SolutionFile {
    const problem = this.problem(id);
    this.assertFramework(problem, framework);
    const starter = renderTemplate(problem.starterTemplate, framework);
    const saved = this.solutions.get(key(id, framework));
    return {
      code: saved ?? starter,
      isStarter: saved === undefined || saved === starter,
      path: `${ROOT}/workspace/${id}/${framework}.py`,
    };
  }

  readSolution = (id: string, framework: Framework): Promise<SolutionFile> =>
    this.respond(() => this.solutionFile(id, framework));

  writeSolution = (id: string, framework: Framework, code: string): Promise<void> =>
    this.respond(() => {
      const problem = this.problem(id);
      this.assertFramework(problem, framework);
      this.solutions.set(key(id, framework), code);
      const state = this.stateOf(id);
      if (state.status === "todo" && code !== renderTemplate(problem.starterTemplate, framework)) {
        state.status = "attempted";
      }
    });

  resetSolution = (id: string, framework: Framework): Promise<SolutionFile> =>
    this.respond(() => {
      this.solutions.delete(key(id, framework));
      return this.solutionFile(id, framework);
    });

  // Reviews -----------------------------------------------------------------

  private reviewFile(id: string, framework: Framework): SolutionFile {
    const problem = this.problem(id);
    this.assertFramework(problem, framework);
    const starter = renderTemplate(problem.starterTemplate, framework);
    const saved = this.reviewSolutions.get(key(id, framework));
    return {
      code: saved ?? starter,
      isStarter: saved === undefined || saved === starter,
      path: `${ROOT}/workspace/${id}/review/${framework}.py`,
    };
  }

  listReviews = (): Promise<Review[]> => this.respond(() => this.activeReviews());

  startReview = (id: string, framework: Framework): Promise<SolutionFile> =>
    this.respond(() => {
      this.activeReview(id);
      this.reviewSolutions.delete(key(id, framework));
      return this.reviewFile(id, framework);
    });

  readReviewSolution = (id: string, framework: Framework): Promise<SolutionFile> =>
    this.respond(() => this.reviewFile(id, framework));

  writeReviewSolution = (id: string, framework: Framework, code: string): Promise<void> =>
    this.respond(() => {
      this.assertFramework(this.problem(id), framework);
      this.activeReview(id);
      this.reviewSolutions.set(key(id, framework), code);
    });

  resolveReview = (id: string, action: ReviewAction): Promise<Review[]> =>
    this.respond(() => {
      const state = this.activeReview(id);
      const now = this.now().getTime();
      if (action === "forgot") {
        state.stage = 0;
        state.dueAt = now + DAY_MS;
      } else if (action === "snooze") {
        state.dueAt = Math.max(state.dueAt, now) + DAY_MS;
      } else {
        state.retired = true;
      }
      // A snoozed attempt is kept for the next day.
      if (action !== "snooze") this.clearReviewSolutions(id);
      return this.activeReviews();
    });

  private settings(): Settings {
    return { reviewsEnabled: this.reviewsEnabled, editorIntelligence: this.editor.enabled };
  }

  getSettings = (): Promise<Settings> => this.respond(() => this.settings());

  setReviewsEnabled = (enabled: boolean): Promise<Settings> =>
    this.respond(() => {
      this.reviewsEnabled = enabled;
      return this.settings();
    });

  // Editor intelligence -------------------------------------------------------

  setEditorIntelligence = (enabled: boolean): Promise<Settings> =>
    this.respond(() => {
      this.editor.setEnabled(enabled);
      return this.settings();
    });

  editorStatus = (): Promise<EditorStatus> => this.respond(() => this.editor.status());

  editorStart = (onEvent: (event: EditorEvent) => void): Promise<string> =>
    this.respond(() => this.editor.start(onEvent));

  editorSend = async (sessionId: string, message: string): Promise<void> => {
    this.editor.send(sessionId, message);
  };

  editorStop = (sessionId?: string): Promise<void> => this.respond(() => this.editor.stop(sessionId));

  async editorInstall(onEvent: (event: InstallEvent) => void): Promise<void> {
    await this.sleep(30);
    await this.editor.install(onEvent);
  }

  editorDocs = (source: string, expr: string, member?: string): Promise<EditorDocs> =>
    this.respond(() => this.editor.docs(source, expr, member));

  // Runs --------------------------------------------------------------------

  async startRun(args: StartRunArgs, onEvent: (event: RunEvent) => void): Promise<string> {
    const problem = this.problem(args.problemId);
    this.assertFramework(problem, args.framework);
    await this.sleep(30);
    const review = args.review === true;
    if (review) this.activeReview(args.problemId);
    const runId = `fixture-run-${this.nextRunId++}`;
    const code = review
      ? this.reviewFile(args.problemId, args.framework).code
      : this.solutionFile(args.problemId, args.framework).code;
    const target = this.target(args.target);
    const plan = buildPlan(problem, args.framework, args.mode, code, this.runContext(problem, target));

    void (async () => {
      const results: SubmissionDetail["cases"] = [];
      if (target.kind === "ssh") {
        // The remote steps a real run reports before the judge starts.
        const host = target.host ?? target.label;
        const script = fixtureCheckScript(target);
        const failing = script.steps.find((step) => !step.ok);
        for (const message of [
          `Preparing \`${args.problemId}\` on the ${target.name} target`,
          `Connecting to ${host}`,
          ...(failing
            ? []
            : [`Copying the judge to ${host}`, `Copying the problem and your solution to ${host}`]),
        ]) {
          if (this.canceled.has(runId)) break;
          onEvent({ type: "preparing", message });
          await this.sleep(450);
        }
        if (failing && !this.canceled.has(runId)) {
          onEvent({
            type: "end",
            verdict: "harness_error",
            passed: 0,
            total: 0,
            durationMs: 900,
            metrics: [],
            submissionId: null,
            skipped: 0,
            message: [failing.message, failing.fix].filter(Boolean).join(" "),
            detail: null,
          });
          return;
        }
      }
      const events = simulateRun(plan, {
        sleep: this.sleep,
        isCanceled: () => this.canceled.has(runId),
      });
      for await (const event of events) {
        if (event.type === "caseEnd") {
          const info = plan.cases.find((c) => c.info.id === event.result.id)?.info;
          results.push({
            ...event.result,
            name: info?.name ?? event.result.id,
            public: info?.public ?? true,
            ...(info?.stage ? { stage: info.stage } : {}),
          });
        }
        if (event.type === "end" && args.mode === "submit" && event.verdict !== "canceled") {
          const id = this.nextSubmissionId++;
          this.submissions.push({
            id,
            problemId: args.problemId,
            framework: args.framework,
            mode: "submit",
            verdict: event.verdict,
            passed: event.passed,
            total: event.total,
            skipped: event.skipped,
            durationMs: event.durationMs,
            target: args.target,
            createdAt: this.now().toISOString(),
            review,
            code,
            cases: results,
            metrics: event.metrics,
          });
          const state = this.stateOf(args.problemId);
          if (event.verdict === "accepted") {
            state.status = "solved";
            state.solved.add(args.framework);
            this.recordAccepted(args.problemId, review);
          } else if (state.status === "todo") {
            state.status = "attempted";
          }
          onEvent({ ...event, submissionId: id });
        } else {
          onEvent(event);
        }
      }
      this.canceled.delete(runId);
    })();

    return runId;
  }

  cancelRun = (runId: string): Promise<void> =>
    this.respond(() => {
      this.canceled.add(runId);
    });

  // Submissions -------------------------------------------------------------

  listSubmissions = (problemId?: string): Promise<Submission[]> =>
    this.respond(() =>
      this.submissions
        .filter((s) => problemId === undefined || s.problemId === problemId)
        .map(({ code: _code, cases: _cases, metrics: _metrics, ...submission }) => submission)
        .sort((a, b) => b.createdAt.localeCompare(a.createdAt) || b.id - a.id),
    );

  getSubmission = (id: number): Promise<SubmissionDetail> =>
    this.respond(() => {
      const submission = this.submissions.find((s) => s.id === id);
      if (!submission) throw new Error(`Unknown submission: ${id}`);
      return submission;
    });

  // Editorial ---------------------------------------------------------------

  unlockEditorial = (id: string): Promise<ProblemDetail> =>
    this.respond(() => {
      this.stateOf(id).editorialRevealed = true;
      return this.detail(this.problem(id));
    });

  getReference = (id: string, framework: Framework): Promise<string> =>
    this.respond(() => {
      const problem = this.problem(id);
      this.assertFramework(problem, framework);
      if (!this.detail(problem).editorialUnlocked) {
        throw new Error("The reference is available once the editorial is unlocked.");
      }
      return renderTemplate(problem.referenceTemplate, framework);
    });

  // Environment -------------------------------------------------------------

  envStatus = (): Promise<EnvStatus> =>
    this.respond(() => ({
      uv: "0.13.8",
      python: "3.15.2",
      frameworks: [...this.installed].map(([framework, version]) => ({
        framework,
        installed: version !== null,
        version,
      })),
      root: ROOT,
    }));

  async envInstall(framework: Framework, onEvent: (event: InstallEvent) => void): Promise<void> {
    await this.sleep(30);
    const version = framework === "mlx" ? "0.32.4" : framework === "jax" ? "0.12.0.1" : "2.15.0";
    const lines = [
      `$ uv sync --project python --extra jax --extra ${framework}`,
      "Resolved 61 packages in 4ms",
      `Downloading ${framework} (38.2MiB)`,
      `   Building environment for CPython 3.15.2`,
      `Prepared 3 packages in 2.41s`,
      `Installed 3 packages in 86ms`,
      ` + ${framework}==${version}`,
      ` + ${framework}-metal==${version}`,
      " + typing-extensions==4.15.0",
      `$ python -c "import ${framework === "mlx" ? "mlx.core" : framework}"`,
      "ok",
    ];
    void (async () => {
      for (const text of lines) {
        await this.sleep(280);
        onEvent({ type: "log", text });
      }
      await this.sleep(200);
      this.installed.set(framework, version);
      onEvent({ type: "done", ok: true, message: null });
    })();
  }

  // Stats -------------------------------------------------------------------

  getStats = (): Promise<Stats> =>
    this.respond(() => {
      const byDifficulty: Record<Difficulty, [number, number]> = {
        easy: [0, 0],
        medium: [0, 0],
        hard: [0, 0],
      };
      let solved = 0;
      for (const problem of this.catalog.problems) {
        const bucket = byDifficulty[problem.meta.difficulty];
        bucket[1] += 1;
        if (this.stateOf(problem.meta.id).status === "solved") {
          bucket[0] += 1;
          solved += 1;
        }
      }
      return {
        solved,
        total: this.catalog.problems.length,
        byDifficulty,
        streakDays: this.streak(),
        reviewsDue: this.activeReviews().filter((review) => review.due).length,
      };
    });

  /** One sample issue, so the Settings notice can be seen without a broken catalog. */
  listCatalogIssues = (): Promise<CatalogIssue[]> =>
    this.respond(() => [
      {
        path: `${ROOT}/problems/sampling/min-p-draft/problem.toml`,
        message: "missing starter/torch.py",
      },
    ]);

  /**
   * The fixture path, with core's rule applied: an id the catalog does not know
   * is dropped, and a stage left empty is not listed.
   */
  listPaths = (): Promise<Path[]> =>
    this.respond(() =>
      FIXTURE_PATHS.map((path) => {
        const stages = path.stages
          .map((stage) => ({ ...stage, problems: stage.problems.filter((id) => this.problems.has(id)) }))
          .filter((stage) => stage.problems.length > 0);
        const ids = stages.flatMap((stage) => stage.problems);
        return {
          id: path.id,
          title: path.title,
          description: path.description,
          stages,
          problemCount: ids.length,
          solvedCount: ids.filter((id) => this.stateOf(id).status === "solved").length,
        };
      }),
    );

  /** Serves the figures bundled with authored fixture statements, with core's path rules. */
  getProblemAsset = (id: string, path: string): Promise<string> =>
    this.respond(() => {
      const problem = this.problem(id);
      if (path === "" || path.startsWith("/") || path.includes("\\") || path.split("/").some(isDotSegment)) {
        throw new Error(`invalid asset path \`${path}\`: it must stay inside the problem's assets directory`);
      }
      if (!/\.(svg|png)$/i.test(path)) {
        throw new Error(`invalid asset path \`${path}\`: only .svg and .png files are served`);
      }
      const svg = problem.assets[path];
      if (svg === undefined)
        throw new Error(`missing file ${ROOT}/problems/${problem.meta.track}/${id}/assets/${path}`);
      return `data:image/svg+xml;base64,${base64(svg)}`;
    });

  // Model assets ------------------------------------------------------------

  private assetList(): AssetList {
    const assets: ModelAsset[] = [...this.models.values()]
      .map((asset) => ({
        name: asset.name,
        repo: asset.repo,
        revision: asset.revision,
        files: asset.files,
        sizeMb: asset.sizeMb,
        state: asset.state,
        reason: asset.reason,
        path: `${MODELS_DIR}/${asset.name}`,
        bytesOnDisk: asset.bytesOnDisk,
        problems: this.catalog.problems
          .filter((problem) => problem.meta.assets.includes(asset.name))
          .map((problem) => problem.meta.id)
          .sort(),
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
    return { modelsDir: MODELS_DIR, assets };
  }

  listAssets = (): Promise<AssetList> => this.respond(() => this.assetList());

  /**
   * Plays a download: the environment check, then bytes arriving until the
   * asset is complete. A fetch resumes from what is on disk, a cancel keeps it,
   * and an asset marked `failsOnce` drops its connection on the first attempt.
   */
  async fetchAssets(
    names: string[],
    onEvent: (event: AssetEvent) => void,
    target = "local",
  ): Promise<string> {
    await this.sleep(30);
    const where = this.target(target);
    const models = this.modelsOn(target);
    if (names.length === 0) throw new Error("name at least one model asset to download");
    if (this.fetching) {
      throw new Error("a model download is already running; wait for it to finish or cancel it");
    }
    const job = { jobId: `fixture-fetch-${this.nextJobId++}`, canceled: false };
    this.fetching = job;

    void (async () => {
      const failures: string[] = [];
      const finish = (event: AssetEvent) => {
        this.fetching = null;
        onEvent(event);
      };
      if (where.kind === "ssh") {
        onEvent({ type: "preparing", message: `Connecting to ${where.host}` });
        await this.sleep(500);
        onEvent({ type: "preparing", message: `Copying the judge to ${where.host}` });
      } else {
        onEvent({ type: "preparing", message: "Checking the Python environment" });
      }
      await this.sleep(500);
      for (const name of names) {
        if (job.canceled) break;
        const asset = models.get(name);
        if (!asset) {
          const known = [...models.keys()].sort().join(", ");
          return finish({
            type: "done",
            ok: false,
            canceled: false,
            message: `learnai assets fetch: no asset named ${name} in the catalog (known: ${known})`,
          });
        }
        const total = asset.totalBytes;
        onEvent({ type: "start", asset: name, totalBytes: total });
        const step = Math.ceil(total / FETCH_STEPS);
        const dropAt = asset.failsOnce ? asset.bytesOnDisk + (total - asset.bytesOnDisk) * 0.45 : null;
        while (asset.bytesOnDisk < total && !job.canceled) {
          await this.sleep(FETCH_STEP_MS);
          if (job.canceled) break;
          asset.bytesOnDisk = Math.min(total, asset.bytesOnDisk + step);
          asset.state = asset.bytesOnDisk < total ? "incomplete" : asset.state;
          asset.reason = "not downloaded yet: model.safetensors";
          onEvent({
            type: "progress",
            asset: name,
            file: "model.safetensors",
            doneBytes: asset.bytesOnDisk,
            totalBytes: total,
          });
          if (dropAt !== null && asset.bytesOnDisk >= dropAt) break;
        }
        if (job.canceled) break;
        if (asset.bytesOnDisk < total) {
          asset.failsOnce = false;
          const message =
            "model.safetensors: the connection was reset; fetch again to resume from the bytes already on disk";
          failures.push(`${name}: ${message}`);
          onEvent({ type: "assetEnd", asset: name, ok: false, message });
          continue;
        }
        asset.state = "present";
        asset.reason = null;
        onEvent({ type: "assetEnd", asset: name, ok: true, message: null });
      }
      if (job.canceled) return finish({ type: "done", ok: false, canceled: true, message: null });
      finish({
        type: "done",
        ok: failures.length === 0,
        canceled: false,
        message: failures.length > 0 ? failures.join("; ") : null,
      });
    })();

    return job.jobId;
  }

  cancelAssetFetch = (jobId: string): Promise<void> =>
    this.respond(() => {
      if (this.fetching?.jobId === jobId) this.fetching.canceled = true;
    });

  removeAsset = (name: string): Promise<AssetList> =>
    this.respond(() => {
      const asset = this.models.get(name);
      if (!asset) throw new Error(`\`${name}\` is not a model asset any problem declares`);
      if (this.fetching) {
        throw new Error("a model download is already running; wait for it to finish or cancel it");
      }
      asset.state = "missing";
      asset.reason = null;
      asset.bytesOnDisk = 0;
      return this.assetList();
    });

  /** Consecutive local days, ending today or yesterday, with an accepted submit. */
  private streak(): number {
    const days = new Set(
      this.submissions.filter((s) => s.verdict === "accepted").map((s) => dayKey(new Date(s.createdAt))),
    );
    const cursor = this.now();
    if (!days.has(dayKey(cursor))) cursor.setDate(cursor.getDate() - 1);
    let count = 0;
    while (days.has(dayKey(cursor))) {
      count += 1;
      cursor.setDate(cursor.getDate() - 1);
    }
    return count;
  }

  // Run targets -------------------------------------------------------------

  listSshHosts = (): Promise<SshHost[]> => this.respond(() => FIXTURE_SSH_HOSTS.map((host) => ({ ...host })));

  listTargets = (): Promise<TargetInfo[]> =>
    this.respond(() => this.targets.map((target) => ({ ...target })));

  addTarget = (host: string, remoteDir?: string): Promise<TargetInfo[]> =>
    this.respond(() => {
      if (!FIXTURE_SSH_HOSTS.some((known) => known.alias === host)) {
        throw new Error(
          `\`${host}\` is not a host in your SSH configuration. Add a \`Host ${host}\` entry to ~/.ssh/config first.`,
        );
      }
      if (this.targets.some((target) => target.host === host)) {
        throw new Error(`\`${host}\` is already a run target`);
      }
      this.targets = [
        ...this.targets,
        {
          name: `ssh:${host}`,
          kind: "ssh",
          label: host,
          host,
          remoteDir: remoteDir?.trim() || "~/.learnai",
          lastCheck: null,
        },
      ];
      return this.targets.map((target) => ({ ...target }));
    });

  removeTarget = (name: string): Promise<TargetInfo[]> =>
    this.respond(() => {
      if (name === "local") throw new Error("the local target cannot be removed");
      this.target(name);
      this.targets = this.targets.filter((target) => target.name !== name);
      this.remoteModels.delete(name);
      return this.targets.map((target) => ({ ...target }));
    });

  /** Plays a check: each step ends after a moment, the environment build prints a few lines. */
  async checkTarget(name: string, onEvent: (event: TargetCheckEvent) => void): Promise<string> {
    await this.sleep(30);
    const target = this.target(name);
    if (this.checking.has(name)) throw new Error(`\`${name}\` is already being checked`);
    this.checking.add(name);
    const jobId = `fixture-check-${this.nextJobId++}`;
    const { steps, doctor } = fixtureCheckScript(target);

    void (async () => {
      for (const step of steps) {
        if (step.id === "environment") {
          for (const text of ["Resolved 61 packages in 12ms", "Audited 58 packages in 3ms"]) {
            await this.sleep(350);
            onEvent({ type: "log", text });
          }
        }
        await this.sleep(step.id === "connect" ? 700 : 550);
        onEvent({ type: "step", step });
      }
      const check = {
        ok: steps.every((step) => step.ok),
        checkedAt: this.now().toISOString(),
        steps,
        doctor,
      };
      this.targets = this.targets.map((candidate) =>
        candidate.name === name ? { ...candidate, lastCheck: check } : candidate,
      );
      this.checking.delete(name);
      onEvent({ type: "done", check });
    })();

    return jobId;
  }
}

function key(id: string, framework: Framework): string {
  return `${id}/${framework}`;
}

function isDotSegment(segment: string): boolean {
  return segment === "" || segment === "." || segment === "..";
}

/** Base64 of the UTF-8 encoding of `text`. */
function base64(text: string): string {
  let binary = "";
  for (const byte of new TextEncoder().encode(text)) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function dayKey(date: Date): string {
  return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
}

/**
 * A problem whose seeded outcome is not a solve, for building unsolved history
 * entries. One with cases that need a model or hardware already ends "not
 * judged" on the fixture machine; any other is made to fail.
 */
function failing(problem: FixtureProblem): FixtureProblem {
  if (problem.cases.some((fixtureCase) => fixtureCase.needs)) return problem;
  return problem.scenario === "accepted" ? { ...problem, scenario: "wrong_answer" } : problem;
}
