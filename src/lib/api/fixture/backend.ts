import type { Backend, StartRunArgs } from "../backend";
import type {
  CatalogIssue,
  Difficulty,
  EnvStatus,
  Framework,
  InstallEvent,
  ProblemDetail,
  ProblemStatus,
  ProblemSummary,
  RunEvent,
  SolutionFile,
  Stats,
  Submission,
  SubmissionDetail,
  Track,
} from "../types";
import { buildCatalog, type FixtureProblem, hash, renderTemplate } from "./catalog";
import { buildPlan, simulateRun, verdictOf } from "./runs";

export interface FixtureOptions {
  /** Multiplier on every simulated delay. 0 makes the backend instantaneous (tests). */
  speed?: number;
  now?: () => Date;
}

const ROOT = "/Users/ada/Developer/whetstone";
const DAY_MS = 86_400_000;

interface ProblemState {
  status: ProblemStatus;
  solved: Set<Framework>;
  editorialRevealed: boolean;
}

/**
 * In-memory stand-in for whetstone-core, for component work and tests. It keeps
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
  private readonly submissions: SubmissionDetail[] = [];
  private readonly canceled = new Set<string>();
  private readonly installed = new Map<Framework, string | null>([
    ["torch", "2.15.0"],
    ["jax", "0.12.0.1"],
    ["mlx", null],
  ]);
  private nextSubmissionId = 1;
  private nextRunId = 1;

  constructor(options: FixtureOptions = {}) {
    this.speed = options.speed ?? 1;
    this.now = options.now ?? (() => new Date());
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
      );
      const cases = plan.cases.map((c) => ({ ...c.result, name: c.info.name, public: c.info.public }));
      const verdict = plan.abort ?? verdictOf(cases);
      this.submissions.push({
        id: this.nextSubmissionId++,
        problemId: problem.meta.id,
        framework,
        mode: "submit",
        verdict,
        passed: cases.filter((c) => c.status === "pass").length,
        total: cases.length,
        durationMs: Math.round(cases.reduce((sum, c) => sum + c.durationMs, 0) * 10) / 10,
        target: "local",
        createdAt: new Date(at).toISOString(),
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

  // Runs --------------------------------------------------------------------

  async startRun(args: StartRunArgs, onEvent: (event: RunEvent) => void): Promise<string> {
    const problem = this.problem(args.problemId);
    this.assertFramework(problem, args.framework);
    await this.sleep(30);
    const runId = `fixture-run-${this.nextRunId++}`;
    const code = this.solutionFile(args.problemId, args.framework).code;
    const plan = buildPlan(problem, args.framework, args.mode, code);

    void (async () => {
      const results: SubmissionDetail["cases"] = [];
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
            durationMs: event.durationMs,
            target: args.target,
            createdAt: this.now().toISOString(),
            code,
            cases: results,
            metrics: event.metrics,
          });
          const state = this.stateOf(args.problemId);
          if (event.verdict === "accepted") {
            state.status = "solved";
            state.solved.add(args.framework);
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
}

function key(id: string, framework: Framework): string {
  return `${id}/${framework}`;
}

function dayKey(date: Date): string {
  return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
}

/** A problem whose seeded outcome is a failure, for building failed history entries. */
function failing(problem: FixtureProblem): FixtureProblem {
  return problem.scenario === "accepted" ? { ...problem, scenario: "wrong_answer" } : problem;
}
