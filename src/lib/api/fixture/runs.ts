import type { CaseInfo, CaseResult, Framework, Metric, RunEvent, RunMode, Verdict } from "../types";
import { type FixtureProblem, hash, type Scenario } from "./catalog";

export interface PlannedCase {
  info: CaseInfo;
  result: CaseResult;
  /** Wall-clock time the simulation spends on this case. */
  simulatedMs: number;
}

export interface RunPlan {
  preparing: string[];
  env: Record<string, string>;
  cases: PlannedCase[];
  logs: { stream: "stdout" | "stderr"; text: string }[];
  /** When set, the run ends before any case starts (import and harness failures). */
  abort: Verdict | null;
  metrics: Metric[];
}

const SCENARIOS: Scenario[] = [
  "accepted",
  "wrong_answer",
  "cost_exceeded",
  "too_slow",
  "runtime_error",
  "time_limit",
  "import_error",
  "harness_error",
];

/**
 * Picks the outcome for a run. A `# ws: <scenario>` comment in the code forces
 * one (useful when demonstrating states); untouched starter code raises; any
 * other code gets the problem's seeded outcome.
 */
export function resolveScenario(problem: FixtureProblem, code: string): Scenario {
  const marker = /#\s*ws:\s*([a-z_]+)/.exec(code)?.[1];
  if (marker && (SCENARIOS as string[]).includes(marker)) return marker as Scenario;
  if (/raise NotImplementedError/.test(code)) return "runtime_error";
  return problem.scenario;
}

function envFor(framework: Framework): Record<string, string> {
  const base = { python: "3.15.2" };
  switch (framework) {
    case "torch":
      return { ...base, torch: "2.15.0", device: "mps" };
    case "jax":
      return { ...base, jax: "0.12.0.1", device: "cpu" };
    case "mlx":
      return { ...base, mlx: "0.32.4", device: "gpu" };
    case "python":
      return base;
  }
}

function pass(id: string, durationMs: number, metrics: Metric[] = []): CaseResult {
  return { id, status: "pass", durationMs, message: null, detail: null, stdout: "", metrics };
}

function traceback(problem: FixtureProblem, framework: Framework, starter: boolean): string {
  const entry = problem.entry[0] ?? "solve";
  const path = `workspace/${problem.meta.id}/${framework}.py`;
  if (starter) {
    return [
      "Traceback (most recent call last):",
      `  File "whetstone/harness/runner.py", line 212, in run_case`,
      "    outcome = case.check(solution)",
      `  File "problems/${problem.meta.track}/${problem.meta.id}/tests.py", line 31, in check`,
      `    got = solution.${entry}(*args)`,
      `  File "${path}", line 6, in ${entry}`,
      "    raise NotImplementedError",
      "NotImplementedError",
    ].join("\n");
  }
  return [
    "Traceback (most recent call last):",
    `  File "whetstone/harness/runner.py", line 212, in run_case`,
    "    outcome = case.check(solution)",
    `  File "problems/${problem.meta.track}/${problem.meta.id}/tests.py", line 31, in check`,
    `    got = solution.${entry}(*args)`,
    `  File "${path}", line 9, in ${entry}`,
    "    weights = scores.softmax(dim=-1) @ v",
    "RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x512 and 64x64)",
  ].join("\n");
}

const WRONG_DETAIL: Record<string, { message: string; detail: string; stdout: string }> = {
  "top-p-sampling": {
    message: "Empirical frequencies do not match the nucleus distribution (p-value below 1e-9)",
    detail: [
      "probs = [0.50, 0.30, 0.15, 0.05]   p = 0.9   200,000 draws",
      "",
      "token   expected   observed",
      "    0    0.5263     0.6251",
      "    1    0.3158     0.3749",
      "    2    0.1579     0.0000",
      "    3    0.0000     0.0000",
      "",
      "chi-squared = 38211.4 (2 degrees of freedom)",
      "Token 2 crosses the threshold and belongs to the nucleus, but was never sampled.",
    ].join("\n"),
    stdout: "nucleus sizes: [2]\n",
  },
};

function wrongAnswer(problem: FixtureProblem, hidden: boolean) {
  const authored = WRONG_DETAIL[problem.meta.id];
  if (hidden) {
    return { message: "Output does not match the reference", detail: null, stdout: "" };
  }
  if (authored) return authored;
  return {
    message: "Max absolute error 1.2e-04 exceeds tolerance 1.0e-06",
    detail: [
      "expected: tensor([0.0900, 0.2447, 0.6652])",
      "     got: tensor([0.0900, 0.2446, 0.6653])",
      "",
      "first mismatch at index (1,): expected 0.24472848, got 0.24460852",
    ].join("\n"),
    stdout: "",
  };
}

/** Builds the full, deterministic script for one simulated run. */
export function buildPlan(
  problem: FixtureProblem,
  framework: Framework,
  mode: RunMode,
  code: string,
): RunPlan {
  const scenario = resolveScenario(problem, code);
  const starter = /raise NotImplementedError/.test(code);
  const visible = problem.cases.filter((c) => mode === "submit" || c.public);
  const infos: CaseInfo[] = visible.map((c, i) => ({
    id: `c${i + 1}`,
    name: c.name,
    public: c.public,
  }));
  const plan: RunPlan = {
    preparing: ["Checking the Python environment", "Starting the judge"],
    env: envFor(framework),
    cases: [],
    logs: [],
    abort: null,
    metrics: [],
  };

  if (scenario === "harness_error") {
    plan.logs.push({
      stream: "stderr",
      text: "error: failed to spawn `uv`: No such file or directory (os error 2)\nInstall uv, then check Settings > Environment.",
    });
    plan.abort = "harness_error";
    return plan;
  }
  if (scenario === "import_error") {
    plan.logs.push({
      stream: "stderr",
      text: [
        "Traceback (most recent call last):",
        `  File "workspace/${problem.meta.id}/${framework}.py", line 2, in <module>`,
        "    import einops",
        "ModuleNotFoundError: No module named 'einops'",
      ].join("\n"),
    });
    plan.abort = "import_error";
    plan.cases = infos.map((info) => ({
      info,
      result: { ...pass(info.id, 0), status: "skipped" },
      simulatedMs: 0,
    }));
    return plan;
  }

  const lastPublic = infos.findLastIndex((info) => info.public);
  const lastIndex = infos.length - 1;
  let timedOut = false;

  plan.cases = infos.map((info, index) => {
    const base = 2 + Math.round(hash(`${problem.meta.id}:${info.id}`) * 38);
    const durationMs = Math.round((base + hash(`${info.id}:${framework}`) * 3) * 10) / 10;
    const simulatedMs = 90 + Math.round(hash(`${problem.meta.id}:${info.id}:sim`) * 260);
    const planned = (result: CaseResult, ms = simulatedMs): PlannedCase => ({
      info,
      result,
      simulatedMs: ms,
    });

    if (timedOut) {
      return planned({ ...pass(info.id, 0), status: "skipped" }, 0);
    }

    switch (scenario) {
      case "accepted":
        return planned(pass(info.id, durationMs, caseMetrics(problem, true)));
      case "runtime_error":
        return planned({
          ...pass(info.id, 0.4),
          status: "error",
          message: starter
            ? "NotImplementedError"
            : "RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x512 and 64x64)",
          detail: traceback(problem, framework, starter),
        });
      case "wrong_answer": {
        const failing = index === lastPublic || (!info.public && index === lastIndex);
        if (!failing) return planned(pass(info.id, durationMs));
        return planned({
          ...pass(info.id, durationMs),
          status: "fail",
          ...wrongAnswer(problem, !info.public),
        });
      }
      case "cost_exceeded": {
        if (index !== 1)
          return planned(pass(info.id, durationMs, index === 0 ? [] : caseMetrics(problem, true)));
        return planned({
          ...pass(info.id, durationMs),
          status: "fail",
          message: "model_tokens: 41 exceeds the limit of 36",
          detail: [
            "forward calls (tokens per call): 32, 2, 2, 2, 2, 1",
            "",
            "The output ids are correct, but the model processed 41 tokens.",
            "The optimal decode for a prompt of 32 and 5 new tokens processes 36.",
          ].join("\n"),
          metrics: [{ name: "model_tokens", value: 41, unit: "tokens", limit: 36 }],
        });
      }
      case "too_slow": {
        if (index !== lastIndex || info.public) return planned(pass(info.id, durationMs));
        return planned(
          {
            ...pass(info.id, 1840.5),
            status: "fail",
            message: "212x slower than the reference; the limit is 1.5x",
            detail: null,
            metrics: [{ name: "slowdown", value: 212.4, unit: "x", limit: 1.5 }],
          },
          1400,
        );
      }
      case "time_limit": {
        if (index < lastPublic) return planned(pass(info.id, durationMs, caseMetrics(problem, true)));
        timedOut = true;
        return planned(
          {
            ...pass(info.id, problem.timeLimitS * 1000),
            status: "timeout",
            message: `Exceeded the ${problem.timeLimitS} s time limit`,
            detail: "The process was stopped. Cases after a timeout are skipped.",
            stdout: "step 10000  queue=812\nstep 20000  queue=1604\nstep 30000  queue=2391\n",
          },
          1600,
        );
      }
    }
    return planned(pass(info.id, durationMs));
  });

  const allPassed = plan.cases.every((c) => c.result.status === "pass");
  plan.metrics = allPassed ? caseMetrics(problem, false) : [];
  return plan;
}

function caseMetrics(problem: FixtureProblem, perCase: boolean): Metric[] {
  const judges = problem.meta.judges;
  if (judges.includes("cost")) {
    return [{ name: "model_tokens", value: 36, unit: "tokens", limit: 36 }];
  }
  if (judges.includes("simclock")) {
    return [
      { name: "mean_ttft", value: 41.2, unit: "ms", limit: 41.6 },
      { name: "makespan", value: 9204, unit: "ms", limit: 9296 },
    ];
  }
  if (judges.includes("speed") && !perCase) {
    return [{ name: "slowdown", value: 1.08, unit: "x", limit: 1.5 }];
  }
  return [];
}

/** Verdict precedence from the event protocol: import, runtime, time limit, wrong answer. */
export function verdictOf(results: CaseResult[]): Verdict {
  if (results.some((r) => r.status === "error")) return "runtime_error";
  if (results.some((r) => r.status === "timeout")) return "time_limit";
  if (results.some((r) => r.status === "fail")) return "wrong_answer";
  return "accepted";
}

export interface SimulationOptions {
  /** Resolves after `ms` of simulated time. Tests pass an immediate sleep. */
  sleep: (ms: number) => Promise<void>;
  isCanceled: () => boolean;
}

/**
 * Plays a plan as the ordered event stream the core would produce. The last
 * event is always a single `end`; a cancel yields `end` with verdict "canceled".
 */
export async function* simulateRun(
  plan: RunPlan,
  { sleep, isCanceled }: SimulationOptions,
): AsyncGenerator<RunEvent> {
  const finished: CaseResult[] = [];
  let elapsed = 0;
  const failureLog = plan.abort ? (plan.logs.at(-1)?.text ?? null) : null;
  const end = (verdict: Verdict): RunEvent => ({
    message: verdict === plan.abort ? (failureLog?.split("\n").at(-1) ?? null) : null,
    detail: verdict === plan.abort ? failureLog : null,
    type: "end",
    verdict,
    passed: finished.filter((r) => r.status === "pass").length,
    total: plan.cases.length,
    durationMs: Math.round(elapsed * 10) / 10,
    metrics: verdict === "accepted" ? plan.metrics : [],
    submissionId: null,
  });

  for (const message of plan.preparing) {
    yield { type: "preparing", message };
    await sleep(350);
    if (isCanceled()) return yield end("canceled");
  }

  if (plan.abort === "harness_error") {
    for (const log of plan.logs) yield { type: "log", ...log };
    return yield end("harness_error");
  }

  yield { type: "start", cases: plan.cases.map((c) => c.info), env: plan.env };
  for (const log of plan.logs) yield { type: "log", ...log };
  if (plan.abort) return yield end(plan.abort);

  for (const planned of plan.cases) {
    if (planned.result.status === "skipped") {
      finished.push(planned.result);
      yield { type: "caseEnd", result: planned.result };
      continue;
    }
    yield { type: "caseStart", id: planned.info.id };
    await sleep(planned.simulatedMs);
    if (isCanceled()) return yield end("canceled");
    elapsed += planned.result.durationMs;
    finished.push(planned.result);
    yield { type: "caseEnd", result: planned.result };
  }

  yield end(verdictOf(finished));
}
