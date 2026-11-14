import { describe, expect, it } from "vitest";
import type { RunEvent, RunMode } from "../types";
import { FixtureBackend } from "./backend";
import { buildCatalog, renderTemplate } from "./catalog";
import { buildPlan, resolveScenario, simulateRun } from "./runs";

const instant = { sleep: () => Promise.resolve(), isCanceled: () => false };

async function collect(events: AsyncGenerator<RunEvent>): Promise<RunEvent[]> {
  const out: RunEvent[] = [];
  for await (const event of events) out.push(event);
  return out;
}

function runToEnd(backend: FixtureBackend, problemId: string, mode: RunMode): Promise<RunEvent[]> {
  return new Promise((resolve, reject) => {
    const events: RunEvent[] = [];
    backend
      .startRun(
        {
          problemId,
          framework: problemId === "continuous-batching" ? "python" : "torch",
          mode,
          target: "local",
        },
        (event) => {
          events.push(event);
          if (event.type === "end") resolve(events);
        },
      )
      .catch(reject);
  });
}

const catalog = buildCatalog();
const byId = (id: string) => {
  const problem = catalog.problems.find((p) => p.meta.id === id);
  if (!problem) throw new Error(id);
  return problem;
};

describe("fixture catalog", () => {
  it("is large and internally consistent", () => {
    expect(catalog.tracks.length).toBeGreaterThanOrEqual(3);
    expect(catalog.problems.length).toBeGreaterThanOrEqual(180);
    const ids = new Set(catalog.problems.map((p) => p.meta.id));
    expect(ids.size).toBe(catalog.problems.length);
    for (const problem of catalog.problems) {
      expect(catalog.tracks.some((t) => t.id === problem.meta.track)).toBe(true);
      expect(problem.meta.frameworks.length).toBeGreaterThan(0);
      expect(problem.cases.some((c) => c.public)).toBe(true);
      for (const prerequisite of problem.meta.prerequisites) expect(ids.has(prerequisite)).toBe(true);
    }
  });

  it("has at least twelve authored statements with math, code, and tables", () => {
    const authored = catalog.problems.filter((p) => p.statement.includes("|---"));
    expect(authored.length).toBeGreaterThanOrEqual(12);
    for (const problem of authored) {
      expect(problem.statement).toMatch(/\$[^$]+\$/);
      expect(problem.statement).toContain("```python");
      expect(problem.hints.length).toBeGreaterThanOrEqual(2);
    }
  });

  it("covers all seven judges", () => {
    const judges = new Set(catalog.problems.flatMap((p) => p.meta.judges));
    expect([...judges].sort()).toEqual([
      "cost",
      "numeric",
      "property",
      "service",
      "simclock",
      "speed",
      "statistical",
    ]);
  });

  it("renders code templates per framework", () => {
    const template = "{{import}}\n\ndef f(x: {{T}}) -> {{T}}:\n    return torch.exp(x)\n";
    expect(renderTemplate(template, "torch")).toBe(
      "import torch\n\ndef f(x: torch.Tensor) -> torch.Tensor:\n    return torch.exp(x)\n",
    );
    expect(renderTemplate(template, "mlx")).toContain("return mx.exp(x)");
    expect(renderTemplate(template, "python")).toBe(
      "def f(x: list[float]) -> list[float]:\n    return torch.exp(x)\n",
    );
  });
});

describe("simulated run stream", () => {
  it("is well formed for every scenario: one start at most, ordered cases, exactly one end, last", async () => {
    for (const scenario of [
      "accepted",
      "wrong_answer",
      "cost_exceeded",
      "too_slow",
      "runtime_error",
      "time_limit",
      "import_error",
      "harness_error",
    ]) {
      const plan = buildPlan(byId("kv-cache-decode"), "torch", "submit", `# ws: ${scenario}\n`);
      const events = await collect(simulateRun(plan, instant));
      const types = events.map((e) => e.type);
      expect(types.filter((t) => t === "end")).toHaveLength(1);
      expect(types.at(-1)).toBe("end");
      expect(types.filter((t) => t === "start").length).toBeLessThanOrEqual(1);
      expect(types[0]).toBe("preparing");
      const started = new Set<string>();
      for (const event of events) {
        if (event.type === "caseStart") started.add(event.id);
        if (event.type === "caseEnd" && event.result.status !== "skipped") {
          expect(started.has(event.result.id)).toBe(true);
        }
      }
    }
  });

  it("produces each verdict with the detail the UI needs", async () => {
    const end = async (id: string, code: string, mode: RunMode = "submit") => {
      const events = await collect(simulateRun(buildPlan(byId(id), "torch", mode, code), instant));
      const last = events.at(-1);
      if (last?.type !== "end") throw new Error("no end");
      const results = events.flatMap((e) => (e.type === "caseEnd" ? [e.result] : []));
      return { last, results, events };
    };

    const accepted = await end("stable-softmax", "ok");
    expect(accepted.last).toMatchObject({ verdict: "accepted", passed: 5, total: 5 });

    const wrong = await end("top-p-sampling", "ok", "run");
    expect(wrong.last.verdict).toBe("wrong_answer");
    const failed = wrong.results.find((r) => r.status === "fail");
    expect(failed?.detail).toContain("expected");
    expect(failed?.stdout).toContain("nucleus sizes");

    const starter = await end("scaled-dot-product-attention", "raise NotImplementedError");
    expect(starter.last.verdict).toBe("runtime_error");
    expect(starter.results[0]?.detail).toContain("Traceback");

    const slow = await end("continuous-batching", "ok");
    expect(slow.last.verdict).toBe("time_limit");
    expect(slow.results.some((r) => r.status === "timeout")).toBe(true);
    expect(slow.results.at(-1)?.status).toBe("skipped");

    const cost = await end("kv-cache-decode", "ok");
    expect(cost.last.verdict).toBe("wrong_answer");
    expect(cost.results.flatMap((r) => r.metrics).find((m) => m.name === "model_tokens")).toMatchObject({
      value: 41,
      limit: 36,
    });

    const hiddenOnly = await end("rotary-embeddings", "ok", "run");
    expect(hiddenOnly.last.verdict).toBe("accepted");
    const hiddenFail = await end("rotary-embeddings", "ok", "submit");
    expect(hiddenFail.last.verdict).toBe("wrong_answer");
    // Hidden cases never leak expected values.
    expect(hiddenFail.results.find((r) => r.status === "fail")?.detail).toBeNull();

    const harness = await end("stable-softmax", "# ws: harness_error");
    expect(harness.last).toMatchObject({ verdict: "harness_error", total: 0 });
    expect(harness.events.some((e) => e.type === "start")).toBe(false);
    expect(harness.events.some((e) => e.type === "log" && e.stream === "stderr")).toBe(true);
  });

  it("ends with `canceled` when canceled mid-run", async () => {
    let seen = 0;
    const plan = buildPlan(byId("stable-softmax"), "torch", "submit", "ok");
    const events = await collect(
      simulateRun(plan, { sleep: () => Promise.resolve(), isCanceled: () => ++seen > 3 }),
    );
    const last = events.at(-1);
    expect(last).toMatchObject({ type: "end", verdict: "canceled" });
    expect(events.filter((e) => e.type === "caseEnd").length).toBeLessThan(plan.cases.length);
  });

  it("resolves scenarios from markers, starter code, then the problem default", () => {
    const problem = byId("top-p-sampling");
    expect(resolveScenario(problem, "x = 1  # ws: time_limit")).toBe("time_limit");
    expect(resolveScenario(problem, "raise NotImplementedError")).toBe("runtime_error");
    expect(resolveScenario(problem, "return 1")).toBe("wrong_answer");
    expect(resolveScenario(problem, "# ws: nonsense")).toBe("wrong_answer");
  });
});

describe("FixtureBackend", () => {
  it("serves the starter until a solution is written, and marks the problem attempted", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const id = "scaled-dot-product-attention";
    const starter = await backend.readSolution(id, "torch");
    expect(starter.isStarter).toBe(true);
    expect(starter.path).toMatch(/workspace\/scaled-dot-product-attention\/torch\.py$/);
    expect((await backend.getProblem(id)).status).toBe("todo");

    await backend.writeSolution(id, "torch", "def attention(): ...\n");
    expect((await backend.readSolution(id, "torch")).isStarter).toBe(false);
    expect((await backend.getProblem(id)).status).toBe("attempted");

    const reset = await backend.resetSolution(id, "torch");
    expect(reset).toEqual(starter);
    await expect(backend.readSolution(id, "python")).rejects.toThrow(/does not support/);
  });

  it("stores submits, not runs, and an accepted submit solves the problem and unlocks the editorial", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const id = "int8-affine-quant";
    const before = (await backend.listSubmissions(id)).length;
    expect((await backend.getProblem(id)).editorial).toBeNull();
    await expect(backend.getReference(id, "torch")).rejects.toThrow();

    await backend.writeSolution(id, "torch", "def quantize(x): return x\n");
    const run = await runToEnd(backend, id, "run");
    expect(run.at(-1)).toMatchObject({ type: "end", verdict: "accepted", submissionId: null });
    expect(await backend.listSubmissions(id)).toHaveLength(before);

    const submit = await runToEnd(backend, id, "submit");
    const end = submit.at(-1);
    if (end?.type !== "end" || end.submissionId === null) throw new Error("expected a stored submission");
    const list = await backend.listSubmissions(id);
    expect(list).toHaveLength(before + 1);
    expect(list[0]?.id).toBe(end.submissionId);
    const stored = await backend.getSubmission(end.submissionId);
    expect(stored.code).toContain("def quantize");
    expect(stored.cases).toHaveLength(end.total);
    // Stored cases carry the names announced at the start of the run.
    const start = submit.find((e) => e.type === "start");
    if (start?.type !== "start") throw new Error("expected a start event");
    expect(stored.cases.map((c) => [c.id, c.name, c.public])).toEqual(
      start.cases.map((c) => [c.id, c.name, c.public]),
    );

    const detail = await backend.getProblem(id);
    expect(detail).toMatchObject({ status: "solved", solvedFrameworks: ["torch"], editorialUnlocked: true });
    expect(detail.editorial).not.toBeNull();
    expect(await backend.getReference(id, "torch")).toContain("def quantize");
  });

  it("reveals the editorial on request without solving", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const detail = await backend.unlockEditorial("sse-token-stream");
    expect(detail.editorialUnlocked).toBe(true);
    expect(detail.status).toBe("todo");
  });

  it("keeps tracks, stats, and submissions consistent", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const [tracks, problems, stats, submissions] = await Promise.all([
      backend.listTracks(),
      backend.listProblems(),
      backend.getStats(),
      backend.listSubmissions(),
    ]);
    expect(tracks.reduce((n, t) => n + t.problemCount, 0)).toBe(problems.length);
    expect(tracks.reduce((n, t) => n + t.solvedCount, 0)).toBe(stats.solved);
    expect(stats.total).toBe(problems.length);
    const perDifficulty = Object.values(stats.byDifficulty);
    expect(perDifficulty.reduce((n, [solved]) => n + solved, 0)).toBe(stats.solved);
    expect(perDifficulty.reduce((n, [, total]) => n + total, 0)).toBe(stats.total);
    expect(stats.streakDays).toBeGreaterThanOrEqual(1);
    const times = submissions.map((s) => s.createdAt);
    expect(times).toEqual([...times].sort().reverse());
    for (const problem of problems.filter((p) => p.status === "solved")) {
      expect(submissions.some((s) => s.problemId === problem.id && s.verdict === "accepted")).toBe(true);
    }
  });

  it("serves a 75-problem path over its own catalog, in stages, with progress that follows a solve", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const [paths, problems] = await Promise.all([backend.listPaths(), backend.listProblems()]);
    expect(paths.map((path) => path.id)).toEqual(["inference-75"]);
    const path = paths[0];
    if (!path) throw new Error("no path");

    const ids = path.stages.flatMap((stage) => stage.problems);
    expect(ids.length).toBe(75);
    expect(new Set(ids).size).toBe(75);
    expect(path.problemCount).toBe(75);
    expect(path.stages.length).toBeGreaterThanOrEqual(6);
    const byId = new Map(problems.map((problem) => [problem.id, problem]));
    for (const stage of path.stages) {
      expect(stage.title).not.toBe("");
      expect(stage.description).toMatch(/\.$/);
    }
    // Every entry exists, and nothing comes before one of its prerequisites.
    ids.forEach((id, position) => {
      const problem = byId.get(id);
      expect(problem, id).toBeDefined();
      for (const prerequisite of problem?.prerequisites ?? []) {
        const at = ids.indexOf(prerequisite);
        if (at !== -1) expect(at, `${prerequisite} before ${id}`).toBeLessThan(position);
      }
    });
    expect(path.solvedCount).toBe(ids.filter((id) => byId.get(id)?.status === "solved").length);
    expect(path.solvedCount).toBeGreaterThan(0);
    expect(path.solvedCount).toBeLessThan(75);

    const target = ids.find((id) => {
      const problem = byId.get(id);
      return problem?.status === "todo" && problem.frameworks.includes("torch");
    });
    if (!target) throw new Error("nothing left to solve");
    await backend.writeSolution(target, "torch", "def solution(x): return x\n");
    expect((await runToEnd(backend, target, "submit")).at(-1)).toMatchObject({ verdict: "accepted" });
    expect((await backend.listPaths())[0]?.solvedCount).toBe(path.solvedCount + 1);
  });

  it("serves statement figures as data URLs and refuses anything else", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const statement = (await backend.getProblem("rotary-embeddings")).statement;
    expect(statement).toContain("](assets/pair-rotation.svg)");

    const url = await backend.getProblemAsset("rotary-embeddings", "pair-rotation.svg");
    expect(url.startsWith("data:image/svg+xml;base64,")).toBe(true);
    const svg = new TextDecoder().decode(
      Uint8Array.from(atob(url.slice(url.indexOf(",") + 1)), (char) => char.charCodeAt(0)),
    );
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain("θ");

    await expect(backend.getProblemAsset("rotary-embeddings", "../statement.md")).rejects.toThrow(
      "must stay inside",
    );
    await expect(backend.getProblemAsset("rotary-embeddings", "notes.txt")).rejects.toThrow(
      "only .svg and .png",
    );
    await expect(backend.getProblemAsset("rotary-embeddings", "missing.svg")).rejects.toThrow("missing file");
    await expect(backend.getProblemAsset("nope", "pair-rotation.svg")).rejects.toThrow("Unknown problem");
  });

  it("reports one framework as not installed and installs it with a streamed log", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const before = await backend.envStatus();
    expect(before.frameworks.filter((f) => !f.installed).map((f) => f.framework)).toEqual(["mlx"]);
    const events = await new Promise<string[]>((resolve) => {
      const seen: string[] = [];
      void backend.envInstall("mlx", (event) => {
        seen.push(event.type);
        if (event.type === "done") resolve(seen);
      });
    });
    expect(events.at(-1)).toBe("done");
    expect(events.filter((t) => t === "log").length).toBeGreaterThan(3);
    expect((await backend.envStatus()).frameworks.every((f) => f.installed)).toBe(true);
  });
});
