import { describe, expect, it } from "vitest";
import type { AssetEvent, RunEvent, RunMode, TargetCheckEvent } from "../types";
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

  it("lists one model of each kind, with the problems that use it", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const listing = await backend.listAssets();
    expect(listing.modelsDir).toMatch(/\.whetstone\/models$/);
    expect(listing.assets.map((asset) => [asset.name, asset.state])).toEqual([
      ["granite-4.0-1b", "incomplete"],
      ["granite-4.0-350m", "present"],
      ["granite-4.2-3b-mlx-8bit", "missing"],
    ]);
    const problems = await backend.listProblems();
    for (const asset of listing.assets) {
      expect(asset.path).toBe(`${listing.modelsDir}/${asset.name}`);
      expect(asset.problems.length).toBeGreaterThan(0);
      // The catalog and the listing agree about who declares what.
      expect(asset.problems).toEqual(
        problems
          .filter((problem) => problem.assets.includes(asset.name))
          .map((problem) => problem.id)
          .sort(),
      );
    }
    expect(listing.assets[0]?.reason).not.toBeNull();
    expect(listing.assets[2]?.bytesOnDisk).toBe(0);
  });

  it("skips the cases that need a missing model, stores the submit, and never counts it as solved", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const id = "mlx-8bit-linear";
    // History already holds a not-judged submit with its skipped cases.
    const seeded = (await backend.listSubmissions(id))[0];
    expect(seeded).toMatchObject({ verdict: "assets_missing", skipped: 3 });
    const stored = await backend.getSubmission(seeded?.id ?? -1);
    expect(stored.cases.filter((c) => c.status === "skipped").map((c) => c.skip)).toEqual(
      Array.from({ length: 3 }, () => ({ kind: "asset", asset: "granite-4.2-3b-mlx-8bit" })),
    );

    const solvedBefore = (await backend.getStats()).solved;
    const events = await new Promise<RunEvent[]>((resolve, reject) => {
      const seen: RunEvent[] = [];
      backend
        .startRun({ problemId: id, framework: "mlx", mode: "submit", target: "local" }, (event) => {
          seen.push(event);
          if (event.type === "end") resolve(seen);
        })
        .catch(reject);
    });
    const start = events.find((e) => e.type === "start");
    expect(start).toMatchObject({ requires: "mlx-gpu", requiresMet: true });
    const end = events.at(-1);
    if (end?.type !== "end") throw new Error("expected an end event");
    expect(end).toMatchObject({ verdict: "assets_missing", passed: 3, total: 6, skipped: 3 });
    expect(end.missingAssets).toEqual([
      {
        name: "granite-4.2-3b-mlx-8bit",
        repo: "lmstudio-community/granite-4.2-3b-MLX-8bit",
        sizeMb: 3900,
      },
    ]);
    expect(end.message).toContain("3 of 6 cases were skipped");
    // A skipped case reports `caseEnd` with no `caseStart`, and is not a failure.
    const skipped = events.flatMap((e) =>
      e.type === "caseEnd" && e.result.status === "skipped" ? [e.result] : [],
    );
    expect(skipped).toHaveLength(3);
    expect(skipped[0]?.message).toMatch(/^skipped: model asset/);
    const started = events.flatMap((e) => (e.type === "caseStart" ? [e.id] : []));
    expect(started).not.toContain(skipped[0]?.id);

    expect(end.submissionId).not.toBeNull();
    expect((await backend.getProblem(id)).status).toBe("attempted");
    expect((await backend.getStats()).solved).toBe(solvedBefore);
  });

  it("ends with hardware_missing where the fixture Mac has no NVIDIA GPU", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const id = "cuda-device-placement";
    expect((await backend.listSubmissions(id))[0]).toMatchObject({ verdict: "hardware_missing", skipped: 2 });

    const run = await runToEnd(backend, id, "run");
    const start = run.find((e) => e.type === "start");
    expect(start).toMatchObject({ requires: "cuda", requiresMet: false });
    // Every public case is a CPU case and passes, yet the run is not accepted.
    expect(run.at(-1)).toMatchObject({
      verdict: "hardware_missing",
      passed: 2,
      total: 2,
      skipped: 0,
      missingHardware: "cuda",
    });

    const submit = await runToEnd(backend, id, "submit");
    if (start?.type !== "start") throw new Error("expected a start event");
    const announced = submit.find((e) => e.type === "start");
    if (announced?.type !== "start") throw new Error("expected a start event");
    expect(announced.cases.filter((c) => c.requires === "cuda")).toHaveLength(2);
    expect(submit.at(-1)).toMatchObject({ verdict: "hardware_missing", passed: 3, total: 5, skipped: 2 });
    expect((await backend.getProblem(id)).status).toBe("attempted");
  });

  it("downloads a model with progress, after which the skipped cases run", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const name = "granite-4.2-3b-mlx-8bit";
    const fetch = (names: string[]) =>
      new Promise<AssetEvent[]>((resolve, reject) => {
        const seen: AssetEvent[] = [];
        backend
          .fetchAssets(names, (event) => {
            seen.push(event);
            if (event.type === "done") resolve(seen);
          })
          .catch(reject);
      });

    const events = await fetch([name]);
    expect(events[0]?.type).toBe("preparing");
    expect(events.at(-1)).toEqual({ type: "done", ok: true, canceled: false, message: null });
    expect(events.filter((e) => e.type === "done")).toHaveLength(1);
    const progress = events.flatMap((e) => (e.type === "progress" ? [e] : []));
    expect(progress.length).toBeGreaterThan(10);
    // Bytes only go up, and end at the total.
    for (let i = 1; i < progress.length; i++) {
      expect(progress[i]?.doneBytes).toBeGreaterThanOrEqual(progress[i - 1]?.doneBytes ?? 0);
    }
    expect(progress.at(-1)?.doneBytes).toBe(progress.at(-1)?.totalBytes);
    expect(events).toContainEqual({ type: "assetEnd", asset: name, ok: true, message: null });

    const asset = (await backend.listAssets()).assets.find((a) => a.name === name);
    expect(asset).toMatchObject({ state: "present", reason: null });
    expect(asset?.bytesOnDisk).toBeGreaterThan(3e9);

    // With the model on disk the same submit is judged on every case.
    const events2 = await new Promise<RunEvent[]>((resolve, reject) => {
      const seen: RunEvent[] = [];
      backend
        .startRun(
          { problemId: "mlx-8bit-linear", framework: "mlx", mode: "submit", target: "local" },
          (e) => {
            seen.push(e);
            if (e.type === "end") resolve(seen);
          },
        )
        .catch(reject);
    });
    expect(events2.at(-1)).toMatchObject({ verdict: "accepted", skipped: 0 });

    // Removing it brings the skips back; an undeclared name is refused.
    const after = await backend.removeAsset(name);
    expect(after.assets.find((a) => a.name === name)).toMatchObject({ state: "missing", bytesOnDisk: 0 });
    await expect(backend.removeAsset("../state.db")).rejects.toThrow("not a model asset");
    await expect(backend.fetchAssets([], () => {})).rejects.toThrow("at least one");
  });

  it("fails once with a one-line reason, then resumes from the bytes on disk", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const name = "granite-4.0-1b";
    const fetch = () =>
      new Promise<AssetEvent[]>((resolve, reject) => {
        const seen: AssetEvent[] = [];
        backend
          .fetchAssets([name], (event) => {
            seen.push(event);
            if (event.type === "done") resolve(seen);
          })
          .catch(reject);
      });

    const first = await fetch();
    const failure = first.find((e) => e.type === "assetEnd");
    if (failure?.type !== "assetEnd") throw new Error("expected an assetEnd event");
    expect(failure.ok).toBe(false);
    expect(failure.message).toContain("connection was reset");
    expect(failure.message).not.toContain("\n");
    expect(first.at(-1)).toMatchObject({ type: "done", ok: false, canceled: false });
    const partial = (await backend.listAssets()).assets.find((a) => a.name === name);
    expect(partial?.state).toBe("incomplete");

    const second = await fetch();
    expect(second.at(-1)).toMatchObject({ type: "done", ok: true });
    const resumed = second.find((e) => e.type === "progress");
    if (resumed?.type !== "progress") throw new Error("expected progress");
    expect(resumed.doneBytes).toBeGreaterThan(partial?.bytesOnDisk ?? 0);
    expect((await backend.listAssets()).assets.find((a) => a.name === name)?.state).toBe("present");
  });

  it("runs one fetch at a time, and a cancel keeps what was downloaded", async () => {
    const backend = new FixtureBackend({ speed: 0.01 });
    const name = "granite-4.2-3b-mlx-8bit";
    const events: AssetEvent[] = [];
    const done = new Promise<void>((resolve) => {
      void backend
        .fetchAssets([name], (event) => {
          events.push(event);
          if (event.type === "done") resolve();
        })
        .then(async (jobId) => {
          await expect(backend.fetchAssets(["granite-4.0-1b"], () => {})).rejects.toThrow("already running");
          await expect(backend.removeAsset(name)).rejects.toThrow("already running");
          await backend.cancelAssetFetch("not-a-job");
          await backend.cancelAssetFetch(jobId);
        });
    });
    await done;
    expect(events.at(-1)).toEqual({ type: "done", ok: false, canceled: true, message: null });
    expect(events.some((e) => e.type === "assetEnd")).toBe(false);
    const asset = (await backend.listAssets()).assets.find((a) => a.name === name);
    expect(asset?.state === "missing" || asset?.state === "incomplete").toBe(true);
    // The slot is free again.
    await expect(backend.fetchAssets([name], () => {})).resolves.toMatch(/^fixture-fetch-/);
  });
});

describe("FixtureBackend run targets", () => {
  const checkToEnd = (backend: FixtureBackend, name: string): Promise<TargetCheckEvent[]> =>
    new Promise((resolve, reject) => {
      const events: TargetCheckEvent[] = [];
      backend
        .checkTarget(name, (event) => {
          events.push(event);
          if (event.type === "done") resolve(events);
        })
        .catch(reject);
    });

  it("lists this computer first, a healthy GPU machine, and one failing at uv with a fix", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const targets = await backend.listTargets();
    expect(targets.map((target) => target.name)).toEqual(["local", "ssh:gpu-box", "ssh:studio"]);
    expect(targets[0]).toMatchObject({ kind: "local", host: null, remoteDir: null });
    expect(targets[1]?.lastCheck).toMatchObject({ ok: true, doctor: { hardware: ["cpu", "cuda"] } });
    expect(targets[1]?.lastCheck?.doctor?.gpuName).toContain("RTX 4090");
    const failing = targets[2]?.lastCheck;
    expect(failing?.ok).toBe(false);
    expect(failing?.steps.at(-1)).toMatchObject({ id: "uv", ok: false });
    expect(failing?.steps.at(-1)?.fix).toContain("astral.sh/uv/install.sh");
    expect(failing?.doctor).toBeNull();
  });

  it("adds a host from the configuration, checks it, and removes it", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const hosts = await backend.listSshHosts();
    expect(hosts.map((host) => host.alias)).toEqual(["gpu-box", "studio", "lab-a100"]);
    await expect(backend.addTarget("stranger")).rejects.toThrow("not a host in your SSH configuration");
    await expect(backend.addTarget("gpu-box")).rejects.toThrow("already a run target");

    const added = await backend.addTarget("lab-a100");
    expect(added.at(-1)).toMatchObject({ name: "ssh:lab-a100", remoteDir: "~/.whetstone", lastCheck: null });

    const events = await checkToEnd(backend, "ssh:lab-a100");
    expect(events.filter((event) => event.type === "step")).toHaveLength(5);
    const done = events.at(-1);
    expect(done).toMatchObject({ type: "done", check: { ok: true } });
    expect((await backend.listTargets()).at(-1)?.lastCheck?.doctor?.gpuName).toContain("A100");

    await expect(backend.removeTarget("local")).rejects.toThrow("cannot be removed");
    await expect(backend.removeTarget("ssh:stranger")).rejects.toThrow("unknown target");
    expect((await backend.removeTarget("ssh:lab-a100")).map((target) => target.name)).toEqual([
      "local",
      "ssh:gpu-box",
      "ssh:studio",
    ]);
  });

  it("streams a check that stops at its failing step", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const events = await checkToEnd(backend, "ssh:studio");
    expect(events.map((event) => event.type)).toEqual(["step", "step", "done"]);
    expect(events.at(-1)).toMatchObject({ check: { ok: false, doctor: null } });
  });

  it("runs on a remote target with its own steps, hardware, and models", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const run = (target: string): Promise<RunEvent[]> =>
      new Promise((resolve, reject) => {
        const events: RunEvent[] = [];
        backend
          .startRun({ problemId: "stable-softmax", framework: "torch", mode: "submit", target }, (event) => {
            events.push(event);
            if (event.type === "end") resolve(events);
          })
          .catch(reject);
      });

    const events = await run("ssh:gpu-box");
    const preparing = events.flatMap((event) => (event.type === "preparing" ? [event.message] : []));
    expect(preparing).toContain("Connecting to gpu-box");
    expect(preparing).toContain("Copying the problem and your solution to gpu-box");
    const end = events.at(-1);
    if (end?.type !== "end" || end.submissionId === null) throw new Error("expected a stored submit");
    expect((await backend.getSubmission(end.submissionId)).target).toBe("ssh:gpu-box");

    // A machine whose check fails cannot run: the reason and the fix come back.
    const refused = (await run("ssh:studio")).at(-1);
    expect(refused).toMatchObject({ type: "end", verdict: "harness_error", submissionId: null });
    expect(refused?.type === "end" ? refused.message : "").toContain("astral.sh/uv/install.sh");

    await expect(
      backend.startRun(
        { problemId: "stable-softmax", framework: "torch", mode: "run", target: "ssh:stranger" },
        () => {},
      ),
    ).rejects.toThrow("unknown target");
  });

  it("downloads a model on a remote target without touching this computer's", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const before = (await backend.listAssets()).assets.find(
      (asset) => asset.name === "granite-4.2-3b-mlx-8bit",
    );
    const events = await new Promise<AssetEvent[]>((resolve, reject) => {
      const seen: AssetEvent[] = [];
      backend
        .fetchAssets(
          ["granite-4.2-3b-mlx-8bit"],
          (event) => {
            seen.push(event);
            if (event.type === "done") resolve(seen);
          },
          "ssh:gpu-box",
        )
        .catch(reject);
    });
    expect(events[0]).toEqual({ type: "preparing", message: "Connecting to gpu-box" });
    expect(events.at(-1)).toEqual({ type: "done", ok: true, canceled: false, message: null });
    const after = (await backend.listAssets()).assets.find(
      (asset) => asset.name === "granite-4.2-3b-mlx-8bit",
    );
    expect(after).toEqual(before);
  });
});
