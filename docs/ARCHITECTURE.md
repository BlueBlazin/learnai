# Whetstone architecture

Whetstone is a native desktop app for mastering LLM inference and serving through
LeetCode-style practice problems. This document is the contract between components.
If you change a contract here, change every side of it in the same piece of work.

Working name: **Whetstone**. Bundle identifier: `dev.whetstone.app`. Python package and
CLI: `whetstone` / `ws`.

Language: American English everywhere (code, comments, docs, UI text, commits).
Library versions: see `docs/VERSIONS.md`. Never pin a version from memory; look it up.

## 1. Repository layout

```
learnai/
  docs/                    ARCHITECTURE.md, VERSIONS.md, PROBLEM_AUTHORING.md, DESIGN.md
  package.json, vite.config.ts, components.json, index.html     front-end project root
  src/                     React + TypeScript front end
  src-tauri/               Tauri shell: window, command layer only (thin)
  crates/core/             whetstone-core: all app logic, no Tauri dependency
  crates/devbridge/        dev-only HTTP bridge exposing core to a plain browser
  Cargo.toml               Cargo workspace (src-tauri, crates/*)
  python/                  uv project: the `whetstone` package (judge, kit, CLI)
  problems/<track>/<id>/   the problem catalog
  workspace/               the user's solutions (gitignored)
  .whetstone/              local state: SQLite database, caches (gitignored)
```

## 2. Processes

```
React UI  --invoke/Channel-->  src-tauri (thin)  -->  whetstone-core (Rust)
                                                        |  SQLite (.whetstone/state.db)
                                                        |  files (problems/, workspace/)
                                                        '--spawns-->  uv run ... python -m whetstone.harness
                                                                         (one subprocess per run; JSONL events on stdout)
```

- **whetstone-core** owns: catalog loading, solution files, submissions and progress,
  environment setup, run orchestration (spawn, stream, cancel, timeout), targets
  (local now; SSH in milestone 4), spaced re-solve (milestone 5).
- **src-tauri** only maps Tauri commands to core functions and forwards run events
  over a Tauri Channel. No logic lives here.
- **devbridge** exposes the same commands over HTTP (plus a server-sent event stream for
  run events) on `127.0.0.1:1421`, so the real UI and real core can be exercised in a
  normal browser against the Vite dev server. Dev and test only; never bundled.
- The **front end** talks to a `Backend` interface with two implementations:
  `TauriBackend` and `HttpBackend` (devbridge). A third, `FixtureBackend`, serves static
  sample data for component work. Selection is automatic: Tauri when
  `window.__TAURI_INTERNALS__` exists, else HTTP when `VITE_BACKEND=http`, else fixtures.

**Content root.** Core needs the directory holding `python/`, `problems/`, `workspace/`.
Resolution order: env `WHETSTONE_ROOT`, then the saved setting, then the repository path
captured at compile time.

## 3. Python side

### 3.1 Environment

One uv project at `python/` with ONE virtual environment.

- Base dependencies always present: `torch` (the oracle runtime), `numpy`, `httpx`,
  `uvicorn`, `starlette`, plus judge internals.
- Optional extras: `jax`, `mlx`. Later: `models` (safetensors, tokenizers,
  huggingface_hub) for real-weight problems.
- The Python version is pinned in `python/.python-version` (see VERSIONS.md).
- All tensors in tests are **float32** unless a problem is about dtypes. MLX has no
  float64 on GPU and JAX defaults to 32-bit; tolerances must be chosen for float32.

### 3.2 Frameworks

A problem lists the frameworks a solution may be written in:

| framework | meaning |
|---|---|
| `torch` | PyTorch. Canonical: the oracle reference and mutants are written in it. |
| `jax` | JAX (CPU on Apple Silicon). APIs must therefore be functional: state in, state out. |
| `mlx` | MLX (Apple GPU). |
| `python` | Framework-free problems (tokenizers, schedulers, servers). |

Rule for framework-neutral problems: the function signatures are identical across
frameworks and differ only in tensor type. **Randomness is an input**: a solution never
draws from a framework RNG; tests pass in uniform numbers or keys so results are exactly
checkable in every framework.

### 3.3 Problem directory

```
problems/<track>/track.toml          id, title, description, order
problems/<track>/<id>/
  problem.toml
  statement.md                        Markdown + KaTeX ($...$, $$...$$) + fenced code
  starter/<framework>.py              one per supported framework
  reference/<framework>.py            one per supported framework; torch.py (or python.py) is the oracle
  tests.py                            public and hidden cases
  hints.md                            progressive hints separated by a line containing only `---`
  editorial.md                        explanation of the reference approach
  mutants/<bug-name>.py               wrong solutions (oracle framework only); line 1: `# BUG: <what is wrong>`
```

`problem.toml`:

```toml
id = "stable-softmax"            # equals the directory name; globally unique
title = "Numerically Stable Softmax"
track = "foundations"            # equals the parent directory name
order = 10                       # position within the track
difficulty = "easy"              # easy | medium | hard
frameworks = ["torch", "jax", "mlx"]     # or ["python"]
judges = ["numeric"]             # numeric | property | statistical | cost | speed | simclock | service
tags = ["softmax", "numerics"]
prerequisites = []               # problem ids
requires = "cpu"                 # cpu | mlx-gpu | cuda
entry = ["softmax"]              # names the solution must define
time_limit_s = 10                # default per-case limit
```

`forbidden` (optional): a list of dotted names the solution must not call, for example
`["torch.softmax", "torch.nn.functional.softmax"]`. A violation fails the case.

### 3.4 Judge types

| judge | checks | determinism rule |
|---|---|---|
| numeric | output matches the oracle within tolerance | float32 tolerances |
| property | an invariant holds on seeded random inputs | seeded |
| statistical | output distribution matches the target | randomness is an input; fixed seeds; false-failure rate below 1e-9 |
| cost | an instrumented kit object counts work done | exact counts |
| speed | time relative to the same-framework reference, same process, interleaved, median of k | never an absolute wall-clock threshold |
| simclock | discrete-event simulation with virtual time | no real sleeping |
| service | the judge is an HTTP client against the solution's ASGI app on an ephemeral local port | bounded timeouts |

### 3.5 Harness command (called by core and by the CLI)

```
uv run --project <root>/python [--extra jax] [--extra mlx] -- \
  python -m whetstone.harness \
    --problem <abs problem dir> --solution <abs solution file> \
    --framework torch|jax|mlx|python --mode run|submit [--case <case id>] [--reveal]
```

- `run` executes public cases only; `submit` executes all cases.
- `--case` takes an id listed in `start.cases`. `--reveal` (authors only) shows real
  hidden-case ids and names.
- Exit code 0 whenever the harness itself worked (even if cases failed); non-zero only
  for harness or usage failure.
- The solution runs inside this subprocess. Core enforces an overall wall-clock limit
  and kills the process group on cancel or timeout.

### 3.6 Event protocol

The harness writes protocol lines to stdout, each one `@@WS@@ ` followed by a single-line
JSON object. Anything the solution prints is captured by the harness (file-descriptor
level) and delivered inside events, so it never corrupts the stream. Core treats any
non-prefixed stdout line and all stderr as diagnostic log output.

```jsonc
{"event":"start","problem":"stable-softmax","framework":"torch","mode":"run","cases":[{"id":"c1","name":"...","public":true}],"env":{"python":"3.x","torch":"...","device":"..."}}
{"event":"case_start","id":"c1"}
{"event":"case_end","id":"c1","status":"pass","duration_ms":12.3,
   "message":null,             // short failure reason, one line
   "detail":null,              // optional longer text: expected vs. got excerpt, traceback
   "stdout":"",                // captured prints, truncated to 64 KB
   "metrics":[{"name":"model_tokens","value":37,"unit":"tokens","limit":37}]}
{"event":"end","status":"accepted","passed":5,"total":5,"duration_ms":210.0,"metrics":[...]}
```

- Case `status`: `pass | fail | error | timeout | skipped`.
- Final `status`: `accepted | wrong_answer | runtime_error | time_limit | import_error | harness_error`.
  Precedence when mixed: import_error, runtime_error, time_limit, wrong_answer.
- Hidden cases have the opaque id `hidden-N` and the name "Hidden case N", keep the
  one-line `message`, and always have `detail: null`. For a hidden `error`, `message` is
  the exception line plus the location in the solution file. Skipped cases have
  `detail: null`.
- `end` may carry optional `message` and `detail` (used for `import_error` and
  `harness_error`: the traceback). Core forwards them on the `end` RunEvent as
  `message: string | null` and `detail: string | null`.
- A skipped case emits `case_end` with no preceding `case_start`.
- Output printed while the solution module is imported arrives as non-prefixed log lines.
- `start.env` also carries `platform`.
- With no saved solution file, `ws run` judges the starter and says so.

### 3.7 CLI

`uv run --project python ws <command>`: `list`, `run <id> [-f framework]`,
`submit <id> [-f framework]`, `validate [<id> | --all]`, `new <track>/<id>`.

`validate` is the quality gate. For every supported framework of a problem: the reference
is accepted; the starter is not accepted. Every mutant is rejected. Metadata is complete
and consistent. A problem that fails validation must not be merged.

Authoring details (the test API, the framework adapter, the kit) are specified in
`docs/PROBLEM_AUTHORING.md`, owned by the Python judge.

## 4. Core API

Types are shown in TypeScript; Rust mirrors them with serde (`camelCase` on the wire).

```ts
type Framework = "torch" | "jax" | "mlx" | "python";
type Difficulty = "easy" | "medium" | "hard";
type JudgeKind = "numeric" | "property" | "statistical" | "cost" | "speed" | "simclock" | "service";
type Requires = "cpu" | "mlx-gpu" | "cuda";
type ProblemStatus = "todo" | "attempted" | "solved";
type RunMode = "run" | "submit";
type CaseStatus = "pass" | "fail" | "error" | "timeout" | "skipped";
type Verdict = "accepted" | "wrong_answer" | "runtime_error" | "time_limit"
             | "import_error" | "harness_error" | "canceled";

interface Track { id: string; title: string; description: string; order: number;
                  problemCount: number; solvedCount: number; }

interface ProblemSummary {
  id: string; title: string; track: string; order: number; difficulty: Difficulty;
  frameworks: Framework[]; judges: JudgeKind[]; tags: string[]; prerequisites: string[];
  requires: Requires; status: ProblemStatus; solvedFrameworks: Framework[];
}

interface ProblemDetail extends ProblemSummary {
  statement: string;            // Markdown
  hints: string[];              // Markdown each
  editorial: string | null;     // null while locked
  editorialUnlocked: boolean;   // true once solved or explicitly revealed
  entry: string[];
  timeLimitS: number;
}

interface SolutionFile { code: string; isStarter: boolean; path: string; }

interface Metric { name: string; value: number; unit?: string; limit?: number; }

interface CaseInfo { id: string; name: string; public: boolean; }
interface CaseResult { id: string; status: CaseStatus; durationMs: number;
                       message: string | null; detail: string | null;
                       stdout: string; metrics: Metric[]; }

type RunEvent =
  | { type: "preparing"; message: string }                      // environment sync, spawn
  | { type: "start"; cases: CaseInfo[]; env: Record<string, string> }
  | { type: "caseStart"; id: string }
  | { type: "caseEnd"; result: CaseResult }
  | { type: "log"; stream: "stdout" | "stderr"; text: string }  // diagnostics
  | { type: "end"; verdict: Verdict; passed: number; total: number;
      durationMs: number; metrics: Metric[]; submissionId: number | null;
      message: string | null; detail: string | null };          // import/harness errors

interface Submission { id: number; problemId: string; framework: Framework; mode: RunMode;
                       verdict: Verdict; passed: number; total: number; durationMs: number;
                       target: string; createdAt: string; }               // ISO 8601
interface StoredCase extends CaseResult { name: string; public: boolean; }
interface SubmissionDetail extends Submission { code: string; cases: StoredCase[]; metrics: Metric[]; }

interface FrameworkEnv { framework: Framework; installed: boolean; version: string | null; }
interface EnvStatus { uv: string | null; python: string | null; frameworks: FrameworkEnv[];
                      root: string; }
interface Stats { solved: number; total: number; byDifficulty: Record<Difficulty, [number, number]>;
                  streakDays: number; }
```

Commands (Tauri command names are snake_case; the `Backend` interface uses camelCase):

| command | arguments | returns |
|---|---|---|
| `list_tracks` | | `Track[]` |
| `list_problems` | | `ProblemSummary[]` |
| `get_problem` | `id` | `ProblemDetail` |
| `read_solution` | `id, framework` | `SolutionFile` (the starter if none saved) |
| `write_solution` | `id, framework, code` | |
| `reset_solution` | `id, framework` | `SolutionFile` |
| `start_run` | `problemId, framework, mode, target` + event sink | `runId: string`; stream of `RunEvent` |
| `cancel_run` | `runId` | |
| `list_submissions` | `problemId?` | `Submission[]` newest first |
| `get_submission` | `id` | `SubmissionDetail` |
| `unlock_editorial` | `id` | `ProblemDetail` |
| `get_reference` | `id, framework` | `string` (only when editorial is unlocked) |
| `env_status` | | `EnvStatus` |
| `env_install` | `framework` + event sink | stream of `{type:"log",...}` then `{type:"done", ok}` |
| `get_stats` | | `Stats` |

| `list_catalog_issues` | | `{ path: string; message: string }[]` (malformed problems) |

Rules: only `submit` runs are stored as submissions with a code snapshot; `run` results
are ephemeral. A canceled submit, or one whose process never launched, is not stored.
`cancel_run` on an unknown or finished run is a no-op. `EnvStatus.frameworks` includes a
`python` entry. The saved content-root setting lives in `root.json` in the Tauri app
config directory (it cannot live in the database, which sits under the root). Overall
run limit: 600 s until the harness `start` event (environment sync can be slow), then
30 s + cases x `time_limit_s`. Streak: consecutive local days with an accepted submit,
ending today or yesterday. A problem is `solved` once any framework has an accepted submit, and
`attempted` once a solution file differs from the starter or any submit exists.
Solutions live at `workspace/<problem id>/<framework>.py` so they can be edited in any
editor; the app re-reads the file when the window regains focus.

### 4.1 Transports

**Tauri.** One `#[tauri::command]` per row above. JavaScript passes camelCase argument
names (Tauri maps them to snake_case Rust parameters). Streaming commands take an extra
argument `onEvent: Channel<RunEvent>` (or `Channel<InstallEvent>`); `start_run` returns
the `runId` immediately and events flow over the channel until `end`. Errors are
returned as a string message.

**Devbridge (HTTP, dev and test only).** Listens on `127.0.0.1:1421`, CORS open to the
Vite dev origin.

- `POST /api/<command>` with a JSON object body of the camelCase arguments (empty object
  when there are none). Success: `200` with the JSON result (`null` for no result).
  Failure: `4xx/5xx` with `{"error": "<message>"}`.
- `POST /api/start_run` returns `{"runId": "..."}`. Events:
  `GET /api/runs/<runId>/events` as server-sent events, one `data: <RunEvent JSON>` per
  event. The bridge buffers events per run and replays from the first event to a
  subscriber that connects late; the stream closes after `end`.
- `POST /api/env_install` returns `{"jobId": "..."}`; events at
  `GET /api/jobs/<jobId>/events`, same rules, closing after `done`.

`InstallEvent = { type: "log"; text: string } | { type: "done"; ok: boolean; message: string | null }`.
The Tauri `env_install` command resolves with no value once the install has finished
(after the `done` event).

## 5. Front end

Stack: React, TypeScript, Vite, Tailwind CSS, shadcn/ui, CodeMirror 6, TanStack Router
and Query. Fonts are self-hosted (the app must work offline). Design direction and tokens
live in `docs/DESIGN.md`.

Component rules (enforced in review):

1. Every interactive element comes from a shadcn primitive in `src/components/ui/`.
   Do not hand-roll buttons, dialogs, menus, tabs, tooltips, inputs, or scroll areas.
2. App-specific UI is built as named, reusable components composed from those
   primitives under `src/components/<area>/`. A pattern used twice becomes a component.
3. Screens under `src/routes/` compose components; they contain layout and data wiring,
   not styling one-offs. No inline `style=` except for dynamic values.
4. Colors, type sizes, spacing, and radii come from design tokens. No raw hex values or
   arbitrary Tailwind values in components.
5. Data access goes through the `Backend` interface via TanStack Query hooks in
   `src/lib/api/`. Components never call `invoke` or `fetch` directly.

## 6. Milestones

1. Vertical slice: judge, CLI, app, local runs, 12 problems covering all seven judges.
2. Core tracks to about 80 problems; curated "Inference 75" path.
3. MLX GPU track; real-model assets (current open models up to 4B parameters,
   safetensors; 8-bit at the 4B size).
4. SSH target; CUDA track (unit-tested only until an NVIDIA GPU is available).
5. Capstones; spaced re-solve; remaining catalog (about 210 problems).
