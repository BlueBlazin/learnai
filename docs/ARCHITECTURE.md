# Learn AI architecture

Learn AI is a native desktop app for mastering LLM inference and serving through
LeetCode-style practice problems. This document is the contract between components.
If you change a contract here, change every side of it in the same piece of work.

Name: **Learn AI**. Bundle identifier: `dev.learnai.app`. Python package and CLI:
`learnai`.

Language: American English everywhere (code, comments, docs, UI text, commits).
Library versions: see `docs/VERSIONS.md`. Never pin a version from memory; look it up.

## 1. Repository layout

```
learnai/
  docs/                    ARCHITECTURE.md, VERSIONS.md, PROBLEM_AUTHORING.md, DESIGN.md
  package.json, vite.config.ts, components.json, index.html     front-end project root
  src/                     React + TypeScript front end
  src-tauri/               Tauri shell: window, command layer only (thin)
  crates/core/             learnai-core: all app logic, no Tauri dependency
  crates/devbridge/        dev-only HTTP bridge exposing core to a plain browser
  Cargo.toml               Cargo workspace (src-tauri, crates/*)
  python/                  uv project: the `learnai` package (judge, kit, CLI)
  problems/<track>/<id>/   the problem catalog
  workspace/               the user's solutions (gitignored)
  .learnai/              local state: SQLite database, caches (gitignored)
```

## 2. Processes

```
React UI  --invoke/Channel-->  src-tauri (thin)  -->  learnai-core (Rust)
                                                        |  SQLite (.learnai/state.db)
                                                        |  files (problems/, workspace/)
                                                        '--spawns-->  uv run ... python -m learnai.harness
                                                                         (one subprocess per run; JSONL events on stdout)
```

- **learnai-core** owns: catalog loading, solution files, submissions and progress,
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
Resolution order: env `LEARNAI_ROOT`, then the saved setting, then the repository path
captured at compile time.

## 3. Python side

### 3.1 Environment

One uv project at `python/` with ONE virtual environment.

- Base dependencies always present: `torch` (the oracle runtime), `numpy`, `httpx`,
  `uvicorn`, `starlette`, plus judge internals.
- Optional extras: `jax`, `mlx`, `models` (huggingface_hub, safetensors, tokenizers,
  transformers, and mlx-lm on macOS). `models` is needed only by `learnai assets fetch` and
  by authors generating fixtures; the harness and the kit read downloaded files with
  numpy.
- The `mlx` extra and `mlx-lm` carry the marker `sys_platform == 'darwin' and
  platform_machine == 'arm64'`, so the same extras can be passed on every host; on Linux
  they install nothing and torch resolves to its normal Linux wheels. `uv.lock` is
  universal.
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

A problem may be MLX-only (`frameworks = ["mlx"]`). Its oracle is `reference/mlx.py`
and its mutants are written in MLX. Every other tensor problem includes `torch`.

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
requires = "cpu"                 # cpu | mlx-gpu | cuda: hardware needed to be accepted
entry = ["softmax"]              # names the solution must define
time_limit_s = 10                # default per-case limit
```

`forbidden` (optional): a list of dotted names the solution must not call, for example
`["torch.softmax", "torch.nn.functional.softmax"]`. A violation fails the case.
`forbidden_source` (optional): identifiers that may not appear in the solution's source
(an AST check), for names that cannot be guarded at call time, for example
`["bfloat16"]`.

`[[assets]]` (optional, repeatable): model files the problem needs.

```toml
[[assets]]
name = "granite-4.0-350m"        # directory under <root>/.learnai/models/
repo = "ibm-granite/granite-4.0-350m"
revision = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"   # full commit hash
files = ["config.json", "tokenizer.json", "model.safetensors"]   # names or globs; safetensors, tokenizer, and config files only
size_mb = 712                    # approximate download size
```

Model assets live in `<root>/.learnai/models/<name>/` (gitignored) beside a
`learnai-asset.json` manifest. An asset is present when the manifest matches the
declared repository and revision, covers the declared files, and every file has its
recorded size. Cases declare what they need with `@case(requires=..., assets=[...])`;
the default is `requires="cpu"` and every asset of the problem.

Optional `fixtures/` directory: committed reference outputs loaded by the tests.

Optional `assets/` directory: figures for the statement (SVG preferred, PNG allowed),
referenced from Markdown as `![caption](assets/<file>)`. SVG figures must read well in
both light and dark themes (use `currentColor`, no fixed backgrounds).

**Paths.** A path is a curated, ordered study plan stored at `paths/<id>.toml` in the
content root:

```toml
id = "inference-75"
title = "Inference 75"
description = "One sentence."

[[stages]]
title = "Numerics and tensors"
description = "One sentence."
problems = ["stable-softmax", "rmsnorm"]
```

A problem id that does not exist in the catalog is skipped and reported as a catalog
issue; it never breaks the path. Paths are listed in file-name order; the first is the
learner's study path.

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
  python -m learnai.harness \
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
- Environment variables: `LEARNAI_MODELS_DIR` replaces `<root>/.learnai/models`;
  `LEARNAI_HARDWARE` is a comma-separated list of hardware to treat as present (tests
  and previews only); `LEARNAI_JAX_CACHE` is the directory of JAX's persistent
  compilation cache (default `<root>/.learnai/cache/jax`; `0` disables). Anything
  under `<root>/.learnai/cache/` is disposable.

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

Skipped cases and the two "not judged" endings:

```jsonc
{"event":"start", ..., "cases":[{"id":"c1","name":"...","public":true,"requires":"cuda"}],   // `requires` only when not "cpu"
   "requires":"cuda","requires_met":false}
{"event":"case_end","id":"hidden-2","status":"skipped","duration_ms":0,
   "message":"skipped: needs an NVIDIA GPU (CUDA), which this machine does not have",
   "detail":null,"stdout":"","metrics":[],
   "skip":{"kind":"hardware","requires":"cuda"}}      // or {"kind":"asset","asset":"<name>"} or {"kind":"aborted"}
{"event":"end","status":"hardware_missing","passed":2,"total":3,"skipped":1,"duration_ms":210.0,"metrics":[...],
   "message":"2 of 2 cases that could run passed; 1 of 3 cases were skipped because ...",
   "missing_hardware":"cuda"}                         // with assets_missing: "missing_assets":[{"name","repo","size_mb","fetch_command"}]
```

- Case `status`: `pass | fail | error | timeout | skipped`.
- Final `status`: `accepted | wrong_answer | runtime_error | time_limit | import_error
  | harness_error | assets_missing | hardware_missing`. Precedence when mixed:
  import_error, runtime_error, time_limit, wrong_answer, assets_missing,
  hardware_missing, accepted.
- A case is `skipped`, without being run, when the hardware it requires is absent
  (`skip.kind = "hardware"`) or an asset it needs is not downloaded (`"asset"`); cases
  skipped because the run stopped early have `skip.kind = "aborted"`. Every skipped
  `case_end` carries `skip` and a one-line `message`.
- `accepted` means every selected case ran and passed on a machine that has the
  problem's `requires`. If nothing that ran failed but a case was skipped for an asset,
  the status is `assets_missing`; if a case was skipped for hardware, or the problem's
  `requires` is not met, it is `hardware_missing`. A failure among the cases that ran
  decides the status as before. When no case can run, the solution is not imported.
- `end` always carries `skipped` (the number of skipped cases). For the two new
  statuses it carries `message` (what was skipped, why, and what to do) and
  `missing_assets` or `missing_hardware`.
- A metric may carry `"higher_is_better": true` (throughput); the run-level value of
  such a metric is the smallest across cases instead of the largest.
- Presentation: `assets_missing` and `hardware_missing` are neither success nor a
  failure of the learner's code. Show "N of M cases passed; K skipped", the `message`,
  and for `assets_missing` an action that fetches the asset. They never mark a problem
  solved; they are stored like any other non-accepted submit.
- Hidden cases have the opaque id `hidden-N` and the name "Hidden case N", keep the
  one-line `message`, and always have `detail: null`. For a hidden `error`, `message` is
  the exception line plus the location in the solution file. Skipped cases have
  `detail: null`.
- `end` may carry optional `message` and `detail` (used for `import_error` and
  `harness_error`: the traceback). Core forwards them on the `end` RunEvent as
  `message: string | null` and `detail: string | null`.
- A skipped case emits `case_end` with no preceding `case_start`.
- The one-line `message` of a case that ended in `error` is the exception type and the
  first meaningful line of its text.
- Output printed while the solution module is imported arrives as non-prefixed log lines.
- `start.env` also carries `platform`.
- With no saved solution file, `learnai run` judges the starter and says so.

### 3.7 CLI

`uv run --project python learnai <command>`: `list`, `run <id> [-f framework]`,
`submit <id> [-f framework]`, `validate [<id> | --all] [--require-assets]`,
`new <track>/<id>`, `assets list [--json]`,
`assets fetch <name>... | --all [--json] [--no-verify]`, `doctor [--json]`.

`learnai doctor --json` prints one JSON document describing the environment it runs in:
`schema`, `learnai`, `python`, `executable`, `platform{system,machine,release}`,
`frameworks{torch,jax,mlx}` (installed version or null), `packages`,
`hardware{cpu{present,cores}, mlx-gpu{present,name,memory_bytes},
cuda{present,name,memory_bytes,devices[],torch_cuda_version}}`, `hardware_override`,
`root`, `models_dir`. It needs no extra and no network and never fails because a
package is missing (full schema: PROBLEM_AUTHORING section 19).

`learnai assets list --json` prints one JSON document (`models_dir`, `assets[]` with `name`,
`repo`, `revision`, `files`, `size_mb`, `state` = `present | missing | incomplete`,
`present`, `reason`, `path`, `bytes_on_disk`, `problems`, `fetch_command`, and
`issues[]`); it needs no extra and no network. `learnai assets fetch ... --json` needs
`--extra models` and writes `@@WS@@ `-prefixed events `asset_start`, `asset_progress`,
`asset_file`, and `asset_end` (exactly one per asset, last, with `ok` and `message`);
exit code 0 when every asset succeeded, 1 when one failed, 2 for a usage error. A fetch
is pinned to the declared revision, verified, and resumable per file. Full schemas:
PROBLEM_AUTHORING section 15.
Core passes `LEARNAI_ROOT=<root>` to `learnai assets` commands.

`validate` is the quality gate. For every supported framework of a problem: the reference
is accepted; the starter is not accepted. Every mutant is rejected. Metadata is complete
and consistent. A problem that fails validation must not be merged. `validate` reports
each problem as `ok`, `FAIL`, or `SKIP` (nothing could run on this machine; does not
fail the command) and warns about every case it could not run.

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
             | "import_error" | "harness_error" | "canceled"
             | "assets_missing" | "hardware_missing";

interface Track { id: string; title: string; description: string; order: number;
                  problemCount: number; solvedCount: number; }

interface ProblemSummary {
  id: string; title: string; track: string; order: number; difficulty: Difficulty;
  frameworks: Framework[]; judges: JudgeKind[]; tags: string[]; prerequisites: string[];
  requires: Requires; status: ProblemStatus; solvedFrameworks: Framework[];
  assets: string[];             // names of the model assets the problem declares
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

interface Metric { name: string; value: number; unit?: string; limit?: number;
                   higherIsBetter?: boolean; }

interface CaseInfo { id: string; name: string; public: boolean; requires?: Requires; }
type Skip = { kind: "hardware"; requires: Requires } | { kind: "asset"; asset: string }
          | { kind: "aborted" };
interface CaseResult { id: string; status: CaseStatus; durationMs: number;
                       message: string | null; detail: string | null;
                       stdout: string; metrics: Metric[]; skip?: Skip; }
interface MissingAsset { name: string; repo: string; sizeMb: number; }

type RunEvent =
  | { type: "preparing"; message: string }                      // environment sync, spawn
  | { type: "start"; cases: CaseInfo[]; env: Record<string, string>;
      requires: Requires; requiresMet: boolean }
  | { type: "caseStart"; id: string }
  | { type: "caseEnd"; result: CaseResult }
  | { type: "log"; stream: "stdout" | "stderr"; text: string }  // diagnostics
  | { type: "end"; verdict: Verdict; passed: number; total: number;
      durationMs: number; metrics: Metric[]; submissionId: number | null;
      skipped: number;                                          // cases not run
      missingAssets?: MissingAsset[]; missingHardware?: Requires;
      message: string | null; detail: string | null };          // import/harness errors, not-judged reasons

interface Submission { id: number; problemId: string; framework: Framework; mode: RunMode;
                       verdict: Verdict; passed: number; total: number; skipped: number;
                       durationMs: number; target: string; createdAt: string; }   // ISO 8601
interface StoredCase extends CaseResult { name: string; public: boolean; }
interface SubmissionDetail extends Submission { code: string; cases: StoredCase[]; metrics: Metric[]; }

interface PathStage { title: string; description: string; problems: string[]; }  // existing ids only
interface Path { id: string; title: string; description: string; stages: PathStage[];
                 problemCount: number; solvedCount: number; }

interface FrameworkEnv { framework: Framework; installed: boolean; version: string | null; }
interface EnvStatus { uv: string | null; python: string | null; frameworks: FrameworkEnv[];
                      root: string; }
interface Stats { solved: number; total: number; byDifficulty: Record<Difficulty, [number, number]>;
                  streakDays: number; }

type AssetState = "present" | "missing" | "incomplete";
interface ModelAsset { name: string; repo: string; revision: string; files: string[];
                       sizeMb: number;          // declared approximate download size
                       state: AssetState; reason: string | null;   // why it is incomplete
                       path: string; bytesOnDisk: number; problems: string[]; }
interface AssetList { modelsDir: string; assets: ModelAsset[]; }
type AssetEvent =
  | { type: "preparing"; message: string }                      // environment sync
  | { type: "start"; asset: string; totalBytes: number }
  | { type: "progress"; asset: string; file: string; doneBytes: number; totalBytes: number }
  | { type: "assetEnd"; asset: string; ok: boolean; message: string | null }
  | { type: "log"; text: string }
  | { type: "done"; ok: boolean; canceled: boolean; message: string | null };  // always last
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

| `list_paths` | | `Path[]` |
| `get_problem_asset` | `id, path` | `string`: a `data:` URL. `path` is relative to the problem's `assets/` directory and may not escape it; only `.svg` and `.png`, at most 1 MiB; an SVG must be UTF-8 and a PNG must have the PNG signature |
| `list_catalog_issues` | | `{ path: string; message: string }[]` (malformed tracks, problems, and paths, and path entries that were skipped) |
| `list_assets` | | `AssetList` (runs `learnai assets list --json`) |
| `fetch_assets` | `names: string[], target?` + event sink | `jobId: string`; stream of `AssetEvent` (runs `learnai assets fetch <names> --json` with `--extra models`) |
| `cancel_asset_fetch` | `jobId` | |
| `remove_asset` | `name` | `AssetList`; deletes `<models dir>/<name>` after checking that `name` is a declared asset |

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

Model assets: one fetch job at a time (a second `fetch_assets` fails with a clear
message). Canceling kills the process group; completed files are kept, so a later fetch
resumes. `cancel_asset_fetch` on an unknown or finished job is a no-op. `remove_asset`
fails while a fetch is running, and `env_install` and `fetch_assets` refuse to run at
the same time (both change the one environment). Every uv call core makes passes the
full set of installed extras, so no call removes another's packages. The app never
downloads anything without an explicit click. `assets_missing` and `hardware_missing` submits are stored (with `skipped`) but
never count as solved or toward the streak.

### 4.1 Run targets

A target is where the harness runs. `local` always exists. An **SSH target** runs the
same harness on a remote POSIX host (Linux with or without an NVIDIA GPU, or another
Mac) and streams the same events back; its name is `ssh:<alias>`.

```ts
interface SshHost { alias: string; hostName: string | null; user: string | null;
                    port: number | null; }                // from ~/.ssh/config, no wildcards
type CheckStepId = "connect" | "uv" | "sync" | "environment" | "doctor";
interface CheckStep { id: CheckStepId; ok: boolean; message: string;
                      fix: string | null; }               // fix: a command or instruction for the user
interface RemoteDoctor { python: string; system: string; machine: string;
                         frameworks: Partial<Record<Framework, string | null>>;
                         hardware: Requires[];            // what is present, always with "cpu"
                         gpuName: string | null; gpuMemoryBytes: number | null; }
interface TargetCheck { ok: boolean; checkedAt: string; steps: CheckStep[];
                        doctor: RemoteDoctor | null; }
interface TargetInfo { name: string;                      // "local" | "ssh:<alias>"
                       kind: "local" | "ssh"; label: string;
                       host: string | null; remoteDir: string | null;
                       lastCheck: TargetCheck | null; }
type TargetCheckEvent =
  | { type: "step"; step: CheckStep }
  | { type: "log"; text: string }
  | { type: "done"; check: TargetCheck };                 // always last
```

| command | arguments | returns |
|---|---|---|
| `list_ssh_hosts` | | `SshHost[]` |
| `list_targets` | | `TargetInfo[]`, `local` first |
| `add_target` | `host, remoteDir?` | `TargetInfo[]`; `host` must be an alias from `list_ssh_hosts`; `remoteDir` defaults to `~/.learnai` |
| `remove_target` | `name` | `TargetInfo[]`; removing `local` fails; nothing is deleted on the remote host |
| `check_target` | `name` + event sink | `jobId: string`; stream of `TargetCheckEvent` |

Rules:

- The app uses the system `ssh` and `rsync` and the user's own SSH configuration and
  agent. It **never** stores, reads, or asks for keys or passwords: every invocation
  uses `BatchMode=yes` (no prompts) and a connect timeout. A host that needs a password
  or an unknown host key fails the `connect` step with a `fix` that tells the user what
  to do in a terminal. Host aliases and remote directories are validated and passed as
  separate arguments, never spliced into a shell string unquoted.
- Remote layout: `<remoteDir>/root/` mirrors a content root: `python/` (without the
  virtual environment and caches), `problems/<track>/<id>/` for the problem being run,
  and `workspace/<id>/<framework>.py` for the solution. `launch` syncs these (only
  what changed), then runs the section 3.5 command there with `LEARNAI_ROOT` set and
  the extras the job's framework needs, emitting `preparing` events for each step.
- `check_target` runs the bootstrap: connect; find `uv` (the `fix` gives the official
  install command, which the user runs; the app does not install software on a remote
  host unasked); sync `python/`; build the environment; run `learnai doctor --json`. The
  result is saved as `lastCheck`. A run on a target never requires a prior check.
  The environment step runs `uv sync --inexact`. `check_target` on `local` runs only
  the `uv` and `doctor` steps and installs nothing. A check cannot be canceled.
- Cancel and timeouts terminate the remote process tree, not only the local `ssh`:
  the harness runs under a small POSIX supervisor in its own process group, and the
  supervisor kills the group when its stdin (the SSH channel core holds open) closes.
  On total network loss the remote side cleans up when its sshd notices. A connection
  dropped mid-run ends the run with `harness_error` and a message saying so.
- Targets are stored in the settings table. Submissions record the target name.
  `start_run` with an unknown target fails with a clear message.
- Model assets on a remote target live under `<remoteDir>/root/.learnai/models/`.
  `fetch_assets` takes an optional `target` (default `local`) and runs the same fetch
  there, syncing the problem directories that declare the assets first; `list_assets`
  and `remove_asset` are local only.
- Devbridge: the four plain commands as `POST /api/<command>`; `POST /api/check_target`
  returns `{"jobId"}` with events at `GET /api/jobs/<jobId>/events`.

### 4.2 Spaced re-solve

Solving a problem once is not mastery. After a problem is first solved, the app asks the
learner to solve it again **from the blank starter** at growing intervals.

```ts
interface Review { problemId: string; stage: number;        // 0-based index into the intervals
                   intervalDays: number; dueAt: string;     // ISO 8601
                   lastReviewedAt: string | null; due: boolean; }
type ReviewAction = "forgot" | "snooze" | "retire";
```

| command | arguments | returns |
|---|---|---|
| `list_reviews` | | `Review[]`, active reviews only, soonest due first |
| `start_review` | `id, framework` | `SolutionFile`: the review file, reset to the starter |
| `read_review_solution` | `id, framework` | `SolutionFile` (the starter if none saved) |
| `write_review_solution` | `id, framework, code` | |
| `resolve_review` | `id, action` | `Review[]` |
| `get_settings` | | `Settings` (`{ reviewsEnabled: boolean }`) |
| `set_reviews_enabled` | `enabled` | `Settings` |

Rules:

- Intervals are 3, 7, 21, 60, and 180 days. The first accepted submit of a problem (any
  framework) creates its review at stage 0, due 3 days later. Problems solved before the
  feature existed get a review dated from their first accepted submit.
- A review attempt is written to `workspace/<id>/review/<framework>.py`; the learner's
  saved solution is never touched. `start_run` takes `review?: boolean`; a review run
  judges the review file. `Submission` gains `review: boolean`.
- An accepted review **submit** advances the stage and sets the next due date from the
  time of the submit; after the last interval the review is retired (the problem is
  mastered). A non-accepted review submit changes nothing. `forgot` (the learner gave up
  and looked at their solution or the editorial) resets to stage 0, due the next day;
  `snooze` moves the due date to one day after the later of now and the current due date,
  without changing the stage; `retire` ends reviews for the problem.
- The review file is removed when a review is resolved (accepted review submit, `forgot`,
  `retire`); `snooze` keeps it, and `start_review` is the explicit reset. A review run or
  `start_review` fails when the problem has no active review or reviews are off. A review
  need not be due to be run. A normal accepted submit does not advance a review.
- While a review is in progress the workspace hides hints, the editorial, the reference,
  and the learner's earlier submissions' code until the learner chooses `forgot`.
- `Stats` gains `reviewsDue: number`. Review submits count toward the streak. Reviews can
  be turned off in Settings (stored setting `reviews_enabled`, default on); when off,
  nothing is scheduled or shown, and existing schedules are kept. Turning reviews on
  schedules problems solved while they were off, dated from their first accepted submit.
- Hiding during a review is done by the UI; core still serves the editorial and reference.

**Case stages (capstones).** `CaseInfo` and `StoredCase` gain `stage?: string`, taken
from `start.cases[].stage`. When any case has a stage, results group cases by stage in
first-appearance order and show per-stage pass counts ("3/5 passed"); a stage label is
public even for hidden cases. The heading reads "Stage <label>" (the label as written if
it already begins with "Stage"); in a staged problem, cases without a stage are grouped
last under "Other cases". Stages gate nothing: every selected case runs and the
verdict is computed from all cases. `case_end`, `end`, and the verdict do not mention
stages; consumers derive per-stage progress from `start.cases` and case statuses.
`case_end.duration_ms` excludes one-time library initialization the judge pays inside a
case with the clock stopped; `end.duration_ms` is wall time and includes it.

### 4.3 Transports

**Tauri.** One `#[tauri::command]` per row above. JavaScript passes camelCase argument
names (Tauri maps them to snake_case Rust parameters). Streaming commands take an extra
argument `onEvent: Channel<RunEvent>` (or `Channel<InstallEvent>`, `Channel<AssetEvent>`, `Channel<TargetCheckEvent>`);
`start_run` returns the `runId` immediately and events flow over the channel until `end`
(`fetch_assets` likewise returns the `jobId` and streams until `done`). Errors are
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
- `POST /api/fetch_assets` returns `{"jobId": "..."}`; `AssetEvent`s at the same
  `GET /api/jobs/<jobId>/events`, closing after `done`.

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
