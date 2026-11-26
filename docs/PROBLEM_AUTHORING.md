# Problem authoring guide

This is the authoritative guide for writing Learn AI problems. It covers the directory
format, the test API for all seven judge types, the framework adapter, the kit, the
rules that keep judging deterministic, and the validation workflow. You should not need
to read the judge's source to write a problem.

How a batch of problems is produced (research, specs, kit work, the brief an author
gets, every check in order, review, and commit) is in
[`PROBLEM_PROCESS.md`](PROBLEM_PROCESS.md).

The exemplar to copy is `problems/foundations/stable-softmax/`. Read it once, in full,
before writing your first problem. Working miniature problems for the other judge types
live under `python/tests/fixtures/catalog/problems/judges/`:

| judge | fixture |
|---|---|
| numeric | `problems/foundations/stable-softmax` (the exemplar) |
| property | `unit-normalize` |
| statistical | `sample-categorical` |
| cost | `greedy-decode` |
| speed | `pairwise-distances` |
| simclock | `dynamic-batcher` |
| service | `sse-tokens` |

Miniatures of the milestone 3 features live under
`python/tests/fixtures/m3/problems/`: `gpu/mlx-scale` (an MLX-only problem with a
memory limit and a throughput metric, section 13), `gpu/cuda-scale` (CPU cases plus a
case that needs CUDA, section 14), and `real/asset-vocab` (a model asset, a fixture
file, and cases that need no download, section 15). CUDA problems, which are judged on
the CPU with a kit of stand-ins, are section 18.

All commands below run from the repository root. Always pass both extras; a bare
`uv sync` removes the optional frameworks.

```sh
uv run --project python --extra jax --extra mlx learnai <command>
```

The rest of this guide abbreviates that prefix to `learnai`. Downloading model weights and
generating fixtures from a reference implementation additionally need `--extra models`
(section 15); nothing else does.

## 1. Workflow

```sh
learnai new decoding/top-p-filter --title "Nucleus (Top-p) Filtering"   # scaffold
# ... write the problem; replace every TODO(author) marker ...
learnai submit top-p-filter -f torch --reference        # iterate on the tests
learnai submit top-p-filter --mutant forgets_renormalize
learnai run top-p-filter -f mlx --starter               # what a learner sees first
learnai validate top-p-filter -v                        # the gate; must pass before merging
uv run --project python --extra jax --extra mlx ruff check problems/decoding/top-p-filter
```

`learnai new <track>/<id>` takes `--frameworks torch,jax,mlx` (the default),
`--frameworks python`, or `--frameworks mlx` (an MLX-only problem, section 13), plus
`--title` and `--difficulty`. It writes a complete toy
problem in the house layout and creates `track.toml` if the track is new. `learnai validate`
refuses any problem that still contains `TODO(author)`.

`learnai run` executes public cases, `learnai submit` all cases. Both take `-f <framework>` and one
of `--reference`, `--starter`, `--mutant <name>`, or `--solution <path>`; with none of
them they judge `workspace/<id>/<framework>.py`. `--case <id>` runs a single case and
`--events` prints the raw protocol.

`learnai run` and `learnai submit` show what a learner sees. A hidden case is reported like
a public one, under its own id and name and with the full failure detail; the only
difference is that `learnai run` does not execute it. `--case` takes the case id:
`--case very_negative`.

### What `learnai validate` enforces

For every framework the problem lists:

- the reference is **accepted** on a full submit;
- the starter runs and is **not accepted** (it must import and define every entry name,
  so a starter that fails to import is an error, not a pass).

Every mutant runs and is rejected with `wrong_answer`, `runtime_error`, or `time_limit`.
A mutant that is accepted means the tests have a hole: add a case, do not delete the
mutant.

Metadata must be complete and consistent:

- `problem.toml` has every key with the right type; `id` equals the directory name, is
  lowercase words joined by hyphens, and is unique across the whole catalog; `track`
  equals the parent directory name; `difficulty`, `judges`, `requires`, and `frameworks`
  use known values; `entry` is a non-empty list of Python names; `time_limit_s` is
  positive; every prerequisite id exists.
- `frameworks` is `["python"]`, or a set of tensor frameworks that includes `"torch"`
  (the oracle is written in torch), or exactly `["mlx"]` (an MLX-only problem, whose
  MLX reference is the oracle; section 13).
- `requires = "mlx-gpu"` needs `"mlx"` in `frameworks`, and `requires = "cuda"` needs
  `"torch"`. A case may require only `"cpu"` or what the problem requires (section 14).
- Every `[[assets]]` table is well formed (section 15): a lowercase `name`, a `repo`,
  a full 40-character `revision`, `files` that can only select safetensors, tokenizer,
  and config files, a positive `size_mb`. Two problems that use one asset name agree on
  its repository and revision, and a house model's name means the house model.
- `track.toml` exists with `id` (equal to the directory name), `title`, `description`,
  and `order`.
- `statement.md`, `hints.md`, `editorial.md`, and `tests.py` exist and are not empty; no
  hint is empty.
- `starter/<framework>.py` and `reference/<framework>.py` exist for every listed
  framework, and for no other; a starter is not identical to its reference.
- At least one mutant exists, and line 1 of every mutant is `# BUG: <what is wrong>`.
- The tests define at least one public and at least one hidden case.
- Either every case has a `stage=` or none has (section 4, "Stages").
- Every name in `forbidden` (section 4.1) resolves to a real attribute that the harness
  can guard safely, in every framework the names mention. A typo, a function that a
  framework release renamed, a dtype or constant (not callable), or a C type that
  cannot be guarded without breaking it for everyone fails validation with the reason,
  instead of silently enforcing nothing. Every entry of `forbidden_source` is a single
  identifier.
- Nothing contains `TODO(author)`.

It warns (without failing) when the cases of a reference submit take more than 3
seconds in total.

**When this machine cannot run every case.** A case may need hardware the machine does
not have (section 14) or a model asset that is not downloaded (section 15). Such cases
are skipped by the harness, and validation never counts what it did not see:

- The reference must pass every case that *can* run. The cases that could not are
  listed in a warning on the problem ("only 2 of 3 cases could run on this machine
  (all passed); 1 was NOT validated: needs an NVIDIA GPU (CUDA), which this machine
  does not have (GPU cases not validated: on_cuda)"). The problem is still `ok`, but
  the warning is printed on every run, not only with `-v`, and the summary line
  counts such problems: "110 of 110 problems passed validation; 14 passed with cases
  that could NOT be validated on this machine (see the warnings above)".
- When every case ran and only the problem's `requires` is unmet (no case is marked,
  or `learnai run` selected only CPU cases), nothing was skipped and the warning says so:
  "all 9 cases ran and passed, but not on the hardware the problem requires (an Apple
  GPU (MLX with Metal)), which this machine does not have". In that situation a
  starter or mutant that passes every case is an error, exactly as on a machine with
  the hardware: every test ran.
- The starter must be rejected by the cases that run here. A starter that survives
  them is an error, whatever the skipped cases would have said.
- A mutant that survives the cases that run here is reported as "NOT validated": only
  a machine that runs the skipped cases can tell whether a test catches it.
- If nothing could run (every case needs the missing asset or hardware), the problem
  is reported as `SKIP`, with the reason, and the summary line counts it separately:
  "80 of 81 problems passed validation; 1 skipped (not validated on this machine, see
  above)". A skip does not fail the command. Static checks (metadata, files, the
  `forbidden` list) still run.
- A framework this platform can never have is not run at all. MLX is installed only on
  Apple silicon, so on Linux and Windows a problem's MLX reference and starter (and the
  mutants of an MLX-only problem) are left out, and the problem lists "skipped:
  reference (mlx): not validated here: MLX is installed only on Apple silicon". The
  problem is `ok` when its other frameworks pass and `SKIP` when MLX is its only one.
  The MLX names on a `forbidden` list are not resolved there, and a mutant that imports
  MLX is reported as "NOT validated". A framework that is only not installed, on a
  machine that could have it, is still an error.
- `learnai validate --require-assets` makes a missing asset an error. Use it on the machine
  that is supposed to validate everything (and in CI once that machine has the weights).

Harness runs execute in parallel (`-j N`, default 4), with one exception that needs no
flag: problems whose `judges` include `speed` or `service` are timing-sensitive, so
their runs are held back and executed one at a time after the parallel batch has
finished, with nothing else competing for the machine. That is why `judges` must list
every judge type the tests use. If a speed case still flakes, the gap it measures is
too small: widen `max_ratio` or make the input larger (section 5.5).

## 2. Directory format

```
problems/<track>/track.toml
problems/<track>/<id>/
  problem.toml
  statement.md
  starter/<framework>.py      one per framework
  reference/<framework>.py    one per framework; torch.py (or python.py) is the oracle
  tests.py
  hints.md
  editorial.md
  mutants/<bug_name>.py       oracle framework only; line 1: # BUG: <what is wrong>
  fixtures/                   optional: committed reference outputs (section 15)
```

The **oracle framework** is torch when the problem lists it, `python` for a
framework-free problem, and `mlx` for an MLX-only problem. The oracle reference and
every mutant are written in it.

`track.toml`:

```toml
id = "foundations"
title = "Foundations"
description = "One or two sentences shown on the track page."
order = 10
```

`problem.toml` (every key is required, except `forbidden`, `forbidden_source`, and
`[[assets]]`):

```toml
id = "stable-softmax"
title = "Numerically Stable Softmax"
track = "foundations"
order = 10                        # position within the track; leave gaps (10, 20, ...)
difficulty = "easy"               # easy | medium | hard
frameworks = ["torch", "jax", "mlx"]      # or ["python"]
judges = ["numeric"]              # every judge type the tests use
tags = ["softmax", "numerics", "logits"]
prerequisites = []                # problem ids
requires = "cpu"                  # cpu | mlx-gpu | cuda
entry = ["softmax", "log_softmax"]        # names the solution must define
time_limit_s = 10                 # per-case limit
forbidden = [                     # optional: calls the statement rules out (section 4.1)
    "torch.softmax", "torch.nn.functional.softmax", "torch.Tensor.softmax",
    "jax.nn.softmax", "mlx.core.softmax",
]
forbidden_source = ["bfloat16"]   # optional: identifiers ruled out of the source (section 4.1)

[[assets]]                        # optional, repeatable: model weights (section 15)
name = "granite-4.0-350m"
repo = "ibm-granite/granite-4.0-350m"
revision = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
files = ["config.json", "tokenizer.json", "model.safetensors"]
size_mb = 712
```

`requires` is the hardware a solution needs before it can be *accepted* (section 14).

A solution that fails to import, or lacks an `entry` name, gets the `import_error`
verdict before any case runs. Entry names can be functions, classes, or (for service
problems) the ASGI object `app`.

## 3. The two rules that keep judging exact

### Float32 everywhere

All tensors are float32 unless the problem is about dtypes. MLX has no float64 on the
GPU and JAX defaults to 32-bit, so a test that only passes in float64 fails for two of
three frameworks. Consequences:

- Build inputs as float32 (`ctx.randn`, `ctx.uniform`, or `np.array(..., dtype=np.float32)`).
- Choose tolerances for float32 (section 9).
- The oracle computes in float32 too. Do not write a float64 reference "for accuracy";
  design the test so that float32 is enough.

### Randomness is an input

A solution never draws from a framework random number generator. Torch, JAX, and MLX
generators produce different streams, so their outputs could never be compared. Instead
the test draws the randomness and passes it in:

```python
def sample(probs, u):            # u: a uniform number in [0, 1), supplied by the test
def top_p_sample(logits, p, u):
def speculative_step(p, q, token, u_accept, u_resample):
```

Every case has its own seeded generator, `ctx.rng`, derived from the problem id and the
case id. It is identical for every framework and every run, so each case is fully
deterministic, and a failure is reproducible.

Likewise, function signatures are identical across frameworks and differ only in tensor
type. State is passed in and returned (never mutated in place), so the same signature
works in JAX.

## 4. Writing tests

`tests.py` registers cases with `@case`. A case is a function of one argument, the
context. It passes by returning and fails by raising `CheckFailure`, which every `check`
helper does for you.

```python
"""Tests for stable-softmax. Every case compares the solution with the oracle."""

import numpy as np

from learnai.judge import case

BOTH = ("softmax", "log_softmax")


@case("the worked example: logits [1, 2, 3]", public=True)
def worked_example(ctx):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    for fn in BOTH:
        ctx.check(fn, x)


@case("very negative logits")
def very_negative(ctx):
    x = np.array([[-1001.0, -1002.0, -1003.0], [-9000.0, -9000.5, -1e4]], dtype=np.float32)
    for fn in BOTH:
        ctx.check(fn, x, axis=-1)
```

### `@case(name, *, public=False, id=None, time_limit_s=None, frameworks=None, requires="cpu", assets=None, stage=None)`

- `name`: what the learner reads. State what is checked: "large logits: a row spanning
  -1e4 to 1e4", not "test 3".
- `public=True`: runs on Run and Submit. `public=False` (the default, a "hidden"
  case): runs on Submit only. That is the only difference: either way the learner
  sees the id, the name, the failure message, the detail with the inputs, and the
  metrics. So name hidden cases as descriptively as public ones.
- `id`: defaults to the function name. Ids are stable identifiers; do not rename them
  casually: the id seeds the case's random generator.
- `time_limit_s`: overrides the problem's per-case limit.
- `frameworks`: restricts the case to some solution frameworks. Rarely needed.
- `requires`: the hardware the case needs, `"cpu"` (the default: it runs on every
  machine) or the problem's own `requires` (`"mlx-gpu"`, `"cuda"`). Where the hardware
  is absent the case is `skipped` with the reason. Section 14.
- `assets`: the names of the problem's `[[assets]]` the case needs. The default is all
  of them; `assets=[]` marks a case that needs no download. A case whose asset is not
  on disk is `skipped` with the fetch command. Section 15.
- `stage`: the stage of a multi-part problem this case grades. See "Stages" below.

Cases run in definition order. Put the worked example from the statement first, then
progressively harder public cases, then hidden ones.

**Public versus hidden.** Public cases are for debugging with Run: they are the
examples from the statement and the main edge-case families. Hidden cases add coverage
on Submit: more edge cases, random sweeps, and whatever each mutant needs. A good
split is 3 to 5 public and 4 to 8 hidden.

**Stages.** A capstone is one problem built in parts: stage 1 is an allocator, stage 2
a batching loop on top of it, and so on. Give each case the stage it grades:

```python
@case("fork shares every block by reference", public=True, stage="1: block allocator")
def fork_shares(ctx): ...

@case("FCFS: the trace equals the oracle's", public=True, stage="2: continuous batching")
def fcfs_trace(ctx): ...

@case("a preempted sequence resumes with the right token index", stage="4: preemption")
def resumes(ctx): ...
```

- A stage is its label. Cases with the same label are one stage, and stages are
  ordered by the first case that names them, so write the number into the label
  (`"2: continuous batching"`; the app and the terminal show it as "Stage 2:
  continuous batching") and keep a stage's cases together.
- **The label is shown to the learner**, as in "Stage 4: preemption, 2 of 5 passed".
  It names the part of the build. At most 60 characters, one line.
- Either every case of a problem has a stage or none has; `learnai validate` rejects a
  mix. Nothing else about a case changes: ids, time limits, and the verdict are as
  before. A stage is not a gate:
  every case runs whatever happened in earlier stages, so make each stage's cases
  fail cleanly (not crash the file) when a later entry point is still a stub.
- `learnai run` and `learnai submit` print a heading whenever the stage changes and, under the
  verdict, one line per stage with its pass count. The curriculum's convention for
  capstones still holds: early stages public-heavy, the last one hidden-heavy.

### The context

| member | what it is |
|---|---|
| `ctx.check(name, *args, **kwargs)` | Call the solution and the oracle with the same inputs and compare. The one-line numeric test. Returns the solution's result (numpy). |
| `ctx.call(name, *args, **kwargs)` | Call the solution. numpy in, numpy out. |
| `ctx.expected(name, *args, **kwargs)` | Call the oracle reference the same way. |
| `ctx.baseline(name, *args, **kwargs)` | Call the reference written in the solution's framework. |
| `ctx.check_speed(name, *args, max_ratio=..., tokens=None, **kwargs)` | Speed judge (section 5.5). |
| `ctx.compare_speed(run_solution, run_reference, *, max_ratio=..., label=...)` | Speed judge for prepared callables: compiled functions, loops (section 5.5). |
| `ctx.randn(*shape, scale=1.0, loc=0.0)` | float32 normal array from `ctx.rng`. |
| `ctx.uniform(*shape, low=0.0, high=1.0)` | float32 uniform array in `[low, high)`. |
| `ctx.randint(low, high, shape)` | int64 array in `[low, high)`. |
| `ctx.rng`, `ctx.seed` | The case's numpy `Generator` and its seed. |
| `ctx.child_rng(i)` | An independent generator for trial `i` (property tests). |
| `ctx.metric(name, value, unit=None, limit=None, *, higher_is_better=False)` | Report a number with the case. |
| `ctx.asset(name, file=None)` | Path of a downloaded model asset, or of one file in it (section 15). |
| `ctx.fixture(name)`, `ctx.fixture_path(name)` | A committed reference output from the problem's `fixtures/` directory, loaded by extension; or just its path (section 15). |
| `ctx.solution`, `ctx.oracle`, `ctx.reference` | The modules: the learner's, the oracle reference, the same-framework reference. |
| `ctx.fw`, `ctx.oracle_fw` | Framework adapters (section 6) for the solution and for the oracle. |
| `ctx.framework`, `ctx.problem` | The solution's framework name; the problem metadata. |
| `ctx.simclock()` | A fresh simulated clock, closed after the case (section 5.6). |
| `ctx.serve(app=None)` | Serve the solution's ASGI app (section 5.7). |
| `ctx.defer(fn)` | Run `fn` after the case, whatever the outcome. |

`ctx.call`, `ctx.expected`, and `ctx.baseline` convert for you: every numpy array in the
arguments (also inside lists, tuples, and dicts) becomes a tensor of the target
framework, and every tensor in the result becomes a numpy array. Python scalars,
strings, and other objects pass through unchanged.

> [!IMPORTANT]
> **Containers are rebuilt, so `ctx.call` cannot test in-place mutation.** To convert
> the arrays inside a list, tuple, or dict, `ctx.call` builds a new container, even
> when there is nothing in it to convert. The solution therefore never receives the
> very object your test holds: `ctx.call("push", heap, 3)` leaves the test's `heap`
> untouched however correct the solution is. (This is deliberate. `ctx.check` passes
> the same arguments to the oracle next, and they must not have been changed.) When
> the specification is "modifies its argument", call the solution directly, which
> converts nothing:
>
> ```python
> @case("push keeps the heap property, in place", public=True)
> def push_in_place(ctx):
>     heap = [1, 5, 3]
>     result = ctx.solution.push(heap, 0)       # the solution gets this very list
>     check.that(result is None, "push: must return None and modify the list")
>     check.equal(heap, [0, 1, 3, 5], label="heap after push")
> ```
>
> For a tensor problem, convert by hand around the direct call:
> `x = ctx.fw.tensor(array)`, `ctx.solution.fn(x)`, then `ctx.fw.to_numpy(x)`. Calls
> made this way are still covered by the `forbidden` guards. Remember that JAX arrays
> are immutable, so an in-place specification is only possible for `python`, torch,
> and MLX-only problems.

> [!NOTE]
> **Dtypes of converted inputs.** Floating arrays arrive as float32. Integer arrays
> **keep their dtype**: an `np.int8` array arrives as an int8 tensor, `np.uint8` as
> uint8, `np.int32` as int32, in every framework. The one exception is 64-bit integers,
> numpy's default (`np.array([1, 2])`, `ctx.randint`, `np.arange`): they become the
> framework's default integer, int64 in torch and int32 in JAX and MLX. So to hand a
> solution int8 codes, pass an int8 array to `ctx.call`; to hand it ordinary token ids,
> pass the int64 array numpy gives you. Section 6 has the full rules.

 Kit objects such as `TinyLM`
are rebound to the target framework, and the oracle and baseline always get a fresh twin
with its own counters, so calling `ctx.expected` never pollutes the counters the
solution is judged by.

A solution that returns the wrong type fails with a clear message, for example
`` `softmax` returned a numpy.ndarray, expected a torch.Tensor ``.

`ctx.check(name, *args, rtol=None, atol=None, dtype=True, **kwargs)` compares float
results with `check.allclose`, integer and bool results exactly, and tuples, lists of
tensors, and dicts element by element (labels such as `softmax[1]`). With `dtype=True`
the result must also have the oracle's dtype kind and float width, which enforces the
float32 rule. The keyword names `rtol`, `atol`, and `dtype` are reserved; if your
function has a parameter with one of those names, use `ctx.call` and `ctx.expected`
with `check.allclose` instead.

### Failure messages

Every failure has a one-line `message` and a longer `detail`.

```
softmax: got NaN where a number is expected (4 of 6 values, first at index [0, 0])

call
  softmax(x, axis)
    x       = float32 array, shape (2, 3)
              [[  1001.,   1002.,   1003.],
               [ 10000.,      0., -10000.]]
    axis    = -1

result
  shape            (2, 3), dtype float32
  mismatched       4 of 6 values
  max abs error    inf at index [0, 0]   (expected 0.09003057, got nan)
  tolerance        |got - expected| <= 1e-06 + 0.0001 * |expected|

expected
  [[0.090031, 0.244728, 0.665241],
   [1.      , 0.      , 0.      ]]

got
  [[nan, nan, nan],
   [nan,  0.,  0.]]
```

The detail is laid out in groups, one blank line apart, always in this order: `call`
(when the test went through `ctx.check`), the facts about the `result` in two columns,
then `expected` and `got` (with the slice they show when the arrays are large:
`expected[3, 0:8]`). Text that a test passes itself (the `detail` of `check.that`, a
`context=` string) is shown as written, as a group of its own.

A shape mismatch is reported first and alone (`softmax: got shape (4,), expected shape
(2, 3)`), then dtype, then values with the worst index and an excerpt around the worst
element.

**Message and detail are shown for every case, public or hidden.** The message is one
line: what the solution produced, what was expected where that is short (a shape, a
length, a dtype, an exception type, the worst value), and where it went wrong. The
detail is the longer evidence: the call with its inputs, the numbers, and the
excerpts. All `check` helpers follow this:

| failure | message | detail |
|---|---|---|
| wrong shape | `softmax: got shape (4,), expected shape (2, 3)` | both shapes |
| wrong length | `tokens: got 3 items, expected 5` (scalars and nested values: `x: got 3 values, expected 5`) | both lengths, both lists |
| wrong values | `y: 2 of 60 values differ from the expected result (largest error 0.1234567 at index [1, 13]: expected 0.5, got 0.6234567)` | tolerance, expected and got |
| wrong dtype | `y: got dtype float64, expected dtype float32` | both dtypes |
| wrong dict keys | `stats: got a dict whose keys are not the expected keys` | both key lists |
| wrong exception | `decode: raised KeyError, expected ValueError` | the expected type |

**Limits.** When `check.cost`, `check.at_most`, `check.memory`, or the speed judge
fails, the message repeats the limit (`tokens_processed: 114 tokens exceeds the limit
of 15 tokens`), so the learner can tell how far off they are. State limits in the
statement.

**Metrics.** A case carries its metrics (value and limit, from which the app draws
"114 / 15 tokens") whether it passes or fails, public or hidden. The run-level metrics
at the end of a run are the worst values over all cases that reported some.

When you write your own failure with `check.that(condition, message, detail)`, follow
the same split: `message` is one line saying what the solution produced and what was
expected, `detail` is the longer evidence. Expected values and inputs may appear in
either. Plain `assert` works too (it counts as a failed case), but its text becomes
the message, so prefer `check.that`.

An exception raised by the solution is an `error` (verdict `runtime_error`). The
*exception line* below is always one line: the exception type and the first
meaningful line of its text (`RuntimeError: size mismatch for layer 3`). Framework
errors are often many lines (a summary, then shapes, hints, a stack of their own);
blank and decoration-only lines are skipped, and a first line that only introduces
the rest (it ends with a colon) is joined with the next one. A `SyntaxError` keeps
its message line.

The message is the exception line and the detail is a traceback trimmed to the
learner's own frames (or to the frames of `tests.py` when the learner's file is not on
the stack at all), in public and hidden cases alike.

When the solution fails to import, the verdict is `import_error`; the message and the
traceback are on the run's final `end` event, once, and every case is reported as
skipped with the one-line reason only.

### 4.1 Forbidden calls

When the statement says "do not call your framework's built-in `softmax`", list the
calls in `problem.toml` and the harness enforces the sentence:

```toml
forbidden = [
    "torch.softmax", "torch.log_softmax", "torch.logsumexp",
    "torch.nn.functional.softmax", "torch.nn.functional.log_softmax",
    "torch.Tensor.softmax", "torch.Tensor.log_softmax", "torch.Tensor.logsumexp",
    "jax.nn.softmax", "jax.nn.log_softmax", "jax.nn.logsumexp",
    "mlx.core.softmax", "mlx.core.logsumexp", "mlx.core.array.logsumexp",
    "mlx.nn.softmax", "mlx.nn.log_softmax", "mlx.nn.Softmax", "mlx.nn.LogSoftmax",
]
```

A name is the full dotted path, starting at the package, of one of:

| kind | example | what is guarded |
|---|---|---|
| a function in a module | `torch.softmax`, `jax.nn.softmax`, `time.sleep` | calling it, under any alias: `from torch import softmax`, `F.softmax` when it is the same object, private re-exports |
| a method on a tensor or other class | `torch.Tensor.softmax` | `x.softmax(-1)` |
| a class | `torch.nn.RMSNorm`, `mlx.nn.RoPE` | constructing it (also through a subclass) |

Two kinds of names cannot be guarded, and `learnai validate` refuses them with the reason
rather than accept a rule it cannot enforce:

- **Something that is not called**: a dtype (`torch.bfloat16`), a constant. There is
  no call to intercept. Use `forbidden_source` (below).
- **A type implemented in C that takes its arguments in `__new__`**, such as
  `json.scanner.make_scanner`. Replacing the constructor of such a type breaks it for
  every caller in the process (an earlier version of the harness did, and `json.loads`
  stopped working). Validation says: "is a type implemented in C that takes its
  arguments in `__new__`; a guard on it would break it for every caller. Rule out its
  identifier with `forbidden_source = ["make_scanner"]` instead". Python classes, and
  C types with an `__init__`, are guarded as before.

How it works, so you can predict it:

- Each name is replaced, for the whole run, by a guard that inspects the call stack.
  If a frame of the **solution file** is on the stack, the case fails with
  `` `torch.softmax` is not allowed in this problem (called at torch.py line 6, in
  softmax) `` and the verdict is `wrong_answer`. Catching the exception does not help: the violation is recorded, and the case
  fails even if the solution then computes the right answer another way.
- Any other caller gets the original function. The oracle, the same-framework
  reference (`ctx.baseline`, `ctx.check_speed`), your tests, and the kit are not the
  solution file, so they are unaffected. **References must not need a forbidden call**
  all the same: `learnai validate` judges each reference as a solution, and it would fail.
- The guard is transitive. `torch.nn.Softmax(dim)(x)` ends up in
  `torch.nn.functional.softmax`, so it is caught without being listed. The price: do
  not forbid a low-level utility that allowed library functions use internally
  (forbidding `time.monotonic` in a JAX problem fails correct solutions, because JAX
  reads the clock while it compiles). Forbid the specific shortcut, and let
  `learnai validate` confirm that the references still pass.
- Names from all frameworks go in one list. Only the names of packages that are
  loaded are guarded, and the list is checked again before every case, so a JAX
  solution that imports torch to borrow its `softmax` is caught as well.
- List every spelling a learner would reach for. Different objects need separate
  entries: `torch.softmax`, `torch.nn.functional.softmax`, `torch.special.softmax`, and
  the method `torch.Tensor.softmax` are four functions. Aliases of one object
  (`jax.nn.logsumexp` is `jax.scipy.special.logsumexp`) need one entry, and listing
  both is harmless.
- **A kit operation needs one entry.** `learnai.kit.ops.eb_accept` only routes the call
  to `learnai.kit.ops._torch.eb_accept` (or `_jax`, `_mlx`, `_numpy`), and a solution
  can call those directly. When a name is `learnai.kit.ops.<operation>`, the harness
  guards the four functions behind it as well, each as soon as its framework is
  loaded, and the message names the operation you listed. Do not list the backends
  yourself. The guard is transitive here too: an allowed operation that is built on
  the forbidden one inside a backend (`gated_delta_layer` calls `causal_conv1d`) is
  refused with it.
- `mlx.core.array` is an immutable type, so its methods cannot be guarded at run time.
  For those names only (`mlx.core.array.logsumexp`), the harness scans the solution's
  source before running anything and fails every case if it finds
  `<expression>.logsumexp` where the expression is not an imported module. This is
  cruder than the run-time guard (it also flags a method of the same name on the
  learner's own class), so list MLX array methods only when they really are the
  shortcut.
- It is a guard against taking the shortcut, not a sandbox. Code written to defeat it
  can; the tests still have to make sure the answer is right.

**`forbidden_source`: identifiers that may not appear in the solution.** For a rule
about something that is mentioned rather than called, list plain identifiers:

```toml
forbidden_source = ["bfloat16"]
```

Before anything runs, the harness parses the solution file and fails every case if the
identifier appears as a bare name (`bfloat16`), an attribute (`torch.bfloat16`,
`x.bfloat16()`), any part of an import (`from torch import bfloat16`; for
`"safetensors"`, also `import safetensors.torch`), a keyword argument, or a string that
is exactly the identifier (`dtype="bfloat16"`, `getattr(torch, "bfloat16")`). Comments,
docstrings, and longer strings do not count. The message is
`` `bfloat16` is not allowed in this problem (used at torch.py line 6) ``.

It is a blunt instrument, by design: it cannot tell `torch.bfloat16` from a local
variable the learner happened to call `bfloat16`. Use it for identifiers nobody uses by
accident (a dtype name, a package name such as `safetensors` or `tokenizers`), prefer
`forbidden` whenever the thing is called, and never hand-roll a source scan in
`tests.py`: `problems/foundations/bf16-decode` shows the supported way. The starter
and the references must not contain the identifier either (docstrings may).

**`forbidden_source` is for names that are never ordinary words.** `counters`, `peek`,
`log`, `_state`, and `_k` are words a learner uses for their own variables, and a
solution with `counters = {}` in it would fail every case with a message about a rule
it did not break. What those entries were meant to protect, the attributes of a kit
object handed to the solution, is protected at run time by `Guarded` (sections 5.4
and 7), which refuses the access itself, whatever the name was spelled or computed as.

Every problem with a `forbidden` or `forbidden_source` list needs a mutant that takes
the shortcut (section 11), and the statement must still state the rule in words.

### The `check` helpers

`from learnai.judge import check`

| helper | use |
|---|---|
| `check.allclose(got, expected, *, rtol=1e-4, atol=1e-6, label="output", dtype=False)` | Float comparison. NaN matches NaN; infinities must match exactly. |
| `check.equal(got, expected, *, label="output")` | Exact equality: int/bool/token arrays (integer width ignored), lists, strings, dicts. A list of 0-d tensors is a list of scalars (see below). |
| `check.match(got, expected, ...)` | Dispatches to the two above by structure; what `ctx.check` uses. |
| `check.shape(got, shape, *, label)` | Shape only. |
| `check.finite(got, *, label)` | No NaN or infinity. |
| `check.that(condition, message, detail=None)` | Any other assertion. |
| `check.raises(ExcType, fn, *args, label="call")` | Specified error behavior. |
| `check.forall(prop, *, trials=25)` | Property judge. |
| `check.distribution(samples, probs, *, delta=1e-10, label)` | Statistical judge, categorical. |
| `check.mean(values, expected, *, low, high, delta=1e-10, label)` | Statistical judge, bounded mean. |
| `check.cost(counters, name, *, at_most=None, exactly=None, at_least=None, unit=None, metric=None)` | Cost judge. `metric=` names the reported metric (default: the counter's name). |
| `check.at_most(name, value, limit, *, unit=None)` | Bound a measured value and report it as a metric. |
| `check.memory(measurement, *, at_most_mib, what="peak", metric=None)` | Memory limit on an `mlx_memory` measurement (section 13). |

Always pass a `label` that names the quantity (`label="tokens"`); it starts the message.

**Lists of scalar tensors.** A solution that collects scalar tensors (`tokens.append(
logits.argmax())`) returns a list of 0-d tensors, which `ctx.call` turns into a list of
0-d arrays. `check.equal` and `ctx.check` compare such a list as a list of scalars:
each 0-d array counts as the number it holds, so it equals `[5, 6, 7]`, a list of 0-d
arrays from the oracle, or `np.array([5, 6, 7])`, and a mismatch is reported once, with
its position, as for any list. Nothing else is coerced: an array with one or more
dimensions inside a list (`[tensor([5]), ...]`) is compared as an array and fails on
shape against a scalar. If the statement requires Python ints rather than scalar
tensors, assert that separately (`check.that(all(type(t) is int for t in tokens), ...)`).

## 5. The seven judge types

List every judge type your tests use in `judges`. Most problems use `numeric` plus at
most one other.

### 5.1 numeric

The solution's output matches the oracle within tolerance. A typical test is one to
four lines:

```python
@case("a random batch of logits, last axis", public=True)
def random_batch(ctx):
    x = ctx.randn(4, 8, scale=3.0)
    ctx.check("softmax", x, axis=-1)
```

Exact comparison for integers and tokens is automatic in `ctx.check`, or explicit:

```python
@case("encodes the worked example", public=True)
def worked_example(ctx):
    check.equal(ctx.call("encode", "low lower lowest", MERGES), [12, 7, 12, 9, 3], label="tokens")
```

Compare against literal values for the statement's worked example when the numbers are
part of the specification; compare against the oracle (`ctx.expected`) everywhere else,
so tests stay correct if the specification is clarified.

Numeric tests must include the inputs that break naive implementations: extreme
magnitudes, size-one axes, ties, masked (`-inf`) entries, square shapes (which hide
broadcasting bugs), and every axis position. A sweep over random shapes is a good final
hidden case:

```python
@case("random shapes, axes, and scales")
def random_shapes(ctx):
    for _ in range(8):
        shape = tuple(int(n) for n in ctx.rng.integers(1, 7, size=ctx.rng.integers(1, 5)))
        axis = int(ctx.rng.integers(-len(shape), len(shape)))
        x = ctx.randn(*shape, scale=float(10.0 ** ctx.rng.uniform(-1, 3.5)))
        ctx.check("softmax", x, axis=axis)
```

### 5.2 property

An invariant that must hold on seeded random inputs. Use it when there is no single
right answer to compare with, or to state what the function means independently of the
reference. `check.forall(prop, trials=N)` calls `prop(rng)` N times, each with its own
generator; during a trial `ctx.rng`, `ctx.randn`, and friends draw from that generator.
The first failing trial is reported with the line needed to replay it:
`np.random.default_rng([seed, trial])`.

```python
@case("every output vector has unit length and keeps its direction")
def unit_length(ctx):
    def prop(rng):
        shape = tuple(int(n) for n in rng.integers(2, 6, size=rng.integers(1, 4)))
        axis = int(rng.integers(-len(shape), len(shape)))
        x = ctx.randn(*shape, scale=float(10.0 ** rng.uniform(-2, 2)))
        y = ctx.call("normalize", x, axis=axis)
        check.shape(y, x.shape, label="normalize")
        norms = np.linalg.norm(y, axis=axis)
        check.allclose(norms, np.ones_like(norms), label="norm of the output")

    check.forall(prop, trials=10)
```

Compute the invariant in numpy on the converted outputs, as above. Keep `trials` small
for JAX's sake (section 10).

### 5.3 statistical

The output distribution matches a target. The test supplies independent uniform numbers
(randomness is an input), collects the outcomes, and compares the empirical
distribution with the exact target.

```python
@case("the samples follow the distribution")
def follows_distribution(ctx):
    probs = ctx.rng.dirichlet(np.ones(6)).tolist()
    uniforms = ctx.rng.random(20_000)
    samples = [ctx.call("sample", probs, float(u)) for u in uniforms]
    check.distribution(samples, probs, label="sample")
```

For a speculative-sampling step, the target is the big model's distribution `p`,
whatever the draft distribution `q` was:

```python
@case("accepted-or-resampled tokens are distributed as p")
def output_follows_p(ctx):
    p, q = ctx.rng.dirichlet(np.ones(8)), ctx.rng.dirichlet(np.ones(8))
    n = 4000
    drafts = ctx.rng.choice(8, size=n, p=q)                  # the draft model's proposals
    u = ctx.rng.random((n, 2)).astype(np.float32)
    tokens = [
        int(ctx.call("speculative_step", p.astype(np.float32), q.astype(np.float32),
                     int(drafts[i]), float(u[i, 0]), float(u[i, 1])))
        for i in range(n)
    ]
    check.distribution(tokens, p, label="output token")
```

**The bound.** `check.distribution(samples, probs, delta=1e-10)` takes outcomes in
`range(K)` (or `counts=`) and the exact target probabilities. For outcome $i$ with
target probability $p_i$ and observed frequency $f_i$ over $N$ independent draws, the
Chernoff bound gives $P(f_i \ge a) \le e^{-N\,\mathrm{KL}(a \| p_i)}$ for $a > p_i$ and
the mirror image below, where KL is the Bernoulli relative entropy. Hence
$P\big(N\,\mathrm{KL}(f_i \| p_i) \ge t\big) \le 2e^{-t}$. The check fails when any
outcome has $N\,\mathrm{KL}(f_i \| p_i) \ge t$ with $t = \ln(2K/\delta)$; by the union
bound a correct solution fails with probability at most $\delta$. This is a
finite-sample guarantee (unlike a chi-square test) and stays sharp for rare outcomes
(unlike Hoeffding). An outcome with $p_i = 0$ must never occur.

`check.mean(values, expected, low=, high=, delta=1e-10)` covers bounded real values
(an acceptance indicator, a clipped score) with Hoeffding's inequality: the allowed
deviation is $(high - low)\sqrt{\ln(2/\delta) / (2N)}$.

Rules:

- **Budget.** The whole judge run must have a false-failure rate below `1e-9`. The
  default `delta=1e-10` allows ten statistical checks per problem. With more, pass
  `delta=1e-9 / number_of_checks`.
- **Independence.** The samples must be independent and identically distributed when
  the solution is correct: one fresh set of uniforms per call, the same target each
  time.
- **Power.** The acceptance half-width is about $\sqrt{2\,p(1-p)\,t/N}$ with
  $t \approx 26$ for $K = 8$. With $N = 4000$ and $p = 0.3$ that is $\pm 0.052$; with
  $N = 20000$, $\pm 0.023$. Choose $N$ so that each mutant's distortion is outside the
  band, and confirm it with `learnai validate`. In a tensor framework a call costs tens of
  microseconds to a millisecond, so $N$ of 2000 to 5000 is the practical range there;
  framework-free problems can afford 20000.
- **Pin down how the randomness is used.** Because the seed is fixed, a given solution
  either always passes or always fails. The bound covers every correct solution only
  if "correct" is unambiguous, so the statement must say exactly which uniform number
  decides what (for example: "`u_accept` decides acceptance; `u_resample` is used for
  inverse-CDF sampling from the residual distribution, and only on rejection").

### 5.4 cost

An instrumented kit object counts the work the solution asks of it, and the test
asserts on the count. Counts are exact, so there is no tolerance.

```python
from learnai.kit import TinyLM

@case("each token goes through the model once", public=True)
def linear_cost(ctx):
    lm = TinyLM(ctx.fw, seed=3)
    prompt = [0, 17, 42, 5]
    ctx.call("greedy_decode", lm, prompt, 12)
    # The prompt once, then every generated token except the last one.
    check.cost(lm.counters, "tokens_processed", at_most=len(prompt) + 12 - 1, unit="tokens")
    check.cost(lm.counters, "forward_calls", at_most=12)
```

`check.cost` reports the counter as a metric (with the limit, so the UI can draw
"114 / 15 tokens") whether or not it passes. Check correctness in a separate case, or
before the cost assertion, so the learner sees "wrong tokens" before "too expensive".
State the cost limit in the statement, with the reasoning, and make the bound tight
enough that the classic waste (re-feeding the prefix, one extra forward pass) exceeds
it.

**Two kit objects with the same counter.** A case reports one value per metric name,
and by default the metric is named after the counter, so the second of two
`check.cost(..., "tokens_processed")` calls replaces the first in what the learner
sees. When a case has a draft and a target model (or two caches, two engines), name
each metric:

```python
draft, target = draft_target_pair(ctx.fw, seed=4)
ctx.call("speculative_decode", draft, target, prompt, 24)
check.cost(draft.counters, "tokens_processed", at_most=60, unit="tokens", metric="draft_tokens")
check.cost(target.counters, "tokens_processed", at_most=20, unit="tokens", metric="target_tokens")
```

The failure message uses the metric's name too
(`target_tokens: 31 tokens exceeds the limit of 20 tokens`).

**Guard the counting object.** A count is evidence only if the solution cannot reset
the counter, read the arrays behind it, or build an uncounted twin. When the count is
what separates the intended solution from one that returns the same numbers, hand the
object over behind the kit's `Guarded` (section 7):

```python
from learnai.kit import CountingKV, Guarded, check_guard, guard

class Store(Guarded, CountingKV):
    label = "store"                     # the argument's name in the statement

store = Store(ctx.fw, keys, values)
guard(ctx)                              # before the solution is called
got = ctx.call("sparse_attend", q, store, idx)
check_guard()                           # right after, before the other checks
check.cost(store.counters, "rows_read", at_most=int((idx >= 0).sum()), unit="rows")
```

Keep the object's test-only methods (`reference`, the `*_ref` oracles) in `forbidden`
as before, so that the message names the method, and say in the statement which
attributes are the interface. Do not use `forbidden_source` for the counters or the
private attributes (section 4.1). The cost problems written before `Guarded` were
converted on 2026-11-21. Eight cost problems have no guard because the solution is
handed nothing that leads to the count: the cost is computed from what it returns
(`dynamic-draft-length`, `reasoning-budget-processor`), or measured by a spy or a
memory reading that the test owns (`mlx-weights-load`, `mlx-preallocated-kv-cache`,
`mlx-dtype-discipline`, `mlx-memory-accounting`, `mlx-quantized-linear`,
`capstone-quantized-runner`). A guard protects the object, not the measurement: when
the solution is also given the data the object would be asked about (the vectors of
a search problem, the public submodules of an `MLXTinyLM`), it can do the arithmetic
by hand, and only a check on the result's cost elsewhere catches that.

### 5.5 speed

The solution is timed against the reference in the same framework, and must be within a
stated factor. There are no absolute thresholds, ever: they mean nothing across
machines.

```python
@case("within 5x of the vectorized reference")
def fast_enough(ctx):
    ctx.check_speed("pairwise_sq_dists", ctx.randn(48, 16), ctx.randn(48, 16), max_ratio=5.0)
```

`ctx.check_speed(name, *args, max_ratio, rounds=7, warmup=2, tokens=None, **kwargs)`
converts the inputs once, runs `warmup` untimed calls of each implementation, then
`rounds` rounds in which both are measured, alternating which goes first, and compares
the medians. Each measurement repeats the call until it lasts about 5 ms, and forces
the result before the clock stops (`fw.sync`: `mx.eval` then `mx.synchronize` for MLX,
`block_until_ready` for JAX, an MPS or CUDA synchronize for torch), so lazy frameworks
are timed doing the work. It reports the `slowdown` metric and fails beyond
`max_ratio`.

**What "forcing the result" means on MLX.** MLX computes nothing until asked, so an
unforced result would be timed building a graph. `fw.sync` evaluates every array it
can reach from the returned value: inside tuples, lists, and dicts, in the parameters
of an `mlx.nn.Module`, and in the attributes of a plain object such as a cache class
the solution defines (a few levels deep). Then it calls `mx.synchronize()`, which
waits for work a solution scheduled with `mx.async_eval` and did not wait for. So a
solution cannot look fast by returning a lazy cache or by leaving work in flight, and
a solution is not charged for anything it does not return or hold. If the value that
matters is not reachable from the return value, return it from your callable
(`ctx.compare_speed`).

**Tokens per second.** Pass `tokens=N` (how many tokens one call produces or
processes) and the solution's throughput is reported as a second metric,
`tokens_per_s` (unit `tok/s`). It is informational: it has no limit and never fails a
case, because an absolute rate means nothing across machines. The verdict is still
decided by `slowdown`. Across the cases of a run the app shows the *lowest*
`tokens_per_s` (the metric is marked higher-is-better) and the highest `slowdown`.

```python
@case("cached generation is within 3x of the reference", time_limit_s=30)
def cached_speed(ctx):
    ctx.check_speed("generate", weights, cfg, prompt, 24, max_ratio=3.0, tokens=24, rounds=3)
```

**Compiled functions, loops, anything that is prepared first:
`ctx.compare_speed`.** When the entry point returns the thing to time (a function
built with `mx.compile`, a step closure), or the timed work is a loop the test drives,
build it outside the timer and hand over two zero-argument callables. Whatever they
return is forced as above.

```python
@case("the compiled step is within 2x of the reference's", requires="mlx-gpu")
def compiled_speed(ctx):
    def loop(make_step):
        lm = MLXTinyLM(seed=5)
        step = make_step(lm)                      # built, not yet traced

        def run():
            token, state = mx.array([3], dtype=mx.int32), lm.init_state(max_len=64)
            for _ in range(32):
                token, state = step(token, state)
            return token, state                   # forced by the judge

        return run

    ctx.compare_speed(loop(ctx.solution.make_step), loop(ctx.reference.make_step),
                      max_ratio=2.0, label="32 compiled steps", tokens=32)
```

How many calls that is: `warmup` calls of each side, one *sizing* call of each (to
decide how often a measurement repeats the callable), then `rounds` measurements. When
one call already lasts 5 ms on both sides, the sizing pair is itself a valid
measurement (after warm-up, garbage collection off, reference first) and counts as
the first round, so a slow callable runs `warmup + rounds` times, not one more. Size
slow speed cases with that number.

The first `warmup` calls absorb tracing, kernel compilation, and allocator growth for
both sides; keep `warmup` at 2 or more for anything compiled. A function compiled with
`mx.compile` retraces when an input's shape or dtype changes, so a callable that feeds
different shapes on every call measures tracing, not the compiled code: that is the
right test for a retracing bug and the wrong one for steady-state speed. On MLX the
default `rounds=7` is fine for work in the millisecond range; lower it (3) when one
call takes hundreds of milliseconds.

Rules:

- The gap you are testing for must be large. A Python loop versus a vectorized
  expression is 50x to 1000x; set `max_ratio` between 3 and 10. Never use the speed
  judge to separate implementations that differ by less than about 3x.
- Size the input so the reference takes at least tens of microseconds (so call overhead
  does not dominate) and the slow mutant still finishes in well under a second per
  call.
- Check correctness in a separate case. A fast wrong answer must fail as wrong.
- The baseline is always the same-framework reference, so `reference/<framework>.py`
  must be the idiomatic fast solution in that framework. Never compare across
  frameworks: JAX runs on the CPU here, MLX and torch may use the GPU.
- For custom timing (a class, a multi-step protocol), use `ctx.compare_speed` (above).
  The lower-level pair `learnai.judge.speed.compare(run_solution, run_reference)`
  and `speed.require(result, max_ratio=..., label=..., ctx=ctx, tokens=None)` remains;
  with it each callable must call `ctx.fw.sync(result)` itself.
- A speed case that only makes sense on a GPU gets `requires=` (section 14), so a
  machine without one skips it instead of timing a CPU fallback.

### 5.6 simclock

A deterministic discrete-event simulation with virtual time, for schedulers, batchers,
rate limiters, and anything else whose behavior is about time. `ctx.simclock()` returns
a `SimClock`: an asyncio event loop whose clock is virtual. When nothing is ready to
run, time jumps to the next timer. A simulated hour takes milliseconds and every run is
identical.

Solutions are written with ordinary asyncio (`asyncio.sleep`, `asyncio.Queue`,
`asyncio.wait_for`, `loop.call_later`, `loop.time()`), which is how real serving code is
written. They must not use the real clock (`time.sleep`, `time.monotonic`) or real I/O;
say so in the statement.

| member | meaning |
|---|---|
| `sim.now()` | Virtual seconds since the start (starts at 0). |
| `sim.call_at(when, fn, *args, priority=0)`, `sim.call_later(delay, fn, *args, priority=0)` | Schedule a plain callback; returns a handle with `cancel()`. |
| `sim.spawn(coro)` | Start a coroutine as a task. An exception escaping it fails `run()`. |
| `sim.spawn_at(when, make, priority=0)` | At time `when`, call `make()` and start the coroutine it returns. |
| `sim.sleep(delay)` | Awaitable; the same as `asyncio.sleep(delay)`. |
| `sim.run(until=None)` | Run until idle (nothing ready, no timer pending), or until virtual time `until`. Tasks blocked forever on a queue do not keep it alive. |
| `sim.run_until_complete(awaitable)` | Run until it finishes and return its result; fails with a deadlock message if the simulation goes idle first. |
| `sim.trace.record(kind, **fields)` | Append an event stamped with the virtual time. |
| `sim.trace.of(kind)`, `.values(kind, field)`, `.times(kind)` | Query the trace. |
| `sim.trace.latencies(start_kind, end_kind, key="id")` | Per-item latency; fails if an item never finished. |
| `percentile(values, q)` | Nearest-rank percentile (an actual observation; no interpolation). |

**Order within one instant.** Events that share a virtual time are where schedulers
differ, so the order is defined:

- Timers due at exactly the same time fire in the order they were scheduled, first
  scheduled first, whoever scheduled them (`sim.call_at`, `sim.call_later`,
  `sim.spawn_at`, `asyncio.sleep`, `loop.call_later`, a `wait_for` timeout).
- `priority=` on `sim.call_at`, `sim.call_later`, and `sim.spawn_at` overrides that:
  among timers due at the same time, a lower priority runs first, and schedule order
  breaks ties within a priority. Timers created by plain asyncio code have priority 0.
  Use it to state a rule such as "a completion is delivered before an arrival at the
  same instant" in one place, instead of relying on the order of your loops.
- One step of the simulation: everything that is ready runs first in, first out
  (`call_soon` callbacks, tasks woken by a future or a queue, tasks just spawned); when
  nothing is ready, time jumps to the next timer and all timers due at that instant
  become ready together, in (priority, schedule) order.
- Two consequences. A coroutine started with `spawn_at` begins after the other timers
  of its instant have fired, because the timer only creates the task. A timer
  scheduled during instant `t` for `t` itself (`call_later(0, ...)`) runs after
  everything already ready, whatever its priority.
- "The same time" means equal floats: `0.1 + 0.2` is not `0.3`. Build times that must
  coincide from the same arithmetic, or count in a unit that is exact (the
  `dynamic-batcher` problem counts integer milliseconds).

If the outcome of a problem depends on the order of simultaneous events, the statement
must specify that order, and a case must exercise a tie.

The kit's `SimModel(sim, base_s=, per_item_s=, max_batch_size=, fn=)` is a fake model
server: `await model(batch)` takes `base_s + per_item_s * len(batch)` virtual seconds,
returns `[fn(item) for item in batch]`, records `batch` and `batch_done` trace events,
and raises if called while busy or with an oversized batch.

```python
from learnai.judge import case, check, percentile
from learnai.kit import SimModel


def simulate(ctx, batcher_class, arrivals, *, max_batch_size, max_wait_s):
    sim = ctx.simclock()
    model = SimModel(sim, base_s=0.004, per_item_s=0.0005, max_batch_size=max_batch_size,
                     fn=lambda item: item * 2)
    batcher = batcher_class(model, max_batch_size=max_batch_size, max_wait_s=max_wait_s)

    async def client(i):
        sim.trace.record("arrive", id=i)
        result = await batcher.submit(i)
        sim.trace.record("done", id=i, result=result)

    for i, when in enumerate(arrivals):
        sim.spawn_at(when, lambda i=i: client(i))
    sim.run()
    return sim


@case("a lone request is dispatched after max_wait_s", public=True)
def lone_request(ctx):
    sim = simulate(ctx, ctx.solution.Batcher, [0.0], max_batch_size=4, max_wait_s=0.050)
    check.equal(sim.trace.values("batch", "size"), [1], label="batch sizes")
    check.allclose(sim.trace.times("batch"), [0.050], label="dispatch times")


@case("every request gets its own result, and tail latency is bounded")
def steady_load(ctx):
    arrivals = [0.002 * i for i in range(100)]
    sim = simulate(ctx, ctx.solution.Batcher, arrivals, max_batch_size=8, max_wait_s=0.005)
    latencies = sim.trace.latencies("arrive", "done")
    check.that(max(sim.trace.values("batch", "size")) <= 8, "a batch exceeded max_batch_size")
    check.at_most("p99_latency", percentile(latencies, 99) * 1000, 15.0, unit="ms")
```

Rules: compare virtual times with `check.allclose` (they are floats); when the policy
is fully specified, also run the oracle through the same simulation and compare the
traces exactly; derive latency limits from the reference's result with clear headroom
over it and clear distance from every mutant's.

### 5.7 service

The solution defines an ASGI application named `app` (`entry = ["app"]`). The judge
serves it with uvicorn on `127.0.0.1` and an ephemeral port, inside the harness process,
and the test talks to it with an `httpx.Client`.

```python
from learnai.judge import case, check, read_sse


@case("streams one event per token, then [DONE]", public=True)
def streams_tokens(ctx):
    with ctx.serve() as client:
        events = read_sse(client, "GET", "/stream", params={"prompt": "the quick brown fox"},
                          stop_at="[DONE]")
    check.equal([e.data for e in events[-1:]], ["[DONE]"], label="last event")
    check.equal([e.json()["token"] for e in events[:-1]], ["the", "quick", "brown", "fox"],
                label="tokens")
```

- `ctx.serve(app=None, timeout_s=5.0)` is a context manager yielding the client. The
  default app is `ctx.solution.app`; pass another ASGI app to serve something the
  solution builds (`ctx.serve(ctx.solution.make_app(source))`). The server always shuts
  down when the block exits, also on failure or timeout. `timeout_s` bounds every
  connect and read, so a stalled endpoint fails the case instead of hanging it.
- `read_sse(client, method, url, *, max_events=None, stop_at=None, **request)` makes
  the request, checks for status 200 and the `text/event-stream` content type, and
  returns the events as `SSEEvent(data, event="message", id=None, retry=None)`; `.json()`
  parses the data. It stops at end of stream, after `max_events`, or at the first event
  whose data equals `stop_at` (included).
- `iter_sse(response)` yields events as they arrive, inside
  `with client.stream("GET", url) as response:`. Use it to prove that the endpoint
  really streams: have the test control a gate (a `threading.Event` the app's token
  source waits on), read the first event, open the gate, read the rest. Do not assert on
  wall-clock arrival times.
- `parse_sse(lines)` parses event-stream text you obtained some other way.
- Anything else is plain httpx: `client.get`, `client.post(json=...)`, status codes,
  headers.

Starlette is available to solutions and references; so is any raw ASGI callable. Keep
service cases few (starting a server costs about 0.1 s each) and reuse one
`with ctx.serve()` block for several requests.

> [!WARNING]
> **A send after a disconnect does not fail.** Once the client has gone, uvicorn's
> `send` returns without doing anything: it does not raise (in uvicorn 0.54, both HTTP
> implementations start `send` with `if self.disconnected: return`). The ASGI
> specification only says a server *should* raise `OSError` there. So a streaming
> solution cannot detect a closed connection by sending; it has to watch `receive()`,
> which returns `{"type": "http.disconnect"}` when the client is gone (after the
> request body has been read, the next `await receive()` blocks until then). If a
> problem requires stopping on disconnect, say this in the statement, write the
> reference with a `receive()` watcher next to the sender, and test it by closing the
> response early and observing a side effect (the token source was closed, no further
> token was requested), never by expecting an exception. `problems/serving/
> sse-token-stream` does all three.

## 6. The framework adapter

`ctx.fw` is the adapter for the solution's framework and `ctx.oracle_fw` for the
oracle's. You rarely need them directly, because `ctx.call` converts for you. Use them
when a test builds tensors itself or hands a kit object to the solution.

| member | meaning |
|---|---|
| `fw.name` | `"torch"`, `"jax"`, `"mlx"`, or `"python"`. |
| `fw.xp` | The array namespace: `torch`, `jax.numpy`, `mlx.core`; `None` for python. |
| `fw.tensor(array, dtype=None)` | numpy (or anything `np.asarray` accepts) to a tensor. |
| `fw.numpy(tensor)` | One tensor to numpy. |
| `fw.is_tensor(value)` | Whether `value` is this framework's tensor type. |
| `fw.to_framework(obj)` | Convert every array inside lists, tuples, and dicts. |
| `fw.to_numpy(obj)` | Convert every tensor inside lists, tuples, and dicts; rejects foreign tensor types. |
| `fw.sync(*values)` | Block until the tensors are computed. Required before stopping a timer. |

Conversion rules:

- Floating arrays become float32 unless `dtype` is given.
- Integer arrays keep their dtype. int8, uint8, int16, uint16, int32, and uint32 arrive
  unchanged in every framework, without `dtype=`. This is what `ctx.call`,
  `ctx.expected`, and `ctx.baseline` do too, since they convert with `fw.tensor`.
- 64-bit integers are the exception, because they are numpy's default and usually mean
  "an integer", not "64 bits": int64 becomes the framework's default integer (int64 in
  torch, int32 in JAX and MLX, where 64-bit integers are unavailable or unidiomatic),
  and uint64 stays uint64 in torch and becomes uint32 in JAX and MLX. To force a
  width, pass `dtype=` (`fw.tensor(ids, dtype=np.int64)` gives MLX an int64 tensor).
- Bool stays bool. Exact comparisons (`check.equal`) ignore integer width, so a result
  may come back as int32 from one framework and int64 from another.
- A consequence for torch: indices for `gather`, `scatter`, and `take_along_dim` must be
  int64 there, so pass index arrays as numpy's default int64, not as `np.int32`.
- The `python` adapter turns arrays into nested lists and numpy scalars into Python
  scalars; results come back unchanged.
- Inputs are copied. A solution that works in place cannot corrupt the array the test
  passes to the oracle next.

Torch solutions receive CPU tensors. MLX uses its default device (the GPU on Apple
silicon). JAX runs on the CPU.

### Sorting and ties

When a specification breaks ties by index ("equal probabilities are ranked by token
index, lowest first"), each reference needs a stable sort, and each framework spells it
differently:

- torch: `torch.sort(x, stable=True)` and `torch.argsort(x, stable=True)`. The default
  is not stable.
- JAX: `jnp.sort` and `jnp.argsort` are stable by default (`stable=True`).
- MLX: `mx.sort` and `mx.argsort` are stable, and you may rely on it. It is documented
  behavior in MLX 0.32.4, the version this project uses ("The sort is stable, meaning
  equal elements preserve their relative order. `NaN` values are placed at the end.",
  in the docstring and on the `mlx.core.argsort` page of the official documentation);
  the CPU backend implements it with `std::stable_sort` plus an explicit index
  tie-break (`mlx/backend/cpu/sort.cpp`), and the GPU agrees with numpy's stable sort
  on tie-heavy inputs. The guarantee belongs to that version: re-read the docstring
  (`print(mx.argsort.__doc__)`) when the MLX pin changes. There is no `stable=` or
  `descending=` argument: sort the negated values for a descending order that still
  ranks ties by lowest index, as `problems/sampling/top-p-filtering` does.

Whatever the framework, a case must contain ties when the tie rule matters, and a
mutant must break them the other way.

## 7. The kit

Kit objects are what problems hand to solutions, the way LeetCode hands you a
`ListNode`. They live in `learnai.kit`, are small, and are documented for learners,
who are told to read them. Link to the relevant one in your statement.

### TinyLM

A tiny seeded decoder-only transformer (64-token vocabulary, 2 layers, 4 heads, width
32, context 128 by default). Its arithmetic is numpy, so logits are bit-identical
whatever framework the solution uses; at its boundary it accepts and returns the
solution's tensors.

```python
lm = TinyLM(ctx.fw, seed=3)             # also: vocab_size=, d_model=, n_layers=, n_heads=, max_context=
logits, cache = lm(tokens)              # no cache: tokens are the whole prefix
logits, cache = lm(next_tokens, cache)  # with cache: tokens continue the prefix
```

- `tokens`: a list of ints or an integer tensor, shape `(T,)` or `(B, T)`, `T >= 1`.
- `logits`: float32 tensor of the framework, one row per token passed in this call:
  `(T, vocab_size)` or `(B, T, vocab_size)`. `logits[-1]` scores the next token.
- `cache`: an immutable `KVCache` (`cache.length` is the number of tokens it covers).
  The call never modifies the cache it was given; it returns a new one. State in, state
  out, so it is usable from JAX, and an old cache can be reused to branch.
- `lm.counters.forward_calls`, `lm.counters.tokens_processed` (batch size times length,
  summed over calls); `lm.counters.snapshot()`, `.reset()`.
- `lm.vocab_size`, `lm.max_context`, `lm.bos_id` (0), `lm.eos_id` (1).
- Invalid input raises `ValueError` (token out of range, empty call, context overflow,
  batch size not matching the cache).

Authoring notes:

- Pass the model as an argument (`ctx.call("greedy_decode", lm, prompt, 12)`). Keep the
  instance: its counters are what you assert on. `ctx.expected` runs the oracle on a
  twin with the same weights and separate counters.
- Feeding a prefix in one call and token by token through the cache gives logits that
  agree to about `1e-5`, not bit for bit. The default models keep the top two logits
  further apart than that for typical prompts, but when a problem depends on exact
  token agreement between two valid call patterns, confirm it for your seed and prompt
  by running the reference in all frameworks, and prefer comparing tokens over
  comparing logits.
- TinyLM cannot be traced by `jax.jit` or `mx.compile` (it runs numpy). Problems that
  use it are about decoding logic, not about compiling the model.
- Different seeds give different models; use two seeds across cases so that a solution
  cannot special-case one.

### KVCache.truncate and KVCache.select

Two operations on the cache `TinyLM` returns, for loops that roll back or branch. Both
return a new cache and leave the original untouched.

```python
_, cache = lm([5, 6, 7, 8])
short = cache.truncate(2)              # as if only [5, 6] had been fed
logits, _ = lm([9], short)             # continues [5, 6, 9]

beams = cache.select([0, 0, 0])        # one prompt expanded to three beams
logits, cache = lm(next_tokens, beams) # next_tokens has shape (3, 1)
cache = cache.select([2, 0, 0])        # reorder (and repeat) rows after a beam reshuffle
```

- `cache.truncate(n)`: `0 <= n <= cache.length`, else `ValueError`; a non-int is a
  `TypeError`. `truncate(cache.length)` is an equal cache. `truncate(0)` is an empty
  cache, and continuing from it equals starting without a cache; a model call still
  needs at least one token, which is why `prefix-cache-reuse` keeps one prompt token
  to feed.
- `cache.select(rows)`: `rows` is a non-empty 1-D list or integer tensor of row
  indices in `[0, cache.batch_size)`; rows may repeat or be dropped. Negative indices,
  an empty list, and anything not 1-D raise `ValueError`. The result always counts as
  batched, also when the source came from 1-D tokens (one row), so feed it tokens of
  shape `(len(rows), T)`.
- `cache.batch_size` is the number of rows.
- Logits after `truncate` or `select` agree with the direct computation to about
  `1e-5`, like any two call patterns of `TinyLM`: compare tokens, not logits.

### draft_target_pair

Two `TinyLM`s for speculative decoding: a target, and a draft that agrees with it at a
chosen rate.

```python
from learnai.kit import draft_target_pair

draft, target = draft_target_pair(ctx.fw, seed=3, agreement=0.8)   # also TinyLM's size arguments
ctx.call("speculative_greedy", target, draft, prompt, 4, 24, target.eos_id)
check.cost(target.counters, "forward_calls", at_most=...)
check.cost(draft.counters, "tokens_processed", at_most=...)
```

- `target` is exactly `TinyLM(fw, seed)`. `draft` has the same shape and vocabulary and
  the target's weights plus seeded noise, tuned so that along the target's greedy
  continuations the draft's greedy token equals the target's at about `agreement`.
- `agreement` is in `(0, 1]`; `1.0` gives a draft with identical weights. The rate is a
  statistical property: over a few hundred positions of prompts the pair was not tuned
  on it lands within about 0.05 of the request, but one short prompt can be far off.
  So derive cost limits from the reference's measured counters on your fixture, not
  from the nominal rate, and pick `agreement=0.9` or more for a "high agreement" case.
- Each model has its own counters, both zero when returned. Building a pair takes about
  0.1 s; build it once per case.

### ops

`learnai.kit.ops` holds reference implementations of concepts from earlier problems,
so a later problem can build on them without asking for them again. One signature in
every framework: the call is routed by the type of the first argument (torch, JAX,
MLX, or numpy), and each framework has a native implementation written from elementary
operations, so `ops` keeps working in problems whose `forbidden` list rules out the
built-in `softmax`, `topk`, or attention kernel. Solutions, references, and tests may
all use it.

```python
from learnai.kit import ops

h = ops.rms_norm(x, weight, eps=1e-5)
q, k = ops.rope(q, positions), ops.rope(k, positions)
y = ops.attention(q, k, v, mask)
logits = ops.min_p_filter(ops.top_p_filter(ops.top_k_filter(logits, 40), 0.95), 0.05)
probs = ops.softmax(logits)
```

| function | contract |
|---|---|
| `softmax(x, axis=-1)`, `log_softmax(x, axis=-1)` | Stable; `-inf` entries get probability 0. An all-`-inf` slice gives NaN. |
| `rms_norm(x, weight, eps=1e-5)` | `x / sqrt(mean(x**2, -1) + eps) * weight`. The plain formula: it does not have the overflow protection the `rmsnorm` problem asks for. |
| `rope(x, positions, base=10000.0)` | `x` is `(..., T, D)`; pairs `(2i, 2i+1)` rotated by `positions[t] * base**(-2i/D)`, as in the `rope` problem. `positions` is `(T,)`, or `(B, T)` for `x` of shape `(B, H, T, D)` (one row of positions per batch row). |
| `attention(q, k, v, mask=None)` | `softmax(q k^T / sqrt(d) + mask) v`. `mask` broadcasts to `(..., T, S)`: boolean (True = may attend) or additive float (0 or `-inf`). A query that may attend to nothing gets a zero output, not NaN. Grouped-query attention is built in: with `q` `(B, H, T, d)` and `k`, `v` `(B, H_kv, S, d)`, query head `h` uses KV head `h // (H // H_kv)`. |
| `swiglu(x, w_gate, w_up, w_down)` | `(silu(x W_gate^T) * (x W_up^T)) W_down^T`, weights `(out, in)`. |
| `top_k_filter(logits, k)` | Keeps the `k` largest per row, ties to the lowest index; `k <= 0` or `k >= V` is a no-op. |
| `top_p_filter(logits, p)` | Keeps a token while the mass ranked strictly before it is below `p` (ties: lowest index first); the top token always survives. |
| `min_p_filter(logits, min_p)` | Keeps tokens with probability at least `min_p` times the largest. |
| `softplus(x)` | `log(1 + exp(x))` as `max(x, 0) + log1p(exp(-abs(x)))`: no overflow, and accurate to 2e-7 relatively where the result is tiny (large negative `x`). See the note on MLX below. |
| `l2norm(x, eps=1e-6)` | `x * rsqrt(sum(x**2, -1) + eps)`, the `l2norm` of Qwen3.5's delta layers (eps inside the root). |
| `causal_conv1d(x, w, state=None)` | Depthwise causal convolution of `x` `(..., T, C)`; returns `(y, state)`. `w` is `(C, K)` or the checkpoint's `(C, 1, K)`; no bias, no flip: `y[t, c] = sum_j w[c, j] * padded[t + j, c]` with `padded` the state followed by `x`, so the **last** tap multiplies the current input. `state` is the `K - 1` inputs before `x`, `(..., K - 1, C)` (`None`: zeros); the returned state is the last `K - 1` rows of `padded`, also when `T < K - 1`. |
| `partial_rope(x, positions, base=10000.0, factor=1.0)` | The `partial_rope` of the `partial-rope` problem: the first `r = int(D * factor)` channels are a complete RoPE of width `r` in the **half-split** layout (pair `i` is channels `(i, i + r/2)`, frequency `base**(-2i / r)`), the rest pass through unchanged. `r` odd raises `ValueError`. `positions` as for `rope`. Not `rope` with a factor: `rope` pairs adjacent channels, so `partial_rope(x, p, base, 1.0)` is a permutation of `rope(x, p, base)`. There is no kit version of the proportional (Gemma 4) form. |
| `gated_delta_layer(x, params, state=None, n_key_heads=None, norm_eps=1e-6)` | The Gated DeltaNet mixer of Qwen3.5 on `x` `(..., T, D)`; returns `(out, (conv_state, S))`. See [GDNConfig](#gdnconfig-gdn_params-and-the-delta-layer-oracles). |
| `eb_accept(logits, entropy_bound)` | DiffusionGemma's entropy-bound acceptance for logits `(..., L, V)`: a boolean `(..., L)` mask. Positions are ranked by entropy in nats, ascending, ties to the lowest index; a position is accepted while the entropies ranked strictly before it sum to at most the bound. The lowest-entropy position is always accepted; `-inf` logits contribute no entropy. Pass the logits the tokens are sampled from (after the temperature). |

The three filters work on the last axis and return **logits**: removed entries become
`-inf`, kept entries are unchanged, already-masked entries stay masked. That makes
them chain, and `ops.softmax` of the result is the renormalized distribution. Note the
difference from the `top-p-filtering` problem, whose `top_p_filter` returns
probabilities: `ops.softmax(ops.top_p_filter(logits, p))` equals that problem's output.

If your problem *teaches* one of these, forbid the kit version too, by saying so in the
statement and adding a mutant that calls it; `forbidden` entries such as
`learnai.kit.ops.softmax` resolve like any other name. For the milestone 6 batches that
means: `gated-delta-layer` forbids `learnai.kit.ops.gated_delta_layer` (and
`gdn_layer_ref`), `gated-conv-block` forbids `learnai.kit.ops.causal_conv1d`,
`entropy-bound-denoise-step` forbids `learnai.kit.ops.eb_accept`, and
`hybrid-decoder-forward` and `block-diffusion-decode` are built on them.

**`softplus` in MLX: `mlx.nn.softplus` is inexact for negative inputs** (MLX 0.32.4,
measured 2026-11-21). It evaluates `log(1 + exp(x))` in float32, so the result carries
an absolute error of up to 6e-8, which is a large *relative* error where the result is
small: 4.5427e-5 against the exact 4.5399e-5 at `x = -10` (6e-4), up to 0.8% for `x` in
`[-12, -8]`, and up to 39% near `x = -16`. `mx.logaddexp(x, 0)`, `torch`'s and JAX's
`softplus`, and `ops.softplus` in every framework are accurate to 2e-7 relatively over
`[-30, 10]`. What follows for a problem whose learners will call a built-in softplus:

- In `g = exp(-exp(A_log) * softplus(a + dt_bias))` (the delta rule) and in
  `h * exp(-dt * A)` (Mamba-2) the error is harmless: it is an absolute 6e-8 on a
  number that is then compared near 1. No constraint is needed.
- Where a small softplus is the *output* or is multiplied into it (`sqrt(softplus(.))`
  of the `sqrtsoftplus` router, `dt * x B^T` in `mamba2-ssd-step`), keep the inputs of
  softplus above about -5 in the cases (error below 1e-5 relatively), or give the
  comparison an `atol` that covers 6e-8 times the factor it is multiplied by. The
  `sigmoid-router-shared-expert` problem keeps router logits within 5 in magnitude for
  this reason.
- A reference solution may call `ops.softplus`; say in the statement that
  `mx.logaddexp(x, 0)` is the accurate built-in if the case needs one.

### DecoderConfig, decoder_params, decoder_forward_ref

Weights for a small modern decoder (RMSNorm pre-norm blocks, RoPE, grouped-query
attention, SwiGLU, tied head), named like a Hugging Face checkpoint, plus the numpy
oracle. Use them when the solution must run the forward pass itself.

```python
from learnai.kit import DecoderConfig, decoder_forward_ref, decoder_params, layer_params

cfg = DecoderConfig(n_layers=2)                  # vocab_size=64, d_model=32, n_heads=4,
params = decoder_params(cfg, seed=1)             #   n_kv_heads=2, d_mlp=48, rope_base=1e4,
tokens = ctx.randint(0, cfg.vocab_size, (2, 6))  #   norm_eps=1e-5, tie_embeddings=True
got = ctx.call("forward", params, cfg, tokens)   # the dict's arrays become tensors
check.allclose(got, decoder_forward_ref(cfg, params, tokens), rtol=1e-3, atol=1e-4)
```

- `decoder_params(cfg, seed)` returns `dict[str, np.ndarray]` (float32, checkpoint
  order). Linear weights are `(out, in)`, no biases:

  | key | shape |
  |---|---|
  | `model.embed_tokens.weight` | `(vocab_size, d_model)` |
  | `model.layers.{i}.input_layernorm.weight` | `(d_model,)` |
  | `model.layers.{i}.self_attn.q_proj.weight` | `(n_heads * head_dim, d_model)` |
  | `model.layers.{i}.self_attn.k_proj.weight`, `v_proj.weight` | `(n_kv_heads * head_dim, d_model)` |
  | `model.layers.{i}.self_attn.o_proj.weight` | `(d_model, n_heads * head_dim)` |
  | `model.layers.{i}.post_attention_layernorm.weight` | `(d_model,)` |
  | `model.layers.{i}.mlp.gate_proj.weight`, `up_proj.weight` | `(d_mlp, d_model)` |
  | `model.layers.{i}.mlp.down_proj.weight` | `(d_model, d_mlp)` |
  | `model.norm.weight` | `(d_model,)` |
  | `lm_head.weight` | `(vocab_size, d_model)`; present only with `tie_embeddings=False` |

  With tied embeddings there is no `lm_head.weight` key at all, as in a real tied
  checkpoint. Norm gains are drawn from `[0.5, 1.5]`, not set to 1, so a missing or
  swapped norm changes the output. `cfg.head_dim` is `d_model // n_heads` (must be even).
- `decoder_forward_ref(cfg, params, tokens)`: `tokens` `(T,)` or `(B, T)` to float32
  logits `(T, V)` or `(B, T, V)`; causal, positions from 0. The arithmetic is exactly
  the kit's `ops`: RoPE on queries and keys only, adjacent pairs; scores divided by
  `sqrt(head_dim)`; logits have a standard deviation of about 2.
- `layer_params(params, i)` strips the `model.layers.{i}.` prefix (works on numpy and
  on tensor dicts), for single-block problems.
- `decoder_block_ref(cfg, block, x, mask=None, positions=None)` is one block of the
  oracle: `x` `(B, T, d_model)` in and out, `mask` as in `ops.attention` (`None` means
  no mask, not a causal one), `positions` as in `ops.rope` (default `0..T-1`). For
  `padded-batch-invariance`, compare each row's last-token logits with
  `decoder_forward_ref` on the unpadded sequence.
- `cfg` is a frozen dataclass and passes through `ctx.call` unchanged. Use `rtol=1e-3`:
  the forward pass is a deep stack (section 9).

### ToyVocab

The byte strings behind `TinyLM`'s token ids, for detokenizing, stop strings, and
constrained decoding.

```python
from learnai.kit import ToyVocab

vocab = ToyVocab(size=lm.vocab_size, seed=0)      # size 64 to 282
ids = vocab.encode('{"a": true}')                 # byte-pair merging; never a special id
vocab.decode(ids)                                 # '{"a": true}'
vocab.bytes(ids[0])                               # b'{"'
euro = [vocab.id_of(b"\xe2\x82"), vocab.id_of(b"\xac")]     # "€" split over two tokens
check.cost(vocab.counters, "bytes_decoded", at_most=...)
```

- Ids 0 to 4 are special tokens, fixed: `vocab.specials == {"<|bos|>": 0, "<|eos|>": 1,
  "<|pad|>": 2, "<|im_start|>": 3, "<|im_end|>": 4}` (`vocab.bos_id`, `eos_id`,
  `pad_id`; 0 and 1 match `TinyLM`). **Their bytes are `b""`**, and `decode` skips them.
- The other ids are shuffled by `seed`. At size 64 they are 38 single bytes (digits,
  `{}[]:,"`, space, `-`, `.`, the letters `aeflnrstu`, and the nine bytes of `é`, `€`,
  `😀`) and 21 merged pieces: `tr tru true fa fal fals false nu nul null`, `":`, `": `,
  `, `, `{"`, `"}`, `é`, `€`, `😀`, and three pieces that are only part of a character
  (`E2 82`, `F0 9F`, `98 80`). Larger sizes add single bytes, printable ASCII first;
  at 282 every byte has a token.
- `vocab.bytes(id)` and `vocab.decode(ids)` (UTF-8 with `errors="replace"`, so a
  cut-off character is U+FFFD) are **counted**: `counters.lookups` per id and
  `counters.bytes_decoded` per byte handed out. Re-decoding a growing sequence at every
  step is quadratic in `bytes_decoded`. An id outside `[0, size)` raises `ValueError`.
- Not counted, for tests: `vocab.pieces` (a tuple of every token's bytes),
  `vocab.id_of(piece)` (`KeyError` if absent; also accepts a special's string),
  `vocab.is_special(id)`, `vocab.merges` (ranked `(left, right)` byte pairs), and
  `vocab.encode(text)`. `encode` follows the `bpe-encode` rule (lowest-rank adjacent
  pair first, all its occurrences left to right) and raises `ValueError` for a byte
  without a token, so below size 282 write test text from the pieces above. It treats
  special strings as ordinary text: it is the `encode_ordinary` of
  `special-tokens-split` (use size 282 there, so that `<|im_start|>` is encodable).
- Token ids are plain ints in every framework. `ctx.expected` gets a twin with fresh
  counters.

### CharDFA

A deterministic automaton over **bytes** for constrained generation.

```python
from learnai.kit import DEAD, CharDFA

dfa = CharDFA.from_regex(r'\{"[a-z]+": (true|false|-?\d+)\}')
state = dfa.step(dfa.start, ord("{"))        # next state, or DEAD (-1); one counted step
state = dfa.walk(state, b'"ab')              # several bytes; stops when the walk dies
dfa.is_accepting(state)
check.cost(dfa.counters, "steps", exactly=0)

by_hand = CharDFA({0: {"a": 1}, 1: {"b": 1}}, start=0, accepting=[1])    # a b*
```

- States are non-negative ints; `DEAD` is -1 and stepping from it stays `DEAD`.
  `dfa.start`, `dfa.accepting` (frozenset), `dfa.states` (sorted tuple), `dfa.n_states`.
- `dfa.step(state, byte)` counts one in `dfa.counters.steps`; `byte` is an int in
  `[0, 256)`. An unknown state or a bad byte raises `ValueError`. `dfa.walk(state,
  data)` counts one step per byte consumed, including the byte that kills the walk and
  none after it. The transition table is private, so `step` is the only way a solution
  learns it; that is what makes a step count meaningful.
- Oracles for tests, **not** counted: `dfa.accepts(data)`, `dfa.accepts_prefix(data)`
  (can `data` be extended to a string of the language?), `dfa.is_live(state)`. They
  take `bytes` or `str`. Tell learners in the statement not to call them.
- `CharDFA.from_regex(pattern)` returns the minimal DFA, **trimmed**: no state from
  which acceptance is unreachable. So for a regex DFA "the walk is not `DEAD`" means
  exactly "this is a prefix of a string in the language", which is the rule
  `dfa-token-mask` states. States are numbered breadth-first from the start (0),
  following bytes in increasing order, so equal languages give identical automata.
- A hand-built DFA may have trap states (alive, but nothing accepts from there);
  `dfa.trim()` removes them, keeping state numbers.
- Regex subset (the whole string must match): literals; `.`; classes `[abc]`,
  `[a-z0-9_]`, `[^"]` (ASCII ranges only); `\d \w \s` (ASCII) and, outside classes,
  `\D \W \S`; `\n \t \r \f \v \0 \xHH`; escaped punctuation; concatenation, `|`,
  `( )`, `(?: )`; `* + ?`, `{m}`, `{m,}`, `{m,n}` with bounds up to 100. Non-ASCII
  literals match their UTF-8 bytes, and `.`, negated classes, and `\D \W \S` match one
  whole character (ASCII or a well-formed multi-byte sequence), never half of one: a
  token that ends inside a character leaves the walk alive in an intermediate state,
  and a stray continuation byte kills it. Everything else raises `ValueError`:
  anchors, lazy, possessive, or stacked quantifiers, backreferences, lookaround, named
  groups, flags, Unicode classes. The metacharacters `. ^ $ * + ? { } [ ] ( ) | \`
  must be escaped to be literal. On valid text the subset agrees with
  `re.fullmatch(pattern, text, re.ASCII)`.

### TokenConstraint and DFAConstraint

`TokenConstraint` is the protocol a constrained decoding loop is written against;
`DFAConstraint(dfa, vocab)` is the reference implementation.

```python
from learnai.kit import CharDFA, DFAConstraint, ToyVocab

vocab = ToyVocab(lm.vocab_size, seed=1)
constraint = DFAConstraint(CharDFA.from_regex(r"(true|false)"), vocab)
tokens = ctx.call("constrained_greedy", lm, prompt, constraint, 16, lm.eos_id)

state = constraint.start()
constraint.allowed(state)            # sorted token ids that may come next
state = constraint.advance(state, token)     # ValueError if the token was not allowed
constraint.is_accepting(state)       # may the output end here?
```

- States are hashable and immutable; `advance` returns a new one.
- `allowed` never contains EOS or any other special token (their bytes are empty). The
  loop permits EOS exactly when `is_accepting(state)`. An empty list in a
  non-accepting state is possible only when the vocabulary lacks the bytes the
  language needs next (at size 64, a regex that requires the letter `z`): write
  patterns from the vocabulary's pieces, or use a larger vocabulary.
- A token is allowed when its bytes, walked from the state, end where acceptance is
  still reachable, so it is correct for hand-built DFAs with trap states too.
- `constraint.counters` has `allowed_calls` and `advance_calls` (a loop that forgets to
  advance shows up here). It answers from its own index and never touches
  `dfa.counters` or `vocab.counters`.
- `isinstance(obj, TokenConstraint)` checks that the four methods exist. Write your own
  constraint class in `tests.py` when a case needs another language.

### SimEngine and CostModel

The simulated inference engine behind the scheduling problems: the forward pass of a
continuous-batching server with the model taken out. The scheduler under test decides
what goes into each step; the engine keeps the bookkeeping, the virtual clock, the
trace, and the rules.

```python
from learnai.kit import CostModel, SimEngine, workloads

def simulate(ctx, run, requests, *, max_running):
    sim = ctx.simclock()
    engine = SimEngine(sim, CostModel(base_s=2**-6, prefill_s_per_token=2**-12,
                                      decode_s_per_seq=2**-9), max_seqs=max_running)
    queue = asyncio.Queue()
    workloads.drive(sim, requests, queue.put_nowait)      # records `arrive` events
    sim.spawn(run(engine, queue, max_running))
    sim.run()
    check.that(not engine.violations, "the scheduler broke the engine's protocol",
               "\n".join(f"{v.code}: {v}" for v in engine.violations))
    return sim, engine

# inside the scheduler:
engine.admit(request)                                     # no time passes
result = await engine.step(prefill=[(request.id, request.prompt_tokens)], decode=[3, 5])
result.tokens, result.finished                            # ids, in report order
```

`SimEngine(sim, cost, *, max_batch_tokens=None, max_seqs=None, kv_blocks=None,
block_tokens=16, name=None)`. With a `name`, every trace event of the engine carries
`engine=name` (several engines on one clock: `SimCluster`).

**The life of a sequence.**

1. `engine.admit(request)` makes it *live* (anything with `id`, `prompt_tokens`,
   `output_tokens`; normally a `SimRequest`). It may be called while a step runs.
2. `step(prefill=[(id, n)])` processes `n` more prompt tokens, all at once or in chunks
   over several steps.
3. The step that completes the prefill produces the **first output token**.
4. From the next step on the sequence goes in `decode`, one token per step. It cannot
   be decoded in the step that completes its prefill.
5. After `output_tokens` tokens it finishes at the end of that step: it is no longer
   live and its memory is free, without a call. Its id may be admitted again.

**Sequences that do not start from nothing** (milestone 5). `engine.admit(request,
cached_tokens=c)` with `0 <= c < prompt_tokens`: the first `c` prompt tokens are a
prefix-cache hit; the sequence holds their blocks at once and `prompt_tokens - c`
tokens remain to prefill (at least one always is: its forward pass produces the first
token). `engine.admit(request, resume_from=g)` with `1 <= g < output_tokens`: the
sequence was prefilled and produced `g` tokens elsewhere; it arrives holding
`prompt_tokens + g - 1` tokens of KV, needs no prefill, emits no `first_token`, and
its next decode produces token index `g`. Out of range (or both at once) is
`bad_request`; blocks the pool cannot hold are `out_of_blocks`, at `admit`. The
`admit` event then has `cached` or `resumed`.

**Time.** A step takes `cost.step_s(prefill_tokens, decode_seqs)` =
`base_s + prefill_s_per_token * prefill_tokens + decode_s_per_seq * len(decode)`
virtual seconds. `CostModel` has no defaults: state the three numbers in the statement.
Tokens appear when the step ends. One step runs at a time; it runs to its end even if
the task awaiting it is canceled. The step's completion timer has priority
`STEP_DONE_PRIORITY` (10), so at a shared instant every ordinary timer (an arrival, a
deadline, a `sleep`) runs first: a request arriving exactly when a step ends is already
queued when the scheduler resumes. State that rule in the statement and exercise it
(`workloads.bursty`, or `quantum_s=`). Use costs that are exact in binary (`2**-6`,
0.125) if a test compares times with `==`; otherwise compare with `check.allclose`, or
compare the trace with the oracle's from the same simulation.

**Limits.** `max_batch_tokens` bounds prompt tokens plus decoded sequences in one
step. `max_seqs` bounds live sequences. `kv_blocks` is a pool of KV blocks of
`block_tokens` tokens: a sequence holds `ceil(kv_tokens / block_tokens)` blocks, and
`kv_tokens` grows by `n` for a prefill of `n` and by 1 for a decode, when the step
*starts*. After `g` produced tokens a sequence holds `prompt_tokens + g - 1`.

**Preemption and cancellation.** `engine.preempt(id)` frees the blocks but keeps the
sequence live and keeps its produced tokens; its whole context, `prompt_tokens +
generated` tokens, must be prefilled again, and the step that completes that produces
its *next* token (indices continue; `counters.recomputed_tokens` counts the repeated
work). `engine.release(id)` drops a sequence altogether. Either may be called during a
step: the sequence then gets nothing from that step, whose duration is unchanged, so
no token is recorded for a request after its cancellation.

**State a scheduler may read:** `engine.busy`, `engine.live` (ids in admission order),
`engine.free_blocks` (None if unlimited), `engine.remaining_prefill(id)`,
`engine.generated(id)`, `engine.kv_tokens(id)`, `engine.blocks(id)`,
`engine.block_tokens`, `engine.max_batch_tokens`.

**Violations.** Anything the engine cannot do raises `ProtocolViolation` (a
`RuntimeError`) with a `.code`. `step` checks in this order and raises for the first
failure:

| code | when |
|---|---|
| `busy` | another step is running |
| `empty_step` | nothing to prefill and nothing to decode |
| `duplicate_sequence` | an id listed twice in the step, in either list or both (so also "decoded in the step that completes its prefill") |
| `unknown_sequence` | a prefill id that is not live: never admitted, finished, or released |
| `bad_prefill` | a chunk below 1 or above the remaining prefill tokens |
| `unknown_sequence` | a decode id that is not live |
| `prefill_incomplete` | a decode id that still needs prefill |
| `budget_exceeded` | more than `max_batch_tokens` |
| `out_of_blocks` | the step needs more KV blocks than the pool has; the class is `OutOfBlocks`, a subclass |

`admit` raises `duplicate_sequence` (the id is live), `bad_request` (a token count that
is not an int >= 1), or `too_many_sequences`; `preempt` and `release` raise
`unknown_sequence`. A refused call takes no time and changes nothing (no partial
allocation). Every violation is appended to `engine.violations` and recorded in the
trace, so a solution that catches the exception is still caught by the test: assert
`engine.violations == []` in every case, as `simulate` does above. The read-only
queries raise `unknown_sequence` too, without recording anything.

**Trace events** (in `sim.trace`, section 5.6):

| kind | fields | recorded |
|---|---|---|
| `admit` | `id` | at `admit()` |
| `step` | `index`, `prefill` (tuple of `(id, n)`), `decode` (tuple of ids), `tokens`, `duration` | when the step starts |
| `first_token` | `id`, `step` | at the end of the step, just before the sequence's first `token` |
| `token` | `id`, `index` (0-based within the sequence), `step` | at the end of the step |
| `finish` | `id`, `tokens`, `step` | right after the sequence's last `token` |
| `step_done` | `index` | after the step's token events |
| `preempt`, `release` | `id`, `generated` | at the call |
| `violation` | `code`, `message` | at the refused call |

At the end of a step, sequences are reported in the order of the `prefill` list, then
the `decode` list; `result.tokens` and `result.finished` use the same order.
`workloads.drive` adds `arrive` events, so
`sim.trace.latencies("arrive", "first_token")` is TTFT and
`sim.trace.latencies("arrive", "finish")` is end-to-end latency.

**Counters:** `steps`, `prefill_tokens`, `decode_tokens`, `output_tokens`,
`recomputed_tokens`, `preemptions`; `engine.busy_s` is the total step time.

`python/tests/test_kit_simengine.py` ends with a complete continuous-batching
scheduler, a static-batching baseline, and the checks a problem would make on them
(exact trace, token accounting, latency comparison). Start from it.

### SimRequest and workloads

```python
from learnai.kit import SimRequest, workloads

requests = workloads.poisson(ctx.rng, rate=12.0, n=150, prompt_tokens=(16, 128),
                             output_tokens=(2, 40), priorities=(0, 1), tenants=("a", "b"))
requests = workloads.bursty(ctx.rng, bursts=5, burst_size=8, period_s=1.0)   # exact ties
requests = workloads.replay([(0.0, 4, 3), (0.5, 2, 2)])   # (arrival_s, prompt, output)
requests = workloads.phases(ctx.rng, [(2.0, 5.0), (4.0, 40.0), (2.0, 5.0)])   # a load step
requests = workloads.shared_prefixes(ctx.rng, rate=20.0, n=200, groups=4)     # prompts with token ids
workloads.drive(sim, requests, scheduler.submit)
```

- `SimRequest(id, arrival_s, prompt_tokens, output_tokens, priority=0, tenant=None,
  deadline_s=None, tokens=None)`: frozen. `tokens` is the prompt's token ids (a
  tuple; `prompt_tokens` must equal its length), for problems that look at what a
  prompt contains: prefix caches and cache-aware routers. `None` when only the
  length matters. Token counts are ints >= 1. `priority`: smaller is more
  urgent. `deadline_s` is an absolute virtual time, like `arrival_s`. `output_tokens`
  is what the engine will generate; say in the statement whether the scheduler may
  read it (a real server cannot).
- Every generator returns a list sorted by arrival with ids `0..n-1` in that order,
  and takes `prompt_tokens=` and `output_tokens=` (an int, or an inclusive `(low,
  high)` range), `priorities=`, `tenants=` (drawn uniformly), `deadline_after_s=`
  (deadline that long after arrival), `start_s=`, and `quantum_s=` (round arrivals
  down to a multiple; with `2**-10` times are exact and some coincide). All randomness
  comes from the generator you pass, and every column is always drawn, so adding
  `tenants=` does not change the arrival times.
- `poisson(rng, *, rate, n, ...)`: exponential gaps with mean `1 / rate`.
  `bursty(rng, *, bursts, burst_size, period_s, jitter_s=0.0, ...)`: burst `b` at
  `start_s + b * period_s`, each request a uniform `[0, jitter_s)` later; with no
  jitter a burst arrives at one instant, in id order. `replay(rows)`: rows are
  `SimRequest`s, dicts of fields, or tuples `(arrival_s, prompt_tokens, output_tokens[,
  priority[, tenant[, deadline_s]]])`; ids default to the row's position; equal
  arrival times keep their order; duplicate ids raise `ValueError`.
- `phases(rng, [(duration_s, rate), ...], ...)`: a load that changes in steps: a
  Poisson process of each phase's rate (0 for silence) for that phase's duration, one
  phase after another from `start_s`. The number of requests is what the draw gives.
  This is the "step-load fixture" of the autoscaling and overload problems.
- `shared_prefixes(rng, *, rate, n, groups=4, prefix_tokens=(32, 96),
  suffix_tokens=(4, 24), vocab_size=64, skew=0.0, ...)`: Poisson traffic in which
  every request has `tokens`: one of `groups` shared prefixes followed by a suffix
  of its own. `skew > 0` makes a few prefixes hot (group `g` has weight `(g + 1) **
  -skew`). `token_sequences(rng, n, ...)` returns just the token tuples, for cache
  problems without a clock (`block-hash-prefix-cache`, `radix-prefix-cache`).
- `workloads.drive(sim, requests, on_arrival, *, priority=0)`: at each `arrival_s`,
  records an `arrive` event (`id`, `request`) and calls `on_arrival(request)`; if that
  returns a coroutine it is started with `sim.spawn`. Same-instant arrivals are
  delivered in list order. Call it before `sim.run()`.

> [!IMPORTANT]
> **Arrival callbacks must be synchronous.** `drive` calls `on_arrival` from a timer
> at the arrival instant. A plain function runs right there, among the timers of that
> instant. An `async def` callback does not: calling it only creates a coroutine,
> which `drive` hands to `sim.spawn`, and a spawned task starts once *all* timers due
> at that instant have fired. One of those timers may be the completion of an engine
> step (`SimEngine` completes a step that ends at `t` after the ordinary timers at
> `t`, priority `STEP_DONE_PRIORITY`, precisely so that an arrival at `t` is already
> there when the scheduler plans its next step). Measured, for a request that arrives
> at the instant a step ends:
>
> | `on_arrival` | when its body runs | what the scheduler sees when it wakes |
> |---|---|---|
> | plain function | before the step completes (`engine.busy` is still true) | the request |
> | `async def`, enqueues at once | after the step has completed (`engine.busy` is false, the step's tokens are in the trace) | the request |
> | `async def` with any `await` before it enqueues | after the step has completed | **nothing**: the scheduler resumes first and plans its next step without the request |
>
> So an async callback observes the engine one step later than a synchronous one, and
> with a single `await` in front of the enqueue the request misses the step entirely.
> A reference and a solution that differ only in this then disagree on exactly the
> cases with coinciding times, which are the cases `quantum_s` and `bursty` exist to
> produce.
>
> Specify `submit(request)` as a plain method that only enqueues (append to a list,
> `queue.put_nowait`, set an `asyncio.Event`) and returns, and pass that to `drive`.
> If the solution's API has to be `async def submit`, do not pass it to `drive`:
> pass a plain function that records the arrival
> (`workloads.drive(sim, requests, pending.append)`) and let a task you spawn feed the
> solution, with the statement saying when a same-instant arrival counts.

### ScriptedLM

A scripted token source for service problems: the "model" behind `make_app(lm)`.

```python
from learnai.kit import ScriptedLM

lm = ScriptedLM(["Hello", ",", " world", "!"], gated=True)
with ctx.serve(ctx.solution.make_app(lm)) as client:
    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        events = iter_sse(response)
        lm.release(1)                                  # let exactly one piece through
        first = next(events)
        check.equal(lm.calls[0].pieces, ["Hello"], label="pieces generated so far")
        lm.release()                                   # open the gate
        rest = list(events)
check.that(lm.wait_closed(), "the server did not close the model's stream")
check.that(not lm.stalled, "the server stopped reading the model's stream")

# what a solution does with it:
async for piece in lm.stream(prompt, max_tokens):      # stream.aclose() to stop early
    ...
```

- `ScriptedLM(script, *, gated=False, delay_s=0.0, gate_timeout_s=5.0)`. `script` is a
  list of pieces (used for every prompt), a dict from prompt to pieces (key `None` is
  the default; otherwise an unknown prompt raises `KeyError`), or a function of the
  prompt.
- `lm.stream(prompt, max_tokens=None)` returns an async iterator with `aclose()`. One
  piece is one generated token; at most `max_tokens` are produced (`None`: no limit; a
  negative or non-int value raises `ValueError`). **Finish reason:** a generation that
  produced exactly `max_tokens` pieces was cut by the limit (`length`), also when the
  script happens to end there; one that ended earlier ended by itself (`stop`). The
  solution derives it by counting; the test reads `lm.calls[i].finish_reason`.
- `lm.count_tokens(text)` is the number of whitespace-separated words: the definition
  of prompt tokens for `usage`.
- `lm.calls`: one `StreamRecord` per call, with `prompt`, `max_tokens`, `script`,
  `pieces` (produced so far), `closed`, `finish_reason` (None until complete).
  `lm.active` counts open streams (for a "running requests" gauge); `lm.closed` is true
  when at least one stream was started and none is open; `lm.counters` has `streams`
  and `pieces`.
- A stream is closed when it has produced everything, when `aclose()` was called, or
  when a cancellation or exception interrupted it while it was producing a piece. A
  server that just stops reading leaves it open; so "stops generating on disconnect"
  is tested with `lm.wait_closed(timeout_s=2.0)`, which blocks the test thread until
  every stream is closed (False on timeout).
- The gate: with `gated=True` nothing is produced until the test calls
  `lm.release(n)` (n more pieces) or `lm.release()` (open for good), from the test's
  thread. A stream that waits longer than `gate_timeout_s` ends early and sets
  `lm.stalled`, so a buffering server fails instead of hanging. `delay_s` pauses
  before each piece in the event loop's time (virtual under `ctx.simclock()`). Neither
  uses `asyncio.sleep`, so a `forbidden` list may contain it.

### LayeredLM

A `TinyLM` with the stack opened, for techniques that live between layers: early exit
and layer skipping, KV shared across layers.

```python
from learnai.kit import LayeredLM

lm = LayeredLM(ctx.fw, seed=3)                    # 4 layers; also TinyLM's size arguments
h = lm.embed(tokens, offset=cache.length)         # (T, d_model); positions offset, offset + 1, ...
h, cache = lm.layers(h, 0, 2, cache)              # layers 0 and 1
h, cache = lm.layers(h, 2, lm.n_layers, cache)    # layers 2 and 3
logits = lm.head(lm.norm(h))                      # exactly lm(tokens, cache)
check.cost(lm.counters, "layer_tokens", at_most=...)
```

- For equal seeds and sizes the weights are `TinyLM`'s, and `lm(tokens, cache)` gives
  the same logits. Hidden states are float32 tensors of the framework, `(T, d_model)`
  or `(B, T, d_model)`; nothing is modified in place.
- `lm.embed(tokens, offset=0)`: the model has learned position embeddings, so tokens
  that continue a prefix need `offset=` the number of tokens before them.
- `lm.layers(h, start, stop, cache=None, *, kv_from=None)` runs layers `start` to
  `stop - 1` and returns `(h_out, new_cache)`. The rows of `h` continue what those
  layers have cached, so the layers of one call must have equal cached lengths.
- The cache is a `LayerCache`: keys and values per layer, `cache.lengths` (tokens
  cached by each layer; they may differ, the early layers of a self-speculating model
  run ahead), `cache.length` (their maximum), `cache.truncate(n)` (every layer back
  to at most `n` tokens), `cache.select(rows)`, `lm.empty_cache()`.
- `lm.norm(h)` is the final RMSNorm and `lm.head(h)` the projection of *normalized*
  states. They are separate so that an exit after any layer can be scored (and so
  that forgetting the norm there is a mistake a solution can make).
- `kv_from=j` runs the layers as readers of layer `j`: each computes queries from its
  own input, attends over the keys and values layer `j` has cached, and stores none
  of its own. The `T` rows of `h` are the last `T` positions layer `j` has cached, so
  run layer `j` on them first. `lm(tokens, cache, share_from=s)` is the whole forward
  pass of the model in which every layer from `s` up reads layer `s - 1`: the oracle
  for `cross-layer-kv-sharing`, whose prefill then costs `s * P + (L - s)`.
- Counters: `layer_calls` (layers run), `layer_tokens` (layers run times the tokens
  they ran on, batch included), and `forward_calls`, `tokens_processed` for whole
  passes through `lm(...)` (which add to the first two as well). `embed`, `norm`, and
  `head` are free.
- `residual_scale=` (a number, or one per layer) multiplies what each layer adds to
  the hidden state. With the default 1.0 an exit after two of four layers predicts
  the full model's token about 40 percent of the time; with `(1, 1, 0.25, 0.25)`
  about 80 percent, with `(1, 1, 0.1, 0.1)` above 90. Pick the agreement a
  self-speculation case needs, and derive cost limits from the reference's measured
  counters, as with `draft_target_pair`.
- Misuse raises `ValueError` (a range outside `0 <= start < stop <= n_layers`, hidden
  states of the wrong shape, a batch that does not match the cache, layers with
  different cached lengths, `kv_from` not below `start` or behind `h`) or `TypeError`
  (a layer index that is not an int).

**Interface change from the curriculum** (`lm.embed(tokens)`, `lm.layers(h, start,
stop, cache)`, `lm.head(h)`): `embed` takes `offset=`, because positions are learned
and a decode step embeds one token at position `P`; the final norm is `lm.norm`,
apart from `head`, because the "exit state unnormalized" mutant of
`early-exit-self-speculation` needs it to be the solution's call; `kv_from=`,
`share_from=`, and per-layer cache lengths were added for `cross-layer-kv-sharing`,
where sharing is the solution's job and the cost formula counts the layers it skips;
`residual_scale=` because random layers give an early exit nothing to agree with.

### CountingExperts and TinyMoE

```python
from learnai.kit import CountingExperts, TinyMoE, moe_dense_ref, moe_route_ref

moe = TinyMoE(ctx.fw, seed=2, n_experts=8, top_k=2)       # a LayeredLM with MoE MLPs
x = moe.mlp_inputs(ctx.randint(0, 64, (2, 10)), layer=1)  # (20, d_model): real routing inputs
experts, w_router = moe.experts(1), moe.router(1)         # CountingExperts, (8, d_model)
y = ctx.call("moe_sparse", x, w_router, experts, moe.top_k)
check.allclose(y, moe_dense_ref(x, w_router, experts, moe.top_k), rtol=1e-4, atol=1e-5)
check.cost(experts.counters, "expert_rows", exactly=moe.top_k * len(x))
```

- `experts(i, rows)` runs expert `i` (a SwiGLU MLP with its own seeded weights) on
  rows `(n, d_model)` and returns `(n, d_model)`; one `expert_calls`, `n`
  `expert_rows`. `experts.calls` is the log of `(expert, n_rows)`. A call with no
  rows is allowed and counted, so "at most one call per expert that has a token" is
  a check on `expert_calls`. `CountingExperts(fw, seed, n_experts=8, d_model=32,
  d_hidden=48)` stands alone when no model is needed.
- Not counted, for tests: `experts.reference(i, rows)` (numpy), `experts.weights`
  (`(w_gate, w_up, w_down)`, shaped as the `experts_w` of `moe-router-forward`).
- `moe_route_ref(x, w_router, k)` returns `(experts, weights)`, both `(n_tokens, k)`:
  the routing rule of `moe-router-forward` (softmax over the router logits, the `k`
  largest logits per token with ties to the lowest index, listed from the largest,
  weights renormalized over the kept ones). `moe_dense_ref(x, w_router, experts, k)`
  is the layer evaluated densely. Neither touches the counters.
- A row's output depends on that row only, but not bit for bit on what it was batched
  with: compare with `check.allclose`, not exactly.
- `TinyMoE` exists to give dispatch problems inputs whose routing is uneven, as a
  model's is: `moe.mlp_inputs(tokens, layer)` are the rows the MoE of that layer
  receives. `moe.experts(layer)` is the same object on every call; the model's own
  forward pass uses the dense oracle and counts nothing on it.

### SimDevices

Several devices with memory budgets and a metered network: a ledger, for problems
about models that do not fit on one device.

```python
from learnai.kit import SimDevices

devs = SimDevices(ctx.fw, memory_bytes=[shard_bytes, shard_bytes])     # or n_devices=3
for i, (up, down) in enumerate(shards):
    devs.put(i, "up", up)                         # charged to device i; over budget raises
    devs.put(i, "down", down)
y = ctx.call("tp_mlp", x, devs)                   # the solution: devs.get(i, "up"), ...
check.cost(devs.counters, "all_reduces", exactly=1)
check.cost(devs.counters, "comm_bytes", exactly=devs.n_devices * y.nbytes)
```

- Nothing moves: every tensor lives in this process. The object records what the
  solution says it stores and sends, and gives each operation a cost.
- Memory: `put(device, name, array)` stores and charges elements times item size;
  the put that would exceed the device's budget raises `OutOfDeviceMemory` and stores
  nothing; putting an existing name replaces it. `get`, `names(device)`, `free`.
  `devs.used` (bytes per device now), `devs.peak`, `counters.bytes_placed`.
- `send(src, dst, array)` is one message: it returns `array` and adds its size to
  `comm_bytes`. `devs.messages` is the log of `(src, dst, nbytes)`; `devs.sent(src,
  dst)` counts a pair. Sending to oneself is a `ValueError`.
- Collectives over all devices, one tensor per device in device order:
  `all_reduce(parts)` is their sum and costs `n_devices * nbytes(part)`;
  `all_gather(parts, axis=-1)` is their concatenation and costs `(n_devices - 1) *
  nbytes(result)`. So two shards of a hidden layer four times as wide as the output
  cost four times as much to gather as the output costs to reduce, which is the
  point of `tensor-parallel-linear`.
- `ctx.expected` gets a twin with the same stored tensors and a clean ledger.
- What it does **not** do: check that a computation used only what its device holds.
  Tensors are ordinary tensors, and a solution could add a shard of device 0 to one
  of device 1 without a `send`. The cost checks are what judge: assert the message
  log (`devs.sent(0, d) == 1` for every device with tokens) and `comm_bytes`
  exactly, so that work done without communication shows as bytes that are missing.

**Interface change from the curriculum** (`put`, `send`, counts, budgets):
`all_reduce` and `all_gather` were added because `tensor-parallel-linear` judges
"exactly one all-reduce of output size" against an all-gather mutant, and `get`,
`free`, `messages`, `sent`, `used`, `peak` so tests can state their checks.

### Grammar

A context-free grammar in a subset of GBNF (the notation of llama.cpp), parsed, with
an oracle.

```python
from learnai.kit import Grammar, ToyVocab

g = Grammar.parse('''
    root  ::= "[" ws (value ("," ws value)*)? "]"
    value ::= [0-9]+ | "true" | "false" | root      # nested lists
    ws    ::= " "?
''')
g.accepts("[1, [true]]")                 # True
g.accepts_prefix(b"[1, [tr")             # True: can still become a member
matcher = ctx.solution.GrammarMatcher(g, vocab)
for token in matcher.allowed(state):
    check.that(g.accepts_prefix(text_so_far + vocab.pieces[token]), ...)
```

The subset, checked against `grammars/README.md` in the llama.cpp repository:

| supported | notation |
|---|---|
| rules; the language is what `root` matches | `name ::= ...`; names are dashed lowercase words (digits allowed after the first letter) |
| string literals and character sets, over Unicode code points | `"O-O"`, `[1-9]`, `[NBKQR]`, `[^\n]`, ranges of any code points; a `-` just before `]` is literal |
| escapes | `\n \t \r \\ \" \[ \]`, `\xXX`, `\uXXXX`, `\UXXXXXXXX` |
| sequence, alternatives, grouping | `a b`, `a \| b`, `( ... )` |
| repetition of the preceding symbol or group | `*`, `+`, `?`, `{m}`, `{m,}`, `{m,n}`, counts up to 100; after a literal, the whole literal |
| comments | `# ...` |
| rules over several lines | inside parentheses, and directly after `\|` |

Not supported, each a `ValueError` naming the line: token references (`<[1000]>`,
`<think>`, `!<...>`) and `.` (the README uses it in an example and never defines it;
write `[^\n]`). Also errors: a rule used but not defined, defined twice, no `root`,
and a rule that can never finish (`a ::= "x" a`).

- `g.rules` is the grammar in the shape a matcher wants: `{name: (alternative, ...)}`,
  an alternative a tuple of symbols, a symbol either `Chars(ranges, negated)` (one
  character; `chars.matches(c)`, `chars.overlaps(low, high)`) or `Ref(name)`. A
  literal becomes one `Chars` per character; a group becomes a new rule; `x*` becomes
  `r ::= x r | (empty)`, `x+` is `x` then that, `x?` is `r ::= x | (empty)`, `x{m,n}`
  is `m` copies and `n - m` nested optional ones (how llama.cpp rewrites them).
  Generated rules are named `<rule>_<n>`, which the notation cannot produce. An
  alternative may be empty.
- Recursion of every kind is accepted, **left recursion included** (llama.cpp's
  README does not mention it; a stack machine that expands the first symbol loops on
  it, which is the mutant). `g.left_recursive` names the rules that can begin with
  themselves and `g.nullable` those that can match nothing.
- `accepts(text)` and `accepts_prefix(text)` take `str` or `bytes`. Bytes that end
  inside a character are a prefix when some character starting with those bytes
  would be accepted there, so a `ToyVocab` token that is half of `€` is judged
  correctly; bytes that are not UTF-8 are never accepted. Both are an Earley
  recognizer, exact for ambiguous and left-recursive grammars. Not counted; tell
  learners not to call them.

### ScriptedChatModel and ScriptedTool

An OpenAI-style chat model whose replies are scripted, and tools with a latency and a
log, for agent loops, routers, and speculative tool calls.

```python
from learnai.kit import ScriptedChatModel, ScriptedTool, reply, tool_call

model = ScriptedChatModel([
    reply(tool_calls=[tool_call("get_weather", {"city": "Paris"}),
                      tool_call("get_time", {"zone": "CET"})]),     # two calls at once
    reply(tool_calls=[tool_call("get_weather", '{"city": ')]),      # malformed arguments
    "It is 21 degrees in Paris.",
])
tools = {"get_weather": ScriptedTool("get_weather", lookup, delay_s=0.5), ...}
transcript = sim.run_until_complete(ctx.solution.run_agent(model, tools, messages, 8))
check.cost(model.counters, "calls", exactly=3)
check.that(not model.violations, "the conversation was not valid", ...)
```

- Messages are plain dicts in the chat-completions shape. `model.complete(messages)`
  returns `{"role": "assistant", "content": str | None, "tool_calls": [{"id":
  "call_0", "type": "function", "function": {"name": ..., "arguments": "<JSON
  string>"}}]}`; `tool_calls` is absent when there are none. Ids are `call_0`,
  `call_1`, ... in emission order unless `tool_call(..., id=)` names one. `arguments`
  given as a dict is encoded with `json.dumps`; given as a string it is passed
  through, valid or not.
- The script: a list (one reply per call; when it runs out `ScriptExhausted` is
  raised, which is how a loop without a step limit ends, unless `repeat_last=True`),
  a dict keyed on the content of the conversation's last message (`None`: default),
  or a function `(messages, call_index) -> reply`. A reply is a string, `reply(...)`,
  or a list of `tool_call(...)`.
- With `strict=True` (default) every call validates the conversation as an API would
  and raises `ChatProtocolError` with `.code`: `bad_message`, `missing_tool_result`
  (an assistant turn's tool calls are not each answered by one tool message directly
  after it), `unknown_tool_call_id`, `duplicate_tool_result`. Errors are kept in
  `model.violations`, so a solution that catches them is still caught. The *order*
  of tool results is not checked by the model: compare the transcript.
- `model.calls` (a copy of the messages of every call), `model.replies`,
  `model.counters` (`calls`, `tool_calls`), `model.cost` (price of one call, known
  before calling: `cascade-router` checks its budget against it) and `model.spent`.
  `reply(..., logprobs=[...])` puts token log-probabilities on the message.
- Async: `await model.acomplete(messages)` takes `delay_s` of the loop's time;
  `model.stream(messages)` yields the same reply in pieces, one every `delay_s`:
  `{"content": ...}` (`chunk_chars` characters at a time), `{"tool_calls": [{"index":
  i, "id": ..., "type": "function", "function": {"name": ..., "arguments": ""}}]}`,
  argument fragments `{"tool_calls": [{"index": i, "function": {"arguments":
  ...}}]}`, and last `{"finish_reason": "tool_calls" | "stop"}`. `assemble(deltas)`
  rebuilds the message. Timers, not `asyncio.sleep`, as in `ScriptedLM`.
- `ScriptedTool(name, fn, *, delay_s=0.0, side_effects=False, parameters=None,
  description="")`: `await tool(**arguments)` records a `ToolRun`, waits `delay_s`,
  then returns `fn(**arguments)` (an exception propagates, after the delay). If the
  awaiting task is canceled first, the run is `"canceled"` and `fn` never runs.
  `tool.runs` (`arguments`, `started_s`, `finished_s`, `status`, `result`),
  `tool.counters` (`started`, `done`, `failed`, `canceled`), `tool.schema` (the
  `tools` entry), `tool.side_effects`, and `tool.run(**arguments)` for synchronous
  code. "No canceled result is ever used" and "a side-effecting tool was never
  started early" are assertions on `tool.runs`.

**Interface change from the curriculum** (`model.complete(messages)`, call counts):
`stream` and `acomplete` for `speculative-tool-calls`, whose input is a model
stream; `cost`, `spent`, and `logprobs` for `cascade-router`; protocol validation,
because "ids mismatched" and "only the first of parallel calls executed" are
otherwise visible only by reading transcripts; and `ScriptedTool`, which both
tool-calling problems would otherwise write twice.

### TinyEncoder and CountingScorer

```python
from learnai.kit import CountingScorer, TinyEncoder

enc = TinyEncoder(ctx.fw, seed=1)                  # bidirectional; vocab 64, width 32, 2 layers
states = enc(tokens, mask)                         # (B, T, d_model); mask: nonzero = real token
emb = ctx.call("pool", states, mask, "mean")

scorer = CountingScorer(seed=1, max_length=48, max_batch=8)
ranked = ctx.call("rerank", query, docs, scorer, 8, 5)        # scorer([(query, doc), ...])
check.cost(scorer.counters, "padded_tokens", at_most=...)
```

- `TinyEncoder`: a real token attends to real tokens only, and its position is
  counted among the real tokens of its row. So the states of the real tokens do not
  depend on where the padding is (to about `1e-6`), which is what lets
  `embedding-pooling` state "padding never changes the embedding" for left and right
  padding alike. States at padded positions are arbitrary: a pooling mutant that
  reads them is wrong by a visible amount. `enc.reference(tokens, mask)` is the same
  as numpy, uncounted. Counters: `forward_calls`, `tokens_processed` (padding
  included), `real_tokens`. `enc.pad_id` is 0 and `enc.cls_id` 1 by convention only.
- `CountingScorer` is framework-free (the problem that uses it is `python`). A pair
  is `(query_tokens, doc_tokens)`, two non-empty sequences of ints; `scorer(pairs)`
  returns a list of floats. It charges for the rectangle a cross-encoder batch is:
  `padded_tokens += len(batch) * max(len(q) + len(d))`; `real_tokens` is what the
  pairs hold; `calls`, `pairs`; `scorer.batches` logs the pair lengths of each call.
- A pair's score is a function of that pair alone, **exactly**: the scorer encodes
  each pair by itself, so equal pairs get equal floats in any batch, and "stable
  ties" in a top-n is testable with duplicate documents.
- A pair longer than `max_length` is a `ValueError` ("truncate the document first"),
  as are an empty batch, a batch over `max_batch`, and an empty query or document.
  `scorer.score(query, doc)` is the uncounted oracle.

### RecurrentLM

A language model whose memory of a prefix is a fixed-size state, for
`hybrid-state-checkpoints`.

```python
from learnai.kit import RecurrentLM

lm = RecurrentLM(ctx.fw, seed=3)                   # vocab 64, 2 layers, state 48 per layer
logits, state = lm(tokens)                         # the TinyLM call convention
logits, state = lm(more_tokens, state)
store = {}                                         # the checkpoint store is the problem's own
tokens = ctx.call("generate_with_checkpoints", lm, store, prompt, 8)
check.cost(lm.counters, "tokens_processed", at_most=P - (lcp // 8) * 8 + g - 1)
```

- A `RecurrentState` is opaque, immutable, and the same size at every length
  (`state.nbytes`, `state.length`). `state.truncate(n)` exists only to raise a
  `TypeError` that says why: there is nothing to cut. `state.select(rows)` and
  `lm.initial_state(batch_size=None)` complete the picture.
- Continuing from a saved state is **bit-identical** to reading the whole prefix in
  one call (every token is folded in by the same arithmetic, one at a time). Tests
  may compare logits with `check.equal`, unlike with `TinyLM`.
- A state from the wrong prefix gives visibly wrong logits (the mutant that resumes
  from the end of a longer cached prompt), but the influence of a token fades over a
  few dozen steps: do not build a case on a difference 100 tokens back.
- Counters: `forward_calls`, `tokens_processed`. The checkpoint store is left to the
  problem (a dict from prefix to state works) because what it holds is what the
  solution is graded on.

### SimCluster

Replicas of `SimEngine`, each with a fixed scheduler and a prefix cache, behind a
`route` hook: for load balancing, cache-aware routing, autoscaling, draining, and
prefill-decode disaggregation.

```python
from learnai.kit import CostModel, SimCluster, workloads

def simulate(ctx, route, requests):
    sim = ctx.simclock()
    cluster = SimCluster(sim, CostModel(2**-6, 2**-12, 2**-9), replicas=3, max_seqs=8,
                         cache_tokens=4096, block_tokens=16, route=route)
    workloads.drive(sim, requests, cluster.submit)
    sim.run()
    check.that(not cluster.violations, "the router misused the cluster", ...)
    return sim, cluster

sim, cluster = simulate(ctx, lambda request, replicas: router.route(request.tokens, replicas),
                        workloads.shared_prefixes(ctx.rng, rate=20.0, n=200, groups=4))
check.cost(cluster.counters, "cache_hit_tokens", at_least=1.5 * round_robin_hits)
ttft = sim.trace.latencies("arrive", "first_token")
```

`SimCluster(sim, cost, *, replicas=1, max_seqs=8, max_batch_tokens=None,
kv_blocks=None, block_tokens=16, cache_tokens=0, startup_s=0.0, route=None,
transfer_s_per_token=0.0)`.

**A replica** is `starting` for `startup_s` after it is added (the initial ones are
ready at once), then `ready`, perhaps `draining`, finally `stopped`. A router may
read `replica.id` (0, 1, ... in order added, never reused), `.state`, `.queue_len`
(waiting plus running), `.waiting_len`, `.running_len`, `.outstanding_tokens` (prompt
tokens still to prefill plus output tokens still to produce), and `.kv_usage` (0.0
without a pool). `replica.cache` is the real prefix cache and is for tests: say in
the statement that a router keeps its own index.

**Its scheduler** is fixed, so that the routing decision is the only variable:
continuous batching, first come first served. When its engine is free it admits
waiting requests in arrival order while fewer than `max_seqs` run (with a KV pool,
while the pool can hold the head request's whole context: nobody overtakes the head;
a request that could never fit fails with `too_large`); then one step in which every
ready sequence decodes and the rest of `max_batch_tokens` goes to prefill in
admission order. State this in the statement in one paragraph; a problem whose
answer depends on a different replica policy should use `SimEngine` directly.

**The prefix cache** (with `cache_tokens > 0`, for requests that have `tokens`): the
longest common prefix with a prompt the replica has prefilled, rounded down to a
multiple of `block_tokens` and to at most `prompt_tokens - 1`, is not prefilled
again. A prompt enters the cache when its prefill completes (two same-prefix
requests arriving together both miss), and least recently used prompts are evicted
beyond `cache_tokens`. `cache_lookup` events and `Outcome.cached_tokens` record the
hits. A cache big enough for every prefix on every replica makes routing irrelevant:
size it so that a replica holds a few prefixes, and measure the round-robin baseline.

**Requests.** `cluster.submit(request)` calls `route(request, replicas)` with the
ready replicas (the starting ones when none is ready) and expects a replica or its
id; without `route`, the least `queue_len`, lowest id on ties. `replica.submit(
request)` sends directly. Both return a future of an `Outcome` (`id`, `replica`,
`phase`, `status` `"finished"` or `"failed"`, `submitted_s`, `first_token_s`,
`finish_s`, `cached_tokens`, `reason`) that never raises; `cluster.outcomes` maps ids
to them. `workloads.drive(sim, requests, cluster.submit)` records the `arrive`
events, so `sim.trace.latencies("arrive", "first_token")` is TTFT as usual.

**Scaling.** `cluster.add_replica()`; `cluster.drain(replica)` (no new work, stops
when what it holds has finished); `cluster.stop(replica)` (now: its unfinished
requests **fail** with `replica_stopped`, which is how the "scale-down kills
in-flight requests" mutant shows); `cluster.scale_to(n)` (adds, or drains the newest,
until `n` are starting or ready). Sending to a draining or stopped replica raises
`ClusterError` (`replica_unavailable`), kept in `cluster.violations`.
`cluster.metrics()` returns `ClusterMetrics(now, ready, starting, draining,
queue_depth, running, outstanding_tokens, kv_usage)`, the input of an autoscaler
policy (`starting` is the pending capacity it must count); `cluster.replica_seconds()`
is the cost so far. The loop that calls a policy every period is three lines of test
code (`sim.spawn` a task that sleeps, reads `metrics()`, calls `scale_to`); run such a
simulation with `sim.run(until=...)`, since a periodic task never goes idle.

**Two phases.** `replica.submit(request, phase="prefill")` runs the prompt only (the
outcome arrives with the first token and the replica drops the sequence); `await
cluster.transfer(request, src, dst)` takes `transfer_s_per_token *
request.prompt_tokens`; `replica.submit(request, phase="decode")` produces the other
`output_tokens - 1` tokens without a prefill. Every engine event carries
`engine=<replica id>`, so "no decode step delayed by a prefill" is a filter on the
decode replicas' `step` events.

**Trace events added:** `route` (`id`, `replica`), `cache_lookup` (`id`, `replica`,
`hit`), `fail` (`id`, `replica`, `reason`: `replica_stopped`, `no_replica`,
`too_large`), `transfer` (`id`, `src`, `dst`, `tokens`, `duration`), `transfer_done`,
`replica_start`, `replica_ready`, `replica_drain`, `replica_stop` (`replica`).
**Counters:** `routed`, `finished`, `failed`, `cache_lookups`, `cache_hit_tokens`,
`scale_ups`, `scale_downs`, `transfers`, `transferred_tokens`.

**Interface change from the curriculum** ("replicas of `SimEngine` with per-replica
prefix caches, startup delay, and a `route` hook"): the replica scheduler is built in
and fixed; `drain`, `stop`, `scale_to`, `metrics`, and `replica_seconds` serve
`autoscaler-policy` and stage 5 of `capstone-inference-gateway`; the `prefill` and
`decode` phases and `transfer` serve `prefill-decode-disaggregation`; requests carry
token ids (`SimRequest.tokens`) because a prefix cannot be matched on a length.
`SimEngine` gained what this needed: `name=` (events tagged `engine=`), and
`admit(request, cached_tokens=c)` and `admit(request, resume_from=g)` for a sequence
that starts with a cached prefix or with a KV cache brought from another engine.

### vectors

Seeded vector sets with known neighbors, an exact top-k oracle, and a distance
function that counts, for the vector-search problems.

```python
from learnai.kit import vectors

data = vectors.clustered(ctx.rng, n=2000, dim=32, clusters=16)      # .vectors .centroids .labels
queries = vectors.queries_near(ctx.rng, data.vectors, [3, 17, 256], noise=0.05)
truth, scores = vectors.exact_topk(queries, data.vectors, k=10)     # float64 scan, ties to the lower index
check.that(vectors.recall(found, truth) >= 0.9, "recall@10 is below 0.9")

distance = vectors.CountingDistance(ctx.fw, metric="l2")
index = ctx.call("build_ivf", data.vectors, data.centroids)
found = ctx.call("search", queries[0], index, 10, 4, distance)
check.cost(distance.counters, "distances", exactly=16 + probed_sizes)
```

- `clustered(rng, n, dim, *, clusters=8, spread=0.25, normalize=True)`: unit
  centroids plus Gaussian noise of relative length `spread`; float32; with
  `normalize` every vector has unit length, so cosine, dot, and L2 rank alike.
- `queries_near(rng, vectors, indices, *, noise=0.05)`: one query per index, that
  vector slightly moved. For small `noise` it is the query's nearest neighbor; when a
  case depends on it, assert it with `exact_topk` in the test.
- `exact_topk(queries, corpus, k, *, metric="cosine")` returns `(indices, scores)`,
  both `(Q, min(k, N))`, best first; metrics `cosine`, `dot`, `l2` (score: negated
  squared distance). `recall(found, truth)` averages the fraction of true neighbors
  found.
- `CountingDistance(fw, metric)`: `distance(queries, candidates)` returns `(N,)` or
  `(Q, N)` as a tensor of the framework (`l2`: squared distances, smaller is closer;
  `dot`, `cosine`: larger is closer) and counts `calls` and `distances` (`Q * N`). An
  empty candidate list is allowed and counts nothing.

### GDNConfig, gdn_params, and the delta-layer oracles

Weights for one Gated DeltaNet layer (the `linear_attn` sublayer of Qwen3.5), named
like the checkpoint, and numpy oracles. The layer itself is `ops.gated_delta_layer`,
one signature in torch, JAX, MLX, and numpy.

```python
from learnai.kit import GDNConfig, gdn_gates_ref, gdn_layer_ref, gdn_params, gdn_step_ref, ops

cfg = GDNConfig()                        # d_model=32, n_key_heads=2, n_value_heads=4,
params = gdn_params(cfg, seed=1)         #   key_head_dim=8, value_head_dim=8, conv_kernel=4,
x = ctx.randn(10, cfg.d_model)           #   norm_eps=1e-6
out, state = gdn_layer_ref(cfg, params, x)               # x (T, D) or (B, T, D)
out2, state = gdn_layer_ref(cfg, params, more, state)    # continue the sequence
y, state = ops.gated_delta_layer(x, params, state, cfg.n_key_heads)   # in a solution
```

- `gdn_params(cfg, seed)` returns `dict[str, np.ndarray]`, float32, linear weights
  `(out, in)`, no biases:

  | key | shape |
  |---|---|
  | `in_proj_qkv.weight` | `(2 * key_dim + value_dim, d_model)`: `q`, `k`, `v` in that order |
  | `in_proj_z.weight` | `(value_dim, d_model)` |
  | `in_proj_a.weight`, `in_proj_b.weight` | `(n_value_heads, d_model)` |
  | `conv1d.weight` | `(2 * key_dim + value_dim, 1, conv_kernel)` |
  | `A_log`, `dt_bias` | `(n_value_heads,)` |
  | `norm.weight` | `(value_head_dim,)`, shared by all value heads |
  | `out_proj.weight` | `(d_model, value_dim)` |

  `cfg.key_dim`, `cfg.value_dim`, and `cfg.conv_dim` are totals over heads
  (`key_dim = n_key_heads * key_head_dim`), as in the shipped configuration; the
  per-head sizes are the fields `key_head_dim` and `value_head_dim`. Nothing is at its
  initialization value: `norm.weight` is in `[0.5, 1.5]`, `dt_bias` in `[-1, 1]`,
  `exp(A_log)` in `[0.05, 0.5]`, so the decay `g` of a typical token is between 0.6 and
  0.99 and each constant changes the output when it is left out.
- **The number of key heads is not in the shapes.** `ops.gated_delta_layer` reads
  every other size from `params`, but `key_dim` could be split into any number of
  heads, so it takes `n_key_heads` (default: the number of value heads). A problem
  whose signature is `gdn_layer(x, params, state)` therefore has to pass the head
  count some other way; pass `cfg` (`gdn_layer(x, params, cfg, state)`), which is a
  frozen dataclass and goes through `ctx.call` unchanged.
- The state is `(conv_state, S)`: the last `conv_kernel - 1` *projected inputs* (the
  rows of `x W_qkv^T` before the convolution), `(K - 1, conv_dim)`, and the memory,
  `(n_value_heads, value_head_dim, key_head_dim)`; a batch axis comes first in both when
  `x` has one. `None` means zeros. The state passed in is not modified.
- **Any chunking of a sequence gives the output of one pass, to float32 rounding**
  (about 2e-6 on outputs of order 1; tests use `rtol=1e-4, atol=1e-5`), not bit for
  bit: the convolution sums its taps in the same order, but a batched matmul may not.
- `gdn_gates_ref(a, b, A_log, dt_bias) -> (g, beta)` and `gdn_step_ref(S, q, k, v, g,
  beta) -> (y, S)` are the numpy oracles of `gated-delta-step` (decay, write, read
  after the write; `q` and `k` used as given). With `g = 1`, `beta = 1`, and a unit
  key, `S k == v` after the step.
- What the layer computes, in order: `in_proj_qkv`; `ops.causal_conv1d` over the saved
  inputs and the new ones; SiLU; split; `ops.l2norm` of `q` and `k`, and `q *
  key_head_dim**-0.5`; each key head repeated over its `n_value_heads / n_key_heads`
  consecutive value heads; `g = exp(-exp(A_log) * softplus(a + dt_bias))`, `beta =
  sigmoid(b)`; per token `S = g S`, `S = S + beta (v - S k) k^T`, `y = S q`;
  `norm.weight * rms_norm(y, eps=norm_eps) * silu(z)` per value head; `out_proj`.

### HybridConfig, hybrid_params, hybrid_forward_ref

Weights and the numpy oracle for a small decoder in the shape of the Qwen3.5 text
models: Gated DeltaNet layers with an attention layer every `full_attention_interval`
layers. For `hybrid-decoder-forward`.

```python
from learnai.kit import HybridConfig, hybrid_forward_ref, hybrid_params

cfg = HybridConfig()                           # 4 layers: linear, linear, linear, full
params = hybrid_params(cfg, seed=1)            # names as below
logits, state = hybrid_forward_ref(cfg, params, prompt)            # tokens (T,) or (B, T)
logits, state = hybrid_forward_ref(cfg, params, [token], state)    # continue
```

- `HybridConfig` is a frozen dataclass (it passes through `ctx.call`): `vocab_size=64,
  d_model=32, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16, d_mlp=48,
  rope_base=10000.0, partial_rotary_factor=0.25, norm_eps=1e-6, tie_embeddings=True,
  full_attention_interval=4, layer_types=None`, and the delta-layer sizes of
  `GDNConfig` (`n_key_heads=2, n_value_heads=4, key_head_dim=8, value_head_dim=8,
  conv_kernel=4`). It is its own class, not a `DecoderConfig`: **`head_dim` is a
  field**, as in Qwen3.5, and `n_heads * head_dim` (64) is not `d_model` (32).
  `cfg.layers` is the resolved tuple of `"linear_attention"` / `"full_attention"`
  (layer `i` is full when `(i + 1) % full_attention_interval == 0`; an explicit
  `layer_types` wins); `cfg.rotary_dim` is `int(head_dim * partial_rotary_factor)`
  (must be even); `cfg.gdn` is the matching `GDNConfig`.
- `hybrid_params(cfg, seed)`: for every layer `model.layers.{i}.input_layernorm.weight`,
  `post_attention_layernorm.weight`, `mlp.{gate,up,down}_proj.weight`; for a linear
  layer the nine `gdn_params` keys under `model.layers.{i}.linear_attn.`; for a full
  layer

  | key | shape |
  |---|---|
  | `model.layers.{i}.self_attn.q_proj.weight` | `(n_heads * 2 * head_dim, d_model)`: per head, query then gate |
  | `model.layers.{i}.self_attn.k_proj.weight`, `v_proj.weight` | `(n_kv_heads * head_dim, d_model)` |
  | `model.layers.{i}.self_attn.o_proj.weight` | `(d_model, n_heads * head_dim)` |
  | `model.layers.{i}.self_attn.q_norm.weight`, `k_norm.weight` | `(head_dim,)` |

  then `model.norm.weight` and `model.embed_tokens.weight` (the head, when tied).
  `layer_params(params, i)` strips the layer prefix as for `decoder_params`.
- **Every norm weight is a plain gain** (`ops.rms_norm(x, weight, eps)`), drawn from
  `[0.5, 1.5]`. The real checkpoint stores the block norms, the final norm, `q_norm`,
  and `k_norm` zero-centered (`1 + weight`); that convention belongs to the real-model
  problems, and `mlx-lm` converts it away when it loads.
- The oracle, block by block: `x + mixer(rms_norm(x))`, then `x + swiglu(rms_norm(x))`.
  A linear mixer is `ops.gated_delta_layer`. A full mixer is gated attention: split
  each head's `2 * head_dim` outputs of `q_proj` into query and gate (inside the
  head, not across the flat projection); `rms_norm` of queries and keys over
  `head_dim` with `q_norm` and `k_norm`; `ops.partial_rope` on both; `ops.attention`
  (grouped-query, scale `head_dim**-0.5`, causal); `o_proj(attended * sigmoid(gate))`.
  `gated_attention_ref(cfg, attn, h, past)` is that mixer alone, and
  `hybrid_block_ref(cfg, params, layer, x, state=None) -> (x, entry)` is one whole
  block with that layer's state entry (chaining the blocks from the embedding, then
  the final norm and the head, is `hybrid_forward_ref`).
- `state` is a list with one entry per layer: `(conv_state, S)` for a linear layer,
  `(k, v)` for a full layer, where `k` are the keys **after the norm and the rotation**
  and both are `(n_kv_heads, S, head_dim)` (batch axis first if the tokens have one).
  New tokens take the positions after the cached keys, so positions never restart.
  The oracle accepts the state a solution returned once it is converted to numpy.
- Prefill in any chunking and then one token at a time agrees with one pass to
  float32 rounding (measured up to 1.2e-4 on logits of magnitude 9 for 8 layers and
  150 tokens): compare with `rtol=1e-3, atol=1e-3` or compare argmax tokens; pick
  prompts whose top two logits are further apart than that if tokens are compared.
- The same forward pass written from `ops` in torch, JAX, and MLX is in
  `python/tests/test_kit_m6_models.py` (`hybrid_forward_with`); it is 40 lines and uses
  only `reshape`, `swapaxes`, `concatenate`, and `exp` besides the kit.

### CountingKV

A key-value store that counts the rows read from it, for `gathered-sparse-attention`
and `indexer-cache-decode`: sparse attention saves work only if the unselected keys are
never read.

```python
from learnai.kit import CountingKV

store = CountingKV(ctx.fw, k, v)                        # k (S, ...), v (S, ...): any row shapes
k_rows, v_rows = store.gather(idx, valid=idx >= 0)      # idx (T, kk) -> (T, kk, ...) each
check.cost(store.counters, "rows_read", exactly=int((idx >= 0).sum()), unit="rows")
store = store.append(k_new, v_new)                      # a longer store, same counters
```

- `CountingKV(fw, k, v)` takes the adapter first, like every kit object (the
  curriculum's sketch omitted it). `CountingKV.empty(fw)` is a store with no rows
  whose first `append` fixes the row shapes: hand it to a decode problem as the
  initial cache.
- `gather(rows, valid=None)`: `rows` is an integer list or tensor of any shape. **Every
  entry is one row read**, also a repeated one, so reading all `S` keys and masking
  afterward costs `S` per query. Indices follow Python: **`-1` is the last row**, read
  and counted (the mutant that uses the padding sentinel as an index reads the last
  key). An index outside `[-n, n)` raises `ValueError`.
- `valid` is a boolean array of the shape of `rows`. Where it is False the slot is
  **not read and not counted**, whatever its index, and comes back as zeros. So the
  intended solution of a padded selection is `store.gather(idx, valid=idx >= 0)`, and
  `rows_read` then equals the number of valid slots exactly. The zeros still need a
  `-inf` score: a slot scored 0 gets weight (another mutant).
- `append(k, v)` returns a new store and leaves the old one usable (branching works).
  **A store made by `append` shares the counters and the log of the one it came
  from**, so a test keeps the initial store and reads the total of a whole decode from
  it. `with_framework(fw)` (what `ctx.expected` uses) gives a twin with fresh counters.
- Counters: `rows_read`, `gather_calls`. `store.log` is one `(shape of rows, rows read)`
  per call, for "at most `k` rows per query":
  `all(read <= k * shape[0] for shape, read in store.log)`.
- `store.n` is the number of rows; `store.reference()` returns numpy copies of all
  keys and values and is not counted. It is for tests: list
  `learnai.kit.sparse.CountingKV.reference` in `forbidden`, and hand the store over
  as `class Store(Guarded, CountingKV)` (below), which keeps the counters, the log,
  and the arrays behind them out of the solution's reach. A store made by `append`
  has the class of the store it came from, so it stays guarded.

### TinyDSA

A seeded stack of sparse-attention layers in the manner of DeepSeek Sparse Attention
(GLM-5.3, DeepSeek-V3.2), handed out in pieces. For `indexer-cache-decode` and
`shared-index-layers`.

```python
from learnai.kit import CountingKV, TinyDSA

dsa = TinyDSA(ctx.fw, seed=1)             # n_layers=4, d_model=32, n_heads=2, head_dim=8,
x = ctx.randn(12, dsa.d_model)            #   index_heads=4, index_dim=8, top_k=4, types=None
k_idx = dsa.index_key(layer, x)           # (T, index_dim); counts index_key_rows
q_idx, w = dsa.index_query(layer, x)      # (T, index_heads, index_dim), (T, index_heads)
q, k, v = dsa.qkv(layer, x)               # (T, n_heads, head_dim) each
out = dsa.out(layer, attended)            # (T, n_heads, head_dim) -> (T, d_model)
idx = dsa.indexer(layer, x, k=None)       # (T, min(k, T)) indices; counts indexer_calls
out = dsa.attend(layer, x, idx)           # (T, d_model)
check.allclose(got, dsa.layer_ref(layer, x, k), rtol=1e-4, atol=1e-5)
check.cost(dsa.counters, "index_key_rows", exactly=len(x), unit="rows")
```

- What a layer computes (the statement can quote this): indexer key `k_I =
  layer_norm(x Wk^T)`, one per token (weight, bias, eps 1e-6); indexer queries `q_I =
  x Wq^T` in `index_heads` heads; raw head weights `w = x Ww^T`; score `I[t, s] = sum_h
  (w[t, h] * H^-0.5) * relu(q_I[t, h] . k_I[s] * D^-0.5)`; later tokens `-inf`; the
  `min(k, S)` best of each row, **best first, ties to the lowest index, `-1` in slots
  that hold no visible token**; attention over the selected rows with scale
  `head_dim**-0.5`; `out`. A token can always select itself.
- `index_key`, `index_query`, `qkv`, `out`, `indexer`, and `attend` take and return
  the framework's tensors and work on rows `(n, d_model)`; a row's result depends on
  that row only (to about 1e-6, not bit for bit across batchings).
- Counters: `index_key_rows` counts the rows passed to `index_key` **and every row of
  every `indexer` call** (which computes its keys itself), so a decode loop cannot
  dodge "each token's indexer key once" by calling `indexer` on the prefix.
  `indexer_calls` counts `indexer`. Nothing else is counted; the rows read from the
  main keys and values are counted by the `CountingKV` the solution keeps them in. A
  refused call (bad layer, shape, or `k`) counts nothing.
- `dsa.types` is `"full"` or `"shared"` per layer. The default is `index_types(n_layers)`
  with `freq=2, offset=1` (full, shared, full, shared, ...); `index_types(n, freq,
  offset)` is GLM's rule `max(i - offset + 1, 0) % max(freq, 1) == 0`, and GLM-5.3
  itself is `index_types(n, 4, 3)`. `indexer` works on every layer whatever its type:
  a shared layer that runs an indexer anyway is a cost mutant, not an error.
- Oracles, numpy in and out, never counted: `index_scores_ref(layer, x)` (`(T, T)`,
  `-inf` above the diagonal), `indexer_ref(layer, x, k)`, `attend_ref(layer, x, idx)`,
  `layer_ref(layer, x, k)` (one layer on a whole sequence: what a chunked decode with
  an indexer cache must reproduce), and `forward_ref(h, types=None, k=None)` (the
  stack `h = h + attend(i, h, idx)`, a shared layer using the nearest preceding full
  layer's indices, `ValueError` when there is none). List the ones a solution could
  shortcut through in `forbidden` (`learnai.kit.sparse.TinyDSA.layer_ref`,
  `...attend_ref`, `...forward_ref`, and for `indexer-cache-decode` also
  `learnai.kit.sparse.TinyDSA.indexer` and `...attend`). The oracles do not go through
  the public methods, so forbidding those does not break `ctx.expected` or a test
  that calls an oracle. `class Layers(Guarded, TinyDSA)` (below) rules out the
  counters and the uncounted internals, also when they are borrowed as plain
  functions (`TinyDSA._index_key(dsa, ...)`).
- **Top-k is discontinuous: choose fixtures with a margin.** The ReLU makes index
  scores that are exactly 0 (about 6% of the visible pairs with the default sizes),
  and exact ties are safe, the lowest index wins in every framework. A `k`-th and
  `(k + 1)`-th score that differ by 1e-7 are not. Draw `x` until, in every row, the
  two scores at the cut are equal or more than 1e-4 apart
  (`dsa.index_scores_ref`; `clear_margin` in `python/tests/test_kit_m6_models.py` is
  the loop).
- The stack has no norm and no MLP: the residual stream grows by about 9% per layer
  (its RMS doubles over 8 layers), which is fine for the 4 to 8 layers these problems
  use.
- Not modeled, compared with the shipped `GlmMoeDsaIndexer`: rotary positions on the
  first indexer channels (TinyDSA has no positions at all), the low-rank query
  residual the shipped indexer projects its queries from (here they come from `x`),
  and the latent (MLA) form of the main attention.

### mtp_params and mtp_draft_ref

The chained multi-token-prediction module that ships inside checkpoints (`mtp.*`), on
top of the kit decoder. For `mtp-chain-draft`.

```python
from learnai.kit import (DecoderConfig, decoder_forward_ref, decoder_hidden_ref,
                         mtp_draft_ref, mtp_layer_params, mtp_params)

cfg = DecoderConfig(n_layers=2)
params = mtp_params(cfg, n_layers=3, seed=1)         # decoder_params + mtp.layers.{0,1,2}.*
hidden = decoder_hidden_ref(cfg, params, tokens)     # (T, d_model), after the final norm
next_token = int(decoder_forward_ref(cfg, params, tokens)[-1].argmax())
draft, logits = mtp_draft_ref(cfg, params, tokens, next_token, hidden, n_draft=3)
```

- `mtp_params(cfg, n_layers=1, seed=0)`: every entry of `decoder_params(cfg, seed)`
  unchanged, plus for stage `k` `mtp.layers.{k}.enorm.weight`, `hnorm.weight`,
  `post_norm.weight` (`(d_model,)`, gains in `[0.5, 1.5]`), `eh_proj.weight` (`(d_model,
  2 * d_model)`), and one decoder layer under `mtp.layers.{k}.mtp_block.` with the nine
  names of `decoder_params` (`input_layernorm.weight`, `self_attn.q_proj.weight`, ...).
  `mtp_layer_params(params, k)` strips the stage prefix.
- `decoder_hidden_ref(cfg, params, tokens)` (new) is the main model's last hidden
  state after the final norm, which is what the module starts from;
  `decoder_forward_ref` is this times the head.
- Stage `j`: ids shifted left by `j + 1`, ending with the latest token (`next_token`,
  then each draft); `eh_proj(concat(rms_norm(E[ids], enorm), rms_norm(h, hnorm)))`,
  **embedding first**; `decoder_block_ref` with a causal mask at positions `j + 1 ..
  j + T`; `rms_norm(., post_norm)`; argmax of the shared head at the last row. Stage
  `j + 1` reads stage `j`'s output and the token it drafted. `n_draft` stages run, one
  per layer present (`KeyError` beyond that).
- Returns `draft_tokens` int64 `(n_draft,)` and `draft_logits` float32 `(n_draft, V)`
  (standard deviation about 2). The drafted token feeds the next stage, so a
  comparison of tokens needs prompts whose top two draft logits are clearly apart in
  every stage; check with the reference in all three frameworks.
- The block is the *kit* block: RoPE on adjacent pairs with `cfg.rope_base`, eps
  `cfg.norm_eps` (1e-5). It is not a Qwen3.5 block.

### BlockDenoiser

A seeded block text-diffusion model with a prefix cache, for `block-diffusion-decode`.

```python
from learnai.kit import BlockDenoiser

model = BlockDenoiser(ctx.fw, seed=3, sharpness=6.0)     # vocab_size=32, d_model=24,
cache = model.encode(prompt)                             #   eos_bias=0.0, max_context=512
logits = model.denoise(canvas, cache)                    # (L, vocab_size)
logits = model.denoise(canvas, cache, scaled_logits)     # with self-conditioning
cache = model.encode(final_canvas, cache)                # extend the prefix
check.cost(model.counters, "encoder_tokens", exactly=P + (canvases - 1) * L, unit="tokens")
```

- `encode(tokens, cache=None)` is causal and folds the tokens in one at a time, so
  the cache after a prefix is **bit-identical** however the prefix was split over
  calls. It returns a new immutable `DenoiserCache` (`cache.length`); the one passed
  in stays valid. Counts `encoder_calls` and `encoder_tokens`.
- `denoise(canvas, cache, self_cond_logits=None)`: `canvas` is `(L,)` token ids placed
  right after the cached prefix, any `L >= 1`. Position `i` reads the prefix, its own
  token, every other canvas token (attention without a mask: bidirectional), and
  `self_cond_logits[i]`. A pure function of its inputs. Counts `denoise_calls` and
  `denoise_tokens`. The cache must not be empty (encode the prompt first).
- `self_cond_logits` is `(L, V)` float: pass **the temperature-scaled logits the
  previous step sampled from**, as the shipped loop does (the model takes their
  softmax). `None` on a canvas's first step.
- The logits are `sharpness * votes + eos_bias on eos_id`; a canvas token adds
  `OWN_VOTE` (0.5) to its own vote and the self-conditioning adds `SELF_COND_VOTE`
  (2.0) times its probabilities. Measured over 8 seeds with 3 canvases of 8, at most
  12 steps, temperatures 1.0 down to 0.2, entropy bound 1.0, a stop after 2 stable
  steps with mean entropy below 0.5, and `eos_bias=-100`. The last four columns count
  the seeds on which a mutant of the spec produced other tokens than the correct loop:

  | `sharpness` | steps per canvas | no self-conditioning | sampled canvas appended | noisy canvas encoded | stop on stability alone |
  |---|---|---|---|---|---|
  | 0.5 | 12, always (mean entropy stays above 0.55) | 8 | 8 | 8 | 2 |
  | 1 | 9 to 12 | 8 | 6 | 4 | 7 |
  | 1.5 | 4 to 11 | 8 | 7 | 5 | 8 |
  | 2 | 3 to 10 | 8 | 5 | 3 | 4 |
  | 3 | 3 to 6 | 8 | 5 | 1 | 0 |
  | 4 | 3 to 6 | 8 | 0 | 0 | 0 |
  | 6, 8 | 3 or 4 (3 is the earliest the rule allows) | 8 | 0 | 0 | 0 |

  So "stops early" fixtures use 6 or more, "never converges" uses 0.5, and the cases
  that must catch the sampled-canvas, noisy-canvas, and stability-only mutants use 1
  to 1.5 (they are invisible at 4 and above, where every position is accepted and
  sampled as its argmax by the time a canvas stops). A smaller entropy bound leaves
  more positions unaccepted: at bound 0.1 the noisy-canvas mutant differed on 7 and 8
  of 8 seeds at sharpness 1 and 1.5. These counts are for one configuration; confirm
  each mutant on your own fixture.
- `eos_bias` is added after the scaling, in logit units. `-100` never ends a text,
  `+100` ends it at the first position of the first canvas; 3 at `sharpness=3` ended
  some of six seeds' texts partway, at different positions. `model.eos_id` is 1 and
  `model.pad_id` 0.
- **Exact tokens across frameworks.** The model's arithmetic is numpy, so its logits
  are the same bits for the same inputs. What can differ between frameworks is the
  solution's own arithmetic: the division by the temperature (which also feeds back
  as self-conditioning), the softmax and cumulative sum of the sampling, and the
  entropies inside `eb_accept`. A draw within 1e-6 of a CDF boundary or two entropies
  within 1e-6 at the cut would flip a token. `learnai validate` runs all three
  references on every case, so a fixture that is not stable shows up there; replace
  it.
- The kit has no oracle for the loop itself: the problem's reference is the loop. A
  complete one (numpy sampling, 30 lines) is `diffusion_generate` in
  `python/tests/test_kit_m6_models.py`, including the rule that the last canvas is
  never encoded.
- It is a toy with the interface and the qualitative behavior of DiffusionGemma, not a
  port: nothing in it was compared with the shipped denoiser. `ops.eb_accept` was.

### Guarded, guard, check_guard

A mixin that makes a kit object safe to hand to a solution whose work is counted
(`python/src/learnai/kit/guarded.py`). Used by every problem that hands the solution
an object whose count is judged: 51 problems as of 2026-11-21 (`grep -l "guard(ctx"
problems/*/*/tests.py`, plus `mmap-lazy-loader`, which arms it as `guard(self.ctx)`).
Short ones to copy from: `gathered-sparse-attention` (a store), `cached-greedy-decode`
(a `TinyLM`), `moe-sparse-dispatch` (`.of` for an object a factory built),
`mlx-compile-step` (an `mlx.nn.Module`), `tool-call-loop` (async methods),
`block-diffusion-decode` (an allowlist and a rule of the test's own).

```python
from learnai.kit import BlockDenoiser, CountingKV, DeltaMemory, Guarded, check_guard, guard

class Store(Guarded, CountingKV):            # Guarded first
    label = "store"

class Memory(Guarded, DeltaMemory):
    label = "memory"
    refused = Guarded.refused | {"peek"}     # a test-only method on top of the defaults

class Model(Guarded, BlockDenoiser):
    label = "model"
    allowed = frozenset({"encode", "denoise", "vocab_size", "eos_id", "pad_id"})

guard(ctx)                                   # arm for this run's solution file
got = ctx.call(FN, q, Store(ctx.fw, k, v), idx)
check_guard()                                # the first refused access fails the case
```

- **What the solution may read.** By default every public attribute except the names
  in `refused` (`counters`, `log`, `with_framework`). Private names (`_x`) and the
  object's machinery (`guarded.MACHINERY`: `__dict__`, `__init__`,
  `__getattribute__`, `__setattr__`, `__delattr__`) are always refused: each of the
  five is a way around the rules (`store.__init__(...)` starts the counters over,
  `store.__getattribute__("counters")` would read through the kit). When the public
  settings would let a solution build a twin (`type(lm)(lm.fw, seed=lm.seed)`),
  refuse the setting (`refused = Guarded.refused | {"seed"}`) or set `allowed`, which
  turns the rule into an allowlist, as `block-diffusion-decode` does. An allowlist
  also refuses `lm.__class__`, which `isinstance(lm, ...)` can ask for: list
  `"__class__"` unless you mean that.
- **What it may never do.** Set or delete an attribute, or copy the object (`copy`,
  `deepcopy`, `pickle`).
- **No counting object of its own.** While the guard is armed, a `Counters` made by
  the solution, or by kit code the solution called some other way than through a
  guarded object, is refused: `` the solution may not build a counting object of the
  kit in this problem: the work is counted on the object it was given ``. Every kit
  object that counts work makes its `Counters` when it is built, so this stops the
  twin whose settings were guessed (`TinyLM(lm.fw, seed=s)` for every `s`), a second
  `MLXTinyLM` filled with the first one's parameters, and `with_framework` borrowed
  from the class, without a `forbidden` entry. One built while the solution was
  imported, before the guard was first armed, is remembered and fails every case,
  and a refused one is remembered for the calls that follow. Test code builds what it likes, except
  in a callback that the solution calls directly (make it a method of a guarded
  object). A problem whose solution is meant to build such objects (it compiles its
  own `CharDFA`, say) arms the guard with `guard(ctx, may_build=True)`.
- **Messages**: `` `store.counters` may not be
  used in this problem ``, `` `store.plain` may not be set in this problem ``,
  `` `model` may not be copied in this problem ``, and, for a borrowed method (next
  point), `` `model._w` may not be read outside the model's own methods ``. `label`
  is the argument's name in the statement.
- **Who is asking is read from the call stack**, as for `forbidden`. Walking out from
  the access, the first frame of the solution file or of a call made through the
  object decides. Frames of `learnai` and of the file that called `guard(ctx)` (your
  `tests.py`) are trusted; frames of anything else (`operator`, a string given to
  `eval`) act for their caller. So `getattr(store, "coun" + "ters")` and
  `eval("store.counters")` are refused like `store.counters`.
- **Kit and test code is not restricted**, with one exception that closes a hole.
  While the solution is running, the restricted names can be read, and attributes
  set, only inside a call the solution made through the object (`store.gather(...)`,
  a property). The kit's methods therefore work, and borrowing one as a plain
  function does not: `CountingKV.reference(store)` and `TinyDSA._index_key(dsa, ...)`
  fail at the first private attribute they read, and `TinyLM.__init__(lm, lm.fw)` at
  the first attribute it sets (`` `lm.fw` may not be set outside the lm's own
  methods ``). Code that the solution has such a call run for it is not part of the
  call either: a callback, or the `__index__` or `__array__` of an argument. Three
  consequences for test code: a callback of yours that the solution reaches some
  other way than through the guarded object may not read restricted names while the
  solution runs; such a callback cannot build a guarded object either (make the
  factory a method of a guarded object); and a special method outside the list of
  the next point (`__lt__`, say) that reads private state is refused during the
  solution's call.
- **Special methods are calls made through the object.** Python calls `lm(tokens)`,
  `len(file)`, `file[a:b]`, and `with file:` on the class, without asking for an
  attribute. `Guarded` wraps the special methods a guarded class defines or inherits
  (`guarded.SPECIAL`: `__call__`, `__len__`, `__length_hint__`, `__bool__`,
  `__contains__`, `__getitem__`, `__setitem__`, `__delitem__`, `__iter__`,
  `__next__`, `__enter__`, `__exit__`, the four `__a...__` names of async
  iteration and `async with`, `__buffer__`, `__eq__`, `__ne__`, `__hash__`) when
  the class is made, so `class LM(Guarded,
  TinyLM)` and `class Experts(Guarded, CountingExperts)` are called as before, and
  the base class's method borrowed as a plain function (`TinyLM.__call__(lm, ...)`)
  is refused at its first private read. `__repr__` is exempt too: `print(store)` is
  a learner debugging. It also shows what the kit class's `__repr__` shows: override
  it in the guarded class if that includes a seed.
- **Refusals are recorded.** The access raises `CheckFailure`, and `check_guard()`
  raises the first one again, so a solution that catches the exception and falls
  back to honest code still fails. Call `guard(ctx)` and `check_guard()` around
  every call of the solution, also in a case that calls `ctx.solution.<fn>`
  directly. A subclass adds a rule of its own with `self.violation(message)` (the
  cache-ownership check of `block-diffusion-decode`).
- **Introspection.** A guarded object's attributes are not stored in the instance but
  in a table in the kit module, so `object.__getattribute__(store, "counters")`,
  `object.__getattribute__(store, "__dict__")`, and a base class's
  `__getattribute__` find nothing (`AttributeError`), and writes made that way are
  never read. `vars(store)` is refused like `store.__dict__`. For test code,
  `store.__dict__` and `vars(store)` are that table. A kit method that returns a new
  version of its object must build it with `object.__new__(type(self))`, as
  `CountingKV.append` and `DeltaMemory.update` do; a twin from `with_framework` is a
  plain, unguarded object, for tests.
- **Coroutines and generators.** A method that is `async def` or a generator does its
  work after the call returned, when it is awaited or iterated. `Guarded` wraps such
  a method in a coroutine or generator of its own, so `await model.acomplete(...)`,
  `await tool(city="Paris")`, `await engine.step(...)`, and a task made from one of
  them run as calls through the object. One limit, of the same kind as threads: kit
  code that the event loop runs from a callback has no solution frame above it and
  is trusted.
- **An `mlx.nn.Module`.** `class LM(Guarded, MLXTinyLM)` works: an attribute is set
  by the kit class's own rule first, so arrays and submodules stay items of the
  module, where `parameters()`, `update`, `nn.quantize`, and `mx.compile(...,
  inputs=lm.state)` look for them, and everything else (the counters, the sizes)
  goes to the table. Two consequences. A record that your own subclass keeps as a
  list, tuple, or dict (`self.traced_lengths = []`) is a module item too
  (`model["traced_lengths"]`, `model.state`), outside the guard: keep it in a small
  holder object, as `mlx-compile-shapes` does. And the parameters are public by
  design: a second `MLXTinyLM` filled with them is refused by the rule against
  building counting objects, but a forward pass written by hand from the public
  submodules is not counted by anything.
- **Cost, and two collisions.** While the solution runs, every restricted read by kit
  code walks the call stack. That is nothing for a model called a few dozen times and
  too much for an engine whose own bookkeeping reads its attributes several hundred
  thousand times in a run (`SimEngine` under a long workload took five times as
  long): hand over a small guarded object that passes the calls on to the plain one,
  as `Handed` in `capstone-continuous-batching-engine` does. A kit class that has an
  attribute called `label`, `refused`, or `allowed` of its own (`TransferSpy.label`)
  cannot be mixed with `Guarded` directly, for the same reason: put a guarded object
  in front of it (`Hands` in `cuda-pinned-transfer`).
- **An object a factory built.** `Experts.of(moe.experts(1))` and `LM.of(draft)` (for
  `draft, target = draft_target_pair(...)`) are guarded objects with the attributes
  of the plain one. The attributes are shared, so the test can read the counters
  from either.
- **It is a guard against taking the shortcut, not a sandbox.** The solution runs in
  the judge's process. What still gets through: importing `learnai.kit.guarded` and
  reading its table or replacing its functions, replacing a method on the guarded
  class or `sys._getframe`, walking the garbage collector's object graph
  (`gc.get_referrers`), and another thread. What it stops is plain and computed
  attribute access, introspection of the instance, copies, and borrowed methods: the
  cheats the reviews of the model-architecture problems found. The tests still have
  to make sure the answer is right.

### DeltaMemory

The memory of a gated delta layer behind two counted calls, for `gated-delta-chunked`:
a token-by-token loop returns the numbers of the chunked form, and needs one read per
token however it is written or compiled.

```python
from learnai.kit import DeltaMemory

memory = DeltaMemory(ctx.fw, state, max_rows=2 * chunk_size)   # state (H, Dv, Dk)
out = memory.read(rows)                          # rows (H, R, Dk) -> (H, R, Dv): rows @ S^T
memory = memory.update(decay, values, keys)      # decay * S + values^T keys; a new memory
check.cost(memory.counters, "reads", at_most=chunks, unit="calls")
```

- `read` and `update` take `1 <= R <= max_rows` rows and count **one call whatever
  `R`** (`counters.reads`, `counters.writes`); a wrong shape or too many rows raises
  `ValueError` and counts nothing. `update(decay, values, keys)` takes `decay`
  `(H,)`, `values` `(H, R, Dv)`, `keys` `(H, R, Dk)`, leaves the memory it was called
  on unchanged, and returns one that **shares its counters**, so a test reads the
  total from the memory it passed in.
- Choose `max_rows` small against `Dk` in at least one case: with keys of width 64
  and a limit of 6 reads of 8 rows, the memory cannot be read out whole and
  processed elsewhere.
- `shape`, `max_rows`, and `fw` are public. `peek()` is the memory as a numpy copy,
  not counted, for tests; `with_framework(fw)` is a twin with fresh counters.
- Hand it over guarded, with `peek` refused (the `Memory` class in the example
  above). The counters, the array, `peek`, and `with_framework` are then out of the
  solution's reach without a `forbidden_source` entry.

### Checked against the shipped code (milestone 6)

The milestone 6 objects mirror `transformers` 5.23.0 and `mlx-lm` 0.32.2. Each was
checked on 2026-11-21 by building the shipped module with a tiny configuration,
copying the kit's seeded weights into it, and comparing outputs. The checks are unit
tests (`python/tests/test_kit_m6_shipped.py`, skipped when the library is missing);
running that file as a script prints the differences again. No weights were
downloaded. "Diff" is the largest absolute difference.

| kit | shipped | what was compared, on what | diff |
|---|---|---|---|
| `ops.l2norm`, `ops.causal_conv1d` | `l2norm`, `causal_conv1d_fn`, `causal_conv1d_update` (tf `qwen3_5`) | `(3, 5, 12)`; `(2, 9, 10)` with kernel 4: one prefill call, then 9 one-token calls with the saved state | below 1e-6; below 1e-5; states equal |
| `ops.partial_rope` | `Qwen3_5TextRotaryEmbedding` + `apply_rotary_pos_emb` (tf) | queries and keys `(2, 3, 7, D)` for `(D, factor, theta)` = `(16, 0.25, 1e7)`, `(32, 0.5, 1e4)`, `(8, 1.0, 100)`, positions up to 202 | below 2e-5 |
| `ops.partial_rope` | the `partial-rope` problem's three references | `(2, 7, D)` for four `(D, factor, base)`, positions up to 1000 | below 1e-6 in the same framework |
| `ops.gated_delta_layer`, `gdn_layer_ref` | `Qwen3_5GatedDeltaNet` (tf), **chunked path** (one call, no cache; asserted that `torch_chunk_gated_delta_rule` ran) | `x` `(2, T, D)` for `T` = 1, 5, 64, 70, 150 (chunk size 64, so padding and three chunks) and head layouts 2 key/4 value, 3/3, 1/4 with kernel 2 | 3.6e-6 |
| the same | `Qwen3_5GatedDeltaNet` (tf), **recurrent path** (prefill 70 with a `DynamicCache`, then 12 one-token calls; asserted that `torch_recurrent_gated_delta_rule` ran 12 times) | outputs; final memory against theirs transposed; conv state | 1.3e-6; 2.5e-7; 0 |
| `ops.gated_delta_layer` in torch, JAX, MLX | the same module | `x` `(70, 32)` | below 2e-4 (asserted) |
| `hybrid_forward_ref`, `HybridConfig.layers` | `Qwen3_5ForCausalLM` (tf) with 4 layers (3 linear + 1 full), 8 layers (factor 0.5, theta 1e7), and an explicit mixed `layer_types` of 5; norm gains stored as `gain - 1` | logits for `T` = 6, 70, 150 in one pass, and prefill in chunks then token by token with a `DynamicCache` on their side and the kit's state on ours; cached keys, values, memories | logits 2.3e-4 at magnitude 9 (4.8e-5 for the default config); keys 1.3e-4; argmax equal at every position |
| the same | `mlx_lm.models.qwen3_5.TextModel` with its `make_cache()` | logits for `T` = 6 and 70, one pass and chunks, 4 and 8 layers | 1.4e-4 |
| `TinyDSA` index scores and top-k, `index_types` | `GlmMoeDsaIndexer` (tf), `GlmMoeDsaConfig.indexer_types` | the `(24, 24)` scores handed to `topk`, for three seeds and layers; the selected sets for `k` = 1, 4, 7, 24, 34 after intersecting theirs with the causal mask; the same with their `DynamicCache` indexer cache over chunks of 6, 1, 1, 8, 8 | scores 6e-7; sets equal in every row without a tie at the cut (353 of 360 rows; in the other 7 the selected scores are equal); cached keys 5e-7 |
| `mtp_draft_ref`, `mtp_params` | `MtpLayer` / `MtpModel.forward` (tf) on a `LlamaForCausalLM` holding the kit decoder (rows of `q_proj`, `k_proj` reordered for Llama's half-split RoPE) | draft tokens and logits for `(T, stages)` = `(1, 1)`, `(5, 2)`, `(12, 3)`, `(20, 3)`, without a cache and with an `MtpCache`; also that `mtp_hidden_states_first` gives other logits | tokens equal; logits 2.7e-6 |
| `decoder_forward_ref`, `decoder_hidden_ref` | the same `LlamaForCausalLM` | logits and final-normed hidden states, `T` up to 20 | 3.7e-6; 2.3e-6 |
| `ops.eb_accept` | `EntropyBoundSampler.accept_canvas` (tf `diffusion_gemma`) | the accepted mask for logits `(3, 16, 40)`, 4 seeds, 7 bounds from 1e-6 to 1e3; and in torch, JAX, MLX on `(12, 24)` | 0 positions differ |

Facts these checks turned up that the specs did not have:

- `transformers` keeps **`K` rows** of convolution state per layer (`conv_kernel`
  projected inputs), one more than the next token's convolution reads; `mlx-lm` and
  the kit keep `K - 1`. The last `K - 1` of theirs equal the kit's. A real-model
  problem that compares states must say which.
- `transformers` stores the delta memory as `(heads, d_key, d_value)`, the transpose
  of the kit's and `mlx-lm`'s `(heads, d_value, d_key)`.
- `GlmMoeDsaIndexer` applies its rotation in a form that also reorders the rotated
  channels (evens, then odds) of queries and keys alike, so its cached indexer keys
  are a permutation of `layer_norm(wk(x))` even at position 0.
- `torch.topk` does not promise an order among equal scores, and the ReLU makes equal
  (zero) scores common; "ties to the lowest index" is the kit's rule, not the
  shipped one.
- What the generator hands `MtpModel` as `hidden_states[-1]` is the final-normed
  state (checked on Llama: equal to `last_hidden_state` bit for bit).

**Not verified against shipped code:** `BlockDenoiser` (a toy, compared with nothing);
`TinyDSA`'s main attention, output projection, and stack (plain attention over the
selected rows; the shipped layer is latent attention inside a full decoder block, and
only its indexer was run); cross-layer sharing as executed by `GlmMoeDsaModel` (only
the `indexer_types` rule was compared); `gdn_gates_ref` and `gdn_step_ref` on their own
(they are covered through the layer); the MTP module of a Qwen3.5 or GLM checkpoint
(the check used a Llama decoder, whose block is the kit's block; no shipped `mtp.*`
weights were loaded); anything on real weights (batch F); `ops.softplus` beyond the
float64 formula.

### SimModel and Counters

`SimModel` is described in section 5.6. `Counters("a", "b")` is the counter object for
new kit objects: `counters.add("a", n)`, attribute access, `snapshot()`, `reset()`.

### Real models, MLX tools, CountingFile

The kit objects added for milestone 3 are documented where they are used:
`MLXTinyLM`, `EvalSpy`, `TraceCounter`, and `mlx_memory` in section 13; `RealModel`,
`RealLM`, `RealConfig`, `Weights`, `SafetensorsFile`, `ByteLevelBPE`, `dequantize`,
`unpack_codes`, `bf16_to_f32`, `CountingFile`, and the house models `SMALL` and
`LARGE_8BIT` in section 15. Milestone 4 added `pending_ops`, `watch_load`, and
`LazyClock` (section 13), `logit_kl` (section 15), and the CUDA-on-CPU tools
`SyncSpy`, `DeviceSpy`, `FakeDevice`, `RealDevice`, `OOMModel`, `FakeCudaGraph`,
`CompileCounter`, `TransferSpy`, and `tritonshim` (section 18). Milestone 6 added the
`ops` functions `softplus`, `l2norm`, `causal_conv1d`, `partial_rope`,
`gated_delta_layer`, and `eb_accept`, and `GDNConfig`, `HybridConfig`, `CountingKV`,
`TinyDSA`, the MTP parameters, `BlockDenoiser`, `DeltaMemory`, and `Guarded` with
`guard` and `check_guard`, all in this section, and for the real hybrid model
`RealHybrid`, `RealHybridLM`, `checkpoint_layout`, `hybrid_block_ref`, and the asset
`QWEN35` (section 15).

**What milestone 5 left to problem-local test code**, after reading the specifications
of its 90 problems for needs that two or more share: LRU and TTL bookkeeping (it is
what `semantic-cache`, `block-hash-prefix-cache`, `radix-prefix-cache`, and
`kv-tiered-offload` ask the learner to write, and each oracle is that problem's own
reference); brute-force oracles for small inputs (`unigram-viterbi`,
`bit-allocation`, `radix-prefix-cache`); the checkpoint store of
`hybrid-state-checkpoints`; pipeline stages with service times
(`pipeline-parallel-schedule`, one user); attention-weight and Hessian fixtures of
the KV-compression and quantization problems (each has its own shapes); and JSON
Schema fixtures. Shared helpers that were added instead: `SimRequest.tokens`,
`workloads.phases`, `workloads.shared_prefixes`, `workloads.token_sequences`, the
`vectors` module, and `ScriptedTool`.

### Adding a kit object

Add it under `python/src/learnai/kit/`, export it from `learnai/kit/__init__.py`,
write tests under `python/tests/`, and document it here. Requirements: deterministic
(seeded), implemented once (numpy or plain Python), speaking the solution's framework
at its boundary through an adapter passed as `fw`, functional (state in, state out), a
`counters` attribute if it can measure work, and a `with_framework(fw)` method that
returns a twin with fresh counters. A method that returns a new version of its object
builds it with `object.__new__(type(self))`, so that a guarded object stays guarded
(see `Guarded` above). Write it for a reader: learners will open the file.

## 8. Statement house style

`statement.md` is Markdown with KaTeX (`$...$`, `$$...$$`) and fenced code. The title
comes from `problem.toml`; do not repeat it as a heading. Use these sections, in this
order:

1. **Opening paragraph** (no heading). One or two sentences: the task, and the one
   thing that makes it non-trivial.
2. **`## Specification`**. The signature in a `python` code fence, exactly as in the
   starter. The definition, with math. Each parameter and the return value: meaning,
   and anything the function must handle (masked entries, empty input, ties). Tie
   breaking and every other choice that affects the output must be specified; if two
   reasonable implementations could disagree, the statement is not finished.
3. **`## Shapes`**. A table of name, shape, dtype for every tensor in and out. Omit for
   framework-free problems.
4. **`## Example`**. A worked example with concrete numbers, small enough to verify by
   hand, and the reasoning for the interesting part. The first public case must be this
   example.
5. **`## Constraints`**. Input ranges and sizes, the tolerance or exactness of the
   comparison, cost or speed limits with their reasoning, and what is not allowed.
6. **`## Why this matters`**. One short paragraph that connects the problem to real
   inference systems: where this code runs, and what goes wrong in production when it
   is wrong.

Style:

- Precise and plain. Say "returns a float32 tensor of shape `(B, V)`", not "returns the
  result".
- American English. Present tense. Address the reader as "you" only in instructions.
- State the rule of the exercise when a built-in would trivialize it ("do not call your
  framework's built-in `softmax`"), and make the harness enforce it with `forbidden`
  in `problem.toml` (section 4.1).
- Framework-neutral: describe tensors, not `torch.Tensor`s. Framework specifics belong
  in the starter's type hints.
- No hints in the statement. The statement specifies; `hints.md` helps.

### Markdown the app renders

Statements, hints, and editorials are rendered by the same component, which supports:

- GFM tables, `$...$` and `$$...$$` math (KaTeX), and fenced code with a language tag
  (`python`, `json`, `bash`). An untagged fence renders as a plain listing.
- `##` for sections and `### Example N` for worked examples. Do not start a file with a
  `#` heading; the app renders the title.
- Callouts, written as GitHub alerts. Three kinds exist: `NOTE`, `TIP`, and `WARNING`.
  The marker goes alone on the first line of a blockquote, and the text follows on the
  next lines:

  ```markdown
  > [!NOTE]
  > Masked entries are `-inf`, not a large negative number.

  > [!TIP]
  > Subtracting the maximum first keeps every exponent at or below zero.

  > [!WARNING]
  > With logits near `1e4`, `exp` overflows float32 and the result is NaN.
  ```

  Each renders as an indented block with a colored hairline and a small label ("Note",
  "Tip", "Warning"). Any other marker, or a blockquote without one, renders as an
  ordinary quotation. Use callouts sparingly: one or two per statement, for something
  the reader must not miss.

### Figures

A statement, hint, or editorial can carry figures. Put the file in the problem's
`assets/` directory and reference it from the Markdown, alone on its own paragraph:

```markdown
![Positions $p = 0, 1, 2, 3$ for $D = 4$. Each pair is a point in its own plane, turned counterclockwise by $p\,\theta_i$.](assets/pair-rotation.svg)
```

- The text in the brackets is the **caption**. The app sets it beneath the drawing
  after "Figure N." (figures are numbered in order within the document, so the prose
  can say "Figure 1"). Write one sentence that says what to see, two at most, not a
  title. It may contain `$...$` math, emphasis, and inline code, and must not
  contain a line break.
- The path is `assets/<file>`, relative to the problem directory. Only `.svg` and
  `.png` files are served, each at most 1 MiB, and a path may not leave `assets/`.
  Subdirectories are allowed.
- An image in the middle of a sentence is not a figure and is not resolved. Give every
  figure its own paragraph.
- Use a figure where a picture carries something the formulas do not: a geometric
  action, a memory layout, a timeline. `problems/transformer/rope/assets/pair-rotation.svg`
  is the model to copy.

There are four kinds of figure. Use the first that fits:

| kind | use it for | example |
|---|---|---|
| map | where the problem's piece sits in the whole model; the first figure, after the opening paragraph | `transformer/pre-norm-block`, `assets/map.svg` |
| mechanism | the inside of one piece: blocks, wires, and the shape of the tensor on each wire | `transformer/causal-self-attention`, `assets/attention-steps.svg` |
| stepped | state that changes over calls or steps | `kv-cache/kv-cache-append`, `assets/cache-steps.svg` |
| interactive | the few ideas a reader has to turn by hand; built in the app, not by authors | `sampling/sampler-chain`, `quantization/clipping-calibration` |

Anything else (a plot, a geometric construction) is an ordinary SVG under the same
rules.

**Text.** A figure is a drawing, not a second statement:

- A caption is one sentence, two at most, and at most 30 words. A step note is one
  sentence of at most 25 words. A `<desc>` is one or two sentences and at most 30
  words. Joining clauses to stay under a sentence count is not shortening: what the
  reader needs beyond that belongs in the statement's own paragraphs.
- Lettering inside a drawing is names, symbols, and shapes, never a sentence, and
  there is no column of notes beside the drawing. Keep the annotations that carry
  information: the tensor shape on a wire, the "This problem" marker on a map.
- Plain wording, as in the rest of the statement.

**Prefer SVG.** An SVG is inlined into the page, so it takes its colors and typefaces
from the app and follows the light and dark themes. A PNG is shown as it is and cannot
adapt: use one only for raster data (a spectrogram, a heat map), with a transparent
background and marks that read on both a near-white and a dark blue-gray page.

SVG rules:

1. **`viewBox`, no `width` or `height`.** Draw a full-width figure in a `viewBox` 480
   units wide, as tall as the content needs: `<svg xmlns="http://www.w3.org/2000/svg"
   viewBox="0 0 480 246">`. The app scales the drawing to the text measure, at most
   4/3 of its natural size, so a 480-unit drawing exactly fills the 640px measure and
   is shown at about 1:1 in the narrowest statement pane. A narrower `viewBox` gives a
   proportionally narrower figure, set flush left.
2. **Ink is `currentColor`.** Use `stroke="currentColor"` for lines. The default fill
   is also the text color, so text and solid shapes need no `fill`; an outline must say
   `fill="none"`. Never draw a background rectangle, and never write a fixed color
   (`#000`, `black`, `white`).
3. **Other colors come from the figure palette**, CSS variables defined by the design
   tokens. A presentation attribute cannot hold `var()`, so set them through `style`.
   Five are for lines and lettering:

   | variable | use |
   |---|---|
   | `--figure-ink` | the text color (the same as `currentColor`) |
   | `--figure-muted` | secondary labels, axis names |
   | `--figure-faint` | axes, grids, guides: lines that must recede |
   | `--figure-accent` | the first series or the thing the figure is about (temper blue) |
   | `--figure-straw` | a second series or an annotation (straw) |

   ```xml
   <line x1="20" y1="142" x2="228" y2="142" fill="none" style="stroke: var(--figure-faint)"/>
   <text x="230" y="87" font-size="13" style="fill: var(--figure-accent)">p = 1</text>
   ```

   Seven more are **role colors**, the fills of blocks. Each has one meaning in every
   figure of the catalog, so a block is orange because it is attention:

   | variable | fills |
   |---|---|
   | `--figure-attn` | attention |
   | `--figure-mlp` | a feed-forward layer |
   | `--figure-norm` | normalization, scaling |
   | `--figure-embed` | an embedding, a mask |
   | `--figure-linear` | a linear layer, a matrix product |
   | `--figure-softmax` | softmax, sampling |
   | `--figure-cache` | a cache, state kept between calls |

   `--figure-panel` is the ground behind a group of blocks. Role colors are fills
   only: outlines, arrows, and lettering stay `currentColor`. Do not use a role color
   for anything but its meaning, and leave a block that has no role unfilled
   (`fill="none"`). Dim a block that is not the subject with `fill-opacity` and
   `stroke-opacity`, never with another color.

   ```xml
   <rect x="74" y="82" width="104" height="22" rx="5" stroke="currentColor" stroke-width="1.5" style="fill: var(--figure-softmax)"/>
   ```

   Never let color carry a distinction alone: label the block or the series, or vary
   the dash pattern as well. Green, red, and amber mean pass, fail, and warning in the
   app; do not use them.
4. **Text is `font-size="13"`**, the size of the caption, in a drawing 480 units wide
   (scale that up in proportion for a wider `viewBox`), and never smaller. The app
   scales lettering with the drawing up to 14px and stops there while the shapes keep
   growing to 4/3, so larger text gains little, and a label that fits its box at 13
   units fits at every pane width. Text is set in the
   interface sans by default; for symbols that also appear in the formulas use
   `style="font-family: var(--figure-serif)"`, with `font-style="italic"` on a
   `<tspan>` for variables, and `var(--figure-mono)` for code. Do not name a font
   yourself and do not convert text to outlines. Use Unicode for Greek letters and
   subscripts (`θ₀`), and keep a label clear of every line it does not belong to.
5. **Strokes are 1 unit for axes and guides, 1.5 for the subject.** Draw arrowheads as
   small filled polygons; a `<marker>` does not take the color of the line that uses it
   in every web view.
6. **Shapes and text only.** The app removes everything else before the figure is
   shown: `<script>`, `<style>`, `<foreignObject>`, `<image>`, links, animation,
   filters, event attributes, `class`, and any reference outside the file (`href` and
   `url(...)` may only point at an `id` in the same file). Allowed elements: `svg`,
   `g`, `defs`, `title`, `desc`, `path`, `line`, `polyline`, `polygon`, `rect`,
   `circle`, `ellipse`, `text`, `tspan`, `marker`, `clipPath`, `mask`,
   `linearGradient`, `radialGradient`, `stop`, `symbol`, `use`. In `style`, only paint
   and text properties survive. A figure that relies on anything else will render
   without it.
7. **Add a `<title>` and a `<desc>`** that say what the figure shows, for readers who
   cannot see it: the `<desc>` in one or two sentences. The caption is the figure's
   accessible name in the app.
8. **Write the file by hand or from a small script, and keep it readable**: grouped,
   commented, coordinates rounded to one decimal. Do not commit an export from a
   drawing program with its metadata, embedded fonts, and inline styles.

Opened on its own, such a file shows only the `currentColor` parts, because the
palette variables exist only inside the app. To check a figure, open the problem in
the app (`pnpm dev:real`) and look at it in both themes, at the default window size and
with the statement pane dragged to its narrowest: every label must be legible and no
two may touch.

**Maps.** The map of a module is one drawing of the whole model, shown by each problem
with its own part at full strength, the rest dimmed, and a "This problem" marker. Do
not draw it by hand. It is described once in `tools/figures/maps/<track>.json`, and
the script writes the file into every problem the spec lists:

```sh
python3 tools/figures/maps.py            # write every map
python3 tools/figures/maps.py --check    # fail when a written file is out of date
```

To give a problem the map, add its id and the parts it highlights to the spec, run the
script, commit the written `assets/map.svg`, and reference it after the opening
paragraph. `tools/figures/README.md` has the spec format. The caption says which part
of the model is this problem's, in one sentence.

**Mechanism figures** are drawn by hand in the vocabulary of the map: a box 20 to 22
units high with `rx="5"` and a role fill, wires that run bottom to top and end in an
arrowhead, an addition as a circle with a plus. Write the shape of the tensor beside
each wire in `var(--figure-mono)`, with the statement's own symbols.

**Stepped figures.** One SVG holds every frame. Each frame is a group directly under
the root with the id `step-1`, `step-2`, and so on, numbered from 1 without a gap, and
its first child is a `<desc>` that is the note of the step:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 206">
  <title>...</title>
  <desc>...</desc>
  <!-- outside the step groups: drawn in every frame -->
  <text x="90" y="41.5" font-size="13" text-anchor="end">K, V</text>
  <g id="step-1">
    <desc>Call 1 appends three tokens to an empty cache, so the mask is the square causal triangle.</desc>
    <!-- the shapes of this frame -->
  </g>
  <g id="step-2">
    <desc>...</desc>
  </g>
</svg>
```

The app shows one frame at a time with Previous and Next, "Step 2 of 4", and the note
beneath the drawing. A note is plain text, one sentence, and it should say what
changed. Put whatever does not change (axes, row names, a key) outside the groups, and
keep every frame inside the same `viewBox`.

**Interactive figures** are components of the app, named in place of a file:

```markdown
![The dashed outline is the model's distribution and a filled bar is the probability the chain samples from.](interactive:sampler-order)
```

The figure is numbered and captioned like any other. There are five:

| name | what the reader turns | what it shows |
|---|---|---|
| `sampler-order` | top-k, top-p, temperature, and the order of the stages | which tokens the chain keeps in either order |
| `quant-grid` | bits, the clipping fraction, a symmetric or an affine grid | a sample of weights, the grid's levels, the clipped tails, and the clipping and rounding error |
| `quant-grid-affine` | the same | the same figure, opening on the affine grid at 8 bits |
| `rope-shift` | the query position, the key position, and one shift added to both | the rotated query and key in each pair's plane, and a score that the shift does not change |
| `draft-overlap` | the sharpness and the shift of the draft `q` against the target `p` | each token's bar split into the overlap `min(p, q)`, the residual, and the rejected draft mass; the acceptance rate and `R` |
| `batch-window` | `max_batch_size` and `max_wait_ms` | ten fixed arrivals on a time axis, the batches they fall into, each request's wait, the mean wait and the mean batch size |

A new one is a change to the app (`src/components/figures/registry.ts`), not something
a problem can carry, and a name the app does not have fails the front-end tests.

**Starters** contain the imports the learner will need, the exact signature with type
hints, a docstring restating argument and return shapes, and `raise NotImplementedError`.
For classes, give the constructor and method signatures. A starter must import
cleanly.

**References** are the solution you would want a learner to end up with: short,
idiomatic for the framework, with comments on the non-obvious steps. The three
framework references should read as translations of each other.

**`hints.md`**: three hints, separated by a line containing only `---`, in increasing
strength: the key observation as a question; the approach, stated concretely; the
formula or the key line. A learner who reads all three should be able to finish without
the editorial.

**`editorial.md`**: sections `## The idea`, `## Reference solution (PyTorch)` (or
Python), `## Common mistakes` (one bullet per mutant, at least), `## Complexity`, and
optionally `## In the wild`. Explain why the approach is correct, not only what it does.

## 9. Tolerances

`check.allclose` passes when `|got - expected| <= atol + rtol * |expected|` for every
element. Defaults: `rtol=1e-4`, `atol=1e-6`.

- float32 has about 7 significant digits. A correct implementation that orders its
  operations differently from the reference typically agrees to `1e-6` relative. The
  default leaves two orders of magnitude of headroom and still catches real bugs.
- Loosen `rtol` to `1e-3` for long accumulations (hundreds of terms, deep stacks of
  layers) and for results built from differences of large numbers. Loosen `atol` (to
  `1e-5` or `1e-4`) when outputs are differences of quantities of order 1 to 100, since
  cancellation leaves absolute, not relative, error.
- Do not loosen a tolerance to make a mutant-shaped answer pass or a flaky case green.
  If the three references disagree beyond the default, find out why first.
- Tiny expected values pass on `atol` alone. If the small values are the point (a
  probability of `1e-30`), test their logarithm instead.
- Infinities must match exactly; NaN matches only NaN. An unexpected NaN is always
  reported first.
- Tokens, indices, counts, and masks are compared exactly with `check.equal`. If
  float ties could change an argmax, make the test inputs tie-free or specify the tie
  break.
- State the tolerance in the statement's Constraints section.

## 10. Time budgets

A full submit should usually finish in a few seconds, wall clock, in every framework.
`learnai validate -v` prints, for each reference submit, the total duration and the time
spent in cases, and warns when the cases take more than 3 s.

- Starting the harness and importing torch (always needed, for the oracle) costs about
  2 s before the first case; JAX adds about 1 s. That is fixed, so the cases are what
  you control: aim for a total under one second in torch and MLX and under two in JAX.
- **JAX costs about 0.15 to 0.2 s for every new input shape.** JAX traces and compiles
  each operation the first time it sees a combination of input shapes and dtypes, and
  keeps the result for the rest of the process. The cost is per *new shape*, not per
  case and not per call: a second case that uses the shapes of the first pays nothing,
  and a property test over 25 random shapes pays 25 times, about 4 s, for work that
  takes torch a few milliseconds. So **share shapes across cases**: pick two or three
  shapes for the whole `tests.py` (module-level constants help), draw fresh *values*
  per trial but not fresh shapes, and when a property needs varied shapes, draw them
  from a short fixed list (`rng.choice([(4, 8), (2, 3, 8)])`) rather than from a range.
  `learnai validate -v` prints the time in cases per framework; when the JAX figure is ten
  times the torch one, count your distinct shapes.
- **The harness keeps JAX's compiled kernels on disk** (`<root>/.learnai/cache/jax`,
  JAX's persistent compilation cache), so that per-shape cost is paid by the first
  run of a problem on a machine and not by later ones. Measured on the four slowest
  JAX references, alone on the machine (first run, then later runs):
  `real-model-cached-generate` 17.4 s then 13.5 s, `quantize-real-weights` 9.9 s then
  4.2 s, `real-model-perplexity` 4.1 s then 2.9 s, `real-model-logits` 3.5 s then
  2.2 s. Budget for the *first* run: that is what a learner's first submit and a fresh
  checkout see. What the cache cannot remove is JAX's cost per operation once
  compiled: an un-jitted 28-layer forward pass is about a thousand dispatches, 0.1 s
  per decode step and 1 s per 768-token prefill on the small real model, which is
  where the rest of `real-model-cached-generate` goes (12 s since milestone 5, when
  its speed case stopped reading the 768-token prompt a second time). The remedy is
  not to make that reference faster than a learner's natural code: its decode step
  is the baseline of a 3x speed limit, and one written to avoid JAX's copies of
  transposed weights (0.074 s instead of 0.119 s per step, measured) or jitted
  (0.072 s) would push an honest eager solution toward the limit. A JAX real-model
  problem that must be quick has to do less: fewer steps, a shorter prompt.
  `LEARNAI_JAX_CACHE=0` turns the cache off (to measure a first run again);
  `learnai validate` times are also inflated by its four parallel runs, so time a single
  problem with `learnai submit`.
- **One-time library costs are not charged to a case.** The first operation on a
  `torch.device("meta")` tensor, the first operation under `FakeCudaGraph` capture,
  and the first `torch.compile` each make torch import `torch._dynamo` (about 2.5 s,
  half of it sympy), and the first compilation costs 0.7 s more. The harness stops
  the case's clock for exactly that: the import runs with the soft time limit
  suspended, the supervisor's deadline is moved, and the time is subtracted from the
  case's `duration_ms`. The run still takes those seconds of wall time, once, and
  only in a run that reaches the code that needs them. A kit object with a one-time
  cost of its own wraps it in `learnai.judge.warmup.once(name, fn)` (as
  `CompileCounter` does for the first compilation); a module whose first import is
  slow is added to `warmup.SLOW_IMPORTS`. Do not use this for anything the solution
  controls.
- Keep tensors small: tests check correctness, not scale. A few thousand elements is
  plenty except in speed cases.
- `time_limit_s` is per case; 10 is right for almost everything. A correct solution
  should use under 5 percent of it. Lower it (2 to 5) when a wrong solution's typical
  failure is an infinite loop.
- Statistical cases dominate their problem's budget: N calls into a tensor framework.
  Keep each statistical case under about 3 s in **every** framework, measured with
  `learnai submit <id> -f <framework> --reference` for each one, not only torch: the same
  cases of `speculative-sampling-step` take 0.15 s in torch, 1.3 s in JAX, and 2.4 s in
  MLX, because per-call overhead (dispatch in JAX, a GPU round trip in MLX whenever a
  value is read back) dominates tiny tensors. Measure first, then pick N; if the slowest framework cannot afford the N
  that the bound needs (section 5.3), make the per-call work smaller (a smaller
  vocabulary, fewer operations) rather than loosening `delta`.

How time limits work, so you can predict the learner's experience: when a case exceeds
its limit, the harness interrupts it, reports `timeout`, and continues with the next
case. If the solution cannot be interrupted (it is stuck inside native code, or it
swallows the interruption), a watchdog process kills the run two seconds later; the
stuck case is reported as `timeout` and the remaining cases as `skipped`.

## 11. Mutants

A mutant is a wrong solution, written in the oracle framework (`torch`; `python` for a
framework-free problem; `mlx` for an MLX-only problem), that the tests must reject. Mutants are how we know the tests test something. Line 1
is `# BUG: <what is wrong>`, one sentence precise enough that a reviewer can see the
bug in the code below it.

```python
# BUG: log_softmax is computed as log(softmax(x)); probabilities that underflow to 0 become -inf although their logarithm is representable.
import torch
...
```

A bug that leans on behavior a library leaves unspecified (the order `torch.topk` gives
equal scores) may not be a bug everywhere: where the library happens to do the right
thing, the mutant is a correct solution and no test can reject it. Say so on line 2,
`# PLATFORM: <where the bug does not show>`. Validation then reports such a mutant as
"NOT validated" on a machine that accepts it, instead of failing. Use it only for
that; every other mutant must be rejected on every platform.

What makes a good mutant:

- **It is a mistake people really make.** Start from the list of common mistakes in
  your editorial: the naive formula, the off-by-one, the wrong axis, the missing
  `keepdim`, the forgotten renormalization, the ignored cache, the stale mask, `>` for
  `>=`, the epsilon that hides a NaN.
- **It is nearly right.** A mutant that fails every case tells you nothing. The best
  mutants pass the worked example and the easy random case and fail only the case
  written for them. Each mutant should justify at least one specific case, and it is
  worth saying which in a comment in `tests.py`.
- **One bug per mutant.** Copy the reference and change one thing.
- **It covers one requirement of the statement.** Walk through the Specification and
  Constraints: for each sentence, is there a mutant that violates only that sentence?
  For non-numeric judges, include the mutant that is correct but too expensive, too
  slow, or too late: correct output with the cost, speed, or latency requirement
  violated.
- **It fails for the right reason.** Prefer mutants that produce a wrong answer over
  ones that crash. Avoid mutants that hang: they make validation slow.

- **One mutant takes the forbidden shortcut.** If the problem has a `forbidden` list
  (section 4.1), include a mutant that produces the right numbers by calling the
  built-in (`mutants/uses_builtin.py`). It is the test that the list works: it must be
  rejected, and `learnai submit <id> --mutant uses_builtin` must show the "is not allowed in
  this problem" message. Mutants are written in the oracle framework, so for a rule
  that only concerns another framework's built-in (`mlx.nn.RoPE`), the mutant borrows
  it from torch code.

Aim for 4 to 7 mutants per problem. When a real learner submission is wrongly accepted
later, it becomes a mutant, together with the case that rejects it.

## 12. Lint

`ruff check problems/<track>/<id>` must be clean; `problems/ruff.toml` holds the
settings (line length 100; starters may have unused imports). Two rules of ruff's
default set surprise authors of otherwise idiomatic code:

- **RUF007, successive pairs.** `zip(xs, xs[1:])` is flagged; write
  `itertools.pairwise(xs)`. This comes up in every tokenizer and merge loop.

  ```python
  from itertools import pairwise

  for left, right in pairwise(tokens): ...
  ```

- **TRY004, the exception type after a type check.** When the only thing an
  `isinstance` test decides is whether to raise, the exception must be `TypeError`.
  Raising another built-in exception there (`ValueError`, usually) is flagged, in the
  `if not isinstance(...)` form and in the `else` of an `if isinstance(...)`:

  ```python
  if not isinstance(max_tokens, int):
      raise TypeError("max_tokens must be an int")      # ValueError here is flagged
  ```

  So specify `TypeError` for wrongly typed arguments in the statement, and `ValueError`
  for values out of range. Two things are not flagged, and are the way out when the
  specification needs something else: a custom exception class
  (`raise BadRequest(...)`), and a condition that tests more than the type
  (`if isinstance(n, bool) or not isinstance(n, int) or n < 1: raise ValueError(...)`).

Fix the code rather than adding `# noqa`; a learner reads references and mutants as
examples of good style.

## 13. MLX-only problems

A problem about MLX itself (laziness, `mx.compile`, unified memory, Metal kernels) has
nothing to say in torch or JAX. Declare it for MLX alone:

```toml
frameworks = ["mlx"]
requires = "mlx-gpu"
```

```sh
learnai new mlx/mlx-lazy-eval-placement --frameworks mlx
```

What changes, compared with a three-framework problem:

- **The oracle is `reference/mlx.py`.** There is no torch reference. `ctx.expected`,
  `ctx.check`, and `ctx.oracle` use the MLX reference; `ctx.oracle_fw` is the MLX
  adapter; `ctx.baseline` and `ctx.reference` are the same module. The harness does not
  import torch for the run.
- **Mutants are written in MLX** (`mutants/<bug>.py` starts with `import mlx.core as
  mx`), since they are judged as MLX solutions.
- **The functional rule relaxes.** "State in, state out" exists for JAX. An MLX-only
  problem may specify classes with mutable state (`class KVCache` with
  `update_and_fetch`), generators, and in-place updates, as the curriculum's MLX track
  does. Randomness is still an input.
- **Numpy stays at the test boundary.** `ctx.call` hands the solution `mx.array`s and
  gives the test numpy arrays, which *evaluates* the result. That is what a numeric
  check wants and exactly what a test about laziness must avoid: there, call
  `ctx.solution.fn(...)` directly and inspect the `mx.array` it returns (the examples
  below do).
- A numeric check against a numpy or kit oracle is as good as one against the
  reference: use `check.allclose(ctx.call(...), numpy_oracle(...))` whenever the
  reference would otherwise be checked against itself.

`python/tests/fixtures/m3/problems/gpu/mlx-scale` is a complete miniature.

### MLXTinyLM

`TinyLM` computes in numpy, which is useless for problems about MLX. `MLXTinyLM` is
the same kind of model built from `mlx.nn`, so its outputs are lazy graphs and it can
be traced by `mx.compile`.

```python
from learnai.kit import MLXTinyLM

lm = MLXTinyLM(seed=0)                       # vocab 64, width 32, 2 layers, 4 heads, RoPE
logits, cache = lm(tokens)                   # tokens: (T,) or (B, T); nothing computed yet
logits, cache = lm(next_tokens, cache)       # continue; logits: (T, V) or (B, T, V)
mx.eval(logits, cache)                       # cache: a list of (keys, values) per layer
```

- Options: `MLXTinyLM(seed, *, vocab_size=64, d_model=32, n_layers=2, n_heads=4,
  n_kv_heads=None, rope=True, dtype=None)`. `n_kv_heads` below `n_heads` gives
  grouped-query attention; `rope=False` adds sinusoidal positions to the embeddings
  instead; `dtype=mx.float16` gives a half-precision model. The weights are a pure
  function of the seed and the sizes.
- It is an `nn.Module`: `lm.parameters()`, `nn.quantize(lm, ...)`, `tree_flatten` work.
- The cache is a plain list of `(keys, values)` tuples of shape `(B, n_kv_heads,
  length, head_dim)`: it can be passed to `mx.eval` and across an `mx.compile`
  boundary. It grows by concatenation, so its shapes change every step.
- **Fixed shapes, for compiled loops:** `state = lm.init_state(max_len, batch_size=1)`
  and `logits, state = lm.step(token, state)` with `token` of shape `(B,)` and
  `logits` of shape `(B, V)`. `state` is `{"keys": [...], "values": [...], "length":
  int32 scalar array}` with buffers of length `max_len` whose shapes never change, so
  a function built on `step` and compiled is traced once.
- **Ragged batches:** `lm(tokens, cache, mask=..., positions=...)`. `mask` is an
  additive mask broadcastable to `(B, 1, T, length + T)` that replaces the causal one;
  `positions` are the absolute positions for RoPE, `(T,)` or `(B, T)`. With left
  padding, the mask hides the pads and the positions start after them; then every row
  equals its single-sequence result.
- `lm.counters.forward_calls` counts runs of the model's **Python body** and
  `tokens_processed` the tokens those runs were given. In an ordinary loop that is the
  number of calls. Inside `mx.compile` the body runs only when the function is traced,
  so the same counter is the number of traces.

> [!NOTE]
> `mx.compile` retraces when an input's **dtype** changes as well as its shape.
> `mx.argmax` returns uint32, so a loop that starts from an int32 token and feeds the
> argmax back is traced twice. Either cast in the step (`.astype(mx.int32)`) or allow
> for it in the limit; say which in the statement.

### EvalSpy: where evaluation happens

```python
from learnai.kit import EvalSpy, MLXTinyLM

@case("one evaluation per generated token, and no hidden reads", public=True)
def evaluation_placement(ctx):
    lm = MLXTinyLM(seed=7)
    with EvalSpy() as spy:                        # watches the solution file
        tokens = ctx.solution.generate(lm, [1, 2, 3], 8)
    check.that(spy.implicit == 0,
               f"generate: {spy.implicit} implicit evaluations (.item(), print, np.array, ...)",
               spy.describe())
    check.at_most("evaluations", spy.evals, 8)
    check.equal([int(t) for t in tokens], expected_tokens, label="tokens")
```

`EvalSpy` records, in order, every evaluation made **by code in the solution file**
(pass `file=` to watch another file): `spy.events` is a list of `SpyEvent(kind, what,
line)`; `spy.evals`, `spy.async_evals`, and `spy.implicit` count the kinds `"eval"`
(`mx.eval`), `"async_eval"` (`mx.async_eval`), and `"read"`; `spy.kinds()` is the
sequence of kinds; `spy.describe()` is one line per event for a failure detail. It
observes calls through `sys.monitoring`, so nothing in MLX is patched, aliases
(`from mlx.core import eval`) are seen, and the kit's and MLX's own evaluations do not
count.

- A **read** is a call that has to compute an array to return: `x.item()`,
  `x.tolist()`, `np.array(x)`, `np.asarray(x)`, `float(x)`, `int(x)`, `bool(x)`,
  `str(x)`, `repr(x)`, `format(x)`, `print(x)`, `memoryview(x)`, with the array as the
  first argument.
- **Not seen**, because Python makes no call for them: `if x:` and `while x:` on an
  array, an array inside an f-string, and an array that is not the first argument
  (`print("loss", x)`). Do not write a mutant whose only fault is one of these.
- Order matters for streaming: with `mx.async_eval`, the kinds of a correct
  `stream_generate` read `["async_eval", "async_eval", "read", "async_eval", "read",
  ...]`: step `t + 1` is scheduled before token `t` is read. Assert on `spy.kinds()`.
- The test's own code is not watched: evaluate and convert freely after the `with`
  block. Call the solution directly inside it; `ctx.call` would add nothing to the
  count (the judge is not the solution file), but it does evaluate the result.

### TraceCounter: how often a compiled function is traced

```python
from learnai.kit import TraceCounter

tc = TraceCounter()
step = ctx.solution.make_step(lambda x: tc.mark(x) * 2)    # mark is the identity
for _ in range(50):
    out = step(mx.array(1.0))
mx.eval(out)
check.at_most("traces", tc.count, 1)
```

`mx.compile` runs the Python function only to trace it; later calls with the same
input shapes and dtypes replay the recorded graph. So the number of times a Python
body runs is the number of traces. `tc.mark(x)` returns `x` and counts the call;
`tc.wrap(fn)` returns `fn` with its runs counted; `tc.reset()` zeroes it. When the
solution compiles a function around a kit model, `lm.counters.forward_calls` is the
same measurement with no extra object:

```python
lm = MLXTinyLM(seed=1)
step = ctx.solution.make_step(lm)
token, state = mx.array([3], dtype=mx.int32), lm.init_state(max_len=64)
for _ in range(50):
    token, state = step(token, state)
    mx.eval(token, state)
check.cost(lm.counters, "forward_calls", at_most=1, metric="traces")
```

A solution that compiles inside the loop shows 50, one that lets the dtype drift
shows 2, one that concatenates the cache shows 50.

### Memory: `mlx_memory` and `check.memory`

```python
from learnai.kit import mlx_memory

# Rejects promotes: a float32 result is 4 MiB, the float16 one 2 MiB.
@case("a float16 input stays float16 and stays small", requires="mlx-gpu")
def half_precision(ctx):
    x = ctx.fw.tensor(ctx.randn(1024, 1024), dtype=np.float16)
    ctx.fw.sync(x)                           # allocate the input before measuring
    with mlx_memory() as mem:
        y = ctx.solution.scale(x, 2.0)
        ctx.fw.sync(y)                       # force the work inside the block
    check.memory(mem, at_most_mib=3.0)
```

`mlx_memory()` empties MLX's buffer cache (`mx.clear_cache`), resets the high-water
mark (`mx.reset_peak_memory`), and notes what is already in use
(`mx.get_active_memory`). After the block (and an `mx.synchronize`):

| attribute | meaning |
|---|---|
| `mem.peak_bytes`, `mem.peak_mib` | `mx.get_peak_memory()` minus what was in use on entry: the most the block held at once, above its inputs. |
| `mem.active_bytes`, `mem.active_mib` | `mx.get_active_memory()` minus the same: what the block left allocated (results still referenced). |
| `mem.cache_bytes` | `mx.get_cache_memory()`: freed buffers MLX keeps for reuse. |
| `mem.baseline_bytes` | what was in use on entry. |

`check.memory(mem, *, at_most_mib, what="peak", metric=None)` reports the value in MiB
as the metric `peak_memory` (or `active_memory` with `what="active"`, or `metric`)
with its limit, and fails beyond it. Rules that keep memory cases honest:

- **Force the work inside the block.** MLX is lazy: an unevaluated result has
  allocated nothing and measures zero. `ctx.fw.sync(result)` (or `ctx.call`, which
  converts to numpy) forces it. A case that forgets this passes everything.
- **Create inputs before the block** and evaluate them, so they are in the baseline.
- **Leave a margin on both sides.** Array sizes are exact, but the allocator rounds
  and kernels use scratch space. Size the test so the reference uses at most two
  thirds of the limit and the approach you want to rule out at least one and a half
  times it; a factor of two between them (float16 versus float32, in place versus a
  copy, quantized versus dequantized) is the natural gap. Never test for equality.
- Use megabytes, not kilobytes: below about 1 MiB the allocator's rounding is the
  signal.
- Memory cases are GPU cases: give them `requires="mlx-gpu"` (section 14) and list
  `cost` in `judges`.
- The names are those of MLX 0.32.4: `mx.get_peak_memory`, `mx.reset_peak_memory`,
  `mx.get_active_memory`, `mx.get_cache_memory`, `mx.clear_cache`, `mx.synchronize`
  (the same functions also exist under `mx.metal`; use the top-level names, which do
  not depend on the backend). After `mx.reset_peak_memory()` the peak reads 0 until
  the next allocation and then tracks total active memory, which is why the kit
  subtracts the baseline. Re-check when the pin moves.

### CountingFile: how much of a file was read

```python
from learnai.kit import CountingFile

@case("only the requested tensor is read", assets=["granite-4.2-3b-mlx-8bit"])
def reads_one_tensor(ctx):
    with CountingFile(ctx.asset("granite-4.2-3b-mlx-8bit", "model.safetensors")) as f:
        weight = ctx.solution.load_quantized(f, "model.layers.0.self_attn.k_proj")
        check.at_most("megabytes_read", f.counters.bytes_read / 1e6, 4.0, unit="MB")
```

`CountingFile(path, page_size=16384)` is a read-only file that counts what is read
from it, usable as a file (`read`, `seek`, `tell`, `readinto`) and as a mapping
(`len(f)`, `f[a:b]`). A slice is a zero-copy `memoryview` of the memory-mapped file
(`np.frombuffer(f[a:b], dtype)` wraps it without copying) and is charged as `b - a`
bytes. The view covers that range only: its `.obj` is opaque and does not lead back
to the mapping, so one counted slice does not hand over the file. `f.counters.bytes_read` and `f.counters.reads` are the totals;
`f.pages_touched` is the number of distinct pages the requested ranges cover. Counting
is by the ranges asked for, not by page faults the operating system saw, so it is
exact and identical on every machine: a loader that takes `f[:]` or `f.read()` and
picks out one tensor is charged for the whole file.

### `pending_ops`, `watch_load`, `LazyClock`

Three helpers that MLX problems kept rewriting:

```python
from learnai.kit import LazyClock, pending_ops, watch_load

check.that(pending_ops(model.parameters()) == 0, "load_into: the parameters are still lazy")

with watch_load() as loaded:                      # {name: array} of what mx.load returned
    ctx.solution.load_into(model, path)
check.that(pending_ops(loaded["unused.weight"]) > 0, "a tensor the module does not use was read")
```

- `pending_ops(*trees)` is the number of operations not yet computed behind the
  arrays in its arguments (arrays, or lists, tuples, and dicts of them); zero means
  evaluated. It counts nodes of `mx.export_to_dot`, so use it for "zero or not" and
  "at most a handful", never for an exact graph size.
- `watch_load()` wraps `mx.load` for the length of the block and collects the arrays
  it returned. `mx.load` of a safetensors file is lazy, so with the arrays in hand a
  test can tell which tensors were actually read.
- `LazyClock(slow=())` is a simulated clock for problems about *measuring* MLX code:
  `clock()` reads it like `time.perf_counter`, `clock.spend(seconds)` advances it at
  once (building a graph), and `clock.owe(array, seconds, group=None)` attaches a cost
  to a lazy array that the clock collects at the first reading after the array has
  been evaluated. An interval around an unevaluated graph therefore contains none of
  its work. `group` chains debts (a later debt of a group waits for the earlier
  ones); `slow[k]` multiplies the k-th group collected (warm-up). The existing
  `mlx-benchmark-harness` keeps its own copy in `tests.py`; new problems use this one.

### MLX behaviors to know (MLX 0.32.4)

Found while writing the MLX track. The first four were re-run for this list on an
M1 Max; the last two are as reported by the authors who hit them. Re-check when the
pin moves.

- **A boolean mask cannot be read, but it can be assigned.** `x[mask]` raises
  `ValueError: boolean indices are not yet supported`; `x[mask] = value` works. Use
  `mx.where(mask, a, b)` for reads; do not write a test whose reference needs
  `x[mask]`.
- **`mx.quantize` accepts only group sizes 32, 64, and 128** (16 and 256 raise
  "[quantize] The requested group size ... is not supported"). And for a group whose
  values are **all negative** (or all positive: zero is outside the group's range)
  the GPU and the CPU disagree: for one such group at 4 bits the GPU chose scale
  0.319 where the CPU chose 0.252, with a correspondingly larger error. A test that
  compares against `mx.quantize` should keep zero inside every group, or compare
  with a numpy oracle of the documented formula instead.
- **Slice assignment copies.** `b = a[1:3]; b[0] = 5` leaves `a` unchanged: an MLX
  slice is a new array, not a view. Only `a[...] = value` on the array itself (or on
  another name bound to the same array object) updates it. A problem about in-place
  cache updates must say which.
- **The fused attention `"causal"` mask is lower-right aligned.**
  `mx.fast.scaled_dot_product_attention(q, k, v, scale=..., mask="causal")` with `T`
  queries and `S > T` keys lets query `i` see keys `j <= i + (S - T)`: the queries
  are taken to be the *last* `T` positions, which is what decoding with a cache
  needs. It is not `j <= i`.
- **`mx.fast.metal_kernel`'s `elem_to_loc` misreads negative strides** (reported): an
  input that is a reversed view reaches the kernel with wrong element locations.
  Make such inputs contiguous first, and do not build a test on reversed views.
- **`atexit` handlers do not run in the harness.** The worker leaves with
  `os._exit` after the `end` event, so that framework teardown cannot delay or break
  the exit. Clean up with `ctx.defer(...)` (runs when the case ends), never `atexit`.
- **`mlx.nn.softplus` loses small results** (4.5427e-5 at -10, exact 4.5399e-5;
  relative error up to 39% near -16); `mx.logaddexp(x, 0)` does not. What this means
  for tolerances is in section 7 under [ops](#ops).

## 14. Hardware requirements and skipped cases

`requires` in `problem.toml` names the hardware a solution needs before it can be
accepted: `"cpu"`, `"mlx-gpu"` (an Apple GPU usable by MLX), or `"cuda"` (an NVIDIA GPU
usable by torch). Each case has a requirement of its own:

```python
@case("the mask is built without a host round trip", public=True)       # requires="cpu"
def cpu_logic(ctx): ...

@case("the result is on the CUDA device", requires="cuda")
def on_cuda(ctx):
    out = ctx.solution.scale(torch.ones(3), 2.0, torch.device("cuda"))
    check.that(out.device.type == "cuda", f"scale: the result is on {out.device.type}")
```

The rules:

1. A case runs when the machine has what the case requires. The default is `"cpu"`:
   the case runs everywhere. A case may require only `"cpu"` or the problem's own
   `requires`; anything else is a harness error (and fails validation).
2. A case whose hardware is absent is reported with status `skipped` and the message
   "skipped: needs an NVIDIA GPU (CUDA), which this machine does not have". It is not
   run, and neither is anything before it skipped on its account: the other cases run
   normally.
3. **A run with a skipped case is never `accepted`.** If something that ran failed,
   the verdict is the failure's, as always (`wrong_answer`, `runtime_error`,
   `time_limit`). If everything that ran passed, the verdict is `hardware_missing`.
4. If the problem's `requires` is not met, the verdict is `hardware_missing` even when
   no case was marked and every case passed (for example `learnai run`, which executes only
   the public cases, on a `cuda` problem whose public cases are all CPU cases). An
   `accepted` verdict therefore always means: every case ran, on the required
   hardware.

What a learner without the hardware sees, in the app and in `learnai submit`:

```
cuda-scale / torch / submit  python 3.15.2, device cpu, torch 2.15.0
  PASS  cpu        scales on the CPU  0.4 ms
  PASS  cpu_random random shapes on the CPU  0.2 ms
  SKIP  gpu_scale  scales on the GPU
      skipped: needs an NVIDIA GPU (CUDA), which this machine does not have
Not accepted: hardware missing  2/3 passed, 1 skipped in 1.21 s
  2 of 2 cases that could run passed; 1 of 3 cases were skipped because they need an
  NVIDIA GPU (CUDA), which this machine does not have. The solution can only be
  accepted on a machine that runs every case.
```

That is the intended presentation of "passed everything this machine can run": a
distinct, neutral outcome that says how many cases passed, how many were skipped and
why, and that the problem is not solved yet. It does not mark the problem as solved
and it is not shown as a failure of the learner's code.

How to split a problem:

- **CPU cases carry the logic and decide validation.** Everything about correctness
  that does not need the device goes in unmarked cases: they run for every learner
  and on the machine that validates the catalog. A CUDA problem's solution takes
  `device` as an argument and must work for `torch.device("cpu")`.
- **Marked cases check only what the hardware makes observable**: where a tensor
  lives, a speed or memory property of the GPU path, a Metal kernel. Keep them few.
- Every mutant should be caught by a CPU case if at all possible. A mutant that only
  a marked case catches is reported by `learnai validate` as "NOT validated" on a machine
  without the hardware (section 1), which is honest and unsatisfying. Section 18
  describes the kit that makes GPU mistakes visible on the CPU for exactly this.
- For the MLX track (`requires = "mlx-gpu"`), mark the cases that need the GPU:
  memory limits, speed, Metal kernels. Apple silicon always has the GPU, so in
  practice everything runs; the marks decide what happens elsewhere, and they tell
  the reader which cases are about the device.

To see how a problem behaves on a machine you do not have, override detection:

```sh
LEARNAI_HARDWARE=cpu learnai submit cuda-scale --reference     # pretend: no GPU of any kind
LEARNAI_HARDWARE=cpu,cuda learnai submit cuda-scale --reference   # pretend CUDA is there
```

`LEARNAI_HARDWARE` is a comma-separated list of what to treat as present (`cpu` is
always included). It changes which cases are skipped, nothing else: pretending CUDA
exists on a Mac makes the CUDA cases run and fail with torch's own error. Without the
variable, `mlx-gpu` means `mx.metal.is_available()` with a GPU default device, and
`cuda` means `torch.cuda.is_available()`.

`python/tests/fixtures/m3/problems/gpu/cuda-scale` is a complete miniature.

## 15. Model assets and real-model problems

Some problems run on the weights of a real open model. Weights are never committed:
a problem *declares* what it needs, the learner downloads it once with `learnai assets
fetch` (or the app's button), and it lives in `.learnai/models/` under the
repository root, which is gitignored.

### The house models

| | small | large (as problems load it) |
|---|---|---|
| asset name | `granite-4.0-350m` | `granite-4.2-3b-mlx-8bit` |
| repository | `ibm-granite/granite-4.0-350m` | `lmstudio-community/granite-4.2-3b-MLX-8bit` |
| pinned revision | `bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2` | `7c630b526561db6b4641767c54595fe494176695` |
| on disk | 712 MB (weights 704.8 MB, BF16) | 3.90 GB (weights 3.89 GB, 8-bit) |
| parameters | 352.4 M | 3.66 B |
| use | CPU tests in every framework | MLX-only problems |

Both are in the kit as `learnai.kit.SMALL` and `LARGE_8BIT` (`AssetSpec` objects),
and `learnai assets fetch <name>` knows them before any problem declares them, so you can
download first and write the problem second:

```sh
uv run --project python --extra models learnai assets fetch granite-4.0-350m
```

Use the house models unless a problem truly cannot. Every extra model is another
download for the learner. One other model is in the kit, for the three problems of the
`real-hybrid-model` module only: `learnai.kit.QWEN35` (`qwen3.5-0.8b`, 1.76 GB), described
in [The hybrid model](#the-hybrid-model-qwen35-08b-verified-2026-11-21). `learnai assets
fetch` knows it the same way (`learnai.kit.realmodel.KIT_ASSETS` holds all three).

### Declaring an asset

```toml
[[assets]]
name = "granite-4.0-350m"
repo = "ibm-granite/granite-4.0-350m"
revision = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
files = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "model.safetensors"]
size_mb = 712
```

Print the block for a house model instead of typing it:

```sh
uv run --project python python -c "from learnai.kit import SMALL; from learnai.kit.realmodel import asset_toml; print(asset_toml(SMALL))"
```

| key | meaning |
|---|---|
| `name` | The directory under `.learnai/models/` and the name used by `ctx.asset`, `@case(assets=...)`, and `learnai assets fetch`. Lowercase letters and digits with `.`, `_`, `-` between them. |
| `repo` | The Hugging Face repository id. |
| `revision` | The full 40-character commit hash. A branch name such as `main` is refused: it can move, and fixtures are only valid for the bytes they were made from. |
| `files` | File names or glob patterns inside the repository. Only `*.safetensors` and the tokenizer and config files can be assets (`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `vocab.json`, `merges.txt`, `chat_template.jinja`, `model.safetensors.index.json`). Nothing that executes or unpickles is ever downloaded: no `.bin`, `.pt`, `.pkl`, `.py`, even when a glob would match it. A pattern that selects nothing is an error. |
| `size_mb` | The approximate download size in megabytes, shown before downloading. |

One asset name means one repository and revision across the whole catalog (validation
checks it); `fetch` downloads the union of the `files` that the problems using a name
ask for. A problem may declare several assets.

### What the learner and the harness do

- `learnai assets list` shows every asset the catalog declares, whether it is present, its
  size, and which problems use it. `learnai assets fetch <name>...` (or `--all`) downloads.
  Fetching needs the `models` extra (`huggingface_hub`); nothing else in the harness
  does. A fetch is pinned to the declared revision, verifies each file's size and (for
  the weights) SHA-256, and can be repeated: files that are complete are not
  downloaded again, so an interrupted fetch continues with the file it was on, and
  fetching a present asset does nothing and needs no network.
- An asset is **present** when its directory holds a manifest for the declared
  repository and revision that covers the declared `files`, and every file has its
  recorded size. Anything else (nothing there, another revision, a missing or
  truncated file) counts as not downloaded.
- **A case whose asset is not present is skipped**, with the message "skipped: model
  asset `granite-4.0-350m` (ibm-granite/granite-4.0-350m, about 712 MB) is not
  downloaded; fetch it with `learnai assets fetch granite-4.0-350m`". The other cases run.
  If nothing that ran failed, the run ends with the verdict **`assets_missing`**
  (shown as "Not judged: model asset missing") and a message naming the asset and the
  command; it is never `accepted`. If every case needs the asset, nothing runs and the
  solution is not even imported.
- By default **every case needs every asset** of the problem. Mark the cases that do
  not with `assets=[]`, and a case that needs only one of several with
  `assets=["name"]`. Do this whenever a problem has cases on synthetic inputs: a
  learner can then work on those before downloading 700 MB.

```python
@case("a file written in the test", public=True, assets=[])
def synthetic(ctx):
    ...                                   # runs with nothing downloaded

@case("the real checkpoint: 226 tensors", public=True)
def real_checkpoint(ctx):
    path = ctx.asset("granite-4.0-350m", "model.safetensors")   # a pathlib.Path
    ...
```

`ctx.asset(name)` is the asset's directory, `ctx.asset(name, file)` the path of one
file in it. Inside a case that needs the asset it is always there (otherwise the case
would have been skipped). Asking for an asset the problem does not declare is an
error naming the declared ones.

To see what a learner without the download sees, point the harness at an empty
directory: `LEARNAI_MODELS_DIR=/nonexistent learnai submit <id> --reference`.
(`LEARNAI_MODELS_DIR` replaces `<root>/.learnai/models` everywhere.)

### The machine-readable forms (what the desktop app calls)

Both commands read the catalog under the content root (`LEARNAI_ROOT`, else the
working directory's repository).

`learnai assets list --json` prints one JSON document on stdout and exits 0. It needs no
extra and no network.

```json
{
  "models_dir": "/abs/learnai/.learnai/models",
  "assets": [
    {
      "name": "granite-4.0-350m",
      "repo": "ibm-granite/granite-4.0-350m",
      "revision": "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2",
      "files": ["config.json", "tokenizer.json", "model.safetensors"],
      "size_mb": 712,
      "state": "missing",
      "present": false,
      "reason": null,
      "path": "/abs/learnai/.learnai/models/granite-4.0-350m",
      "bytes_on_disk": 0,
      "problems": ["real-model-block", "real-model-logits"],
      "fetch_command": "learnai assets fetch granite-4.0-350m"
    }
  ],
  "issues": [{"problem": "some-problem", "message": "asset `x`: `revision` must be ..."}]
}
```

`state` is `present`, `missing` (nothing usable on disk), or `incomplete` (a directory
exists but it is another revision, lacks declared files, or a file is damaged; `reason`
says which). `present` is `state == "present"`. `size_mb` is the declared approximate
download size; `bytes_on_disk` is what is there now. `files` is the union of the
patterns of the problems in `problems`. Assets are sorted by name.

`learnai assets fetch <name>... --json` (or `--all --json`) writes one line per event to
stdout, each `@@WS@@ ` followed by a single-line JSON object, the framing of the
harness protocol; any other line is a log line. It needs `--extra models`. Exit code
0 when every asset ended with `ok: true`, 1 when one failed, 2 for a usage error (an
unknown name; nothing is printed to stdout then).

```jsonc
{"event":"asset_start","asset":"granite-4.0-350m","repo":"ibm-granite/granite-4.0-350m","revision":"bd8a...","path":"/abs/.../granite-4.0-350m","files":[{"path":"config.json","size":1764},{"path":"model.safetensors","size":704786224}],"total_bytes":711966212}
{"event":"asset_progress","asset":"granite-4.0-350m","file":"model.safetensors","file_bytes":150763002,"file_total":704786224,"done_bytes":150771331,"total_bytes":711966212}
{"event":"asset_file","asset":"granite-4.0-350m","file":"model.safetensors","size":704786224,"cached":false}
{"event":"asset_end","asset":"granite-4.0-350m","ok":true,"path":"/abs/.../granite-4.0-350m","bytes":711966212,"message":null}
```

- `asset_start`: once per asset, after the file list is known. Absent when the fetch
  fails before that (no network, `huggingface_hub` missing); then the only event is a
  failed `asset_end`.
- `asset_progress`: at most about four per second per file, plus one when a file
  completes. `done_bytes / total_bytes` is the fraction for the whole asset and never
  decreases; `file_bytes / file_total` is the current file.
- `asset_file`: once per file, in path order; `cached: true` means it was already
  complete and was not downloaded.
- `asset_end`: exactly once per asset, always last. On failure `ok` is false,
  `bytes` is 0, and `message` is one line saying what went wrong; running the command
  again resumes.

### Fixtures: committed reference outputs

Reference logits, token ids, and text for real-model problems come from an
established implementation, are generated once, and are committed as small files in
the problem's `fixtures/` directory, together with the script that made them:

```
problems/transformer/real-model-logits/
  fixtures/make.py          regenerates everything below; needs the `models` extra
  fixtures/logits.npz
  fixtures/prompts.json
```

`ctx.fixture(name)` loads one by extension: `.npz` gives a dict of numpy arrays,
`.npy` an array, `.json` the parsed JSON, `.txt` a string, anything else `bytes`.
Pickled data is never loaded. `ctx.fixture_path(name)` is the path.

```python
@case("matches transformers on the worked prompt", public=True)
def worked_prompt(ctx):
    model = RealModel(ctx.asset("granite-4.0-350m"))
    reference = ctx.fixture("logits.npz")
    ids = reference["ids_0"]
    logits = ctx.call("granite_logits", model.weights(), model.config, ids)
    check.equal(logits.argmax(-1), reference["argmax_0"], label="argmax tokens")
    check.allclose(logits[:, reference["columns"]], reference["logits_0"],
                   rtol=1e-3, atol=1e-3, label="logits")
```

Rules for fixtures:

- **Keep them small.** A full logits row of the house models is 100352 floats; store
  what the test compares: the argmax per position, the logits at a fixed set of
  columns, the top few of the last position, a few columns of a hidden state. The
  kit's own fixture for three prompts is 18 kB.
- **`fixtures/make.py` is the pattern in
  `python/tests/fixtures/realmodel/make.py`**: resolve the asset, run `transformers`
  (float32, CPU) and `tokenizers`, save with `np.savez_compressed`, and record the
  asset's revision in the file so a test can refuse a stale fixture. Run it with
  `uv run --project python --extra models python problems/<track>/<id>/fixtures/make.py`.
- Prompts are at most 16 tokens on the small model so a CPU forward pass stays well
  under a second (the kit's numpy reference takes about 0.1 s for 16 tokens after
  the weights are decoded, which costs about 1 s once per run).
- Fixtures for the large model come from `mlx-lm`. Note that `mlx_lm.load` runs the
  8-bit checkpoint in **bfloat16**; its logits differ from a float32 run of the same
  weights by up to about 0.4 (on logits of magnitude 25), enough to flip a near-tie.
  Call `model.set_dtype(mx.float32)` before generating fixtures (then a from-scratch
  float32 implementation agrees to about 1e-5), choose prompts whose greedy tokens
  have a clear margin, and say in the statement which dtype the fixture used.

### RealModel: config, tokenizer, weights

```python
from learnai.kit import RealModel

model = RealModel(ctx.asset("granite-4.0-350m"))
model.config            # RealConfig: the architecture, normalized (below)
model.config_json       # the raw config.json dict
model.tokenizer         # ByteLevelBPE built from tokenizer.json
model.tokenizer_json    # the parsed tokenizer.json (what a tokenizer problem hands over)
model.chat_template     # the Jinja template text, or None
model.checkpoint        # SafetensorsFile: the memory-mapped checkpoint
model.weights()         # name -> tensor, decoded lazily (below)
model.lm()              # RealLM: the reference forward pass, called like TinyLM
model.dequantized(name) # float32 weights of one layer of a quantized checkpoint
```

Nothing is read until used, and everything needs only numpy: the kit does not import
`transformers`, `tokenizers`, or `safetensors`, so real-model problems run without the
`models` extra once the asset is on disk.

**`model.weights()`** is a lazy mapping from checkpoint names
(`"model.layers.0.self_attn.q_proj.weight"`) to tensors. A tensor is decoded the
first time it is asked for (BF16 and F16 become float32; the packed uint32 words of a
quantized checkpoint stay uint32) and cached for the process, so the cases of a run
share it and a test that touches two layers pays for two layers. Pass it through
`ctx.call` and the solution receives tensors of its own framework, like any kit
object; `model.weights(ctx.fw)` binds it yourself. In a `python` problem the values
are numpy arrays. `weights.raw(name)` is the zero-copy view of the stored bytes (BF16
as uint16), for problems about the file format.

**`model.checkpoint`** is a `SafetensorsFile`: `names()`, `info(name)` (the header
entry: `dtype`, `shape`, `data_offsets`), `raw(name)`, `array(name)`, `metadata`,
`header`, `data_start`, `nbytes`, `n_parameters()`. Use it for oracles of the
`safetensors-parse` and sizing problems. `bf16_to_f32(bits)` decodes uint16 bit
patterns.

**`model.tokenizer`** is a `ByteLevelBPE`: `encode(text) -> list[int]` (nothing is
added around the text), `decode(ids, skip_special_tokens=False)`, `token_bytes(id)`,
`vocab` (token string to id), `ranks` (merge pair to rank), `added` and `specials`
(added-token text to id), `vocab_size`. Its ids equal those of the `tokenizers`
library (the judge's tests compare them on about 220 strings per model).

**`model.config`** is a `RealConfig` with the fields `model_type`, `n_layers`,
`d_model`, `n_heads`, `n_kv_heads`, `head_dim`, `d_ff`, `vocab_size`, `max_positions`,
`norm_eps`, `rope_theta`, `attention_scale`, `embedding_multiplier`,
`residual_multiplier`, `logits_scaling`, `tied_embeddings`, `fused_mlp`,
`quantization` (`(bits, group_size)` or `None`), `bos_id`, `eos_id`, `pad_id`. Hand it
to the solution as `cfg` when the problem is about the forward pass rather than about
reading `config.json`.

### What the house models are (verified)

Facts every real-model author needs; all were checked against the downloaded files
and against `transformers` 5.23 and `mlx-lm` 0.32.2, which load both natively.

| | `granite-4.0-350m` | `granite-4.2-3b-mlx-8bit` |
|---|---|---|
| `model_type` | `granitemoehybrid` (28 attention layers, no experts) | `granite` |
| layers, width | 28, 1024 | 40, 2560 |
| heads, KV heads, head dim | 16, 4, 64 | 40, 8, 64 |
| MLP | SwiGLU 2048, gate and up fused in `shared_mlp.input_linear` `[4096, 1024]` (gate rows first), `shared_mlp.output_linear` | SwiGLU 8192, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` |
| attention scale | 1/64 | 1/64 |
| embedding x, residual x, logits / | 12, 0.263, 4 | 1, 1, 1 |
| output head | tied to the embedding (no `lm_head.weight`) | separate `lm_head` |
| norm, positions | RMSNorm eps 1e-5 pre-norm; RoPE theta 1e7 | the same |
| tensors | 226, all BF16 | 927: packed U32 weights with BF16 scales and biases; BF16 norms |
| vocabulary | 100352 | 100352 |
| special tokens | `<|end_of_text|>` 100257 (BOS and EOS), `<|pad|>` 100256, roles `<|start_of_role|>` 100264 and `<|end_of_role|>` 100265 | `<|im_start|>` 100256, `<|im_end|>` 100257 (EOS), `<s>` 100283 (BOS), `<think>` 100274 |
| pre-tokenizer | `Split` with the cl100k pattern (digit runs cut in threes) | `ByteLevel` with the GPT-2 pattern |
| chat format | `<|start_of_role|>user<|end_of_role|>...<|end_of_text|>` | ChatML with `<think>`; `enable_thinking` defaults to true |

> [!WARNING]
> **The two models do not share a tokenizer.** They have the same 100000 merges and
> the same ordinary vocabulary, but different pre-tokenizers and different special
> tokens at the same ids (100257 is `<|end_of_text|>` in one and `<|im_end|>` in the
> other), so the same text gives
> different ids ("In 2026" is `[644, 220, 2366, 21]` for the small model and
> `[644, 220, 508, 1627]` for the large). A tokenizer or chat-template problem is
> about one named model; never reuse ids across the two.

**What a from-scratch forward pass must do to match the small model's logits.** This
is the complete list, in order; a solution that gets all nine right agrees with
`transformers` to about 1e-4 on logits of magnitude 25 (measured: 8.4e-5), and one
that misses any single item is off by far more than any tolerance:

1. Embed with `model.embed_tokens.weight` and **multiply by 12**
   (`embedding_multiplier`). No position embedding; the tokenizer adds no BOS.
2. 28 pre-norm blocks: `x = x + 0.263 * attn(norm1(x))`, then
   `x = x + 0.263 * mlp(norm2(x))` (`residual_multiplier`, on both branches).
3. RMSNorm: `x / sqrt(mean(x^2) + 1e-5) * weight`, epsilon inside the square root,
   no bias.
4. Grouped-query attention: 16 query heads, 4 key and value heads, each KV head
   serving 4 consecutive query heads. No biases anywhere.
5. RoPE on queries and keys, theta 1e7, in the half-split layout: dimension `i`
   pairs with `i + 32`; `x' = x * cos + rotate(x) * sin` with
   `rotate([a, b]) = [-b, a]`. (`nn.RoPE(64, traditional=False, base=1e7)` and
   `mx.fast.rope` produce exactly this; `traditional=True` does not.)
6. Attention logits are `q @ k^T` **times 1/64** (`attention_multiplier`), which is
   `1 / head_dim`, not `1 / sqrt(head_dim)`; then the causal mask and softmax.
7. SwiGLU with the **fused input matrix**: `u = input_linear(h)` has 4096 columns;
   the first 2048 are the gate, the last 2048 are up;
   `output_linear(silu(gate) * up)`.
8. `model.norm`, then the head, which is the **embedding matrix itself**, then
   **divide by 4** (`logits_scaling`).
9. The weights are bfloat16 on disk and must be used as float32.

`RealLM` implements exactly this list (`learnai/kit/realmodel.py`, about 150 lines,
meant to be read by authors, not linked from statements of problems that ask for it).
The large model needs the same list with the multipliers at 1, separate MLP matrices,
a separate head, and quantized linear layers.

**Tolerances that hold** (small model, 16-token prompts): float32 in numpy, torch,
JAX, or MLX agrees with the fixture to 1e-4 absolute, so `rtol=1e-3, atol=1e-3` is
safe. An MLX forward in **bfloat16** is off by up to 0.4 with the same argmax on
ordinary prompts; it is the safe 16-bit type for this model. For a bfloat16 case
measure the error on the case's own prompts in `fixtures/make.py`, set `atol` with a
margin above it, and compare argmax only where the fixture's top-two margin exceeds 1.

**float16 is not safe on the small model.** On the prompts where it works a float16
forward is off by up to 0.06, closer than bfloat16, but on others it **overflows to
NaN at layer 2** (seen by the milestone 3 authors in MLX). The reason is visible in
float32: the largest hidden-state entry is about 15 after layers 0 and 1 and about
9,400 after layer 2 for every prompt measured, which is after the 0.263 residual
multiplier, so that block's MLP output is about 36,000, within a factor of two of
float16's largest finite value, 65,504. bfloat16 has float32's exponent range and
cannot overflow there. So: never state a float16 tolerance for the whole model
without naming the prompts it was measured on; a float16 case must use prompts
checked to stay finite (assert it in `fixtures/make.py`), and a problem that wants
"half precision" on Granite 4.0 350M should ask for bfloat16.

**KL divergence between two sets of logits** (how far a quantized or half-precision
model's predictions moved): `logit_kl(p_logits, q_logits, axis=-1)` from
`learnai.kit` returns `KL(softmax(p) || softmax(q))` in nats per row, computed in
float64 from log-softmax; `p` is the original model. Report the mean with
`check.at_most("logit_kl", float(kl.mean()), limit, unit="nats")`.

### RealLM: the reference forward pass on demand

```python
lm = RealModel(ctx.asset("granite-4.0-350m")).lm()
logits, cache = lm(tokens)              # tokens: (T,) or (B, T) ids; logits float32
logits, cache = lm(next_tokens, cache)  # continue the prefix; cache is immutable
```

Called like `TinyLM`: numpy inside, tensors of the solution's framework at the
boundary (numpy in a `python` problem), `lm.counters.forward_calls` and
`tokens_processed`, `with_framework`. Use it where a problem needs the real model as
a *given*: the `forward` callable of `real-model-perplexity`, the single-token `step`
of `real-model-chat-turn`, the logit KL divergence of `quantize-real-weights`. Its
pieces are available for oracles: `lm.embed(ids)`, `lm.block(x, layer,
positions=None, past=None)`, `lm.head(x)`, and `lm.hidden_states(tokens)` (the state
after the embedding and after each block; `transformers` applies the final norm to
its last entry, the kit does not). The cache holds keys and values at KV-head width,
keys already rotated. It refuses a quantized checkpoint: the float weights of the 3B
model would be 14.6 GB.

Prefer a committed fixture to `RealLM` as the *oracle* of a problem that asks the
learner to implement the forward pass: the fixture comes from an independent
implementation. Use `RealLM` to cross-check your fixture while authoring.

### The 8-bit MLX checkpoint layout (verified)

`lmstudio-community/granite-4.2-3b-MLX-8bit` is what `mlx_lm.convert -q --q-bits 8`
writes. `config.json` carries `"quantization": {"group_size": 64, "bits": 8, "mode":
"affine"}`; the safetensors metadata is `{"format": "mlx"}`. Every `nn.Linear` and the
embedding (and the separate `lm_head`) is stored as three tensors:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `<name>.weight` | U32 | `[out, in / 4]` | the 8-bit codes, four per little-endian word: code `j` of a word is bits `8j` to `8j + 7`, so the bytes of the buffer in order are the codes in order |
| `<name>.scales` | BF16 | `[out, in / 64]` | one scale per group of 64 consecutive input positions |
| `<name>.biases` | BF16 | `[out, in / 64]` | one bias per group |

`w[o, i] = code[o, i] * scales[o, i // 64] + biases[o, i // 64]`. Norm gains stay
BF16 (`model.norm.weight`, `[2560]`). For `model.layers.0.self_attn.k_proj`:
`weight` `[512, 640]`, `scales` and `biases` `[512, 40]`, dequantized `[512, 2560]`.
The kit's `unpack_codes(words, bits=8)` and `dequantize(words, scales, biases,
group_size=64, bits=8)` implement this in numpy and equal `mx.dequantize` bit for bit
on the real file; `mx.quantized_matmul(x, w, scales, biases, transpose=True,
group_size=64, bits=8)` on the packed tensors reproduces `mlx-lm`'s logits to 1.2e-5
in float32. Scales and biases must be converted from BF16 before use.

`python/tests/fixtures/m3/problems/real/asset-vocab` is a complete miniature of an
asset problem (with a made-up asset that the tests install themselves).

### The hybrid model: Qwen3.5 0.8B (verified 2026-11-21)

`Qwen/Qwen3.5-0.8B` is the real model of `real-gdn-block` and `real-hybrid-logits`: the
smallest shipped decoder that mixes Gated DeltaNet layers with attention layers. It is
not a house model; no other problem should use it. Everything below was established on
this machine from the downloaded files with `transformers` 5.23.0, `mlx-lm` 0.32.2,
torch 2.15.0, JAX 0.12.0.1, and MLX 0.32.4; the unit tests are
`python/tests/test_kit_realhybrid.py` (skipped without the asset) and the committed
`transformers` reference is `python/tests/fixtures/realhybrid/` (`make.py`, five prompts
of 5 to 16 ids, 57 kB).

| | |
|---|---|
| asset name, kit object | `qwen3.5-0.8b`, `learnai.kit.QWEN35` |
| repository, pinned revision | `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` |
| license | Apache-2.0: the repository's `LICENSE` file (11,544 bytes) was read; it is the Apache License 2.0 text with "Copyright 2026 Alibaba Cloud" |
| files the kit fetches | `model.safetensors-00001-of-00001.safetensors` 1,746,942,600 bytes (SHA-256 `04b1c301...1fe4696`, verified by the fetch), `config.json` 2,907, `tokenizer.json` 12,807,982, `model.safetensors.index.json` 50,900; 1.76 GB |
| files a problem declares | `config.json` and the weight file. The other two are for authors: `transformers` cannot find the weight file without the index, and `tokenizer.json` is for making prompt ids with `tokenizers`. A directory fetched with the kit's four files is `present` for a problem that declares two |
| tensors | 488, 873,438,784 parameters: 320 under `model.language_model.` (752,393,024, the text decoder), 153 under `model.visual.`, 15 under `mtp.`. No `lm_head` tensor |
| storage | BF16, except `linear_attn.A_log` `(16,)` and `linear_attn.norm.weight` `(128,)` of the 18 delta layers, which are F32 (36 tensors, 2,592 numbers). `dt_bias` is BF16 |
| decoder | 24 layers, width 1024, SwiGLU 3584, vocabulary 248,320, RMSNorm eps 1e-6, tied head |
| schedule | `layer_types` in `config.json` (under `text_config`): attention at 3, 7, 11, 15, 19, 23, linear attention elsewhere |

The `[[assets]]` block for a problem:

```toml
[[assets]]
name = "qwen3.5-0.8b"
repo = "Qwen/Qwen3.5-0.8B"
revision = "2fc06364715b967f1860aea9cf38778875588b17"
files = ["config.json", "model.safetensors-00001-of-00001.safetensors"]
size_mb = 1750
```

Tensor shapes, all read from the safetensors header (`{i}` is the layer; linear weights
are `(out, in)`, there are no biases):

| name under `model.language_model.` | shape |
|---|---|
| `embed_tokens.weight` | `(248320, 1024)` |
| `norm.weight`, `layers.{i}.input_layernorm.weight`, `layers.{i}.post_attention_layernorm.weight` | `(1024,)` |
| `layers.{i}.mlp.gate_proj.weight`, `up_proj.weight` / `down_proj.weight` | `(3584, 1024)` / `(1024, 3584)` |
| `layers.{i}.linear_attn.in_proj_qkv.weight` | `(6144, 1024)` |
| `layers.{i}.linear_attn.in_proj_z.weight` / `out_proj.weight` | `(2048, 1024)` / `(1024, 2048)` |
| `layers.{i}.linear_attn.in_proj_a.weight`, `in_proj_b.weight` | `(16, 1024)` |
| `layers.{i}.linear_attn.conv1d.weight` | `(6144, 1, 4)` |
| `layers.{i}.linear_attn.A_log`, `dt_bias` | `(16,)` |
| `layers.{i}.linear_attn.norm.weight` | `(128,)` |
| `layers.{i}.self_attn.q_proj.weight` | `(4096, 1024)`: 8 heads of query 256 then gate 256 |
| `layers.{i}.self_attn.k_proj.weight`, `v_proj.weight` / `o_proj.weight` | `(512, 1024)` / `(1024, 2048)` |
| `layers.{i}.self_attn.q_norm.weight`, `k_norm.weight` | `(256,)` |

**Where the checkpoint differs from the kit's synthetic hybrid layout**
(`hybrid_params`). The forward pass is the one of
[HybridConfig](#hybridconfig-hybrid_params-hybrid_forward_ref); these are all the
differences:

1. Names start with `model.language_model.`, not `model.`.
2. The block norms, the final norm, `q_norm`, and `k_norm` store `gain - 1`: they
   compute `rms_norm(x) * (1 + weight)`. The gated norm inside a delta layer
   (`linear_attn.norm.weight`) is a plain gain, used as stored.
3. BF16 storage (two kinds of tensor F32), used as float32.
4. The sizes above; 16 key heads and 16 value heads, so no key head is repeated.
5. 168 tensors that a text-only port never reads (`model.visual.*`, `mtp.*`).
6. `rope_parameters` also carries `mrope_interleaved` and `mrope_section`; for text
   they change nothing (every position axis is the token index), and the plain partial
   RoPE of `ops.partial_rope` matches.

#### RealHybrid and RealHybridLM

```python
from learnai.kit import RealHybrid

model = RealHybrid(ctx.asset("qwen3.5-0.8b"))
model.config              # HybridConfig from config.json's text_config (layer_types set)
model.config_json         # the raw dict
model.checkpoint          # SafetensorsFile over all 488 tensors
model.weights()           # Weights: checkpoint names, stored values, float32, lazy
model.text_names()        # the 320 names under model.language_model.
lm = model.lm()           # RealHybridLM, numpy inside
logits, state = lm(tokens)                 # tokens (T,) or (B, T); state as hybrid_forward_ref
logits, state = lm(more, state)            # continue the sequence
x = lm.embed(tokens)                       # (T, 1024)
x, entry = lm.block(x, layer, entry=None)  # one block; entry is that layer's state
lm.final_norm(x); lm.head(x)               # the last norm; norm and tied head
lm.hidden_states(tokens)                   # 25 arrays: the embedding, then every block
lm.params(layer)                           # that layer in the kit's layout (gains, short names)
```

- `RealHybrid` is a `RealModel` whose `config` is a `HybridConfig` (a frozen dataclass:
  hand it to the solution as `cfg`) and whose weight file is the repository's
  `model.safetensors-00001-of-00001.safetensors`. It has **no tokenizer**
  (`model.tokenizer` raises): prompts are token ids in fixtures.
- `model.weights()` is what a porting problem hands the learner: every tensor under its
  checkpoint name with its stored value, so norm weights are still offsets and the
  learner applies the `1 +`. It always decodes BF16 to float32; `weights.raw(name)` is
  the stored uint16.
- **Hold the framework weights once per framework.** `model.weights(ctx.fw)` decodes
  and converts a tensor the first time it is read and keeps it in that `Weights`
  object. Keep the object in a module-level dict keyed by `ctx.framework` and call
  `ctx.solution.qwen_logits(weights, cfg, ctx.fw.tensor(ids))` directly, as
  `real-model-logits` does; a fresh `model.weights()` through `ctx.call` in every case
  decodes 3 GB again each time (about 0.8 s). Unlike the Granite `Weights`, a
  framework-bound `Weights` of this model does not also keep a numpy copy
  (`Weights(keep_numpy=False)`), which lowers the peak memory of a run from about 9 GB
  to about 7 GB. `RealHybridLM` reads numpy through its own cache, one layer at a time,
  so an oracle for one block costs one block.
- The state is the kit's: `(conv_state, S)` with shapes `(3, 6144)` and
  `(16, 128, 128)` as `(heads, value, key)` for a delta layer, `(k, v)` with
  `(2, seen, 256)` for an attention layer, keys normalized and rotated.
- **Synthetic cases** (`assets=[]`) use `checkpoint_layout(hybrid_params(cfg, seed))`:
  the kit's small hybrid weights renamed and with offset norms, so a solution written
  for the checkpoint runs on them unchanged. The oracles are `hybrid_forward_ref(cfg,
  params, tokens)` and, for one block, `hybrid_block_ref(cfg, params, layer, x,
  state=None)`, both on the *kit-layout* `params`. The synthetic weights are all
  float32 and have twice as many value heads as key heads by default; pass
  `HybridConfig(n_key_heads=4)` for the real model's one-to-one layout.

#### How the shipped libraries load and run it

- **`transformers`.** `Qwen3_5ForCausalLM.from_pretrained(directory,
  dtype=torch.float32)` loads the vision-language checkpoint as text only: with
  `output_loading_info=True` there are no missing, unexpected, or mismatched keys, so
  the class maps the `model.language_model.` prefix, ignores `model.visual.*` and
  `mtp.*`, and ties `lm_head` to the embedding. It needs
  `model.safetensors.index.json` next to the weights (without it: "no file named
  model.safetensors"). The same holds for a miniature directory written by the unit
  tests, so this is tested without the asset too.
- `output_hidden_states=True` gives 25 tensors: entry 0 is the embedding, entry `i + 1`
  the output of layer `i`, and **the last entry (24) has the final norm applied**.
  `lm.hidden_states` applies it to none; compare entry 24 with
  `lm.final_norm(hidden[24])`.
- Its cache keeps, for a delta layer, the convolution state as `(1, 6144, 4)`
  (channels by the last **4** inputs; the kit keeps the last 3 as `(3, 6144)`) and the
  memory as `(1, 16, 128, 128)` in `(heads, key, value)` order, the transpose of the
  kit's. Converted, they agree with the kit's state to 5e-5 and 1.1e-5 after 12 tokens
  in four calls; cached keys and values to 1.9e-5.
- `transformers` runs a multi-token call through the chunked delta rule (the prompt is
  padded to 64) and a one-token call with a cache through the recurrent rule. On the
  real weights its own two paths differ by up to 1.5e-4 on the logits (seven prompts,
  argmax equal everywhere). Fixtures here are **one call, no cache**.
- It is slow: the optional `flash-linear-attention` and `causal_conv1d` packages are
  not installed, so the reference PyTorch delta rule runs, and one forward pass of up
  to 16 tokens takes 4.0 to 4.6 s (a cached one-token step is similar). Loading is
  lazy (0.2 s); peak memory 6.1 GB.
- **`mlx-lm` is not a float32 reference for this model.** `mlx_lm.utils.load_model`
  loads it (the model class is `mlx_lm.models.qwen3_5`; `mlx_lm.load`, which also loads
  a tokenizer, was not tried), but its `sanitize` adds the `1.0` to
  the norm weights **in bfloat16**, which rounds every gain (the final norm's by up to
  0.39 %). Against `transformers` in float32, on eight prompts and 86 positions:

  | how `mlx-lm` was run | largest logit difference | argmax differs |
  |---|---|---|
  | as loaded (bfloat16 weights; the F32 tensors stay float32) | 0.70 | 5 of 86 |
  | after `model.set_dtype(mx.float32)` | 0.17 | 1 of 86 |
  | weights cast to float32 *before* `sanitize`, then `load_weights` | 1.7e-4 | 0 of 86 |

  The `set_dtype` recipe that works for the 8-bit Granite model therefore does not work
  here. Take fixtures from `transformers`.

#### Measured agreement and the tolerances to use

The from-scratch forward (kit `ops`, the raw checkpoint names, the `1 +` applied; the
port is `qwen_forward` in the unit tests, 45 lines, one code path for the three
frameworks) against `transformers` float32, one call per prompt, on eight prompts:
five texts of 5 to 15 ids, a chat-formatted prompt with special tokens, 16 random ids,
and a single token. Largest absolute differences:

| quantity | numpy (`RealHybridLM`) | torch | JAX | MLX |
|---|---|---|---|---|
| hidden state after layers 0 and 1 (early delta blocks), full forward | 1.4e-6 | 1.4e-6 | 4.1e-6 | 3.4e-6 |
| hidden state after layers 12 and 13 (middle delta blocks), full forward | 1.9e-6 | 5.3e-6 | 4.4e-6 | 4.7e-6 |
| hidden state after any of layers 0 to 22 | | 2.0e-5 | 1.8e-5 | 2.0e-5 |
| one delta block (0, 1, 12, 13) run on the fixture's input | 7.2e-7 | 8.3e-7 | 3.0e-6 | 2.7e-6 |
| the same block in two calls through its state | | 8.3e-7 | 3.0e-6 | 2.7e-6 |
| logits (magnitude 12 to 31) | 1.9e-4 | 1.5e-4 | 1.6e-4 | 1.5e-4 |
| argmax tokens | 86 of 86 equal | 86 of 86 | 86 of 86 | 86 of 86 |

Nothing here held for some prompts and not others: the worst logit difference per
prompt ranged from 1e-5 (one token) to 1.9e-4 (14 tokens of prose), and the argmax
agreed at every position, including positions whose top two logits are 0.002 apart.
What does depend on the prompt is listed under "Things that depend on the input" below.

- **`real-hybrid-logits`: `rtol=1e-3, atol=1e-3`**, and the argmax exactly. The
  smallest `atol` that would have passed with `rtol=1e-3` was 1e-4, so there is a
  factor of ten in hand, and the documented single mistakes are off by far more.
  Compare argmax only at positions whose fixture margin (top logit minus second) is at
  least 0.01, about fifty times the measured error; the kit fixture stores
  `margin_<i>` per prompt, and three of the eight prompts above have a position below
  that.
- **`real-gdn-block`: `rtol=1e-3, atol=1e-5` on the hidden state.** The hidden states
  of this model are small: after layers 1 and 13 the median absolute entry is 0.02 to
  0.04 and the largest about 1 (entry 24, before the final norm, reaches 50). An
  `atol` of 1e-3 would be 3 % of a typical entry and is too loose; `1e-5` is three
  times the worst single-block error and a thousand times smaller than a typical
  entry. With this tolerance, on every prompt and each of the four layers: norm
  weights used without the `1 +` fail 99 % of the entries, the `1 +` also applied to
  the gated norm fails 99 %, and a flipped convolution kernel fails 99 %.
- If the block's returned state is compared, use `rtol=1e-3, atol=1e-4`: the memory
  (largest entry 0.2 to 2.6) agreed to 4.3e-6 and the convolution state (largest entry
  13 to 37) to 2.6e-5 across frameworks.

**Things that depend on the input.**

- **`A_log` and `dt_bias` do not affect the first token of a sequence.** The decay
  multiplies a memory that is still zero, so a one-token prompt, or position 0 of any
  prompt, is identical with and without them (measured: exactly 0), and from position
  1 on the output of block 0 moves by 0.02 to 0.07 (one 12-token prompt). A real-weight case meant to catch
  "`dt_bias` omitted" needs at least two tokens and must not compare position 0 only.
- JAX compiles per prompt length. Real-weight cases of one length share a compilation;
  each new length costs 2 to 3.5 s the first time on a machine without the compilation
  cache.

#### Time and memory (Apple M1 Max, 64 GB)

| | numpy | torch | JAX | MLX |
|---|---|---|---|---|
| decode all 320 text tensors to float32 (and convert) | 0.5 s | 0.8 s | 0.9 s | 0.8 s |
| convert only, numpy already decoded | | 0.30 s | 0.33 s | 0.35 s |
| forward pass, 16 tokens, first call | 0.27 s | 0.31 s | 2.7 s | 0.18 s |
| forward pass, 16 tokens, later calls | 0.22 s | 0.26 s | 0.44 s | 0.08 s |
| one delta block, 16 tokens | | 4 ms | 8 ms (0.7 s first) | 1 ms |

Through the harness, a throwaway problem with two synthetic and three real-weight
cases (`learnai submit --reference`): torch 2.2 s in all (first real case 0.9 s, later
ones 0.3 s), MLX 1.7 s (0.8 s, then 0.05 s), JAX 4.0 s with a warm compilation cache
and 12.7 s cold (2.3 to 3.4 s per real case of a new length). Peak resident memory of
the harness process: 6.9 GB (torch, MLX) and 7.9 GB (JAX); 3.0 GB of that is the
float32 weights and 1.7 GB the mapped file. `time_limit_s = 20`, as in
`real-model-logits`, has room for the first real case. A run of `real-gdn-block` reads
only the layers it uses: about 90 MB of float32 and 0.05 s for a delta layer.

Each framework's harness process holds its own 7 GB, so validating this problem with
several jobs in parallel multiplies that.

#### The asset-missing path

Checked with the same throwaway problem (the asset declared with two files, synthetic
cases marked `assets=[]`): with `LEARNAI_MODELS_DIR=/nonexistent` the two synthetic
cases run and pass, the three real cases are skipped with "model asset `qwen3.5-0.8b`
(Qwen/Qwen3.5-0.8B, about 1.75 GB) is not downloaded; fetch it with `learnai assets
fetch qwen3.5-0.8b`", the reference ends "Not judged: model asset missing", and the
starter ends "Runtime error". `learnai assets list` shows the asset as `missing` and,
with the real directory, `present` at 1.76 GB.

In the app, the Models screen lists what `learnai assets list --json` returns, which is
what the catalog's problems declare; this asset appears there once `real-gdn-block` or
`real-hybrid-logits` declares it, with no front-end change (the name is valid for the
Rust core's `is_asset_name`, and the row is generic). Read in the code, not exercised
in the app.

#### Not verified

- The vision tower and the MTP module: their tensors were counted, never loaded or run.
- The tokenizer: `tokenizer.json` was used with the `tokenizers` library to make
  prompt ids and with nothing else; the kit's `ByteLevelBPE` was not tried on it, and
  the chat template was not downloaded.
- Any half-precision forward of the kit's port (bfloat16 or float16). The only 16-bit
  numbers are `mlx-lm`'s above.
- Prompts longer than 16 tokens on the real weights, in particular past 64, where
  `transformers` starts a second chunk (tiny random models were checked to 150 tokens
  in section 7), and a batch of more than one prompt against `transformers`.
- `transformers` with the optional fast kernels installed: the fixtures come from its
  reference PyTorch path.
- Generation: only logits and states were compared, not a decode loop.
- `Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-2B`, or any other revision.

## 16. Checklist before you open a pull request

- `learnai validate <id> -v` passes, with no warning.
- `ruff check problems/<track>/<id>` is clean.
- `learnai run <id> -f <each framework> --starter` shows a failure a beginner can act on.
- `learnai submit <id> --mutant <each>`: every mutant fails for the reason its `# BUG:` line
  gives, and the failure message would make sense to a learner.
- If the statement forbids a call: `forbidden` lists every spelling of it in every
  framework, and a mutant takes the shortcut. If it forbids something that is not
  called (a dtype, a package), `forbidden_source` names the identifier, and no entry
  is an ordinary word.
- If a count decides the verdict: the counting object is handed over behind `Guarded`,
  with `guard(ctx)` and `check_guard()` around every call of the solution.
- If the problem has `requires` other than `cpu`: `LEARNAI_HARDWARE=cpu learnai submit
  <id> --reference` passes every case that still runs, skips the rest with a clear
  reason, and ends with "Not accepted: hardware missing" (section 14).
- If the problem has `[[assets]]`: it validates with the asset present
  (`learnai validate <id> --require-assets`) and behaves sensibly without it
  (`LEARNAI_MODELS_DIR=/nonexistent learnai submit <id> --reference`); fixtures are small
  and `fixtures/make.py` regenerates them (section 15).
- If a case has a memory limit: the reference uses at most two thirds of it and the
  mutant it is written for at least one and a half times it (section 13).
- If the problem is in the `cuda` track: every mutant is rejected by a CPU case
  (`learnai validate <id>` prints no "NOT validated" line for a mutant), the GPU cases are
  marked `requires="cuda"` and few, and the statement says which facts were checked
  only on the CPU (section 18).
- Every statistical case takes under about 3 s in every framework.
- The first public case is the statement's worked example.
- The statement specifies every choice that affects the output, and its Constraints
  section states the tolerance and any cost or speed limit.

## 17. Contract additions of milestone 3 (to mirror in ARCHITECTURE.md)

The judge implements the following; each block is worded to be pasted into the named
section of `docs/ARCHITECTURE.md`. Everything is additive: a run of a problem with
`requires = "cpu"` and no `[[assets]]` produces the same events as before, plus
`end.skipped` and `start.requires`.

**3.1 Environment.** Optional extras: `jax`, `mlx`, `models` (huggingface_hub,
safetensors, tokenizers, transformers, and mlx-lm on macOS). `models` is needed only
by `learnai assets fetch` and by authors generating fixtures; the harness and the kit read
downloaded files with numpy.

**3.2 Frameworks.** A problem may be MLX-only (`frameworks = ["mlx"]`). Its oracle is
`reference/mlx.py` and its mutants are written in MLX. Every other tensor problem
includes `torch`.

**3.3 Problem directory.**

```toml
requires = "cpu"                 # cpu | mlx-gpu | cuda: hardware needed to be accepted
forbidden_source = ["bfloat16"]  # optional: identifiers that may not appear in the solution's source

[[assets]]                       # optional, repeatable: model files the problem needs
name = "granite-4.0-350m"        # directory under <root>/.learnai/models/
repo = "ibm-granite/granite-4.0-350m"
revision = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"   # full commit hash
files = ["config.json", "tokenizer.json", "model.safetensors"]   # names or globs; safetensors, tokenizer, and config files only
size_mb = 712                    # approximate download size
```

Optional `fixtures/` directory: committed reference outputs loaded by the tests.
Cases declare what they need with `@case(requires=..., assets=[...])`; the default is
`requires="cpu"` and every asset of the problem. Assets live in
`<root>/.learnai/models/<name>/` (gitignored) beside a `learnai-asset.json`
manifest; an asset is present when the manifest matches the declared repository and
revision, covers the declared files, and every file has its recorded size.

**3.5 Harness command.** No new flags. Two environment variables:
`LEARNAI_MODELS_DIR` (replaces `<root>/.learnai/models`) and `LEARNAI_HARDWARE`
(comma-separated list of hardware to treat as present, for tests and previews).

**3.6 Event protocol.**

```jsonc
{"event":"start", ..., "cases":[{"id":"c1","name":"...","public":true,"requires":"cuda"}],   // `requires` only when not "cpu"
   "requires":"cuda","requires_met":false}
{"event":"case_end","id":"gpu_scale","status":"skipped","duration_ms":0,
   "message":"skipped: needs an NVIDIA GPU (CUDA), which this machine does not have",
   "detail":null,"stdout":"","metrics":[],
   "skip":{"kind":"hardware","requires":"cuda"}}      // or {"kind":"asset","asset":"<name>"} or {"kind":"aborted"}
{"event":"end","status":"hardware_missing","passed":2,"total":3,"skipped":1,"duration_ms":210.0,"metrics":[...],
   "message":"2 of 2 cases that could run passed; 1 of 3 cases were skipped because ...",
   "missing_hardware":"cuda"}                         // with assets_missing: "missing_assets":[{"name","repo","size_mb","fetch_command"}]
```

- Final `status` adds `assets_missing` and `hardware_missing`. Precedence when mixed:
  import_error, runtime_error, time_limit, wrong_answer, assets_missing,
  hardware_missing, accepted.
- A case is `skipped`, without being run, when the hardware it requires is absent
  (`skip.kind = "hardware"`) or an asset it needs is not downloaded (`"asset"`); cases
  skipped because the run stopped early have `skip.kind = "aborted"`. Every skipped
  `case_end` carries `skip` and a one-line `message`.
- `accepted` means every selected case ran and passed on a machine that has the
  problem's `requires`. If nothing that ran failed but a case was skipped for an
  asset, the status is `assets_missing`; if a case was skipped for hardware, or the
  problem's `requires` is not met, it is `hardware_missing`. A failure among the cases
  that ran decides the status as before. When no case can run, the solution is not
  imported.
- `end` always carries `skipped` (the number of skipped cases). For the two new
  statuses it carries `message` (what was skipped, why, and what to do) and
  `missing_assets` or `missing_hardware`.
- A metric may carry `"higher_is_better": true` (throughput); the run-level value of
  such a metric is the smallest across cases instead of the largest.
- Presentation: `assets_missing` and `hardware_missing` are neither success nor a
  failure of the learner's code. Show "N of M cases passed; K skipped", the `message`,
  and for `assets_missing` an action that runs the fetch. They never mark a problem
  solved; store them like any other non-accepted submit.

**3.7 CLI.** `learnai assets list [--json]`, `learnai assets fetch <name>... | --all [--json]
[--no-verify]` (schemas in section 15), and `learnai validate --require-assets`. `validate`
reports a problem as `ok`, `FAIL`, or `SKIP` (nothing could run on this machine;
does not fail the command) and warns about every case it could not run.

**4 Core API.**

```ts
type Verdict = ... | "assets_missing" | "hardware_missing";
interface Metric { name: string; value: number; unit?: string; limit?: number; higherIsBetter?: boolean; }
interface CaseInfo { id: string; name: string; public: boolean; requires?: Requires; }
type Skip = { kind: "hardware"; requires: Requires } | { kind: "asset"; asset: string } | { kind: "aborted" };
interface CaseResult { ...; skip?: Skip; }
interface MissingAsset { name: string; repo: string; sizeMb: number; fetchCommand: string; }
// RunEvent "start" gains requires: Requires and requiresMet: boolean;
// RunEvent "end" gains skipped: number, missingAssets?: MissingAsset[], missingHardware?: Requires.
```


## 18. CUDA problems

The `cuda` track teaches GPU programming with PyTorch to learners most of whom, like
the machine that validates the catalog, have no NVIDIA GPU. A CUDA problem therefore
has two halves:

```toml
frameworks = ["torch"]
requires = "cuda"
```

- **CPU logic cases** (unmarked) run everywhere and decide validation. They hand the
  solution CPU tensors and `torch.device("cpu")`, and they catch the *GPU* mistake
  with a kit object that makes it observable on the CPU: a spy that counts host
  synchronizations, a device whose work finishes only at `sync()`, a model that runs
  out of memory, a graph that replays on captured memory, a numpy Triton.
- **GPU cases** (`@case(..., requires="cuda")`) run only where
  `torch.cuda.is_available()`. They check what only hardware shows: that tensors are
  on the device, a speed ratio against the same-framework reference, an allocator
  number. Keep them few; no mutant should depend on them (section 14).

Rules for the split:

1. Every solution takes `device` (or a device-like object) as an argument and must
   work for `torch.device("cpu")`. Never call `torch.cuda.*` unconditionally in a
   reference: `torch.cuda.synchronize()` raises `AssertionError: Torch not compiled
   with CUDA enabled` on a CPU build.
2. Things the solution would take from `torch.cuda` are **passed in**, so the CPU
   case can pass the kit's stand-in and the GPU case the real thing: `device_api`
   (`FakeDevice` or `RealDevice`), `empty_cache` (`model.empty_cache` or
   `torch.cuda.empty_cache`), `graph` (`FakeCudaGraph()` or `torch.cuda.CUDAGraph()`),
   `pin` and `to_device` (`TransferSpy` or `torch.Tensor.pin_memory` and `.to`).
3. A GPU case builds its inputs on the device itself (`torch.ones(3, device="cuda")`);
   `ctx.call` and `ctx.fw.tensor` make CPU tensors. `ctx.fw.sync(x)` calls
   `torch.cuda.synchronize()` when a CUDA tensor is among its arguments, so
   `ctx.compare_speed` times GPU work correctly.
4. On this machine `learnai validate` reports each CUDA problem as `ok` with the warning
   "only N of M cases could run ... (GPU cases not validated: <ids>)". That line is
   the honest state of the track: its GPU cases have never run. Say so in the pull
   request, and do not write a GPU case you would not bet on.

The tools, all in `learnai.kit`, all working on CPU tensors
(`python/src/learnai/kit/cudatools.py` and `tritonshim.py`; unit tests in
`python/tests/test_kit_cuda.py` and `test_kit_tritonshim.py`).

### SyncSpy: host synchronizations

```python
from learnai.kit import SyncSpy

@case("at most one host read per check_every steps, and one transfer", public=True)
def sync_free(ctx):
    with SyncSpy() as spy:                        # watches the solution file
        tokens = ctx.solution.decode(model, prompt, 64, uniforms, torch.device("cpu"))
    check.at_most("host_syncs", spy.syncs, 64 // CHECK_EVERY + 1)
    check.equal(tokens.tolist(), expected, label="tokens")
```

On a GPU, `.item()`, `bool(x)`, and friends make Python wait for all queued work and
copy data to the host; in a loop that is once per token. `SyncSpy` replaces the
methods on `torch.Tensor` for the length of the block and records each call **made
from the solution file** (or `file=`): `spy.events` is the ordered list of
`SyncEvent(kind, what, line)`, `spy.syncs` its length, `spy.reads`, `spy.transfers`,
and `spy.shape_syncs` the counts by kind, `spy.describe()` one line per event.

- `"read"`: `x.item()`, `x.tolist()`, `bool(x)` (so also `if x:`, `while x:`, `not
  x`, `x and y`, `max(a, b)`), `int(x)`, `float(x)`, `x` used as a list index, in
  `range(x)`, or as a **slice bound of a tensor** (`row[:x]`, read or written: the
  size of the result depends on the value, so the host must wait for it; `what` is
  `"slice bound"` for a read and `"index"` for a write), `print(x)`, `str(x)`,
  `repr(x)`, an f-string, `y in x`, `x.equal(y)`, `x.allclose(y)`, `torch.equal`,
  `torch.allclose`.
- `"transfer"`: `x.cpu()`, `x.numpy()`, `np.asarray(x)`, and `x.to(...)` **when the
  call names the CPU in the source**: `x.to("cpu")`, `x.to(device="cpu")`,
  `x.to(torch.device("cpu"))`, or a name the solution file only ever assigns such a
  literal to (`HOST = torch.device("cpu")` ... `x.to(HOST)`). `what` is `"to(cpu)"`.
  The spy reads the calling line's source, not the argument's value, because the
  value cannot tell: the `device` a CPU case passes in is the CPU as well.
- `"shape"`: operations whose output size depends on the data, which make the host
  wait to learn it: `nonzero`, `argwhere`, `masked_select`, `unique`,
  `unique_consecutive`, `torch.where(cond)` with one argument, `x[mask]` with a
  boolean mask.
- **Not counted:** `x.to(device)` with the device the test passed in (as a
  `torch.device` or as the string `"cpu"`), `x.to(other.device)`, `x.to(other)`,
  `x.to(dtype)`, and `x.to(where)` for a name that is also bound in any other way (a
  parameter, a second assignment). Here they are no-ops, and whether they would
  synchronize on a GPU depends on where `x` lives. In a GPU case nothing is guessed:
  from a tensor that really is on another device, any `.to` whose target is the CPU
  counts. Two limits to know: a 0-d tensor used as a plain index (`row[n]`) is not
  counted, because the result's size does not depend on it and torch need not read
  it on the host; and `.to("cpu")` on a tensor the solution created on the host is
  counted although it would cost nothing on a GPU. Count the final transfer in the
  limit whichever way the statement spells it.
- Unlike `EvalSpy` (section 13), which watches calls, this spy patches methods, so
  the implicit forms (`if x:`) *are* seen. One call counts once: `print(x)` is one
  event although torch reads many values to format it. The kit's, the test's, and
  torch's own reads are not counted.
- A kit model that is numpy inside (`TinyLM`) reads its inputs on the host; that is
  not the solution's doing and is not counted, but it also means such a model cannot
  run on a CUDA device. A GPU case needs a small `torch.nn.Module` defined in
  `tests.py` and moved to the device.

### DeviceSpy and the meta device: tensors created in the wrong place

```python
from learnai.kit import DeviceSpy

with DeviceSpy() as spy:
    out = ctx.solution.run(model, inputs, torch.device("cpu"))
check.that(not spy.implicit,
           f"run: {len(spy.implicit)} tensors were created without device=", spy.describe())
```

`torch.arange(n)` lands on the default device, the CPU; next to CUDA inputs that is
an error at best and a silent host round trip at worst, and on a CPU run it is
invisible. `DeviceSpy` wraps the factory functions (`arange`, `zeros`, `ones`, `full`,
`empty`, `eye`, `linspace`, `logspace`, `rand`, `randn`, `randint`, `randperm`,
`tensor`, `tril_indices`, `triu_indices`) and records every call from the solution
file as `FactoryEvent(what, line, explicit)`; `spy.implicit` are the calls without
`device=`. The `*_like` functions and `x.new_zeros(...)` inherit the device and are
not watched. Only `torch.<name>` looked up at call time is seen.

There is also a **second device on every machine: `torch.device("meta")`**. A meta
tensor has a shape and a dtype and no data. Verified on torch 2.15: adding,
concatenating, `torch.where`, `masked_fill`, `nn.Linear`, and
`scaled_dot_product_attention` raise "Tensor on device cpu is not on the expected
device meta!" when a CPU tensor meets a meta tensor, exactly as CUDA would, and
`.item()` and `.cpu()` on a meta tensor raise. So a case can run the solution with
inputs moved to `"meta"` and `device=torch.device("meta")` to find a leaf that was
not moved or a mask built on the CPU. It is not complete: `x @ w`, `x[:, index]`,
and `F.embedding` with one CPU and one meta operand do **not** raise. Use it together
with `DeviceSpy`, and check numbers in a separate CPU case (meta tensors have none).

### FakeDevice and RealDevice: asynchronous work on a virtual clock

```python
from learnai.kit import FakeDevice, RealDevice

@case("median seconds per call, warm-up excluded", public=True)
def exact_time(ctx):
    dev = FakeDevice()
    fn = dev.kernel(0.004, first_call_s=0.050)    # each call queues 4 ms of device work
    dev.launch(0.1)                               # stale work queued before the region
    seconds = ctx.solution.time_region(fn, dev, 5)
    check.allclose(seconds, 0.004, rtol=1e-9, label="median seconds per call")

@case("on the GPU the same function reports real time", requires="cuda")
def real_time(ctx):
    seconds = ctx.solution.time_region(gpu_fn, RealDevice("cuda"), 5)
```

A CUDA call returns when the work is *queued*. `FakeDevice` models that with a
virtual clock in integer nanoseconds: `dev.kernel(seconds, first_call_s=None)`
returns a function that queues device work and returns at once (costing the host
`launch_s`, default 0); `dev.launch(seconds)` queues work directly; `dev.sync()`
(for `torch.cuda.synchronize`) moves host time to when the queue is empty;
`dev.clock()` (for `time.perf_counter`) reads host time and never waits. The device
runs its queue concurrently with the host. So a timer without a sync before the stop
measures 0; a region not preceded by a sync pays for the stale work in its first
sample; a first call inside the measurement shows as 0.050 in the first sample.
**A median hides one bad sample**: the last two mistakes change only the first of
the `n` timings, so test them with `n = 1` (a solution that does not sync before starting
the timer then reports 0.154 in the example above, the stale 0.1 and the warm-up
0.050 included, instead of 0.004), or assert on `dev.events` that a `"sync"` precedes the first
`"clock"`. `dev.counters` has
`syncs`, `clock_reads`, `launches`; `dev.pending_s` is unwaited work; `dev.events`
the ordered log. `RealDevice(device)` gives the same two methods over
`torch.cuda.synchronize(device)` (nothing on the CPU) and `time.perf_counter`.

**Interface change from the curriculum:** the curriculum gave `FakeDevice()` with
`sync()` and `clock()` only; `kernel`, `launch`, `pending_s`, `counters`, `events`,
`launch_s`, and `RealDevice` were added because a test needs a way to *queue* work
and a GPU case needs the real counterpart.

### OOMModel: out of memory above a batch size

```python
from learnai.kit import OOMModel

model = OOMModel(lambda x: x * 2, max_items=3)
out = ctx.solution.run_batched(model, items, 8, model.empty_cache)
check.equal(model.batch_sizes, [8, 4, 2, 2, 2, 2, 2], label="batch sizes tried")
check.that(model.counters.live_at_empty_cache == 0,
           "empty_cache() ran while the failed batch was still referenced")
```

`model(batch)` returns `fn(batch)` for `len(batch) <= max_items` and otherwise raises
`torch.OutOfMemoryError("CUDA out of memory. ...")`. `model.batch_sizes` lists every
size tried; `model.events` is the ordered log of `("call", size, "ok" | "oom" |
"error")` and `("empty_cache", live)`; `model.counters` has `calls`, `ooms`, `items`,
`empty_cache_calls`, `live_at_empty_cache`; `model.empty_cache` is the stand-in for
`torch.cuda.empty_cache` to pass to the solution; `errors={call_index: exception}`
makes a given call raise something else.

- `live_at_empty_cache` is the real trap, reproduced with weak references: inside
  `except torch.OutOfMemoryError:` the exception holds the traceback, the traceback
  holds the failed call's frame, and the frame holds the batch, so `empty_cache()`
  called there cannot free it. The fix is to leave the `except` block and drop the
  batch first. It needs a batch that can be weakly referenced (a tensor; a list is
  silently not tracked).
- `torch.OutOfMemoryError` is a subclass of `RuntimeError`; the mutant that catches
  `RuntimeError` is caught by `errors={1: RuntimeError("shape mismatch")}`, which a
  correct solution lets through.
- **Interface change:** the curriculum gave `OOMModel(fn, max_items)` "counts calls";
  `batch_sizes`, `events`, `empty_cache`, `errors=`, and the liveness check were
  added, and the solution receives `empty_cache` from the model.

### FakeCudaGraph: capture, then replay on the captured memory

```python
from learnai.kit import FakeCudaGraph

graph = FakeCudaGraph()
calls = []
def step_fn(x):                                   # the test's wrapper sees when it is called
    calls.append(graph.capturing)
    return model(x)
step = ctx.solution.GraphedStep(step_fn, example, graph)
check.equal(calls[-1:], [True], label="the last call of step_fn is the capture")
check.that(False in calls, "step_fn was never run before capturing (no warm-up)")
for x in inputs:
    check.allclose(step(x), model(x), label="replayed output")
check.equal(len(calls), n_before, label="step_fn must not run again after capture")
```

A CUDA graph records the kernels launched during capture with the memory addresses
they read and wrote; `replay()` reruns them on those addresses and runs no Python.
`FakeCudaGraph` does the same on CPU tensors: between `capture_begin()` and
`capture_end()` a `TorchDispatchMode` records every operation with the tensor
objects it read and returned; `replay()` re-executes them on those same tensors and
writes each result into the tensor that held it during capture. Consequences, as on
the GPU:

- rebinding a placeholder (`static_in = x`) instead of `static_in.copy_(x)` replays
  the old input: stale output;
- outputs are overwritten in place by every replay: without `clone()` an earlier
  result changes at the next call (and `data_ptr()` of the output stays the same);
- a host read during capture (`.item()`, `bool(x)`, `print(x)`) raises
  `RuntimeError: CUDA error: operation not permitted when stream is capturing`;
- a replay whose recomputed shape differs raises: shapes are frozen.

Surface: `capture_begin(pool=None, capture_error_mode="global")`, `capture_end()`,
`replay()`, `reset()`, as in torch 2.15, plus `graph.capturing`, `graph.captured`,
`graph.n_ops`, and `graph.counters` (`captures`, `replays`).

**Use the context manager in solutions.** Torch's documentation says not to call
`capture_begin` yourself but to use `with torch.cuda.graph(g):`.
(Torch's CUDA notes: warm-up and capture happen on a side stream, and
`torch.cuda.graph` sets one for you; its source also synchronizes and empties the
cache first.)
`with graph.patched():` makes `torch.cuda.graph(graph)` accept a `FakeCudaGraph`, so
a solution written `with torch.cuda.graph(self.graph): ...` runs unchanged in both
halves: wrap the CPU case's calls in `with graph.patched():`. Not imitated: streams,
memory pools, and any failure of a capture made without warm-up (check the warm-up
through `graph.capturing`, as above).

### CompileCounter: how many graphs `torch.compile` built

```python
from learnai.kit import CompileCounter

with CompileCounter() as cc:
    step = ctx.solution.make_step(model, cache, backend="eager")
    for pos in range(64):
        logits = step(token, torch.tensor(pos))
check.at_most("compilations", cc.compilations, 1)
check.that(cc.graph_breaks == 0, "the step has a graph break", cc.describe())
```

`cc.compilations` is the number of frames Dynamo compiled (a function compiled once
with no break is 1; a graph break makes each piece a frame), the same with and
without `fullgraph=True`; `cc.graph_breaks` the number of graph breaks with
`cc.break_reasons`, `cc.graphs` the distinct graphs.
`cc.snapshot()` updates the numbers inside the block. Entering and leaving call
`torch._dynamo.reset()`. Have the solution accept `backend` and pass `"eager"` on the
CPU: Dynamo traces and guards exactly as for any backend and no C++ compiler is
needed (`mode=` is ignored by the eager backend, with a warning).

Measured on torch 2.15, CPU, `backend="eager"` (the numbers a limit must be set
against):

| the step function is called with | `compilations` |
|---|---|
| a 0-d tensor position, fixed shapes, 6 calls | 1 |
| a Python `int` position that changes, default `dynamic=None` | **2**: specialized on the first value, then recompiled once with the value symbolic |
| a Python `int` that changes, `dynamic=False` | 8, then Dynamo gives up at `recompile_limit` (8) and runs the frame uncompiled |
| an input whose length grows every call, default | 2 (the size becomes symbolic) |
| an input whose length grows, `dynamic=False` | 8, then the limit |
| `if x.sum() > 0:` on a tensor (one call) | 2 frames, `graph_breaks == 1` ("Data-dependent branching") |

So the curriculum's "at most 2 compilations" does **not** reject a Python-int
position under the default settings: it compiles exactly twice. Either require
`compilations <= 1` for the decode steps (with the prompt compiled outside the
block), or specify `dynamic=False` in the statement, under which the int mutant
reaches 8. `fullgraph=True` turns a graph break into an exception
(`torch._dynamo.exc.Unsupported`), which is another way to test for one.

The counts come from `torch._dynamo.utils`, internal API that torch's own tests read:
`get_compilation_metrics()` (one record per frame Dynamo tried to compile; those
without a `fail_type` are the compilations) and `counters` (`["stats"]
["unique_graphs"]`, `["unimplemented"]`). Until milestone 5 `compilations` was
`counters["frames"]["ok"]`, which torch 2.15 does not increment under
`fullgraph=True`: it read 0 there, and tests worked around it with
`max(cc.compilations, cc.graphs)`. That is no longer needed (it is harmless where it
is still written): measured in both modes, a tensor position compiles once, a
changing Python `int` with `dynamic=False` once per value, and a refused graph break
under `fullgraph=True` is 0 compilations and 1 graph break. `CompileCounter` is the
one place that touches these internals.

### TransferSpy: pinning and non-blocking transfers

```python
from learnai.kit import TransferSpy

spy = TransferSpy()
batches = [spy.label(torch.randn(4, 8), i) for i in range(5)]
fetcher = ctx.solution.Prefetcher(batches, torch.device("cpu"), spy.to_device, spy.pin)
for i, batch in enumerate(fetcher):
    spy.mark(("consumed", i))
check.that(not spy.unpinned, "a batch was transferred from pageable memory", spy.describe())
order = spy.order()                # [("pin", 0), ("transfer", 0), ("pin", 1), ("transfer", 1),
                                   #  ("mark", ("consumed", 0)), ...]
```

`spy.pin(tensor)` returns a copy that counts as pinned; `spy.to_device(tensor,
device, non_blocking=False)` moves it (a real `.to`) and records whether its
argument was pinned and whether the call was non-blocking; `spy.label(tensor, label)`
names a source tensor, and tensors derived from it carry the label; `spy.mark(name)`
records a moment chosen by the test. `spy.events` is the ordered `TransferEvent`
list, `spy.transfers`, `spy.unpinned`, and `spy.blocking` filter it, `spy.order()` is
`[(kind, label), ...]`. On a GPU pass `pin=torch.Tensor.pin_memory` and
`to_device=lambda t, device, non_blocking=False: t.to(device,
non_blocking=non_blocking)`.

### tritonshim: Triton kernels in numpy

```python
# solution
from learnai.kit import tritonshim as tl

@tl.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * n_cols
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + offs, mask=offs < n_cols, other=0.0)
        acc += x * x
    ...

# tests.py
info = tl.launch(ctx.solution.rmsnorm_kernel, (n_rows,), x, w, out, n_cols, 1e-5, BLOCK=64)
check.allclose(out.numpy(), oracle, rtol=1e-5, atol=1e-6, label="rmsnorm")
check.equal(info.elements_stored, n_rows * n_cols, label="elements written")
```

`tritonshim` runs a kernel written in a subset of `triton.language` on CPU tensors
or numpy arrays, one program after another, writing into the output in place.
`tl.launch(kernel, grid, *args, **kwargs)` (or Triton's own spelling,
`kernel[grid](*args, **kwargs)`) returns counters for the launch: `programs`,
`loads`, `stores`, `elements_loaded`, `elements_stored` (masked-off elements do not
count), which is how "never materializes the weights" becomes a cost check.

What is modeled, from the Triton documentation:

- `program_id(axis)`, `num_programs(axis)`; a grid of one to three ints, or a
  function of the keyword (meta-)parameters.
- Pointers: a tensor argument arrives as a pointer to its first element; `ptr +
  offsets` (in elements) is a block of pointers; `load(pointer, mask=None,
  other=None, *, cache_modifier="", eviction_policy="", volatile=False)` and
  `store(pointer, value, mask=None, *, cache_modifier="", eviction_policy="")`.
- `arange(start, end)` (see below for which bounds), `zeros(shape, dtype)`,
  `zeros_like`, `full(shape, value, dtype)` with `shape` a tuple (as the pages for
  `zeros` and `full` document) or a list (as the layer-norm tutorial writes:
  `tl.zeros([BLOCK_SIZE], dtype=tl.float32)`), `where(condition, x, y)`, `cast(input, dtype, fp_downcast_rounding=None,
  bitcast=False)` and `x.to(dtype)`, `sum(input, axis=None, keep_dims=False,
  dtype=None)`, `max` and `min` (`axis`, `keep_dims`), `maximum`, `minimum`, `clamp`,
  `abs`, `sqrt`, `rsqrt`, `exp`, `log`, `sigmoid`, `static_range`; the operators `+ -
  * / // % & | ^ << >> ~` and comparisons; `x[:, None]` and `x[None, :]`; dtypes
  `int1`, `int8` to `int64`, `uint8` to `uint64`, `float16`, `float32`, `float64`;
  host helpers `cdiv` and `next_power_of_2` (`triton.cdiv`, `triton.next_power_of_2`).
- Type rules of the "Triton Semantics" page: promotion by kind (bool < integer <
  float), then width, then unsigned; a Python number next to a tensor takes the
  tensor's dtype unless its kind is higher (`uint8 + 300` stays uint8, `int16 * 4.0`
  is float32); `/` and `%` with a float operand compute in at least float32 (a
  float16 operand is promoted); integer `//` and `%` round toward zero as in C, while
  arithmetic on `constexpr` values is Python's; `sum` widens integers narrower than
  32 bits to int32 or uint32.
- `tl.constexpr` parameters stay Python numbers and are the only values allowed as
  shapes and `arange` bounds. The shim also refuses `//` on floats (use `/`).

**Where the documentation stops, and what the shim does there.** Re-read for
milestone 5 on triton-lang.org: the pages `triton.language`, `tensor`, `arange`,
`zeros`, `full`, "Triton Semantics", and the layer-norm tutorial. Each item below is
a guess or a refusal, not a documented fact, and is the first thing to check on a
GPU:

| question | what the documentation says | what the shim does |
|---|---|---|
| bounds of `tl.arange(start, end)` | "start: Must be a power of two", "end: Must be a power of two greater than start"; yet every tutorial writes `tl.arange(0, BLOCK)`, and 0 is not a power of two | accepts only what passes both the wording (with 0 allowed as a start) and a power-of-two *length*: `(0, 2**m)` and `(2**k, 2**(k + 1))`. So `tl.arange(3, 7)` (length 4, refused by the wording) and `tl.arange(2, 8)` (allowed by the wording, length 6) are both errors. Write `start + tl.arange(0, BLOCK)`. |
| shapes of `zeros` and `full` | "tuple of ints"; the tutorial passes a list | accepts both; a list anywhere else in a kernel is still refused |
| must every block shape be a power of two? | not stated (only `arange`'s bounds are constrained) | yes: a guess, the strict reading |
| `>>` on signed integers | the `tensor` page says only that the operators exist ("`x+y`, `x << 2`, etc.") | zeros come in for unsigned values and for non-negative signed ones, where there is nothing to decide; `>>` on a **negative signed value raises**, because arithmetic and logical shifts differ there and nothing says which it is. A shift count outside `[0, bits)` raises too. |
| `/` on two integers | not stated | float32: a guess |
| dtype of scalar launch arguments | not stated | int32 (int64 when it does not fit), float32, int1: a guess |

The shift rule is why the `signed_shift` mutant of `cuda-triton-dequant-linear` is
rejected whatever real Triton does: a kernel that isolates a field with
`(word.to(tl.int32) << a) >> b` stops at the shift. A kernel that needs sign
extension has to do it with arithmetic the documentation defines (`tl.where(code >=
8, code - 16, code)`).

Where the shim is strict because a GPU would not tell you:

- an **unmasked access outside the tensor raises `tl.OutOfBounds`** (with the
  program, the element, and the tensor's size); negative offsets never wrap;
- a **masked-off element of a load without `other=` is poison**: NaN for floats, the
  largest value for integers. Triton documents it as undefined;
- **two programs storing to the same element raise `tl.DataRace`**, and programs run
  in a fixed shuffled order, so nothing may depend on program order;
- `tl.jit` **checks the kernel's source** and rejects what Triton could not compile
  in this subset: calling numpy or any function that is not `tl.<name>`, a block
  method, another `@tl.jit` function, or one of `range`, `min`, `max`, `int`,
  `float`, `bool`, `abs`, `print`; reaching into a block (`x.value`); lists, dicts,
  comprehensions, lambdas, `with`, `try`, imports, nested functions, f-strings.

How the mutants of the two Triton problems show: no tail mask reads past the end of
the tensor (`OutOfBounds`), or into the next row (a wrong answer or a `DataRace`);
`other=1.0` pollutes the sum (numeric); eps outside the root, nibble order, and the
group index are numeric; a float16 accumulator loses precision (numeric).

**What is and is not verified.** The signatures and semantics above were read from
triton-lang.org (`triton.language`, the pages for `load`, `store`, `sum`, `max`,
`cast`, `arange`, `zeros`, `where`, `full`, `tensor`, `program_id`, "Triton
Semantics", and the vector-add and layer-norm tutorials; the vector-add kernel is the
first unit test, and the first loop of the layer-norm kernel is another). **The shim has not been
run against real Triton**, which does not install on macOS; torch's Linux wheels
depend on it (`triton` 3.9.0 in `uv.lock`), so it is present on a CUDA host. On such
a host `tl.launch` with CUDA tensors hands the same kernel function to real Triton
(`tl.compile_with_triton`: the function's globals are copied with the shim replaced
by `triton.language`, and `triton.jit` is applied); that path is exercised only with
a stand-in `triton` module in the unit tests. Known gaps, where a kernel accepted
here might still be refused or behave differently there: Triton's rule that a
variable used after an `if` or `for` must be assigned on every path (documented, not
checked by the shim); the guesses in the table above; float results of GPU math
functions, which need not round like numpy's; float-to-int casts out of range (undefined in Triton);
bfloat16 (numpy has none). Treat the first GPU run of each Triton problem as a test
of the shim, and tell the judge's owner what differs.

### Facts about torch 2.15 behind this section

Verified here by running torch 2.15.0 on the CPU (macOS arm64, no CUDA) and from
the docstrings it ships:

- `torch.OutOfMemoryError` exists, is a subclass of `RuntimeError`, and
  `torch.cuda.OutOfMemoryError` is the same class.
- `torch.cuda.CUDAGraph(keep_graph=False)` has `capture_begin(pool=None,
  capture_error_mode="global", check_input_liveness=False)`, `capture_end()`,
  `replay()`, `reset()`, `pool()`, `enable_debug_mode()`, `debug_dump(path)`; on a
  CPU build the constructor raises "Tried to instantiate dummy base class
  CUDAGraph". `torch.cuda.graph(cuda_graph, pool=None, stream=None,
  capture_error_mode="global", ...)` is the documented way to capture.
- `torch.cuda.memory_stats()` returns an empty `OrderedDict` without CUDA, and
  `memory_allocated`, `max_memory_allocated`, `memory_reserved` return 0. Its
  documented keys are `"<stat>.<pool>.<metric>"` with stats `allocated`,
  `allocated_bytes`, `segment`, `reserved_bytes`, `active`, `active_bytes`,
  `inactive_split`, `inactive_split_bytes`, `requested_bytes`; pools `all`,
  `large_pool`, `small_pool` (the small pool takes allocations under 1 MB); metrics
  `current`, `peak`, `allocated`, `freed`; plus the counters `num_alloc_retries`,
  `num_ooms`, `num_sync_all_streams`, `num_device_alloc`, `num_device_free`,
  `max_split_size`, and `oversize_allocations.*`, `oversize_segments.*`.
- `torch.cuda.empty_cache()` is a no-op without CUDA; `torch.cuda.synchronize()`,
  `get_device_name()`, and `mem_get_info()` raise `AssertionError`.
  `torch.cuda.Event` has `record`, `elapsed_time`, `query`, `synchronize`, `wait`.
- `torch.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)`. On
  the CPU both `dtype=torch.bfloat16` and `torch.float16` are accepted: `nn.Linear`
  and `@` produce the 16-bit type, and so does `torch.softmax` (**on the CPU softmax
  is not kept in float32 by autocast**; on CUDA it is, per the op lists below). So a CPU case that requires float32
  probabilities is checking the solution's explicit `.float()`, which is the point.
  `device_type="cuda"` on a machine without CUDA only warns ("CUDA is not
  available ... Disabling autocast") and computes in float32: the hard-coded-device
  mutant is caught by checking the dtype of a hidden activation.
  `torch.is_autocast_enabled("cpu")` and `torch.get_autocast_dtype("cpu")` report the
  state. `torch.tensor(70000.0).half()` is `inf`; as bfloat16 it is 70144.
- `x.pin_memory()` and `torch.zeros(n, pin_memory=True)` succeed on this CPU-only
  build and report `is_pinned() == True`; `x.to("cpu", non_blocking=True)` and
  `x.cpu()` on a CPU tensor return the same object. Pinning therefore cannot be
  judged from the tensor on a CPU: use `TransferSpy`.
- `F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
  is_causal=False, scale=None, enable_gqa=False)`; passing `attn_mask` together with
  `is_causal=True` raises "Explicit attn_mask should not be set when
  is_causal=True".
- `torch.compile(fn, backend="eager")` works on the CPU with the numbers in the
  `CompileCounter` table; `torch.compiler.list_backends()` is `['cudagraphs',
  'inductor', 'openxla', 'tvm']`; the default inductor backend also compiles on this
  Mac.
- `torch.device("meta")` behaves as described under `DeviceSpy`.

Read from torch 2.15's documentation (`notes/cuda.html`, `amp.html`), not run:

- GPU operations are asynchronous; "time measurements without synchronizations are
  not accurate"; use `torch.cuda.synchronize()` or a pair of
  `torch.cuda.Event(enable_timing=True)`.
- Host-to-GPU copies are faster from pinned memory, and once a tensor is pinned
  `non_blocking=True` gives an asynchronous copy that can overlap with computation.
  The notes do not say what `non_blocking=True` does from pageable memory (it is
  commonly described as falling back to a synchronous copy).
- CUDA graphs: warm up with a few eager iterations before capture, on a side stream
  (`torch.cuda.graph` and `make_graphed_callables` set one); operations that
  synchronize the CPU with the GPU (`.item()`) and dynamic shapes are prohibited
  during capture; "every replay reads from and writes to the same (virtual) memory
  addresses"; the whole-network example fills static inputs with
  `static_input.copy_(data)` and calls `g.replay()`.
- `memory_allocated` counts memory occupied by tensors, `memory_reserved` everything
  the caching allocator holds; `empty_cache()` releases unused cached memory and
  cannot free memory occupied by tensors.
- Autocast: `softmax` and `log_softmax` are in the list of CUDA ops that autocast to
  float32 and in neither CPU list; the CPU lists are written for bfloat16 and
  "float16 shares the lists of bfloat16"; the default dtype is float16 for CUDA and
  bfloat16 for the CPU.

**Not verifiable without a GPU**, and therefore to be treated as documentation, not
fact, until a GPU case has run: everything about real `CUDAGraph` behavior (what an
un-warmed capture or a capture on the default stream does, speedups); the
values in a real `memory_stats()` dict and the allocator's rounding to 512-byte
blocks (the fixture dicts of `cuda-memory-accounting` cannot be captured here: build
them by hand from the documented keys and say so in the problem); what
`non_blocking=True` does from pageable memory; how CUDA autocast behaves in practice;
`mode="reduce-overhead"`; every speed ratio; and real Triton.

## 19. Contract additions of milestone 4 (to mirror in ARCHITECTURE.md)

Everything is additive; no event changes shape.

**3.1 Environment.** The `mlx` extra and `mlx-lm` in `models` carry the marker
`sys_platform == 'darwin' and platform_machine == 'arm64'`, so the same extras can be
passed on every host: `uv run --project python --extra jax --extra mlx ...` on Linux
installs torch's normal Linux wheels (CUDA 13 runtime libraries and `triton` come
with them) and no MLX. `uv.lock` stays universal. The `models` extra needs a source
build of `pyyaml` on Linux for Python 3.15 (no wheel is published); the harness and
the kit do not need that extra.

**3.5 Harness command.** One more environment variable: `LEARNAI_JAX_CACHE`, a
directory for JAX's persistent compilation cache, or `0`/`off` to disable it. Default:
`<root>/.learnai/cache/jax`, created on the first JAX run. Anything under
`<root>/.learnai/cache/` is disposable. A `JAX_COMPILATION_CACHE_DIR` set by the
user is respected instead.

**3.6 Event protocol.** No new fields. The one-line `message` of a case that ended
in `error` is the exception type and the first meaningful line of its text (a first
line that ends with a colon is joined with the next), for public and hidden cases
alike. The traceback is in the case's `detail`.

**3.7 CLI.** `learnai doctor [--json]`: what this environment has. It needs no extra and
no network, imports torch (a base dependency) and, when installed, MLX, to detect
hardware, and never fails because a package is missing. `learnai doctor --json` prints
one JSON document:

```jsonc
{
  "schema": 1,                      // bumped on an incompatible change
  "learnai": "0.1.0",
  "python": "3.15.2",
  "executable": "/abs/path/python/.venv/bin/python3",
  "platform": {"system": "Linux", "machine": "x86_64", "release": "6.8.0"},   // platform.system(), machine(), release()
  "frameworks": {"torch": "2.15.0", "jax": "0.12.0.1", "mlx": null},          // installed version, or null
  "packages": {"numpy": "2.5.4", "triton": "3.9.0", "huggingface_hub": null,
               "safetensors": null, "tokenizers": null, "transformers": null, "mlx-lm": null},
  "hardware": {
    "cpu": {"present": true, "cores": 32},
    "mlx-gpu": {"present": false, "name": null, "memory_bytes": null},        // "Apple M1 Max", unified memory size
    "cuda": {"present": true, "name": "NVIDIA GeForce RTX 4090", "memory_bytes": 25757220864,   // device 0
             "devices": [{"index": 0, "name": "NVIDIA GeForce RTX 4090", "memory_bytes": 25757220864}],
             "torch_cuda_version": "13.0"}                                    // torch.version.cuda, null on a CPU build
  },
  "hardware_override": null,        // LEARNAI_HARDWARE as a sorted list (always with "cpu"), or null
  "root": "/abs/content/root",
  "models_dir": "/abs/content/root/.learnai/models"                         // LEARNAI_MODELS_DIR applied
}
```

- `frameworks` and `packages` are read from installed package metadata; nothing is
  imported to fill them. A `null` framework means its extra is not installed here.
- `hardware.<name>.present` is real detection and is exactly what decides whether
  cases with that `requires` run: `torch.cuda.is_available()` for `cuda`,
  `mx.metal.is_available()` with a GPU default device for `mlx-gpu`.
  `LEARNAI_HARDWARE` does not change it; it is reported in `hardware_override`.
- `cuda.name` and `cuda.memory_bytes` describe device 0 (`null` when absent, or when
  the driver does not answer); `devices` lists all of them.
- Exit code 0. Without `--json` the same facts are printed as text.

`learnai validate` prints, per problem, which cases this machine could not validate
(`... (GPU cases not validated: <ids>)`), a different warning when every case ran but
the problem's `requires` is unmet, and a summary count of problems that passed with
unvalidated cases.

**Remote runs.** The harness is POSIX code with nothing specific to macOS on the
paths a Linux run takes: time limits use `signal.setitimer`/`SIGALRM` in the worker
(skipped where missing) and a supervising process that kills the worker; output is
captured by redirecting file descriptors 1 and 2 to a file in a `tempfile`
directory, with `fflush` through `ctypes.CDLL(None)`; no path is hard-coded. MLX is
imported only by the MLX adapter, the MLX kit tools, and hardware detection, each
guarded. A remote host needs Python 3.15, `uv`, the `python/` directory, and the
problem directory at `<root>/problems/<track>/<id>` (or `LEARNAI_ROOT`); assets are
looked up under that root.

## 20. Contract additions of milestone 5 (to mirror in ARCHITECTURE.md)

Everything is additive. A run of a problem that uses no stage produces exactly the
events it produced before.

**3.3 Problem directory.** `tests.py` may group its cases into stages:
`@case("...", stage="2: paged KV cache")`. A stage is identified by its label (one
line, at most 60 characters); stages are ordered by the first case that names them.
A problem gives every case a stage or none.

**3.6 Event protocol.** Each entry of `start.cases` may carry `"stage": "<label>"`;
the key is absent for a case without a stage. Nothing else changes: `case_start`,
`case_end`, and `end` do not mention stages, and a consumer derives per-stage
progress from `start.cases` and the `case_end` statuses.

```jsonc
{"event":"start", ..., "cases":[
   {"id":"alloc_fork","name":"fork shares every block","public":true,"stage":"1: block allocator"},
   {"id":"alloc_reuse","name":"freed blocks are reused","public":false,"stage":"1: block allocator"},
   {"id":"fcfs_trace","name":"FCFS: the trace equals the oracle's","public":true,"stage":"2: continuous batching"}]}
```

- A stage label is sent as written, for public and hidden cases alike.
- Stage order is the order of first appearance in `start.cases`. With `--case` or in
  `run` mode, `start.cases` lists only the selected cases, each with its stage.
- Stages do not gate anything: every selected case runs, and the final `status` is
  computed from all cases as before.
- Presentation: when any case has a stage, show the label as "Stage <label>" (as
  written if it already begins with "Stage"), group cases under it in
  first-appearance order, and show a pass count per stage ("3/5 passed").
- `case_end.duration_ms` does not include time the judge spent, inside the case, on
  one-time library initialization that is not the solution's doing (the first import
  of `torch._dynamo`, about 2.5 s, and the first `torch.compile`); the case's time
  limit is suspended for that time as well. `end.duration_ms` is wall time and
  includes it.

**3.7 CLI.** `learnai run` and `learnai submit` (text output) print a heading line `Stage
<label>` whenever the stage of the next case differs from the previous one's, and,
after the verdict line, one line per stage in first-appearance order: `<passed>/<total>
passed[, <k> skipped]  Stage <label>`. `--events` output is the protocol, unchanged
but for the `stage` key. `learnai validate` fails a problem in which some cases have a
stage and others do not (`tests.py: N of M cases have no stage (<ids>); give every
@case a stage= or none`).
