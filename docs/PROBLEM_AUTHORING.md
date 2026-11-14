# Problem authoring guide

This is the authoritative guide for writing Whetstone problems. It covers the directory
format, the test API for all seven judge types, the framework adapter, the kit, the
rules that keep judging deterministic, and the validation workflow. You should not need
to read the judge's source to write a problem.

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
file, and cases that need no download, section 15).

All commands below run from the repository root. Always pass both extras; a bare
`uv sync` removes the optional frameworks.

```sh
uv run --project python --extra jax --extra mlx ws <command>
```

The rest of this guide abbreviates that prefix to `ws`. Downloading model weights and
generating fixtures from a reference implementation additionally need `--extra models`
(section 15); nothing else does.

## 1. Workflow

```sh
ws new decoding/top-p-filter --title "Nucleus (Top-p) Filtering"   # scaffold
# ... write the problem; replace every TODO(author) marker ...
ws submit top-p-filter -f torch --reference        # iterate on the tests
ws submit top-p-filter --mutant forgets_renormalize
ws run top-p-filter -f mlx --starter               # what a learner sees first
ws validate top-p-filter -v                        # the gate; must pass before merging
uv run --project python --extra jax --extra mlx ruff check problems/decoding/top-p-filter
```

`ws new <track>/<id>` takes `--frameworks torch,jax,mlx` (the default),
`--frameworks python`, or `--frameworks mlx` (an MLX-only problem, section 13), plus
`--title` and `--difficulty`. It writes a complete toy
problem in the house layout and creates `track.toml` if the track is new. `ws validate`
refuses any problem that still contains `TODO(author)`.

`ws run` executes public cases, `ws submit` all cases. Both take `-f <framework>` and one
of `--reference`, `--starter`, `--mutant <name>`, or `--solution <path>`; with none of
them they judge `workspace/<id>/<framework>.py`. `--case <id>` runs a single case and
`--events` prints the raw protocol.

`ws run` and `ws submit` show what a learner sees: a hidden case is `hidden-3` /
"Hidden case 3", with its one-line message and nothing else. Add `--reveal` while you
write a problem to see hidden cases under their real ids and names, with the failure
detail. `--case` takes the id as it is displayed: `--case hidden-3`, or
`--reveal --case very_negative`. `ws validate` always reports real ids.

### What `ws validate` enforces

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
  (all passed); 1 was NOT validated: needs an NVIDIA GPU (CUDA) ..."). The problem is
  still `ok`, but the warning is printed on every run, not only with `-v`.
- The starter must be rejected by the cases that run here. A starter that survives
  them is an error, whatever the skipped cases would have said.
- A mutant that survives the cases that run here is reported as "NOT validated": only
  a machine that runs the skipped cases can tell whether a test catches it.
- If nothing could run (every case needs the missing asset or hardware), the problem
  is reported as `SKIP`, with the reason, and the summary line counts it separately:
  "80 of 81 problems passed validation; 1 skipped (not validated on this machine, see
  above)". A skip does not fail the command. Static checks (metadata, files, the
  `forbidden` list) still run.
- `ws validate --require-assets` makes a missing asset an error. Use it on the machine
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

from whetstone.judge import case

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

### `@case(name, *, public=False, id=None, time_limit_s=None, frameworks=None, requires="cpu", assets=None)`

- `name`: what the learner reads. State what is checked: "large logits: a row spanning
  -1e4 to 1e4", not "test 3".
- `public=True`: runs on Run and Submit; name, failure detail, and inputs are shown.
  `public=False` (the default): runs on Submit only; the learner sees "Hidden case N",
  the one-line failure message, and never the detail.
- `id`: defaults to the function name. Ids are stable identifiers; do not rename them
  casually: the id seeds the case's random generator. An id of the form `hidden-N` is
  reserved.

**What leaves the harness for a hidden case.** Neither the id nor the name: a function
called `masked_logits` would tell the learner what the hidden test checks. On the wire,
in the app, and in `ws run` / `ws submit`, a hidden case is `hidden-N` and "Hidden case
N", where N counts the hidden cases of `tests.py` in definition order (all of them,
whatever the framework, so `hidden-3` is the same test for everyone). Inserting or
reordering hidden cases renumbers the ones after it. So name hidden cases and their
functions for yourself and for reviewers, as descriptively as public ones.
- `time_limit_s`: overrides the problem's per-case limit.
- `frameworks`: restricts the case to some solution frameworks. Rarely needed.
- `requires`: the hardware the case needs, `"cpu"` (the default: it runs on every
  machine) or the problem's own `requires` (`"mlx-gpu"`, `"cuda"`). Where the hardware
  is absent the case is `skipped` with the reason. Section 14.
- `assets`: the names of the problem's `[[assets]]` the case needs. The default is all
  of them; `assets=[]` marks a case that needs no download. A case whose asset is not
  on disk is `skipped` with the fetch command. Section 15.

Cases run in definition order. Put the worked example from the statement first, then
progressively harder public cases, then hidden ones.

**Public versus hidden.** Public cases teach: they are the examples from the statement
and the main edge-case families, so a learner can debug with Run. Hidden cases make
sure the solution is general: more edge cases, random sweeps, and whatever each mutant
needs. A good split is 3 to 5 public and 4 to 8 hidden.

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

call: softmax(x, axis)
  x = float32 array, shape (2, 3):
    [[  1001.,   1002.,   1003.],
     [ 10000.,      0., -10000.]]
  axis = -1
shape (2, 3), dtype float32
tolerance: |got - expected| <= 1e-06 + 0.0001 * |expected|
mismatched values: 4 of 6
max abs error: inf at index [0, 0] (expected 0.09003057, got nan)
expected =
  [[0.090031, 0.244728, 0.665241],
   [1.      , 0.      , 0.      ]]
got =
  [[nan, nan, nan],
   [nan,  0.,  0.]]
```

A shape mismatch is reported first and alone (`softmax: got shape (4,), which is not the
expected shape`), then dtype, then values with the worst index and an excerpt around
the worst element; the expected shape, the expected dtype, and the exact errors are in
the detail.

**The message is shown for hidden cases; the detail is not.** So a message must never
reveal what the test expects or what it passed in: no expected value, shape, length,
dtype, set of keys, or exception type, and no input. A message says what the solution
*produced* and where it went wrong; the detail carries the expected side. All `check`
helpers follow this:

| failure | message (always shown) | detail (public cases only) |
|---|---|---|
| wrong shape | `tokens: got shape (4,), which is not the expected shape` | both shapes |
| wrong length | `tokens: got 3 items, which is not the expected number` | both lengths, both lists |
| wrong values | `y: 2 of 60 values differ from the expected result (largest error on the order of 1e-01, at index [1, 13])` | exact errors, expected and got |
| wrong dtype | `y: got dtype float64, which is not the expected dtype` | both dtypes |
| wrong dict keys | `stats: got a dict whose keys are not the expected keys` | both key lists |
| wrong exception | `decode: raised KeyError, which is not the exception the statement specifies here` | the expected type |

The error of a float comparison is given as an order of magnitude only: next to the
learner's own value, the exact error would give the expected value away, while the
magnitude still tells "float noise, loosen nothing, fix the formula" (1e-01) from "a
precision problem" (1e-05).

One deliberate exception: **limits**. The limit of `check.cost`, `check.at_most`,
`check.memory`, and the speed judge is reported with the metric on every case, hidden
or not, because the app draws the bar from it, and the failure message repeats it
(`tokens_processed: 114 tokens exceeds the limit of 15 tokens`). Never use a secret as
a limit; state limits in the statement.

When you write your own failure with `check.that(condition, message, detail)`, follow
the same rule: describe what the solution did in `message`, put the specifics in
`detail`. Plain `assert` works too (it counts as a failed case), but its text becomes
the message, so prefer `check.that`.

An exception raised by the solution is an `error` (verdict `runtime_error`).

- In a **public** case the message is the exception line and the detail is a traceback
  trimmed to the learner's own frames (or to the frames of `tests.py` when the
  learner's file is not on the stack at all).
- In a **hidden** case there is no detail, like for every hidden outcome: a traceback
  through `tests.py` would show the test's source and inputs. Everything the learner
  needs to debug their own code goes into the one-line message: their exception type
  and text (cut to 200 characters), then the innermost line of their own file that
  was running.

  ```
  IndexError: index 8 is out of bounds for dimension 1 with size 8 (torch.py line 14, in top_p_filter)
  ```

  When the exception was raised outside the learner's file and none of their frames
  is on the stack (the test tripped over a returned value), the message is the
  exception line alone. Note that the exception text is the learner's own: a solution
  can always print or raise what it was given, so hidden inputs are not secret from a
  learner who goes looking. What stays hidden is what the test is called, what it
  expects, and how it is written.

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

Two kinds of names cannot be guarded, and `ws validate` refuses them with the reason
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
  softmax) `` and the verdict is `wrong_answer`. The message is the same in hidden
  cases. Catching the exception does not help: the violation is recorded, and the case
  fails even if the solution then computes the right answer another way.
- Any other caller gets the original function. The oracle, the same-framework
  reference (`ctx.baseline`, `ctx.check_speed`), your tests, and the kit are not the
  solution file, so they are unaffected. **References must not need a forbidden call**
  all the same: `ws validate` judges each reference as a solution, and it would fail.
- The guard is transitive. `torch.nn.Softmax(dim)(x)` ends up in
  `torch.nn.functional.softmax`, so it is caught without being listed. The price: do
  not forbid a low-level utility that allowed library functions use internally
  (forbidding `time.monotonic` in a JAX problem fails correct solutions, because JAX
  reads the clock while it compiles). Forbid the specific shortcut, and let
  `ws validate` confirm that the references still pass.
- Names from all frameworks go in one list. Only the names of packages that are
  loaded are guarded, and the list is checked again before every case, so a JAX
  solution that imports torch to borrow its `softmax` is caught as well.
- List every spelling a learner would reach for. Different objects need separate
  entries: `torch.softmax`, `torch.nn.functional.softmax`, `torch.special.softmax`, and
  the method `torch.Tensor.softmax` are four functions. Aliases of one object
  (`jax.nn.logsumexp` is `jax.scipy.special.logsumexp`) need one entry, and listing
  both is harmless.
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
`` `bfloat16` is not allowed in this problem (used at torch.py line 6) ``, on public and
hidden cases alike.

It is a blunt instrument, by design: it cannot tell `torch.bfloat16` from a local
variable the learner happened to call `bfloat16`. Use it for identifiers nobody uses by
accident (a dtype name, a package name such as `safetensors` or `tokenizers`), prefer
`forbidden` whenever the thing is called, and never hand-roll a source scan in
`tests.py`: `problems/foundations/bf16-decode` shows the supported way. The starter
and the references must not contain the identifier either (docstrings may).

Every problem with a `forbidden` or `forbidden_source` list needs a mutant that takes
the shortcut (section 11), and the statement must still state the rule in words.

### The `check` helpers

`from whetstone.judge import check`

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
  band, and confirm it with `ws validate`. In a tensor framework a call costs tens of
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
from whetstone.kit import TinyLM

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
  The lower-level pair `whetstone.judge.speed.compare(run_solution, run_reference)`
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
from whetstone.judge import case, check, percentile
from whetstone.kit import SimModel


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
from whetstone.judge import case, check, read_sse


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
`ListNode`. They live in `whetstone.kit`, are small, and are documented for learners,
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
from whetstone.kit import draft_target_pair

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

`whetstone.kit.ops` holds reference implementations of concepts from earlier problems,
so a later problem can build on them without asking for them again. One signature in
every framework: the call is routed by the type of the first argument (torch, JAX,
MLX, or numpy), and each framework has a native implementation written from elementary
operations, so `ops` keeps working in problems whose `forbidden` list rules out the
built-in `softmax`, `topk`, or attention kernel. Solutions, references, and tests may
all use it.

```python
from whetstone.kit import ops

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

The three filters work on the last axis and return **logits**: removed entries become
`-inf`, kept entries are unchanged, already-masked entries stay masked. That makes
them chain, and `ops.softmax` of the result is the renormalized distribution. Note the
difference from the `top-p-filtering` problem, whose `top_p_filter` returns
probabilities: `ops.softmax(ops.top_p_filter(logits, p))` equals that problem's output.

If your problem *teaches* one of these, forbid the kit version too, by saying so in the
statement and adding a mutant that calls it; `forbidden` entries such as
`whetstone.kit.ops.softmax` resolve like any other name.

### DecoderConfig, decoder_params, decoder_forward_ref

Weights for a small modern decoder (RMSNorm pre-norm blocks, RoPE, grouped-query
attention, SwiGLU, tied head), named like a Hugging Face checkpoint, plus the numpy
oracle. Use them when the solution must run the forward pass itself.

```python
from whetstone.kit import DecoderConfig, decoder_forward_ref, decoder_params, layer_params

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
from whetstone.kit import ToyVocab

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
from whetstone.kit import DEAD, CharDFA

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
from whetstone.kit import CharDFA, DFAConstraint, ToyVocab

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
from whetstone.kit import CostModel, SimEngine, workloads

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
block_tokens=16)`.

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
from whetstone.kit import SimRequest, workloads

requests = workloads.poisson(ctx.rng, rate=12.0, n=150, prompt_tokens=(16, 128),
                             output_tokens=(2, 40), priorities=(0, 1), tenants=("a", "b"))
requests = workloads.bursty(ctx.rng, bursts=5, burst_size=8, period_s=1.0)   # exact ties
requests = workloads.replay([(0.0, 4, 3), (0.5, 2, 2)])   # (arrival_s, prompt, output)
workloads.drive(sim, requests, scheduler.submit)
```

- `SimRequest(id, arrival_s, prompt_tokens, output_tokens, priority=0, tenant=None,
  deadline_s=None)`: frozen. Token counts are ints >= 1. `priority`: smaller is more
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
from whetstone.kit import ScriptedLM

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

### SimModel and Counters

`SimModel` is described in section 5.6. `Counters("a", "b")` is the counter object for
new kit objects: `counters.add("a", n)`, attribute access, `snapshot()`, `reset()`.

### Real models, MLX tools, CountingFile

The kit objects added for milestone 3 are documented where they are used:
`MLXTinyLM`, `EvalSpy`, `TraceCounter`, and `mlx_memory` in section 13; `RealModel`,
`RealLM`, `RealConfig`, `Weights`, `SafetensorsFile`, `ByteLevelBPE`, `dequantize`,
`unpack_codes`, `bf16_to_f32`, `CountingFile`, and the house models `SMALL` and
`LARGE_8BIT` in section 15.

### Adding a kit object

Add it under `python/src/whetstone/kit/`, export it from `whetstone/kit/__init__.py`,
write tests under `python/tests/`, and document it here. Requirements: deterministic
(seeded), implemented once (numpy or plain Python), speaking the solution's framework
at its boundary through an adapter passed as `fw`, functional (state in, state out), a
`counters` attribute if it can measure work, and a `with_framework(fw)` method that
returns a twin with fresh counters. Write it for a reader: learners will open the file.

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
  can say "Figure 1"). Write it as one or two full sentences that say what to see, not
  a title. It may contain `$...$` math, emphasis, and inline code, and must not
  contain a line break.
- The path is `assets/<file>`, relative to the problem directory. Only `.svg` and
  `.png` files are served, each at most 1 MiB, and a path may not leave `assets/`.
  Subdirectories are allowed.
- An image in the middle of a sentence is not a figure and is not resolved. Give every
  figure its own paragraph.
- Use a figure where a picture carries something the formulas do not: a geometric
  action, a memory layout, a timeline. `problems/transformer/rope/assets/pair-rotation.svg`
  is the model to copy.

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
3. **Other colors come from the figure palette**, five CSS variables defined by the
   design tokens. A presentation attribute cannot hold `var()`, so set them through
   `style`:

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

   Two hues are the whole budget. Never let color carry a distinction alone: label the
   series, or vary the dash pattern as well. Green, red, and amber mean pass, fail, and
   warning in the app; do not use them.
4. **Text is at least `font-size="13"`**, the size of the caption, in a drawing 480
   units wide (scale that up in proportion for a wider `viewBox`). Text is set in the
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
   cannot see it. The caption is the figure's accessible name in the app.
8. **Write the file by hand or from a small script, and keep it readable**: grouped,
   commented, coordinates rounded to one decimal. Do not commit an export from a
   drawing program with its metadata, embedded fonts, and inline styles.

Opened on its own, such a file shows only the `currentColor` parts, because the
palette variables exist only inside the app. To check a figure, open the problem in
the app (`pnpm dev:real`) and look at it in both themes, at the default window size and
with the statement pane dragged to its narrowest: every label must be legible and no
two may touch.

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
`ws validate -v` prints, for each reference submit, the total duration and the time
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
  `ws validate -v` prints the time in cases per framework; when the JAX figure is ten
  times the torch one, count your distinct shapes.
- Keep tensors small: tests check correctness, not scale. A few thousand elements is
  plenty except in speed cases.
- `time_limit_s` is per case; 10 is right for almost everything. A correct solution
  should use under 5 percent of it. Lower it (2 to 5) when a wrong solution's typical
  failure is an infinite loop.
- Statistical cases dominate their problem's budget: N calls into a tensor framework.
  Keep each statistical case under about 3 s in **every** framework, measured with
  `ws submit <id> -f <framework> --reference` for each one, not only torch: the same
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
  rejected, and `ws submit <id> --mutant uses_builtin` must show the "is not allowed in
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
ws new mlx/mlx-lazy-eval-placement --frameworks mlx
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
from whetstone.kit import MLXTinyLM

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
from whetstone.kit import EvalSpy, MLXTinyLM

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
from whetstone.kit import TraceCounter

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
from whetstone.kit import mlx_memory

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
from whetstone.kit import CountingFile

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
bytes. `f.counters.bytes_read` and `f.counters.reads` are the totals;
`f.pages_touched` is the number of distinct pages the requested ranges cover. Counting
is by the ranges asked for, not by page faults the operating system saw, so it is
exact and identical on every machine: a loader that takes `f[:]` or `f.read()` and
picks out one tensor is charged for the whole file.

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
   no case was marked and every case passed (for example `ws run`, which executes only
   the public cases, on a `cuda` problem whose public cases are all CPU cases). An
   `accepted` verdict therefore always means: every case ran, on the required
   hardware.

What a learner without the hardware sees, in the app and in `ws submit`:

```
cuda-scale / torch / submit  python 3.15.2, device cpu, torch 2.15.0
  PASS  cpu        scales on the CPU  0.4 ms
  PASS  hidden-1   Hidden case 1  0.2 ms
  SKIP  hidden-2   Hidden case 2
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
  a marked case catches is reported by `ws validate` as "NOT validated" on a machine
  without the hardware (section 1), which is honest and unsatisfying.
- For the MLX track (`requires = "mlx-gpu"`), mark the cases that need the GPU:
  memory limits, speed, Metal kernels. Apple silicon always has the GPU, so in
  practice everything runs; the marks decide what happens elsewhere, and they tell
  the reader which cases are about the device.

To see how a problem behaves on a machine you do not have, override detection:

```sh
WHETSTONE_HARDWARE=cpu ws submit cuda-scale --reference     # pretend: no GPU of any kind
WHETSTONE_HARDWARE=cpu,cuda ws submit cuda-scale --reference   # pretend CUDA is there
```

`WHETSTONE_HARDWARE` is a comma-separated list of what to treat as present (`cpu` is
always included). It changes which cases are skipped, nothing else: pretending CUDA
exists on a Mac makes the CUDA cases run and fail with torch's own error. Without the
variable, `mlx-gpu` means `mx.metal.is_available()` with a GPU default device, and
`cuda` means `torch.cuda.is_available()`.

`python/tests/fixtures/m3/problems/gpu/cuda-scale` is a complete miniature.

## 15. Model assets and real-model problems

Some problems run on the weights of a real open model. Weights are never committed:
a problem *declares* what it needs, the learner downloads it once with `ws assets
fetch` (or the app's button), and it lives in `.whetstone/models/` under the
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

Both are in the kit as `whetstone.kit.SMALL` and `LARGE_8BIT` (`AssetSpec` objects),
and `ws assets fetch <name>` knows them before any problem declares them, so you can
download first and write the problem second:

```sh
uv run --project python --extra models ws assets fetch granite-4.0-350m
```

Use the house models unless a problem truly cannot. Every extra model is another
download for the learner.

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
uv run --project python python -c "from whetstone.kit import SMALL; from whetstone.kit.realmodel import asset_toml; print(asset_toml(SMALL))"
```

| key | meaning |
|---|---|
| `name` | The directory under `.whetstone/models/` and the name used by `ctx.asset`, `@case(assets=...)`, and `ws assets fetch`. Lowercase letters and digits with `.`, `_`, `-` between them. |
| `repo` | The Hugging Face repository id. |
| `revision` | The full 40-character commit hash. A branch name such as `main` is refused: it can move, and fixtures are only valid for the bytes they were made from. |
| `files` | File names or glob patterns inside the repository. Only `*.safetensors` and the tokenizer and config files can be assets (`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `vocab.json`, `merges.txt`, `chat_template.jinja`, `model.safetensors.index.json`). Nothing that executes or unpickles is ever downloaded: no `.bin`, `.pt`, `.pkl`, `.py`, even when a glob would match it. A pattern that selects nothing is an error. |
| `size_mb` | The approximate download size in megabytes, shown before downloading. |

One asset name means one repository and revision across the whole catalog (validation
checks it); `fetch` downloads the union of the `files` that the problems using a name
ask for. A problem may declare several assets.

### What the learner and the harness do

- `ws assets list` shows every asset the catalog declares, whether it is present, its
  size, and which problems use it. `ws assets fetch <name>...` (or `--all`) downloads.
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
  downloaded; fetch it with `ws assets fetch granite-4.0-350m`". The other cases run.
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
directory: `WHETSTONE_MODELS_DIR=/nonexistent ws submit <id> --reference`.
(`WHETSTONE_MODELS_DIR` replaces `<root>/.whetstone/models` everywhere.)

### The machine-readable forms (what the desktop app calls)

Both commands read the catalog under the content root (`WHETSTONE_ROOT`, else the
working directory's repository).

`ws assets list --json` prints one JSON document on stdout and exits 0. It needs no
extra and no network.

```json
{
  "models_dir": "/abs/learnai/.whetstone/models",
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
      "path": "/abs/learnai/.whetstone/models/granite-4.0-350m",
      "bytes_on_disk": 0,
      "problems": ["real-model-block", "real-model-logits"],
      "fetch_command": "ws assets fetch granite-4.0-350m"
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

`ws assets fetch <name>... --json` (or `--all --json`) writes one line per event to
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
from whetstone.kit import RealModel

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

`RealLM` implements exactly this list (`whetstone/kit/realmodel.py`, about 150 lines,
meant to be read by authors, not linked from statements of problems that ask for it).
The large model needs the same list with the multipliers at 1, separate MLP matrices,
a separate head, and quantized linear layers.

**Tolerances that hold** (small model, 16-token prompts): float32 in numpy, torch,
JAX, or MLX agrees with the fixture to 1e-4 absolute, so `rtol=1e-3, atol=1e-3` is
safe; an MLX forward in float16 is off by up to 0.06 and in bfloat16 by up to 0.4,
with the same argmax on ordinary prompts. For a float16 case use `atol=0.25` and
compare argmax only where the fixture's top-two margin exceeds 1.

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

## 16. Checklist before you open a pull request

- `ws validate <id> -v` passes, with no warning.
- `ruff check problems/<track>/<id>` is clean.
- `ws run <id> -f <each framework> --starter` shows a failure a beginner can act on.
- `ws submit <id> --mutant <each>`: every mutant fails for the reason its `# BUG:` line
  gives, and the failure message would make sense to a learner.
- `ws submit <id> --mutant <one>` without `--reveal` shows nothing about the hidden
  cases beyond "Hidden case N" and a message free of expected values.
- If the statement forbids a call: `forbidden` lists every spelling of it in every
  framework, and a mutant takes the shortcut. If it forbids something that is not
  called (a dtype, a package), `forbidden_source` names the identifier.
- If the problem has `requires` other than `cpu`: `WHETSTONE_HARDWARE=cpu ws submit
  <id> --reference` passes every case that still runs, skips the rest with a clear
  reason, and ends with "Not accepted: hardware missing" (section 14).
- If the problem has `[[assets]]`: it validates with the asset present
  (`ws validate <id> --require-assets`) and behaves sensibly without it
  (`WHETSTONE_MODELS_DIR=/nonexistent ws submit <id> --reference`); fixtures are small
  and `fixtures/make.py` regenerates them (section 15).
- If a case has a memory limit: the reference uses at most two thirds of it and the
  mutant it is written for at least one and a half times it (section 13).
- Every statistical case takes under about 3 s in every framework.
- The first public case is the statement's worked example.
- No expected values appear in any failure message of a hidden case.
- The statement specifies every choice that affects the output, and its Constraints
  section states the tolerance and any cost or speed limit.

## 17. Contract additions of milestone 3 (to mirror in ARCHITECTURE.md)

The judge implements the following; each block is worded to be pasted into the named
section of `docs/ARCHITECTURE.md`. Everything is additive: a run of a problem with
`requires = "cpu"` and no `[[assets]]` produces the same events as before, plus
`end.skipped` and `start.requires`.

**3.1 Environment.** Optional extras: `jax`, `mlx`, `models` (huggingface_hub,
safetensors, tokenizers, transformers, and mlx-lm on macOS). `models` is needed only
by `ws assets fetch` and by authors generating fixtures; the harness and the kit read
downloaded files with numpy.

**3.2 Frameworks.** A problem may be MLX-only (`frameworks = ["mlx"]`). Its oracle is
`reference/mlx.py` and its mutants are written in MLX. Every other tensor problem
includes `torch`.

**3.3 Problem directory.**

```toml
requires = "cpu"                 # cpu | mlx-gpu | cuda: hardware needed to be accepted
forbidden_source = ["bfloat16"]  # optional: identifiers that may not appear in the solution's source

[[assets]]                       # optional, repeatable: model files the problem needs
name = "granite-4.0-350m"        # directory under <root>/.whetstone/models/
repo = "ibm-granite/granite-4.0-350m"
revision = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"   # full commit hash
files = ["config.json", "tokenizer.json", "model.safetensors"]   # names or globs; safetensors, tokenizer, and config files only
size_mb = 712                    # approximate download size
```

Optional `fixtures/` directory: committed reference outputs loaded by the tests.
Cases declare what they need with `@case(requires=..., assets=[...])`; the default is
`requires="cpu"` and every asset of the problem. Assets live in
`<root>/.whetstone/models/<name>/` (gitignored) beside a `whetstone-asset.json`
manifest; an asset is present when the manifest matches the declared repository and
revision, covers the declared files, and every file has its recorded size.

**3.5 Harness command.** No new flags. Two environment variables:
`WHETSTONE_MODELS_DIR` (replaces `<root>/.whetstone/models`) and `WHETSTONE_HARDWARE`
(comma-separated list of hardware to treat as present, for tests and previews).

**3.6 Event protocol.**

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

**3.7 CLI.** `ws assets list [--json]`, `ws assets fetch <name>... | --all [--json]
[--no-verify]` (schemas in section 15), and `ws validate --require-assets`. `validate`
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

