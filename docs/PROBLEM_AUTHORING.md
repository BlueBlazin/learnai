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

All commands below run from the repository root. Always pass both extras; a bare
`uv sync` removes the optional frameworks.

```sh
uv run --project python --extra jax --extra mlx ws <command>
```

The rest of this guide abbreviates that prefix to `ws`.

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

`ws new <track>/<id>` takes `--frameworks torch,jax,mlx` (the default) or
`--frameworks python`, plus `--title` and `--difficulty`. It writes a complete toy
problem in the house layout and creates `track.toml` if the track is new. `ws validate`
refuses any problem that still contains `TODO(author)`.

`ws run` executes public cases, `ws submit` all cases. Both take `-f <framework>` and one
of `--reference`, `--starter`, `--mutant <name>`, or `--solution <path>`; with none of
them they judge `workspace/<id>/<framework>.py`. `--case <id>` runs a single case and
`--events` prints the raw protocol.

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
- `frameworks` is either `["python"]` or a set of tensor frameworks that includes
  `"torch"` (the oracle is written in torch).
- `track.toml` exists with `id` (equal to the directory name), `title`, `description`,
  and `order`.
- `statement.md`, `hints.md`, `editorial.md`, and `tests.py` exist and are not empty; no
  hint is empty.
- `starter/<framework>.py` and `reference/<framework>.py` exist for every listed
  framework, and for no other; a starter is not identical to its reference.
- At least one mutant exists, and line 1 of every mutant is `# BUG: <what is wrong>`.
- The tests define at least one public and at least one hidden case.
- Nothing contains `TODO(author)`.

It warns (without failing) when the cases of a reference submit take more than 3
seconds in total.
Harness runs execute in parallel (`-j N`, default 4). If a speed-judged problem flakes
under parallel load, check it with `-j 1` and widen its factor.

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
```

`track.toml`:

```toml
id = "foundations"
title = "Foundations"
description = "One or two sentences shown on the track page."
order = 10
```

`problem.toml` (every key is required):

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
```

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

### `@case(name, *, public=False, id=None, time_limit_s=None, frameworks=None)`

- `name`: what the learner reads. State what is checked: "large logits: a row spanning
  -1e4 to 1e4", not "test 3".
- `public=True`: runs on Run and Submit; name, failure detail, and inputs are shown.
  `public=False` (the default): runs on Submit only; the learner sees "Hidden case N",
  the one-line failure message, and never the detail.
- `id`: defaults to the function name. Ids are stable identifiers; do not rename them
  casually.
- `time_limit_s`: overrides the problem's per-case limit.
- `frameworks`: restricts the case to some solution frameworks. Rarely needed.

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
| `ctx.check_speed(name, *args, max_ratio=..., **kwargs)` | Speed judge (section 5.5). |
| `ctx.randn(*shape, scale=1.0, loc=0.0)` | float32 normal array from `ctx.rng`. |
| `ctx.uniform(*shape, low=0.0, high=1.0)` | float32 uniform array in `[low, high)`. |
| `ctx.randint(low, high, shape)` | int64 array in `[low, high)`. |
| `ctx.rng`, `ctx.seed` | The case's numpy `Generator` and its seed. |
| `ctx.child_rng(i)` | An independent generator for trial `i` (property tests). |
| `ctx.metric(name, value, unit=None, limit=None)` | Report a number with the case. |
| `ctx.solution`, `ctx.oracle`, `ctx.reference` | The modules: the learner's, the oracle reference, the same-framework reference. |
| `ctx.fw`, `ctx.oracle_fw` | Framework adapters (section 6) for the solution and for the oracle. |
| `ctx.framework`, `ctx.problem` | The solution's framework name; the problem metadata. |
| `ctx.simclock()` | A fresh simulated clock, closed after the case (section 5.6). |
| `ctx.serve(app=None)` | Serve the solution's ASGI app (section 5.7). |
| `ctx.defer(fn)` | Run `fn` after the case, whatever the outcome. |

`ctx.call`, `ctx.expected`, and `ctx.baseline` convert for you: every numpy array in the
arguments (also inside lists, tuples, and dicts) becomes a tensor of the target
framework, and every tensor in the result becomes a numpy array. Python scalars, lists
of ints, strings, and other objects pass through unchanged. Kit objects such as `TinyLM`
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

A shape mismatch is reported first and alone (`softmax: shape mismatch: expected
(4, 8), got (4,)`), then dtype, then values with the worst index, the maximum absolute
and relative error, and an excerpt around the worst element.

**The message is shown for hidden cases; the detail is not.** So a message must never
contain expected values. All `check` helpers follow this. When you write your own
failure with `check.that(condition, message, detail)`, put the specifics in `detail`.
Plain `assert` works too (it counts as a failed case), but its text becomes the message,
so prefer `check.that`.

An exception raised by the solution is an `error` (verdict `runtime_error`), reported
with a traceback trimmed to the learner's own frames.

### The `check` helpers

`from whetstone.judge import check`

| helper | use |
|---|---|
| `check.allclose(got, expected, *, rtol=1e-4, atol=1e-6, label="output", dtype=False)` | Float comparison. NaN matches NaN; infinities must match exactly. |
| `check.equal(got, expected, *, label="output")` | Exact equality: int/bool/token arrays (integer width ignored), lists, strings, dicts. |
| `check.match(got, expected, ...)` | Dispatches to the two above by structure; what `ctx.check` uses. |
| `check.shape(got, shape, *, label)` | Shape only. |
| `check.finite(got, *, label)` | No NaN or infinity. |
| `check.that(condition, message, detail=None)` | Any other assertion. |
| `check.raises(ExcType, fn, *args, label="call")` | Specified error behavior. |
| `check.forall(prop, *, trials=25)` | Property judge. |
| `check.distribution(samples, probs, *, delta=1e-10, label)` | Statistical judge, categorical. |
| `check.mean(values, expected, *, low, high, delta=1e-10, label)` | Statistical judge, bounded mean. |
| `check.cost(counters, name, *, at_most=None, exactly=None, at_least=None, unit=None)` | Cost judge. |
| `check.at_most(name, value, limit, *, unit=None)` | Bound a measured value and report it as a metric. |

Always pass a `label` that names the quantity (`label="tokens"`); it starts the message.

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

### 5.5 speed

The solution is timed against the reference in the same framework, and must be within a
stated factor. There are no absolute thresholds, ever: they mean nothing across
machines.

```python
@case("within 5x of the vectorized reference")
def fast_enough(ctx):
    ctx.check_speed("pairwise_sq_dists", ctx.randn(48, 16), ctx.randn(48, 16), max_ratio=5.0)
```

`ctx.check_speed(name, *args, max_ratio, rounds=7, warmup=2, **kwargs)` converts the
inputs once, runs `warmup` untimed calls of each implementation, then `rounds` rounds
in which both are measured, alternating which goes first, and compares the medians.
Each measurement repeats the call until it lasts about 5 ms, and forces the result
before the clock stops (`mx.eval` for MLX, `block_until_ready` for JAX, an MPS or CUDA
synchronize for torch), so lazy frameworks are timed doing the work. It reports the
`slowdown` metric and fails beyond `max_ratio`.

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
- For custom timing (a class, a multi-step protocol), use
  `whetstone.judge.speed.compare(run_solution, run_reference)` and
  `speed.require(result, max_ratio=..., label=..., ctx=ctx)`. Each callable must call
  `ctx.fw.sync(result)` itself.

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
| `sim.call_at(when, fn, *args)`, `sim.call_later(delay, fn, *args)` | Schedule a plain callback; returns a handle with `cancel()`. |
| `sim.spawn(coro)` | Start a coroutine as a task. An exception escaping it fails `run()`. |
| `sim.spawn_at(when, make)` | At time `when`, call `make()` and start the coroutine it returns. |
| `sim.sleep(delay)` | Awaitable; the same as `asyncio.sleep(delay)`. |
| `sim.run(until=None)` | Run until idle (nothing ready, no timer pending), or until virtual time `until`. Tasks blocked forever on a queue do not keep it alive. |
| `sim.run_until_complete(awaitable)` | Run until it finishes and return its result; fails with a deadlock message if the simulation goes idle first. |
| `sim.trace.record(kind, **fields)` | Append an event stamped with the virtual time. |
| `sim.trace.of(kind)`, `.values(kind, field)`, `.times(kind)` | Query the trace. |
| `sim.trace.latencies(start_kind, end_kind, key="id")` | Per-item latency; fails if an item never finished. |
| `percentile(values, q)` | Nearest-rank percentile (an actual observation; no interpolation). |

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
- Integer arrays become the framework's default integer type: int64 in torch, int32 in
  JAX and MLX. Bool stays bool. Exact comparisons ignore integer width.
- The `python` adapter turns arrays into nested lists and numpy scalars into Python
  scalars; results come back unchanged.
- Inputs are copied. A solution that works in place cannot corrupt the array the test
  passes to the oracle next.

Torch solutions receive CPU tensors. MLX uses its default device (the GPU on Apple
silicon). JAX runs on the CPU.

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

### SimModel and Counters

`SimModel` is described in section 5.6. `Counters("a", "b")` is the counter object for
new kit objects: `counters.add("a", n)`, attribute access, `snapshot()`, `reset()`.

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
  framework's built-in `softmax`").
- Framework-neutral: describe tensors, not `torch.Tensor`s. Framework specifics belong
  in the starter's type hints.
- No hints in the statement. The statement specifies; `hints.md` helps.

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
- JAX compiles each new operation and shape the first time it is used, about 50 to
  200 ms per case of fresh shapes. Limit random sweeps to about ten shape
  combinations, and prefer few distinct shapes.
- Keep tensors small: tests check correctness, not scale. A few thousand elements is
  plenty except in speed cases.
- `time_limit_s` is per case; 10 is right for almost everything. A correct solution
  should use under 5 percent of it. Lower it (2 to 5) when a wrong solution's typical
  failure is an infinite loop.
- Statistical cases dominate their problem's budget: N calls into a tensor framework.
  Measure, then pick N.

How time limits work, so you can predict the learner's experience: when a case exceeds
its limit, the harness interrupts it, reports `timeout`, and continues with the next
case. If the solution cannot be interrupted (it is stuck inside native code, or it
swallows the interruption), a watchdog process kills the run two seconds later; the
stuck case is reported as `timeout` and the remaining cases as `skipped`.

## 11. Mutants

A mutant is a wrong solution, written in the oracle framework (`torch`, or `python`),
that the tests must reject. Mutants are how we know the tests test something. Line 1
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

Aim for 4 to 7 mutants per problem. When a real learner submission is wrongly accepted
later, it becomes a mutant, together with the case that rejects it.

## 12. Checklist before you open a pull request

- `ws validate <id> -v` passes, with no warning.
- `ruff check problems/<track>/<id>` is clean.
- `ws run <id> -f <each framework> --starter` shows a failure a beginner can act on.
- `ws submit <id> --mutant <each>`: every mutant fails for the reason its `# BUG:` line
  gives, and the failure message would make sense to a learner.
- The first public case is the statement's worked example.
- No expected values appear in any failure message of a hidden case.
- The statement specifies every choice that affects the output, and its Constraints
  section states the tolerance and any cost or speed limit.
