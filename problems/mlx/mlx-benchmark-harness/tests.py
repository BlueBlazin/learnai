"""Tests for mlx-benchmark-harness.

Nothing here depends on how long anything really takes. The solution is handed a
simulated clock and a model that charges simulated time to it, and the expected
throughputs follow exactly from the cost model:

- `WorkClock` is the clock: `clock()` returns simulated seconds.
- `TimedLM` is an `MLXTinyLM` that really computes (the tokens are checked), and on
  every call (1) advances the clock at once by `BUILD`, the cost of building a graph,
  and (2) registers the cost of computing that call with the clock, as a debt attached
  to the lazy logits it returns.
- The clock collects a debt the first time it is read after the logits have been
  computed (`mx.export_to_dot` lists no pending operation behind them). Until then the
  work has not happened and has cost nothing. So an interval timed around an
  unevaluated graph contains only `BUILD`, and the compute time lands in whichever
  interval ends at the first clock reading after someone evaluates.
- Runs that are *computed* earlier are slower: the k-th run whose work the clock
  collects is multiplied by `slow[k]` (compilation, allocation, and ordinary jitter).
  A warm-up run that is never evaluated warms nothing.

Each case checks, in this order: the form of the result; the generated tokens against
uncached greedy decoding done here; the two throughputs against the cost model; that
exactly `n_warmup + n_runs` full runs were made; that `mx.async_eval` was not used;
and that no lazy graph crossed a step (pending operations behind the arguments of
every model call, as in `mlx-lazy-eval-placement`).

The model and the simulated clock are handed over behind the kit's `Guarded`: a
solution may call the model and read its sizes, parameters, and submodules, and may
call the clock. The model's counters, meter, and private methods and everything of the
clock but the call are refused, so the times cannot be taken from the clock's books,
and the solution cannot build a second `MLXTinyLM` to do work uncounted.
"""

import io
import time
from statistics import median

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from learnai.judge import case, check
from learnai.kit import EvalSpy, Guarded, MLXTinyLM, check_guard, guard

FN = "benchmark"

#: Simulated seconds. Building the graph of one model call; computing one prompt
#: token of the prefill; computing one decode step.
BUILD = 0.0001
PREFILL = 0.002
DECODE = 0.004

#: Pending operations allowed behind the arguments of a model call (see
#: mlx-lazy-eval-placement): reshaping an evaluated token is 1.
PENDING_LIMIT = 8

# The worked example of the statement.
PROMPT = [0, 17, 42, 5]
EXAMPLE_SLOW = [5.0, 1.0, 1.5, 1.0]
EXAMPLE_TOKENS = [28, 34, 31, 15, 1, 48, 46, 5, 7]


def pending_ops(*trees):
    """Number of operations not yet computed behind the arrays in `trees` (0: evaluated)."""
    arrays = [leaf for _, leaf in tree_flatten(list(trees)) if isinstance(leaf, mx.array)]
    if not arrays:
        return 0
    dot = io.StringIO()
    mx.export_to_dot(dot, *arrays)
    return dot.getvalue().count("shape=rectangle")


class WorkClock(Guarded):
    """A simulated clock on which time passes only when work is really done."""

    label = "clock"
    # The solution calls it, nothing else.
    refused = Guarded.refused | {"now", "slow", "debts", "order", "reads", "spend", "owe"}

    def __init__(self, slow):
        self.now = 0.0
        self.slow = list(slow)
        self.debts = []  # (lazy array, run it belongs to, seconds when warm)
        self.order = {}  # run -> position among the runs whose work was computed
        self.reads = 0

    def spend(self, seconds):
        self.now += seconds

    def owe(self, array, run, seconds):
        self.debts.append((array, run, seconds))

    def __call__(self):
        self.reads += 1
        waiting = []
        blocked = set()  # runs with an uncomputed call: the calls after it depend on it
        for array, run, seconds in self.debts:
            if run in blocked or pending_ops(array):
                blocked.add(run)
                waiting.append((array, run, seconds))
                continue
            position = self.order.setdefault(run, len(self.order))
            self.now += seconds * (self.slow[position] if position < len(self.slow) else 1.0)
        self.debts = waiting
        return self.now


class Meter:
    """What a TimedLM recorded: the clock, the run counter, pending operations per call."""

    def __init__(self, clock):
        self.clock = clock
        self.run = -1
        self.pending = []


class LM(Guarded, MLXTinyLM):
    """The model the solution gets: everything of an MLXTinyLM but the counters."""

    label = "model"
    refused = Guarded.refused | {"meter"}


class TimedLM(LM):
    """An MLXTinyLM whose calls cost simulated time on a WorkClock."""

    def __init__(self, seed, clock, **options):
        super().__init__(seed, **options)
        self.meter = Meter(clock)

    def __call__(self, tokens, cache=None, **options):
        meter = self.meter
        meter.pending.append(pending_ops(tokens, cache))
        logits, new_cache = super().__call__(tokens, cache, **options)
        meter.clock.spend(BUILD)
        if cache is None:
            meter.run += 1  # a call without a cache starts a run
            meter.clock.owe(logits, meter.run, PREFILL * logits.shape[0])
        else:
            meter.clock.owe(logits, meter.run, DECODE * logits.shape[0])
        return logits, new_cache


def eager_greedy(seed, prompt, n, **options):
    """The definition: uncached greedy decoding, evaluated at once."""
    lm = MLXTinyLM(seed, **options)
    sequence, tokens = list(prompt), []
    for _ in range(n):
        logits, _ = lm(mx.array(sequence))
        row = np.array(logits[-1]).astype(np.float64)
        top = np.sort(row)[-2:]
        if top[1] - top[0] < 1e-3:
            raise RuntimeError("authoring error: the two largest logits are too close")
        tokens.append(int(row.argmax()))
        sequence.append(tokens[-1])
    return tokens


def expected_throughput(n_prompt, n_tokens, n_warmup, n_runs, slow):
    """The throughputs a correct harness reports, from the cost model."""
    factors = [slow[r] if r < len(slow) else 1.0 for r in range(n_warmup, n_warmup + n_runs)]
    prompt_time = median(BUILD + n_prompt * PREFILL * f for f in factors)
    generation_time = median(n_tokens * (BUILD + DECODE * f) for f in factors)
    return n_prompt / prompt_time, n_tokens / generation_time


def run(ctx, seed, prompt, n_tokens, n_warmup, n_runs, slow, **options):
    clock = WorkClock(slow)
    lm = TimedLM(seed, clock, **options)
    guard(ctx)
    with EvalSpy() as spy:
        result = ctx.solution.benchmark(lm, list(prompt), n_tokens, n_warmup, n_runs, clock)
    check_guard()

    check.that(isinstance(result, dict), f"{FN}: must return a dict")
    check.that(
        set(result) == {"prompt_tps", "generation_tps", "tokens"},
        f"{FN}: got a dict whose keys are not the expected keys",
        f"expected ['generation_tps', 'prompt_tps', 'tokens'], got {sorted(map(str, result))}",
    )
    for key in ("prompt_tps", "generation_tps"):
        check.that(
            type(result[key]) is float,
            f"{FN}: {key} must be a Python float",
            f"got {type(result[key]).__name__}",
        )
    tokens = result["tokens"]
    check.that(
        isinstance(tokens, list) and all(type(t) is int for t in tokens),
        f"{FN}: tokens must be a list of Python ints",
    )

    # 1. The run that was timed is a real greedy generation.
    check.equal(tokens, eager_greedy(seed, prompt, n_tokens + 1, **options), label="tokens")

    # 2. The two numbers.
    want_prompt, want_generation = expected_throughput(
        len(prompt), n_tokens, n_warmup, n_runs, slow
    )
    ctx.metric("prompt_tps", round(result["prompt_tps"], 2), "tokens/s", higher_is_better=True)
    ctx.metric(
        "generation_tps", round(result["generation_tps"], 2), "tokens/s", higher_is_better=True
    )
    check.allclose(
        np.float64(result["prompt_tps"]), np.float64(want_prompt), rtol=1e-6, label="prompt_tps"
    )
    check.allclose(
        np.float64(result["generation_tps"]),
        np.float64(want_generation),
        rtol=1e-6,
        label="generation_tps",
    )

    # 3. The work: n_warmup + n_runs full runs, nothing more and nothing less.
    runs = n_warmup + n_runs
    check.cost(lm.counters, "forward_calls", exactly=runs * (1 + n_tokens))
    check.cost(
        lm.counters, "tokens_processed", exactly=runs * (len(prompt) + n_tokens), unit="tokens"
    )

    # 4. How it was evaluated.
    check.that(
        spy.async_evals == 0,
        f"{FN}: called mx.async_eval {spy.async_evals} time(s); it returns before the work "
        "is done, so a clock read after it measures nothing. Evaluate with mx.eval",
        spy.describe(),
    )
    check.at_most("pending_ops_at_model_call", max(lm.meter.pending), PENDING_LIMIT, unit="ops")
    return result


@case("the worked example: one cold warm-up run, then three measured runs", public=True)
def worked_example(ctx):
    result = run(ctx, 3, PROMPT, 8, 1, 3, EXAMPLE_SLOW)
    if result["tokens"] != EXAMPLE_TOKENS:
        raise RuntimeError("authoring error: the worked example changed")
    check.allclose(np.float64(result["prompt_tps"]), 4 / 0.0081, rtol=1e-6, label="prompt_tps")
    check.allclose(
        np.float64(result["generation_tps"]), 8 / 0.0328, rtol=1e-6, label="generation_tps"
    )


@case("five measured runs with two slow outliers: the median ignores them", public=True)
def outliers(ctx):
    # Rejects mean_not_median: the mean of 1, 6, 1, 1, 4 is 2.6 times the median.
    run(ctx, 4, [0, 8, 8, 8], 6, 1, 5, [4.0, 1.0, 6.0, 1.0, 1.0, 4.0])


@case("a 32-token prompt and 4 decode steps: the prompt time is the prefill alone", public=True)
def long_prompt(ctx):
    # Rejects prefill_not_evaluated and generation_includes_prefill most visibly: the
    # prefill is 64 ms of simulated time against 16 ms for the generation.
    run(ctx, 7, [0, *range(20, 51)], 4, 1, 3, [3.0, 1.0, 1.25, 1.5])


@case("no warm-up asked for: the cold first run is measured, and reported", public=True)
def no_warmup(ctx):
    run(ctx, 3, PROMPT, 5, 0, 1, [5.0, 1.0])
    run(ctx, 3, PROMPT, 5, 0, 3, [5.0, 1.0, 2.0])


# Hidden cases.


@case("an even number of runs: the median is the mean of the two middle times")
def even_runs(ctx):
    # Rejects upper_median, and warmup_included (three warm-up runs, all slow).
    run(ctx, 5, [0, 33, 2, 60, 14, 7], 6, 3, 4, [8.0, 4.0, 2.0, 1.0, 3.0, 1.5, 2.0])
    run(ctx, 5, [0, 33, 2, 60, 14, 7], 3, 1, 2, [8.0, 1.0, 2.0])


@case("two warm-up runs that are both slow, then one measured run")
def two_warmups(ctx):
    # Rejects lazy_warmup and warmup_included: with one measured run there is no
    # median to hide behind.
    run(ctx, 9, [0, 12, 40], 4, 2, 1, [6.0, 3.0, 1.0])


@case("a single decode step per run")
def one_decode_step(ctx):
    # Rejects counts_first_token at its largest: 2 tokens exist, 1 step was timed.
    run(ctx, 2, [0, 50, 51], 1, 1, 3, [2.0, 1.0, 1.0, 1.5])


@case("a long generation: 24 decode steps, seven runs with jitter on every other run")
def long_generation(ctx):
    run(ctx, 4, [0, 8, 8, 8], 24, 2, 7, [5.0, 2.0, 1.0, 1.5, 1.0, 1.25, 1.0, 2.5, 1.75])


@case("a single-token prompt, and a model with grouped-query attention")
def other_models(ctx):
    run(ctx, 11, [0], 10, 1, 3, [4.0, 1.5, 1.0, 1.0])
    run(ctx, 6, [0, 3, 9, 27], 8, 1, 4, [3.0, 1.0, 1.5, 2.0, 1.0], n_kv_heads=2, rope=False)


@case("a steady machine: every run takes the same time")
def steady(ctx):
    run(ctx, 3, PROMPT, 8, 2, 5, [])


@case(
    "the real clock on the real GPU: time.perf_counter gives finite, positive rates",
    requires="mlx-gpu",
)
def real_clock(ctx):
    # No threshold on the numbers: real time is not reproducible. The harness must
    # simply work with an ordinary clock and an ordinary model.
    lm = LM(3)
    guard(ctx)
    result = ctx.solution.benchmark(lm, list(PROMPT), 8, 1, 3, time.perf_counter)
    check_guard()
    check.equal(result["tokens"], EXAMPLE_TOKENS, label="tokens")
    for key in ("prompt_tps", "generation_tps"):
        value = result[key]
        check.that(
            type(value) is float and np.isfinite(value) and value > 0,
            f"{FN}: {key} is {value!r} with a real clock, which is not a finite positive rate",
        )
        ctx.metric(f"real_{key}", round(value, 1), "tokens/s", higher_is_better=True)
    check.cost(lm.counters, "forward_calls", exactly=4 * 9)
