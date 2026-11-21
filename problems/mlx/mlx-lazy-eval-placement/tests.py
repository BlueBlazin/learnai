"""Tests for mlx-lazy-eval-placement.

Every case runs the solution once and checks, in this order:

1. the result: the tokens and the summed log-probability, against greedy decoding done
   here eagerly and without a cache (the whole sequence through a fresh model at every
   step, log-softmax in float64 numpy);
2. where evaluation happened: `EvalSpy` counts the `mx.eval` calls and the implicit
   reads made by the solution file;
3. that no lazy graph crossed a step. This does not rely on the spy: the model handed
   to the solution measures, at every call, how many operations are still pending
   behind its arguments (`mx.export_to_dot` lists exactly the unevaluated part of a
   graph), and the returned arrays are measured the same way. An evaluation the spy
   cannot see (`if token:`, an f-string) therefore gains a solution nothing: the count
   of `mx.eval` calls must still be exactly `n`;
4. the work asked of the model, as in `cached-greedy-decode`.

The model is handed over behind the kit's `Guarded`: a solution may call it and read
its sizes, its parameters, and its submodules. Its counters, the probe, and its private
methods are refused, and it cannot build a second `MLXTinyLM` to do the work uncounted.
"""

import io

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from learnai.judge import case, check
from learnai.kit import EvalSpy, Guarded, MLXTinyLM, check_guard, guard

FN = "generate"

# The worked example of the statement.
PROMPT = [0, 17, 42, 5]
CONTINUATION = [28, 34, 31, 15, 1, 48, 46, 5]
LOGPROB = -12.43940

#: Pending operations allowed behind the arguments of a model call. Reshaping or
#: casting an evaluated token is 1 to 3; a step that was never evaluated is over 500.
PENDING_LIMIT = 8


def pending_ops(*trees):
    """Number of operations not yet computed behind the arrays in `trees` (0: evaluated)."""
    arrays = [leaf for _, leaf in tree_flatten(list(trees)) if isinstance(leaf, mx.array)]
    if not arrays:
        return 0
    dot = io.StringIO()
    mx.export_to_dot(dot, *arrays)
    return dot.getvalue().count("shape=rectangle")


class Probe:
    """What the model was handed: pending operations at each call."""

    def __init__(self):
        self.pending = []


class ProbedLM(Guarded, MLXTinyLM):
    """An MLXTinyLM that records how much unevaluated graph each call receives."""

    label = "model"
    refused = Guarded.refused | {"probe"}

    def __init__(self, seed, **options):
        super().__init__(seed, **options)
        self.probe = Probe()

    def __call__(self, tokens, cache=None, **options):
        self.probe.pending.append(pending_ops(tokens, cache))
        return super().__call__(tokens, cache, **options)


def eager_greedy(seed, prompt, n, **options):
    """The definition: uncached greedy decoding, evaluated at once, scored in float64."""
    lm = MLXTinyLM(seed, **options)
    sequence, tokens, total = list(prompt), [], 0.0
    for _ in range(n):
        logits, _ = lm(mx.array(sequence))
        row = np.array(logits[-1]).astype(np.float64)
        top = np.sort(row)[-2:]
        if top[1] - top[0] < 1e-3:
            raise RuntimeError("authoring error: the two largest logits are too close")
        logprobs = row - row.max() - np.log(np.exp(row - row.max()).sum())
        token = int(row.argmax())
        tokens.append(token)
        total += float(logprobs[token])
        sequence.append(token)
    return tokens, total


def run(ctx, seed, prompt, n, expected=None, **options):
    """Run the solution on a probed model and check result, evaluation, and cost."""
    want_tokens, want_logprob = eager_greedy(seed, prompt, n, **options)
    if expected is not None and want_tokens != expected:
        raise RuntimeError("authoring error: the worked example changed")
    lm = ProbedLM(seed, **options)
    guard(ctx)
    with EvalSpy() as spy:
        result = ctx.solution.generate(lm, list(prompt), n)
    check_guard()
    # Measured first: every check below may read (and so evaluate) what was returned.
    left_lazy = pending_ops(result)

    check.that(
        isinstance(result, tuple) and len(result) == 2,
        f"{FN}: must return a tuple (tokens, logprob)",
        f"got {type(result).__name__}",
    )
    tokens, logprob = result
    check.that(
        isinstance(tokens, list)
        and all(isinstance(t, mx.array) and t.ndim == 0 for t in tokens)
        and all(mx.issubdtype(t.dtype, mx.integer) for t in tokens),
        f"{FN}: tokens must be a list of scalar integer mx.arrays",
        f"got {type(tokens).__name__}",
    )
    check.that(
        isinstance(logprob, mx.array) and logprob.ndim == 0 and logprob.dtype == mx.float32,
        f"{FN}: logprob must be a scalar float32 mx.array",
        f"got {type(logprob).__name__}",
    )

    # 1. The result.
    check.equal([int(t.item()) for t in tokens], want_tokens, label="tokens")
    check.allclose(np.array(logprob), np.float32(want_logprob), atol=1e-3, label="logprob")

    # 2. Where evaluation happened.
    check.that(
        spy.implicit == 0,
        f"{FN}: {spy.implicit} implicit evaluations (.item(), .tolist(), int(), np.array, "
        "print, ...); only mx.eval may evaluate",
        spy.describe(),
    )
    check.that(
        spy.async_evals == 0,
        f"{FN}: called mx.async_eval {spy.async_evals} times; this problem asks for mx.eval",
        spy.describe(),
    )
    check.at_most("evaluations", spy.evals, n)
    check.that(
        spy.evals == n,
        f"{FN}: mx.eval was called {spy.evals} time(s), fewer than once per generated token",
        spy.describe(),
    )

    # 3. No lazy graph crossed a step.
    check.at_most("pending_ops_at_model_call", max(lm.probe.pending), PENDING_LIMIT, unit="ops")
    check.that(
        left_lazy == 0,
        f"{FN}: returned arrays that are not evaluated ({left_lazy} operations still pending)",
        "every returned token and the log-probability must be covered by one of the mx.eval calls",
    )

    # 4. The work asked of the model.
    check.cost(lm.counters, "forward_calls", at_most=n)
    check.cost(lm.counters, "tokens_processed", at_most=len(prompt) + n - 1, unit="tokens")
    return tokens, logprob


@case("the worked example: 8 tokens from a 4-token prompt, 8 evaluations", public=True)
def worked_example(ctx):
    _, logprob = run(ctx, 3, PROMPT, 8, expected=CONTINUATION)
    if abs(float(logprob) - LOGPROB) > 1e-3:
        raise RuntimeError("authoring error: the worked example's log-probability changed")


@case("n = 1: one evaluation, and the token is not fed back", public=True)
def single_token(ctx):
    run(ctx, 3, PROMPT, 1)
    run(ctx, 9, [0, 12, 40], 1)


@case("24 tokens: nothing unevaluated is handed to the model at any step", public=True)
def twenty_four_tokens(ctx):
    run(ctx, 4, [0, 8, 8, 8], 24)


# The rest are hidden. Every mutant fails every case below (and the public ones) except
# where noted: the mutants differ in the rule they break, not in the input they need.


@case("another model: 60 tokens from a 6-token prompt")
def long_generation(ctx):
    # eval_at_end hands the model more than 30,000 pending operations by the last step.
    run(ctx, 5, [0, 33, 2, 60, 14, 7], 60)


@case("a 40-token prompt and 3 new tokens: the prompt goes in as one call")
def long_prompt(ctx):
    run(ctx, 7, [0, *range(20, 59)], 3)


@case("a single-token prompt")
def single_token_prompt(ctx):
    run(ctx, 11, [0], 20)


@case("n = 2: the first token is fed back, the second is not")
def two_tokens(ctx):
    # The smallest case that tells one evaluation per token from one evaluation in all.
    run(ctx, 2, [0, 50, 51], 2)


@case("a model with grouped-query attention and sinusoidal positions")
def other_architecture(ctx):
    run(ctx, 6, [0, 3, 9, 27], 16, n_kv_heads=2, rope=False)
