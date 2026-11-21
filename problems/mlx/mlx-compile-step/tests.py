"""Tests for mlx-compile-step.

Every case drives the solution's step in a loop, as a generation loop would (one
`mx.eval` of the token and the state per step, made here), and checks, in this order:

1. the result: every token, exactly, and the final state, against the same loop run
   here on a twin model *without* `mx.compile`; one case also compares with greedy
   decoding that uses no cache at all;
2. tracing: `model.counters.forward_calls` counts runs of the model's Python body,
   which under `mx.compile` happen only when the function is traced, so the counter is
   the number of traces. It is an exact count;
3. evaluation: `EvalSpy` watches the solution file during every call of `step`.

The model is handed over behind the kit's `Guarded`: a solution may call it and read
its sizes, its parameters, its submodules, and `model.state`. Its counters and its
private methods are refused, and it cannot build a second `MLXTinyLM` to run untraced.
"""

import mlx.core as mx
import numpy as np

from learnai.judge import case, check
from learnai.kit import EvalSpy, Guarded, MLXTinyLM, check_guard, guard

FN = "step"
MAX_LEN = 64


class LM(Guarded, MLXTinyLM):
    """The model the solution gets: everything of an MLXTinyLM but the counters."""

    label = "model"


def fresh(model_seed):
    """The parameters of another model of the same shape, to swap in mid-generation."""
    return MLXTinyLM(model_seed).parameters()


def state_arrays(state):
    return [*state["keys"], *state["values"], state["length"]]


def uncompiled(seed, start, n, eos_id, *, max_len=MAX_LEN, swaps=None, **options):
    """The definition: the same loop on a twin model, in plain Python, read eagerly."""
    lm = MLXTinyLM(seed, **options)
    token = np.array(start, dtype=np.int32)
    state = lm.init_state(max_len, len(start))
    tokens = []
    for index in range(n):
        if swaps and index in swaps:
            lm.update(MLXTinyLM(swaps[index], **options).parameters())
        logits, state = lm.step(mx.array(token), state)
        rows = np.array(logits)
        top = np.sort(rows[token != eos_id], axis=-1)[:, -2:]  # rows still generating
        if np.any(top[:, 1] - top[:, 0] < 1e-3):
            raise RuntimeError("authoring error: the two largest logits are too close")
        token = np.where(token == eos_id, eos_id, rows.argmax(axis=-1)).astype(np.int32)
        tokens.append(token.tolist())
    mx.eval(state_arrays(state))
    return tokens, state


def drive(ctx, seed, start, n, eos_id, *, max_len=MAX_LEN, swaps=None, traces=1, **options):
    """Run the solution's step for n calls and check result, traces, and evaluation."""
    want_tokens, want_state = uncompiled(
        seed, start, n, eos_id, max_len=max_len, swaps=swaps, **options
    )
    lm = LM(seed, **options)
    guard(ctx)
    step = ctx.solution.make_step(lm, eos_id)
    check_guard()
    check.that(callable(step), "make_step: must return a function")
    check.that(
        lm.counters.forward_calls == 0,
        "make_step: ran the model before step was called",
        "mx.compile traces on the first call, with the caller's arrays; make_step only builds",
    )

    token = mx.array(start, dtype=mx.int32)
    state = lm.init_state(max_len, len(start))
    mx.eval(token, state_arrays(state))
    got_tokens, dtypes, shapes = [], set(), set()
    with EvalSpy() as spy:
        for index in range(n):
            if swaps and index in swaps:
                lm.update(MLXTinyLM(swaps[index], **options).parameters())
            result = step(token, state)
            check.that(
                isinstance(result, tuple) and len(result) == 2,
                f"{FN}: must return a tuple (next_token, state)",
                f"got {type(result).__name__}",
            )
            token, state = result
            check.that(
                isinstance(token, mx.array)
                and isinstance(state, dict)
                and set(state) == {"keys", "values", "length"},
                f"{FN}: must return an mx.array and the state dict of model.step",
                f"got ({type(token).__name__}, {type(state).__name__})",
            )
            mx.eval(token, state_arrays(state))  # the caller's one evaluation per token
            dtypes.add(str(token.dtype))
            shapes.add(tuple(token.shape))
            got_tokens.append(token.tolist() if token.ndim == 1 else token)
    check_guard()

    # 1. The result.
    check.that(
        shapes == {(len(start),)},
        f"{FN}: next_token has shape {sorted(shapes)}, which is not the shape of token",
        f"expected ({len(start)},) on every call",
    )
    check.equal(got_tokens, want_tokens, label="tokens")
    check.that(
        dtypes == {"mlx.core.int32"},
        f"{FN}: next_token has dtype {', '.join(sorted(dtypes))}, which is not the specified dtype",
        "next_token must be int32 on every call, like the token that was passed in",
    )
    check.equal(int(state["length"].item()), int(want_state["length"].item()), label="length")
    for name in ("keys", "values"):
        for layer, (got, want) in enumerate(zip(state[name], want_state[name], strict=True)):
            check.allclose(
                np.array(got), np.array(want), atol=1e-5, label=f"state['{name}'][{layer}]"
            )

    # 2. Tracing.
    check.cost(lm.counters, "forward_calls", at_most=traces, metric="traces")

    # 3. Evaluation.
    check.that(
        spy.evals + spy.async_evals + spy.implicit == 0,
        f"{FN}: evaluated {spy.evals + spy.async_evals + spy.implicit} time(s) "
        "(mx.eval, .item(), int(), ...); the step must only build a graph",
        spy.describe(),
    )
    return lm, step, token, state


@case("the worked example: 8 steps from token 3, end of sequence at the fifth", public=True)
def worked_example(ctx):
    want, _ = uncompiled(1, [3], 8, 32)
    if [row[0] for row in want] != [7, 48, 0, 53, 32, 32, 32, 32]:
        raise RuntimeError("authoring error: the worked example changed")
    drive(ctx, 1, [3], 8, 32)


@case("50 steps: the Python body runs once", public=True)
def fifty_steps(ctx):
    # Rejects not_compiled and compiles_every_call (50 traces) and dtype_drift (2).
    drive(ctx, 2, [11], 50, 63)


@case("new parameters after 10 steps: the next tokens follow them, with no new trace", public=True)
def new_parameters(ctx):
    # Rejects stale_weights: its compiled graph holds the parameters of the trace.
    drive(ctx, 1, [3], 20, 63, swaps={10: 2})


@case("a batch of three rows that finish at different steps", public=True)
def batch_of_three(ctx):
    # Rejects eos_ignored, and eos_read_in_python (which cannot read three tokens as one).
    want, _ = uncompiled(7, [10, 20, 30], 16, 21)
    first = [next((i for i, row in enumerate(want) if row[b] == 21), None) for b in range(3)]
    if first != [8, 9, 2]:
        raise RuntimeError(f"authoring error: the rows should finish at steps 8, 9, 2, got {first}")
    drive(ctx, 7, [10, 20, 30], 16, 21)


# Hidden cases.


@case("against decoding without a cache: the whole sequence through the model every step")
def against_uncached(ctx):
    # The oracle of `drive` shares `model.step` with the solution; this one does not.
    seed, start, n = 6, 9, 24
    lm = MLXTinyLM(seed)
    sequence = [start]
    for _ in range(n):
        logits, _ = lm(mx.array(sequence))
        sequence.append(int(np.array(logits[-1]).argmax()))
    if 63 in sequence:
        raise RuntimeError("authoring error: the sequence should not contain the eos token")
    want, _ = uncompiled(seed, [start], n, 63)
    if [row[0] for row in want] != sequence[1:]:
        raise RuntimeError("authoring error: cached and uncached decoding disagree")
    drive(ctx, seed, [start], n, 63)


@case("the start token is the end-of-sequence token")
def starts_finished(ctx):
    drive(ctx, 3, [17], 6, 17)
    drive(ctx, 3, [17, 4], 6, 17)


@case("parameters replaced before the first call, and twice more later")
def parameters_replaced_often(ctx):
    drive(ctx, 5, [8], 30, 63, swaps={0: 7, 9: 8, 21: 5})


@case("a second batch size on the same step function: one more trace, not one per call")
def two_batch_sizes(ctx):
    # A new input shape is a legitimate retrace. 12 calls at each size: 2 traces.
    lm, step, _, _ = drive(ctx, 7, [1], 12, 63)
    want, _ = uncompiled(7, [40, 2], 12, 63)
    token, state = mx.array([40, 2], dtype=mx.int32), lm.init_state(MAX_LEN, 2)
    got = []
    for _ in range(12):
        token, state = step(token, state)
        mx.eval(token, state_arrays(state))
        got.append(token.tolist())
    check_guard()
    check.equal(got, want, label="tokens (batch of 2)")
    check.cost(lm.counters, "forward_calls", at_most=2, metric="traces")


@case("two step functions, for two models, do not share anything")
def two_models(ctx):
    first, step_one, token_one, state_one = drive(ctx, 8, [30], 6, 63)
    drive(ctx, 9, [30], 6, 63)
    # The first function still belongs to the first model.
    want, _ = uncompiled(8, [30], 10, 63)
    got = []
    for _ in range(4):
        token_one, state_one = step_one(token_one, state_one)
        mx.eval(token_one, state_arrays(state_one))
        got.append(token_one.tolist())
    check_guard()
    check.equal(got, want[6:], label="tokens of the first model, continued")
    check.cost(first.counters, "forward_calls", at_most=1, metric="traces")


@case("grouped-query attention, sinusoidal positions, a 16-slot state, a batch of 2")
def other_architecture(ctx):
    drive(ctx, 10, [6, 44], 15, 12, max_len=16, n_kv_heads=2, rope=False)
