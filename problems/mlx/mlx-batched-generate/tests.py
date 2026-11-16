"""Tests for mlx-batched-generate.

Every case hands the solution a fresh `MLXTinyLM` and a list of prompts, and checks:

1. the tokens of every row, exactly, against greedy decoding of that prompt *alone*:
   no batch, no padding, no cache, the whole sequence through a fresh model at every
   step, argmax in float64 NumPy (`alone` below). A row then stops after `eos_id`.
   The oracle refuses a prompt whose two largest logits come closer than 1e-3 at any
   step, so the argmax cannot depend on how the batch was laid out;
2. the work asked of the model: one call for the padded prompts, then one call per
   step with one token per row, and no call once every row is finished;
3. where evaluation happened (`EvalSpy`): at most one `mx.eval` and one read per
   step, whatever the number of rows.

Nothing here needs the device, so no case is marked `mlx-gpu`.
"""

import mlx.core as mx
import numpy as np

from learnai.judge import case, check
from learnai.kit import EvalSpy, MLXTinyLM

FN = "batch_generate"
_alone: dict[tuple, list[int]] = {}


def alone(seed, prompt, n, **options):
    """The definition: `n` greedy tokens for one prompt, uncached, evaluated eagerly."""
    key = (seed, tuple(prompt), n, tuple(sorted(options.items())))
    if key not in _alone:
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
        _alone[key] = tokens
    return _alone[key]


def expected(seed, prompts, n, eos_id, **options):
    rows = []
    for prompt in prompts:
        tokens = alone(seed, prompt, n, **options)
        if eos_id in tokens:
            tokens = tokens[: tokens.index(eos_id) + 1]
        rows.append(list(tokens))
    return rows


def run(ctx, seed, prompts, n, eos_id, *, lengths=None, **options):
    """Run the solution on a fresh model; check tokens, model calls, and evaluations."""
    want = expected(seed, prompts, n, eos_id, **options)
    if lengths is not None and [len(row) for row in want] != lengths:
        raise RuntimeError(f"authoring error: rows have lengths {[len(row) for row in want]}")
    lm = MLXTinyLM(seed, **options)
    given = [list(prompt) for prompt in prompts]
    with EvalSpy() as spy:
        got = ctx.solution.batch_generate(lm, given, n, eos_id)
    check.that(
        isinstance(got, list)
        and all(isinstance(row, list) and all(type(t) is int for t in row) for row in got),
        f"{FN}: must return a list with one list of Python ints per prompt",
        f"got {got!r}"[:400],
    )
    check.equal(len(got), len(want), label="number of rows")
    for index, (row, target) in enumerate(zip(got, want, strict=True)):
        check.equal(row, target, label=f"tokens of row {index}")
    check.that(given == [list(p) for p in prompts], f"{FN}: the prompts were modified")

    batch, width, longest = len(prompts), max(len(p) for p in prompts), max(map(len, want))
    check.cost(lm.counters, "forward_calls", at_most=longest)
    check.cost(
        lm.counters,
        "tokens_processed",
        at_most=batch * width + batch * (longest - 1),
        unit="tokens",
    )
    check.that(
        spy.async_evals == 0,
        f"{FN}: called mx.async_eval {spy.async_evals} time(s); this problem asks for "
        "mx.eval, at most once per step",
        spy.describe(),
    )
    check.at_most("evals", spy.evals, longest)
    check.at_most("reads", spy.implicit, longest)
    return got


WORKED = [[5, 17, 42, 9, 3], [8, 1], [30, 31, 32]]


@case("the worked example: three prompts of lengths 5, 2, 3; one row stops at EOS", public=True)
def worked_example(ctx):
    got = run(ctx, 1, WORKED, 6, 1, lengths=[3, 6, 6])
    check.equal(
        got, [[46, 30, 1], [30, 34, 36, 44, 8, 43], [58, 41, 51, 60, 6, 56]], label="tokens"
    )


# No padding here: this case isolates the batched loop from the mask and the positions.
@case("prompts of equal length, without and with an EOS", public=True)
def equal_lengths(ctx):
    prompts = [[3, 14, 15, 9], [2, 6, 5, 35], [8, 9, 7, 9]]
    run(ctx, 1, prompts, 10, None, lengths=[10, 10, 10])
    run(ctx, 1, prompts, 10, 18, lengths=[2, 3, 1])


# Rejects pads_attended, positions_from_padded_start, right_padding, pad_rows_fully_masked
# on the first token; step_mask_dropped and step_positions_shared on a later one.
@case("ragged prompts of lengths 2, 7, 1, 4, no EOS, with rotary and with absolute positions",
      public=True)  # fmt: skip
def ragged_no_eos(ctx):
    prompts = [[11, 4], [7, 7, 7, 7, 7, 7, 7], [50], [20, 21, 22, 23]]
    run(ctx, 1, prompts, 10, None)
    # Sinusoidal positions are absolute: a row whose positions start at its padding
    # count and not at 0 is a different sequence. (RoPE alone would not notice a shift
    # that is the same for the whole row.)
    run(ctx, 1, prompts, 10, None, rope=False)


# Rejects extra_step_after_done, appends_after_eos, stops_at_first_eos, eos_dropped.
@case("every row reaches EOS early: 9 steps of a limit of 16, then no more model calls",
      public=True)  # fmt: skip
def all_rows_finish(ctx):
    prompts = [[12, 40, 3], [9], [33, 34, 35, 36, 37, 38], [1, 2]]
    run(ctx, 11, prompts, 16, 56, lengths=[7, 3, 7, 9])


# Hidden cases.


@case("a single prompt, and max_new_tokens = 1")
def smallest_batches(ctx):
    run(ctx, 1, [[4, 8, 15, 16, 23, 42]], 12, None)
    run(ctx, 1, [[4, 8, 15, 16, 23, 42]], 12, 5, lengths=[5])
    run(ctx, 1, WORKED, 1, None)
    run(ctx, 4, [[7], [1, 2, 3, 4, 5, 6, 7, 8, 9]], 1, 16)


# Rejects pad_by_token_value: token 0 is an ordinary token of these prompts.
@case("prompts that contain token 0 and the EOS token: neither is padding, neither stops a row")
def ordinary_tokens(ctx):
    prompts = [[0, 0, 5], [0], [9, 0, 0, 0, 2, 0], [55, 55, 0, 55]]
    run(ctx, 1, prompts, 10, None)
    run(ctx, 1, prompts, 10, 55)


@case("lengths 1, 24, and 10 with 20 new tokens")
def long_ragged(ctx):
    prompts = [[13], list(range(1, 25)), [60, 61, 62, 63, 60, 61, 62, 63, 60, 61]]
    run(ctx, 1, prompts, 20, None)
    run(ctx, 1, prompts, 20, 60, lengths=[15, 4, 9])


@case("a model with grouped-query attention and sinusoidal positions")
def other_architecture(ctx):
    prompts = [[5, 6, 7, 8, 9], [10, 11], [12, 13, 14]]
    run(ctx, 1, prompts, 12, None, n_kv_heads=2, rope=False)
    run(ctx, 1, prompts, 12, 16, n_kv_heads=2, rope=False)


# Rejects reads_per_row and loops_over_prompts by their counts alone.
@case("eight rows of seven different lengths: still one evaluation and one read per step")
def eight_rows(ctx):
    prompts = [[i + 1] * (1 + (i * 3) % 7) for i in range(8)]
    run(ctx, 2, prompts, 10, None)
    run(ctx, 2, prompts, 10, 46, lengths=[10, 10, 2, 10, 10, 10, 10, 10])


@case("a three-layer model: padding must stay harmless through every layer")
def three_layers(ctx):
    prompts = [[21, 22, 23, 24], [25], [26, 27]]
    run(ctx, 1, prompts, 10, None, n_layers=3, d_model=48)
    run(ctx, 1, prompts, 10, 4, n_layers=3, d_model=48, lengths=[2, 3, 4])
