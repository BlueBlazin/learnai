"""Tests for mlx-mask-free-indexing.

Every call checks three things:

1. the numbers, against a NumPy oracle written here (boolean masks and `np.unique`
   are fine in NumPy);
2. that the function evaluated nothing. `EvalSpy` reports `mx.eval`, `mx.async_eval`,
   and implicit reads made by the solution file, and, independently of the spy, the
   inputs are handed over *lazy* and must still be unevaluated when the function
   returns (`mx.export_to_dot` lists the pending part of a graph). The second check
   also catches evaluations the spy cannot see, such as `if x:` on an array;
3. that the arguments were not modified.

One hidden case runs `top_k_filter` on MLX's CPU device, where `mx.argpartition`
returns its two parts unsorted (on the GPU, MLX 0.32.4 happens to sort them).
"""

import io

import mlx.core as mx
import numpy as np

from whetstone.judge import case, check
from whetstone.kit import EvalSpy

INF = np.inf


def np_top_k(logits, k):
    """The k largest of each row survive (finite entries of a row are distinct)."""
    if k >= logits.shape[-1]:
        return logits.copy()
    out = np.full_like(logits, -INF)
    keep = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
    rows = np.arange(logits.shape[0])[:, None]
    out[rows, keep] = logits[rows, keep]
    return out


def np_penalty(logits, ids, value):
    """Each token that occurs in a row of `ids` is penalized once."""
    out = logits.copy()
    for row, row_ids in zip(out, ids, strict=True):
        row[np.unique(row_ids)] -= np.float32(value)
    return out


def pending_ops(*arrays):
    """Number of operations not yet computed behind `arrays` (0: evaluated)."""
    dot = io.StringIO()
    mx.export_to_dot(dot, *arrays)
    return dot.getvalue().count("shape=rectangle")


def lazy(array):
    """An unevaluated mx.array holding `array` (one pending operation)."""
    x = mx.array(array)
    return x + mx.zeros_like(x)


def run(ctx, fn, logits, *rest, expected):
    """Call `fn(logits, *rest)` on lazy inputs; check numbers, laziness, and purity."""
    x = lazy(logits)
    args = [lazy(a.astype(np.int32)) if isinstance(a, np.ndarray) else a for a in rest]
    inputs = [x, *(a for a in args if isinstance(a, mx.array))]
    with EvalSpy() as spy:
        out = getattr(ctx.solution, fn)(x, *args)
    # Measured before anything is read: reading evaluates.
    inputs_still_lazy = all(pending_ops(a) > 0 for a in inputs)

    check.that(isinstance(out, mx.array), f"{fn}: must return an mx.array")
    check.allclose(np.array(out), expected, label=fn, dtype=True)
    check.that(
        spy.evals == 0 and spy.async_evals == 0 and spy.implicit == 0,
        f"{fn}: evaluated {len(spy.events)} time(s) (mx.eval, .item(), np.array, ...); "
        "the function must only build a graph",
        spy.describe(),
    )
    check.that(
        inputs_still_lazy,
        f"{fn}: the lazy arguments were evaluated during the call",
        "the arguments were handed over unevaluated and must still be unevaluated when "
        "the function returns: build the result with MLX operations only",
    )
    check.that(
        np.array_equal(np.array(x), logits),
        f"{fn}: modified its `logits` argument",
        "return a new array; `logits[...] = ...` changes the caller's array",
    )
    return out


def top_k(ctx, logits, k):
    return run(ctx, "top_k_filter", logits, k, expected=np_top_k(logits, k))


def penalty(ctx, logits, ids, value):
    return run(ctx, "scatter_penalty", logits, ids, value, expected=np_penalty(logits, ids, value))


def distinct_logits(ctx, batch, vocab, scale=3.0):
    """Random float32 logits whose entries are pairwise distinct within each row."""
    logits = ctx.randn(batch, vocab, scale=scale)
    for row in logits:
        if len(np.unique(row)) != vocab:
            raise RuntimeError("authoring error: a row has equal logits")
    return logits


EXAMPLE = np.array([[1.0, 3.0, 0.5, 2.0, -1.0], [0.0, 2.5, -INF, 4.0, 1.5]], dtype=np.float32)


@case("the worked example: top-2 of two rows, and a penalty with a repeated id", public=True)
def worked_example(ctx):
    top_k(ctx, EXAMPLE, 2)
    penalty(ctx, EXAMPLE, np.array([[1, 3, 1], [0, 0, 4]]), 1.5)


@case("top-k of a random batch: (4, 64) logits, k = 10", public=True)
def random_top_k(ctx):
    top_k(ctx, distinct_logits(ctx, 4, 64), 10)


@case("the edges of k: 1, V - 1, V, and far above V", public=True)
def k_edges(ctx):
    logits = distinct_logits(ctx, 3, 16)
    for k in (1, 15, 16, 1000):
        top_k(ctx, logits, k)


@case("a penalty where most ids repeat: 24 ids drawn from 6 tokens", public=True)
def repeated_ids(ctx):
    # Rejects duplicates_accumulate: a token named four times loses 4 * value there.
    logits = ctx.randn(3, 32, scale=2.0)
    ids = ctx.rng.choice([2, 5, 11, 17, 30, 31], size=(3, 24))
    penalty(ctx, logits, ids, 0.75)


@case("masked rows: -inf entries, and a row with fewer than k finite logits")
def masked_rows(ctx):
    logits = distinct_logits(ctx, 4, 16)
    logits[0, [1, 4, 9]] = -INF
    logits[1, 3:] = -INF  # three finite entries, k = 5 keeps all three
    logits[2, :15] = -INF
    top_k(ctx, logits, 5)
    ids = np.array([[1, 2, 2], [0, 5, 0], [15, 15, 3], [7, 7, 7]])
    penalty(ctx, logits, ids, 2.0)  # -inf minus a penalty is still -inf


@case("top-k on the CPU device, where the order within a partition differs")
def top_k_on_cpu(ctx):
    # Rejects argpartition_as_argsort: on this machine's GPU, MLX 0.32.4 returns the
    # indices of mx.argpartition fully sorted whatever `kth` is, so a wrong `kth` goes
    # unnoticed there. The CPU backend partitions without sorting.
    logits = distinct_logits(ctx, 8, 96)
    k = 12
    device = mx.default_device()
    ctx.defer(lambda: mx.set_default_device(device))
    mx.set_default_device(mx.cpu)
    order = np.array(mx.argpartition(-mx.array(logits), kth=k - 1, axis=-1))[:, :k]
    if np.all(np.diff(np.take_along_axis(logits, order, axis=-1), axis=-1) <= 0):
        raise RuntimeError("authoring error: mx.argpartition sorts on the CPU for this input")
    top_k(ctx, logits, k)


@case("a vocabulary of 2000 with k = 40, and a batch of one")
def large_vocabulary(ctx):
    top_k(ctx, distinct_logits(ctx, 2, 2000, scale=5.0), 40)
    top_k(ctx, distinct_logits(ctx, 1, 2000, scale=5.0), 2)


@case("penalties: a single id per row, a negative value, every token of a row")
def penalty_variants(ctx):
    logits = ctx.randn(4, 12)
    penalty(ctx, logits, ctx.randint(0, 12, (4, 1)), 1.25)
    penalty(ctx, logits, ctx.randint(0, 12, (4, 5)), -0.5)  # a bonus
    penalty(ctx, logits, np.tile(np.arange(12), (4, 2)), 3.0)  # all tokens, each twice


@case("random shapes: top-k and penalty on the same logits")
def random_sweep(ctx):
    for batch, vocab, k, n in [(1, 8, 3, 4), (5, 40, 39, 60), (2, 300, 50, 10), (6, 5, 2, 9)]:
        logits = distinct_logits(ctx, batch, vocab)
        top_k(ctx, logits, k)
        penalty(ctx, logits, ctx.randint(0, vocab, (batch, n)), float(ctx.rng.uniform(0.1, 4.0)))
