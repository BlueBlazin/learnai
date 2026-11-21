"""Tests for multi-lora-batching.

The base layer is a local helper, `BaseLayer`: a callable that holds the frozen weight
in the solution's framework, computes `x @ W.T`, and counts its calls and the rows it
was given. Every case compares the result with the oracle and with a plain numpy loop
over the rows (`by_rows`), which is the definition in the statement.

`BaseLayer` is behind the kit's `Guarded`: a solution may call it and nothing else. Its
weight and its two counts are refused, so the shared matmul cannot be done around the
layer and the counts cannot be written.

JAX compiles every new combination of shapes (about 0.15 s each), so the cases share
a few layouts.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import Guarded, check_guard, guard

ATOL = 1e-4


class BaseLayer(Guarded):
    """The frozen base layer: `base(x)` is `x @ W.T`. Counts its calls."""

    label = "base"
    allowed = frozenset({"__class__", "__call__"})

    def __init__(self, fw, w):
        self.weight = fw.tensor(w)
        self.calls = 0
        self.rows = 0  # leading-axis rows over all calls

    def __call__(self, x):
        self.calls += 1
        self.rows += int(x.shape[0])
        return x @ self.weight.T


def adapters(ctx, n, out_dim, in_dim, ranks):
    """Random adapters zero-padded to max(ranks): (a_stack, b_stack, ranks)."""
    r_max = max(ranks)
    a = ctx.randn(n, r_max, in_dim)
    b = ctx.randn(n, out_dim, r_max)
    for k, rank in enumerate(ranks):
        a[k, rank:, :] = 0.0
        b[k, :, rank:] = 0.0
    return a, b, np.array(ranks, dtype=np.int64)


def by_rows(x, w, a_stack, b_stack, ranks, adapter_ids, alpha):
    """The definition, one row at a time, in float64."""
    x64, w64 = x.astype(np.float64), w.astype(np.float64)
    y = x64 @ w64.T
    for row, n in enumerate(adapter_ids):
        if n >= 0:
            a, b = a_stack[n].astype(np.float64), b_stack[n].astype(np.float64)
            y[row] += (alpha / ranks[n]) * (x64[row] @ a.T) @ b.T
    return y.astype(np.float32)


def run(ctx, x, w, a_stack, b_stack, ranks, adapter_ids, alpha, *, label="batched_lora_forward"):
    """Call the solution and the oracle, each with its own base layer; compare the
    result with both the oracle's and the row-by-row definition. Returns the
    solution's base layer, for the cost checks."""
    ids = np.asarray(adapter_ids, dtype=np.int64)
    base = BaseLayer(ctx.fw, w)
    guard(ctx)
    got = ctx.call("batched_lora_forward", x, base, a_stack, b_stack, ranks, ids, alpha)
    check_guard()
    want = ctx.expected(
        "batched_lora_forward", x, BaseLayer(ctx.oracle_fw, w), a_stack, b_stack, ranks, ids, alpha
    )
    check.allclose(got, want, atol=ATOL, label=label, dtype=True)
    check.allclose(got, by_rows(x, w, a_stack, b_stack, ranks, ids, alpha), rtol=1e-3, atol=1e-3,
                   label=f"{label} versus the row-by-row definition")  # fmt: skip
    return base


W = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]], dtype=np.float32)
A_STACK = np.array(
    [[[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32
)
B_STACK = np.array([[[1.0, 0.0], [2.0, -1.0]], [[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
RANKS = np.array([2, 1], dtype=np.int64)
X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], np.float32)

MAIN = (8, 10)  # (out, in) of the shared layout
MAIN_RANKS = [4, 2, 1]


@case("the worked example: four rows, adapters of rank 2 and 1, one row without", public=True)
def worked_example(ctx):
    ids = np.array([0, -1, 1, 0])
    base = BaseLayer(ctx.fw, W)
    guard(ctx)
    got = ctx.call("batched_lora_forward", X, base, A_STACK, B_STACK, RANKS, ids, 2.0)
    check_guard()
    want = np.array([[11.0, 3.0], [7.0, -1.0], [9.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    check.allclose(got, want, label="batched_lora_forward")
    run(ctx, X, W, A_STACK, B_STACK, RANKS, ids, 2.0)


@case("a batch of 6 rows mixing three adapters of ranks 4, 2, and 1", public=True)
def mixed_batch(ctx):
    # Rejects one_adapter_for_batch and scale_by_padded_rank.
    w = ctx.randn(*MAIN)
    a, b, ranks = adapters(ctx, 3, *MAIN, MAIN_RANKS)
    run(ctx, ctx.randn(6, 10), w, a, b, ranks, [2, 0, 1, 1, 0, 2], 6.0)


@case("rows with id -1 get the base output, unchanged", public=True)
def rows_without_adapter(ctx):
    # Rejects no_adapter_wraps_around (index -1 is the last adapter) and
    # no_adapter_rows_zeroed.
    w = ctx.randn(*MAIN)
    a, b, ranks = adapters(ctx, 3, *MAIN, MAIN_RANKS)
    x = ctx.randn(6, 10)
    ids = np.array([-1, 0, -1, 2, 1, -1])
    run(ctx, x, w, a, b, ranks, ids, 6.0)
    guard(ctx)
    got = ctx.call("batched_lora_forward", x, BaseLayer(ctx.fw, w), a, b, ranks, ids, 6.0)
    check_guard()
    plain = [0, 2, 5]
    check.allclose(got[plain], (x @ w.T)[plain], atol=ATOL, label="rows with adapter id -1")
    # A batch without any adapter is the base layer.
    run(ctx, x, w, a, b, ranks, [-1] * 6, 6.0, label="a batch without adapters")


@case("the base layer is called once, with the whole batch", public=True)
def base_called_once(ctx):
    # Rejects per_row_loop (6 calls) and per_adapter_base_calls (one call per adapter
    # in the batch).
    w = ctx.randn(*MAIN)
    a, b, ranks = adapters(ctx, 3, *MAIN, MAIN_RANKS)
    base = run(ctx, ctx.randn(6, 10), w, a, b, ranks, [2, 0, -1, 1, 0, 2], 6.0)
    check.at_most("base_calls", base.calls, 1, unit="calls")
    check.at_most("base_rows", base.rows, 6, unit="rows")


@case("sequences: x of shape (B, T, in), one adapter per request", public=True)
def token_axis(ctx):
    # Rejects scale_broadcast_over_tokens: B == T == 4, so a scale laid along the
    # token axis has a valid shape and the wrong values.
    w = ctx.randn(*MAIN)
    a, b, ranks = adapters(ctx, 3, *MAIN, MAIN_RANKS)
    x = ctx.randn(4, 4, 10)
    ids = [1, -1, 0, 2]
    base = run(ctx, x, w, a, b, ranks, ids, 6.0)
    check.at_most("base_calls", base.calls, 1, unit="calls")


@case("every adapter at its own rank: alpha / rank, not alpha / r_max")
def per_adapter_scale(ctx):
    w = ctx.randn(*MAIN)
    a, b, ranks = adapters(ctx, 3, *MAIN, MAIN_RANKS)
    x = ctx.randn(6, 10)
    for alpha in (4.0, 1.0, -3.0, 0.5):
        run(ctx, x, w, a, b, ranks, [1, 1, 2, 2, 0, -1], alpha)
    # alpha = 0 switches every adapter off.
    run(ctx, x, w, a, b, ranks, [1, 1, 2, 2, 0, -1], 0.0, label="alpha = 0")


@case("one adapter for all rows, an adapter nobody uses, and equal ranks")
def degenerate_batches(ctx):
    w = ctx.randn(*MAIN)
    a, b, ranks = adapters(ctx, 3, *MAIN, MAIN_RANKS)
    x = ctx.randn(6, 10)
    run(ctx, x, w, a, b, ranks, [1] * 6, 6.0, label="all rows on adapter 1")
    run(ctx, x, w, a, b, ranks, [2, 2, 2, 0, 0, 0], 6.0, label="adapter 1 unused")
    # All ranks equal to r_max: no padding at all.
    a, b, ranks = adapters(ctx, 3, *MAIN, [4, 4, 4])
    run(ctx, x, w, a, b, ranks, [0, 1, 2, -1, 2, 1], 8.0, label="equal ranks")


@case("the last adapter and the first: ids 0 and N - 1 next to -1")
def boundary_ids(ctx):
    w = ctx.randn(*MAIN)
    a, b, ranks = adapters(ctx, 3, *MAIN, [1, 2, 4])
    x = ctx.randn(6, 10)
    run(ctx, x, w, a, b, ranks, [2, -1, 0, -1, 2, 0], 5.0)
    run(ctx, ctx.randn(4, 4, 10), w, a, b, ranks, [-1, 2, -1, 0], 5.0, label="sequences")


@case("random batches: adapters, ids, and alpha drawn at random")
def random_batches(ctx):
    def prop(rng):
        w = ctx.randn(*MAIN)
        ranks = [int(r) for r in rng.integers(1, 5, size=3)]
        ranks[int(rng.integers(3))] = 4  # keep r_max = 4, so the shapes are shared
        a, b, ranks = adapters(ctx, 3, *MAIN, ranks)
        alpha = float(np.round(rng.uniform(-8.0, 16.0), 2))
        ids = rng.integers(-1, 3, size=6)
        base = run(ctx, ctx.randn(6, 10), w, a, b, ranks, ids, alpha)
        check.that(base.calls == 1, f"the base layer was called {base.calls} times")
        ids = rng.integers(-1, 3, size=4)
        run(ctx, ctx.randn(4, 4, 10), w, a, b, ranks, ids, alpha, label="sequences")

    check.forall(prop, trials=6)


SPEED = {"batch": 512, "dim": 64, "n": 16, "r_max": 8}


def speed_inputs(ctx):
    dim, n = SPEED["dim"], SPEED["n"]
    w = ctx.randn(dim, dim, scale=0.1)
    a, b, ranks = adapters(ctx, n, dim, dim, [1, 2, 4, 8] * (n // 4))
    x = ctx.randn(SPEED["batch"], dim)
    ids = ctx.rng.integers(-1, n, size=SPEED["batch"])
    return x, w, a, b, ranks, ids


@case("speed: 512 rows over 16 adapters, within 5x of the vectorized reference",
      time_limit_s=30)  # fmt: skip
def fast_enough(ctx):
    # Rejects per_row_updates, which calls the base layer once and then loops over
    # the rows (measured: 16x slower in torch, over 100x in JAX and MLX).
    x, w, a, b, ranks, ids = speed_inputs(ctx)
    run(ctx, x, w, a, b, ranks, ids, 16.0)
    guard(ctx)
    ctx.check_speed("batched_lora_forward", x, BaseLayer(ctx.fw, w), a, b, ranks, ids, 16.0,
                    max_ratio=5.0)  # fmt: skip
    check_guard()
