"""Tests for lora-merge.

JAX compiles every new combination of shapes (about 0.15 s each), so the cases share a
few layouts: (out, in, r).
"""

import numpy as np

from learnai.judge import case, check

ATOL = 1e-4


def adapter(ctx, out_dim, in_dim, rank):
    """A random base weight and adapter: (w, a, b)."""
    return ctx.randn(out_dim, in_dim), ctx.randn(rank, in_dim), ctx.randn(out_dim, rank)


def check_all(ctx, x, w, a, b, alpha):
    """Compare the three functions with the oracle, each on the same inputs."""
    ctx.check("merge", w, a, b, alpha, atol=ATOL)
    merged = ctx.expected("merge", w, a, b, alpha)
    ctx.check("unmerge", merged, a, b, alpha, atol=ATOL)
    ctx.check("lora_forward", x, w, a, b, alpha, atol=ATOL)


W = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]], dtype=np.float32)
A = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]], dtype=np.float32)
B = np.array([[1.0, 0.0], [2.0, -1.0]], dtype=np.float32)


@case("the worked example: out = 2, in = 3, r = 2, alpha = 1", public=True)
def worked_example(ctx):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    merged = np.array([[1.5, 0.0, 2.5], [1.0, 0.0, 0.0]], dtype=np.float32)
    check.allclose(ctx.call("merge", W, A, B, 1.0), merged, label="merge")
    check.allclose(ctx.call("unmerge", merged, A, B, 1.0), W, label="unmerge")
    check.allclose(
        ctx.call("lora_forward", x, W, A, B, 1.0),
        np.array([9.0, 1.0], np.float32),
        label="lora_forward",
    )
    check_all(ctx, x, W, A, B, 1.0)


# Rejects alpha_not_divided, scale_applied_twice, rank_from_b_rows, and unmerge_adds.
@case("a (6, 10) layer with a rank-4 adapter, alpha = 6", public=True)
def random_adapter(ctx):
    w, a, b = adapter(ctx, 6, 10, 4)
    check_all(ctx, ctx.randn(5, 10), w, a, b, 6.0)


# scale_applied_twice is right here; alpha_not_divided is not.
@case("alpha equal to the rank: the scale is exactly 1", public=True)
def alpha_equals_rank(ctx):
    w, a, b = adapter(ctx, 6, 10, 4)
    check_all(ctx, ctx.randn(5, 10), w, a, b, 4.0)


# Rejects factors_swapped where its shapes are valid: a.T @ b.T is the transposed update.
@case("a square (8, 8) layer with a rank-2 adapter", public=True)
def square_layer(ctx):
    w, a, b = adapter(ctx, 8, 8, 2)
    check_all(ctx, ctx.randn(8, 8), w, a, b, 3.0)


@case("a freshly initialized adapter (b = 0) changes nothing", public=True)
def zero_b(ctx):
    w, a, _ = adapter(ctx, 6, 10, 4)
    b = np.zeros((6, 4), dtype=np.float32)
    x = ctx.randn(5, 10)
    merged = ctx.call("merge", w, a, b, 16.0)
    check.that(np.array_equal(merged, w), "merging an adapter with b = 0 must return w unchanged")
    check.allclose(
        ctx.call("lora_forward", x, w, a, b, 16.0),
        x @ w.T,
        atol=ATOL,
        label="lora_forward with b = 0",
    )
    check_all(ctx, x, w, a, b, 16.0)


@case("rank 1, and a rank as large as the output dimension")
def extreme_ranks(ctx):
    w, a, b = adapter(ctx, 6, 10, 1)
    check_all(ctx, ctx.randn(5, 10), w, a, b, 2.0)
    # r == out: reading the rank from b.shape[0] happens to be right here.
    w, a, b = adapter(ctx, 6, 10, 6)
    check_all(ctx, ctx.randn(5, 10), w, a, b, 9.0)
    # r == in.
    w, a, b = adapter(ctx, 6, 10, 10)
    check_all(ctx, ctx.randn(5, 10), w, a, b, 5.0)


@case("fractional, zero, and negative alpha")
def alpha_values(ctx):
    w, a, b = adapter(ctx, 6, 10, 4)
    x = ctx.randn(5, 10)
    for alpha in (0.5, 0.0, -3.0, 32.0):
        check_all(ctx, x, w, a, b, alpha)


@case("batch axes: x of shape (2, 3, in) and of shape (in,)")
def batch_axes(ctx):
    w, a, b = adapter(ctx, 6, 10, 4)
    ctx.check("lora_forward", ctx.randn(2, 3, 10), w, a, b, 6.0, atol=ATOL)
    ctx.check("lora_forward", ctx.randn(10), w, a, b, 6.0, atol=ATOL)


@case("a wide layer and a tall layer: (4, 12) and (12, 4), rank 3")
def wide_and_tall(ctx):
    w, a, b = adapter(ctx, 4, 12, 3)
    check_all(ctx, ctx.randn(5, 12), w, a, b, 7.0)
    w, a, b = adapter(ctx, 12, 4, 3)
    check_all(ctx, ctx.randn(5, 4), w, a, b, 7.0)


LAYOUTS = [(6, 10, 4), (8, 8, 2), (4, 12, 3)]  # (out, in, r)


@case("the merged layer computes the same outputs as lora_forward")
def merged_equals_unmerged(ctx):
    def prop(rng):
        out_dim, in_dim, rank = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
        w, a, b = adapter(ctx, out_dim, in_dim, rank)
        alpha = float(np.round(rng.uniform(-8.0, 24.0), 2))
        x = ctx.randn(5, in_dim)
        merged = ctx.call("merge", w, a, b, alpha)
        check.shape(merged, (out_dim, in_dim), label="merge")
        y = ctx.call("lora_forward", x, w, a, b, alpha)
        check.allclose(
            y, x @ merged.T, rtol=1e-3, atol=1e-3, label="lora_forward versus x @ merge(...).T"
        )
        ctx.check("lora_forward", x, w, a, b, alpha, atol=ATOL)

    check.forall(prop, trials=8)


@case("merge followed by unmerge returns the original weight")
def round_trip(ctx):
    def prop(rng):
        out_dim, in_dim, rank = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
        w, a, b = adapter(ctx, out_dim, in_dim, rank)
        alpha = float(np.round(rng.uniform(-8.0, 24.0), 2))
        merged = ctx.call("merge", w, a, b, alpha)
        ctx.check("merge", w, a, b, alpha, atol=ATOL)
        back = ctx.call("unmerge", merged, a, b, alpha)
        check.allclose(back, w, rtol=1e-4, atol=1e-4, label="unmerge(merge(w))")

    check.forall(prop, trials=8)
