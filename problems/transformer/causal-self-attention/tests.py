"""Tests for causal-self-attention.

Numeric cases compare with the oracle. Property cases state causality and batch
independence directly, without the oracle.
"""

import numpy as np

from whetstone.judge import case, check

FN = "causal_self_attention"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def weights(ctx, d, d_k=None, d_v=None, d_out=None):
    """Projection weights in (out, in) layout, scaled so that scores are of order 1."""
    d_k, d_v, d_out = d_k or d, d_v or d, d_out or d
    s = 1.0 / np.sqrt(d)
    return (
        ctx.randn(d_k, d, scale=s),
        ctx.randn(d_k, d, scale=s),
        ctx.randn(d_v, d, scale=s),
        ctx.randn(d_out, d_v, scale=1.0 / np.sqrt(d_v)),
    )


@case("the worked example: three tokens, identity projections", public=True)
def worked_example(ctx):
    x = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=np.float32)
    eye = np.eye(2, dtype=np.float32)
    got = ctx.check(FN, x, eye, eye, eye, eye, **TOL)
    expected = np.array(
        [[[1.0, 0.0], [0.330238, 0.669762], [0.751745, 0.751745]]], dtype=np.float32
    )
    check.allclose(got, expected, atol=1e-5, label=FN)


@case("a random batch: B = 2, T = 5, d = 8, square projections", public=True)
def random_batch(ctx):
    ctx.check(FN, ctx.randn(2, 5, 8), *weights(ctx, 8), **TOL)


# Rejects scales_by_input_dim (d_k != d) and weights_not_transposed (shapes do not fit).
@case("rectangular projections: d = 8, d_k = 4, d_v = 6, d_out = 10", public=True)
def rectangular(ctx):
    ctx.check(FN, ctx.randn(2, 6, 8), *weights(ctx, 8, d_k=4, d_v=6, d_out=10), **TOL)


# Rejects masks_diagonal and mask_wrong_direction with a message that names the cause.
@case("the first token attends only to itself: out[:, 0] = W_o W_v x[:, 0]", public=True)
def first_token(ctx):
    x = ctx.randn(2, 5, 8)
    w_q, w_k, w_v, w_o = weights(ctx, 8)
    out = ctx.call(FN, x, w_q, w_k, w_v, w_o)
    check.shape(out, (2, 5, 8), label=FN)
    check.allclose(out[:, 0], x[:, 0] @ w_v.T @ w_o.T, atol=1e-5, label="output at position 0")


@case("changing future tokens never changes earlier outputs", public=True)
def causality(ctx):
    def prop(rng):
        x = ctx.randn(2, 6, 8)
        w = weights(ctx, 8, d_k=4)
        t = int(rng.integers(1, 6))
        changed = x.copy()
        changed[:, t:] = ctx.randn(2, 6 - t, 8, scale=3.0)
        before = ctx.call(FN, x, *w)
        after = ctx.call(FN, changed, *w)
        check.shape(before, (2, 6, 8), label=FN)
        check.finite(before, label=FN)
        check.allclose(
            after[:, :t],
            before[:, :t],
            atol=1e-5,
            label=f"outputs before position {t} after changing tokens from position {t} on",
        )

    check.forall(prop, trials=6)


@case("a single token, and a batch of one")
def single_token(ctx):
    ctx.check(FN, ctx.randn(3, 1, 8), *weights(ctx, 8, d_k=4), **TOL)
    ctx.check(FN, ctx.randn(1, 12, 8), *weights(ctx, 8), **TOL)


# Rejects leaks_across_batch even for a solution that only leaks in some layouts.
@case("batch items are independent: each one alone gives the same rows")
def batch_independence(ctx):
    x = ctx.randn(3, 5, 8)
    w = weights(ctx, 8, d_v=6)
    together = ctx.check(FN, x, *w, **TOL)
    for b in range(3):
        alone = ctx.call(FN, x[b : b + 1], *w)
        check.allclose(alone[0], together[b], atol=1e-5, label=f"batch item {b} computed alone")
    # Reordering the batch reorders the outputs and changes nothing else.
    flipped = ctx.call(FN, x[::-1].copy(), *w)
    check.allclose(flipped[::-1], together, atol=1e-5, label="output for the reversed batch")


# B = T = d: every axis has the same length, which hides transposed and broadcast axes.
@case("all axes the same length: B = T = d = 4")
def all_axes_equal(ctx):
    ctx.check(FN, ctx.randn(4, 4, 4), *weights(ctx, 4), **TOL)
    ctx.check(FN, ctx.randn(4, 4, 4, scale=2.0), *weights(ctx, 4), **TOL)


@case("large scores: nearly one-hot attention")
def sharp_attention(ctx):
    got = ctx.check(FN, ctx.randn(2, 5, 8, scale=6.0), *weights(ctx, 8), **TOL)
    check.finite(got, label=FN)


@case("identical tokens share attention equally among the visible positions")
def uniform_attention(ctx):
    x = np.tile(ctx.randn(2, 1, 8), (1, 5, 1))
    w_q, w_k, w_v, w_o = weights(ctx, 8)
    out = ctx.check(FN, x, w_q, w_k, w_v, w_o, **TOL)
    # Every visible value vector is the same, so each output equals the first one.
    check.allclose(out, np.tile(out[:, :1], (1, 5, 1)), atol=1e-5, label=FN)


@case("random sizes for B, T, d, d_k, d_v, and d_out")
def random_sizes(ctx):
    for _ in range(3):
        b, t, d, d_k, d_v, d_out = (int(n) for n in ctx.rng.integers(1, 9, size=6))
        x = ctx.randn(b, t, d, scale=float(ctx.rng.uniform(0.5, 2.0)))
        ctx.check(FN, x, *weights(ctx, d, d_k=d_k, d_v=d_v, d_out=d_out), **TOL)
