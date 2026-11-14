"""Tests for qk-norm-attention. Every case compares the solution with the oracle.

The RoPE tables are built here and passed in, as the statement says: cos[t, i] and
sin[t, i] of positions[t] * base^(-2i / d), the interleaved convention of the `rope`
problem.
"""

import numpy as np

from whetstone.judge import case, check

FN = "qk_norm_attention"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def rope_tables(positions, d, base=10000.0):
    """cos and sin of the RoPE angles, each of shape (T, d/2), float32."""
    theta = base ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    angles = np.asarray(positions, dtype=np.float64)[:, None] * theta
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def gains(ctx, d, low=0.5, high=1.5):
    """Two different gain vectors, each varying from dimension to dimension."""
    return ctx.uniform(d, low=low, high=high), ctx.uniform(d, low=low, high=high)


def inputs(ctx, b, h, t, d, positions=None, base=10000.0):
    """q, k, v, q_weight, k_weight, cos, sin for a random problem."""
    positions = np.arange(t) if positions is None else positions
    return (
        ctx.randn(b, h, t, d),
        ctx.randn(b, h, t, d),
        ctx.randn(b, h, t, d),
        *gains(ctx, d),
        *rope_tables(positions, d, base),
    )


@case("the worked example: two tokens, d = 2, positions 0 and 1", public=True)
def worked_example(ctx):
    q = np.array([[[[3.0, 4.0], [4.0, -3.0]]]], dtype=np.float32)
    k = np.array([[[[1.0, 0.0], [20.0, 20.0]]]], dtype=np.float32)
    v = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    q_weight = np.array([1.0, 0.5], dtype=np.float32)
    k_weight = np.array([2.0, 1.0], dtype=np.float32)
    cos = np.array([[1.0], [0.540302]], dtype=np.float32)
    sin = np.array([[0.0], [0.841471]], dtype=np.float32)
    got = ctx.check(FN, q, k, v, q_weight, k_weight, cos, sin, **TOL)
    expected = np.array([[[[1.0, 0.0], [0.653979, 0.346021]]]], dtype=np.float32)
    check.allclose(got, expected, atol=1e-5, label=FN)


@case("a random batch: shape (2, 3, 5, 8), positions 0 to 4", public=True)
def random_batch(ctx):
    ctx.check(FN, *inputs(ctx, 2, 3, 5, 8), **TOL)


# Rejects norm_over_all_heads by a wide margin: with one root mean square per token,
# the large head dominates it and the small heads are scaled to almost nothing.
@case("heads on very different scales: queries and keys times 0.01, 1, and 100", public=True)
def heads_on_different_scales(ctx):
    q, k, v, q_weight, k_weight, cos, sin = inputs(ctx, 2, 3, 5, 8)
    head_scale = np.array([0.01, 1.0, 100.0], dtype=np.float32)[None, :, None, None]
    ctx.check(FN, q * head_scale, k * head_scale[:, ::-1], v, q_weight, k_weight, cos, sin, **TOL)


# Rejects normalizes_values_too with a case whose name says what went wrong.
@case("the values keep their magnitude: v on the order of 50", public=True)
def large_values(ctx):
    q, k, v, q_weight, k_weight, cos, sin = inputs(ctx, 2, 3, 5, 8)
    ctx.check(FN, q, k, 50.0 * v + 20.0, q_weight, k_weight, cos, sin, **TOL)


# Rejects norm_after_rope: the gains differ a lot between the two dimensions of a pair,
# and the positions are far from 0, so every pair is rotated by a large angle.
@case("gains from 0.1 to 3 that differ within each rotation pair, positions 37 to 41")
def uneven_gains(ctx):
    q, k, v = ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8)
    q_weight = np.array([0.1, 3.0, 2.0, 0.3, 1.0, 2.5, 0.2, 1.5], dtype=np.float32)
    k_weight = np.array([2.0, 0.2, 0.5, 2.5, 3.0, 0.1, 1.0, 0.4], dtype=np.float32)
    cos, sin = rope_tables(37 + np.arange(5), 8)
    ctx.check(FN, q, k, v, q_weight, k_weight, cos, sin, **TOL)
    cos, sin = rope_tables([3, 0, 11, 7, 3], 8, base=100.0)
    ctx.check(FN, q, k, v, q_weight, k_weight, cos, sin, **TOL)


@case("unit gains, and no rotation (every angle 0)")
def identity_parts(ctx):
    q, k, v, q_weight, k_weight, cos, sin = inputs(ctx, 2, 3, 5, 8)
    ones = np.ones(8, dtype=np.float32)
    ctx.check(FN, q, k, v, ones, ones, cos, sin, **TOL)
    ctx.check(FN, q, k, v, q_weight, k_weight, np.ones_like(cos), np.zeros_like(sin), **TOL)


# Rejects eps_outside_sqrt, and any solution that ignores the eps argument.
@case("eps that is not negligible: 0.5 and 0.01 with short vectors")
def explicit_eps(ctx):
    q, k, v, q_weight, k_weight, cos, sin = inputs(ctx, 2, 3, 5, 8)
    ctx.check(FN, q, k, v, q_weight, k_weight, cos, sin, eps=0.5, **TOL)
    ctx.check(FN, 0.1 * q, 0.2 * k, v, q_weight, k_weight, cos, sin, 0.01, **TOL)


@case("a single token at position 9, and a single rotation pair with a single head")
def minimal_sizes(ctx):
    ctx.check(FN, *inputs(ctx, 2, 3, 1, 8, positions=[9]), **TOL)
    ctx.check(FN, *inputs(ctx, 2, 1, 5, 2), **TOL)


# Every broadcastable axis has the same length: (T, d/2) tables against (B, H, T, d).
@case("all axes the same length: B = H = T = d/2 = 4")
def all_axes_equal(ctx):
    ctx.check(FN, *inputs(ctx, 4, 4, 4, 8, positions=[2, 5, 6, 30]), **TOL)


@case("long queries and keys: entries on the order of 500")
def long_vectors(ctx):
    q, k, v, q_weight, k_weight, cos, sin = inputs(ctx, 2, 3, 5, 8)
    got = ctx.check(FN, 300.0 * q, 300.0 * k, v, q_weight, k_weight, cos, sin, **TOL)
    check.finite(got, label=FN)


@case("random shapes, positions, gains, and eps")
def random_sweep(ctx):
    for _ in range(2):
        b, h, t = (int(n) for n in ctx.rng.integers(1, 6, size=3))
        d = 2 * int(ctx.rng.integers(1, 9))
        positions = ctx.rng.integers(0, 64, size=t)
        q, k, v, _, _, cos, sin = inputs(ctx, b, h, t, d, positions=positions)
        q_weight, k_weight = gains(ctx, d, low=0.1, high=3.0)
        eps = float(10.0 ** ctx.rng.uniform(-6, -2))
        ctx.check(FN, q, k, v, q_weight, k_weight, cos, sin, eps=eps, **TOL)
