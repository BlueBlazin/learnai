"""Tests for fused-projection-split.

Numeric cases compare with the oracle. Property cases state what fusing means without
it: a fused weight applied to an input gives the concatenation of what the separate
weights give, and the fused SwiGLU equals the three-matrix SwiGLU computed here in
numpy. A handful of shapes is reused throughout to keep JAX compilation short.
"""

import numpy as np

from learnai.judge import case, check

TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
D, H, D_OUT = 8, 12, 6  # SwiGLU widths used by most cases
QKV = [8, 4, 4]  # query, key, and value widths of a grouped-query layer with d_in = 8


def silu(z):
    return z / (1.0 + np.exp(-z))


def swiglu_numpy(x, w_gate, w_up, w_down):
    """The three-matrix SwiGLU of `swiglu-mlp`, in float64."""
    x = x.astype(np.float64)
    return ((silu(x @ w_gate.T.astype(np.float64)) * (x @ w_up.T)) @ w_down.T).astype(np.float32)


def mlp_weights(ctx, d=D, h=H, d_out=D_OUT, scale=1.0):
    s = scale / np.sqrt(d)
    return ctx.randn(h, d, scale=s), ctx.randn(h, d, scale=s), ctx.randn(d_out, h, scale=0.3)


def check_split(ctx, w, parts):
    got = ctx.check("split_fused", w, parts, rtol=0.0, atol=0.0)
    check.that(
        isinstance(got, (list, tuple)) and len(got) == len(parts),
        "split_fused: the result is not a list with one tensor per part",
    )
    return got


@case("the worked example: a (3, 2) weight split into 1 + 2 rows, and a fused SwiGLU", public=True)
def worked_example(ctx):
    w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    got = check_split(ctx, w, [1, 2])
    check.allclose(got[0], w[:1], rtol=0.0, atol=0.0, label="split_fused[0]")
    check.allclose(got[1], w[1:], rtol=0.0, atol=0.0, label="split_fused[1]")
    fused = ctx.check("fuse", [w[:1], w[1:]], rtol=0.0, atol=0.0)
    check.allclose(fused, w, rtol=0.0, atol=0.0, label="fuse")

    x = np.array([1.0, 2.0], dtype=np.float32)
    w_in = np.array(
        [[1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 0.5]], dtype=np.float32
    )
    w_down = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]], dtype=np.float32)
    got = ctx.check("swiglu_fused", x, w_in, w_down, **TOL)
    expected = np.array([5.050899, -3.334534], dtype=np.float32)
    check.allclose(got, expected, atol=1e-5, label="swiglu_fused")


# Rejects equal_chunks: the three parts have different sizes.
@case("a fused QKV weight of a grouped-query layer: parts [8, 4, 4]", public=True)
def qkv_parts(ctx):
    w = ctx.randn(16, D)
    got = check_split(ctx, w, QKV)
    for i, rows in enumerate(QKV):
        check.shape(got[i], (rows, D), label=f"split_fused[{i}]")
    ctx.check("fuse", [ctx.randn(8, D), ctx.randn(4, D), ctx.randn(4, D)], rtol=0.0, atol=0.0)


# Rejects parts_as_offsets: [2, 3, 3] read as cut points gives rows 0:2, 2:3, 3:3, 3:8.
@case("parts are sizes, not offsets: an (8, 8) weight with parts [2, 3, 3]", public=True)
def sizes_not_offsets(ctx):
    w = ctx.randn(8, D)
    got = check_split(ctx, w, [2, 3, 3])
    check.allclose(got[1], w[2:5], rtol=0.0, atol=0.0, label="split_fused[1]")
    check.allclose(got[2], w[5:8], rtol=0.0, atol=0.0, label="split_fused[2]")


# Rejects splits_input_axis and fuse_along_input_axis: with a square weight the wrong
# axis has the right length, so the mistake must fail on values.
@case("a square fused weight is split by rows, and equal parts are fused by rows", public=True)
def square_weight(ctx):
    w = ctx.randn(8, D)
    check_split(ctx, w, [4, 4])
    check_split(ctx, w, [4, 2, 2])
    a, b = ctx.randn(4, D), ctx.randn(4, D)
    got = ctx.check("fuse", [a, b], rtol=0.0, atol=0.0)
    check.shape(got, (8, D), label="fuse")


# Rejects halves_swapped and activation_before_split: the gate rows give negative
# pre-activations and the up rows positive ones.
@case("the gate rows come first: a negative gate branch and a positive up branch", public=True)
def gate_first(ctx):
    x = np.abs(ctx.randn(2, 5, D)) + 0.5
    w_gate = -np.abs(ctx.randn(H, D, scale=0.3))
    w_up = np.abs(ctx.randn(H, D, scale=0.3))
    w_down = ctx.randn(D_OUT, H, scale=0.3)
    w_in = np.concatenate([w_gate, w_up], axis=0)
    got = ctx.check("swiglu_fused", x, w_in, w_down, **TOL)
    check.allclose(
        got,
        swiglu_numpy(x, w_gate, w_up, w_down),
        label="swiglu_fused against SwiGLU with separate gate and up weights",
        **TOL,
    )


@case("fused and split forms agree: projections, and SwiGLU against three matrices", public=True)
def fused_equals_split(ctx):
    def prop(rng):
        x = ctx.randn(2, 5, D)
        ws = [ctx.randn(rows, D) for rows in QKV]
        fused = ctx.call("fuse", ws)
        check.shape(fused, (16, D), label="fuse")
        separate = np.concatenate([x @ w.T for w in ws], axis=-1)
        check.allclose(
            x @ fused.T, separate, label="x @ fuse(ws)^T against the separate projections", **TOL
        )
        back = ctx.call("split_fused", fused, QKV)
        check.that(len(back) == 3, "split_fused: the result does not have one tensor per part")
        for i, w in enumerate(ws):
            check.allclose(back[i], w, rtol=0.0, atol=0.0, label=f"split_fused(fuse(ws))[{i}]")

        scale = float(10.0 ** rng.uniform(-0.5, 0.7))
        w_gate, w_up, w_down = mlp_weights(ctx, scale=scale)
        w_in = ctx.call("fuse", [w_gate, w_up])
        check.shape(w_in, (2 * H, D), label="fuse")
        got = ctx.call("swiglu_fused", x, w_in, w_down)
        check.allclose(
            got,
            swiglu_numpy(x, w_gate, w_up, w_down),
            label="swiglu_fused(x, fuse([w_gate, w_up]), w_down)",
            **TOL,
        )

    check.forall(prop, trials=6)


@case("one part, parts of one row, and a single weight fused")
def degenerate_parts(ctx):
    w = ctx.randn(8, D)
    got = check_split(ctx, w, [8])
    check.allclose(got[0], w, rtol=0.0, atol=0.0, label="split_fused[0]")
    check_split(ctx, w, [1, 1, 1, 1, 1, 1, 1, 1])
    check_split(ctx, w, [1, 6, 1])
    check_split(ctx, w, [7, 1])
    ctx.check("fuse", [w], rtol=0.0, atol=0.0)
    ctx.check("fuse", [w[i : i + 1] for i in range(8)], rtol=0.0, atol=0.0)


@case("fuse(split_fused(w, parts)) is w, for several ways of cutting 16 rows")
def round_trip(ctx):
    w = ctx.randn(16, D)
    for parts in ([8, 4, 4], [4, 4, 8], [4, 8, 4], [8, 8], [4, 4, 4, 4]):
        pieces = check_split(ctx, w, parts)
        back = ctx.call("fuse", list(pieces))
        check.allclose(back, w, rtol=0.0, atol=0.0, label=f"fuse(split_fused(w, {parts}))")


@case("swiglu_fused: a single vector, a 2-D batch, and d_out different from d")
def swiglu_ranks(ctx):
    for shape in ((D,), (5, D), (2, 5, D)):
        w_gate, w_up, w_down = mlp_weights(ctx)
        ctx.check("swiglu_fused", ctx.randn(*shape), np.concatenate([w_gate, w_up]), w_down, **TOL)


@case("swiglu_fused: saturated gates, and an all-square layer d = h = d_out")
def swiglu_extremes(ctx):
    w_gate, w_up, w_down = mlp_weights(ctx, scale=40.0)
    got = ctx.check("swiglu_fused", ctx.randn(5, D), np.concatenate([w_gate, w_up]), w_down, **TOL)
    check.finite(got, label="swiglu_fused")
    w_gate, w_up, w_down = mlp_weights(ctx, d=D, h=D, d_out=D)
    x = ctx.randn(5, D)
    got = ctx.check("swiglu_fused", x, np.concatenate([w_gate, w_up]), w_down, **TOL)
    check.allclose(got, swiglu_numpy(x, w_gate, w_up, w_down), label="swiglu_fused", **TOL)
