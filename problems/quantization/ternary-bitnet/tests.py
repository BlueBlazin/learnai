"""Tests for ternary-bitnet.

The codes are compared exactly. Two kinds of input make that safe: matrices of small
multiples of a power of two, whose mean and quotients are exact in float32 (these may
sit exactly on the border |w| = scale / 2), and random matrices passed through
`tie_free`, which keeps |w| / scale at least 0.05 away from 0.5, as the statement
promises. The tie at 1.5 needs no care: both neighbors are clamped to 1.

JAX compiles every new shape, so the cases share a few layouts.
"""

import numpy as np

from learnai.judge import case, check

MARGIN = 0.05


def absmean(w):
    """The statement's scale, in float32 like a solution computes it."""
    gamma = np.abs(w).mean(dtype=np.float32)
    return np.float32(gamma if gamma > 0 else 1.0)


def tie_free(w):
    """A copy of w with no |w| / scale within MARGIN of 0.5. Offending elements are
    pushed to 0.8 scales; that moves the mean a little, so repeat until it is clean."""
    w = np.array(w, dtype=np.float32)
    for _ in range(50):
        scale = float(absmean(w))
        ratio = np.abs(w.astype(np.float64)) / scale
        bad = np.abs(ratio - 0.5) < MARGIN
        if not bad.any():
            return w
        w[bad] = (np.sign(w[bad]) * 0.8 * scale).astype(np.float32)
    raise AssertionError("could not make the input tie-free")


def dense(t, scale):
    """The float matrix the codes stand for, in float64."""
    return t.astype(np.float64) * float(scale)


def check_quantize(ctx, w):
    """ternary_quantize against the oracle, plus dtype, value set, and shape of the scale."""
    t, scale = ctx.call("ternary_quantize", w)
    want_t, want_scale = ctx.expected("ternary_quantize", w)
    check.shape(scale, (), label="scale")
    check.allclose(scale, want_scale, dtype=True, label="scale")
    check.shape(t, w.shape, label="t")
    check.that(t.dtype == np.int8, "ternary_quantize must return int8 codes", f"dtype {t.dtype}")
    check.that(
        bool(np.isin(t, (-1, 0, 1)).all()),
        "t contains values other than -1, 0, and 1",
        f"values present: {np.unique(t).tolist()}",
    )
    check.equal(t, want_t, label="t")
    return t, scale


def check_matmul(ctx, x, t, scale):
    """ternary_matmul against the oracle and against x @ (scale * t).T in float64."""
    scale = np.array(scale, dtype=np.float32)
    y = ctx.check("ternary_matmul", x, t, scale, atol=1e-5)
    check.allclose(y, x.astype(np.float64) @ dense(t, scale).T, atol=1e-5, label="ternary_matmul")
    return y


def native_layer(module, fw, w, x):
    """ternary_quantize then ternary_matmul on framework tensors, so that the codes and
    the scale reach the layer exactly as the quantizer returned them."""
    t, scale = module.ternary_quantize(fw.tensor(w))
    y = module.ternary_matmul(fw.tensor(x), t, scale)
    return fw.numpy(t), fw.numpy(scale), fw.numpy(y)


WORKED = np.array([[0.9, -0.2, 0.05, -1.3], [0.4, 0.0, -0.6, 0.6]], dtype=np.float32)
# Sums to 8 over 8 elements: the scale is exactly 1, and +-0.5 are exactly on the border.
BORDER = np.array([[0.5, -0.5, 2.0, -1.0], [0.0, 1.0, 1.5, 1.5]], dtype=np.float32)


@case("the worked example: a (2, 4) matrix and x = [1, 2, 3, 4]", public=True)
def worked_example(ctx):
    t, scale = check_quantize(ctx, WORKED)
    check.allclose(scale, np.float32(0.50625), label="scale")
    check.equal(t, np.array([[1, 0, 0, -1], [1, 0, -1, 1]]), label="t")
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = check_matmul(ctx, x, t.astype(np.int8), np.float32(0.50625))
    check.allclose(y, np.array([-1.51875, 1.0125], np.float32), label="ternary_matmul")


# Rejects absmax_scale and signed_mean (any matrix with both signs and a spread), and
# no_clip (some weights lie beyond 1.5 scales).
@case("a random (4, 8) weight matrix", public=True)
def random_matrix(ctx):
    w = tie_free(ctx.randn(4, 8, scale=0.05))
    t, _ = check_quantize(ctx, w)
    check.that(
        all(int((t == v).sum()) > 0 for v in (-1, 0, 1)),
        "a matrix of normal weights must use all three codes",
        f"count of -1, 0, 1: {[int((t == v).sum()) for v in (-1, 0, 1)]}",
    )


# Rejects ties_away_from_zero.
@case("weights exactly on the border |w| = scale / 2 become 0", public=True)
def border_weights(ctx):
    t, scale = check_quantize(ctx, BORDER)
    check.that(float(scale) == 1.0, "the scale of this matrix is exactly 1", f"scale = {scale}")
    check.equal(t, np.array([[0, 0, 1, -1], [0, 1, 1, 1]]), label="t")
    # The same matrix on other power-of-two scales: every quotient is still exact.
    check_quantize(ctx, BORDER * np.float32(0.25))
    check_quantize(ctx, -BORDER * np.float32(8.0))


# Rejects nonzero_mean (the zeros count in the mean) and a NaN from an all-zero matrix.
@case("exact zeros count in the mean, and an all-zero matrix gets scale 1", public=True)
def zeros(ctx):
    w = ctx.randn(4, 8, scale=0.3)
    w[ctx.rng.random(w.shape) < 0.5] = 0.0
    w[2] = 0.0
    check_quantize(ctx, tie_free(w))
    t, scale = check_quantize(ctx, np.zeros((4, 8), dtype=np.float32))
    check.finite(scale, label="scale")
    check.that(float(scale) == 1.0, "an all-zero matrix must get scale 1", f"scale = {scale}")
    check.that(not t.any(), "an all-zero matrix must quantize to codes 0", f"t = {t}")


# Rejects zero_as_plus_one and scale_dropped.
@case("ternary_matmul: given codes with all three values, on 1-D, 2-D, and 3-D inputs", public=True)
def matmul_given_codes(ctx):
    t = np.array(
        [
            [1, 0, -1, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, 0, 0, -1, 0, 0, -1],
        ],
        dtype=np.int8,
    )
    for shape in ((8,), (3, 8), (2, 3, 8)):
        check_matmul(ctx, ctx.randn(*shape), t, np.float32(0.37))


@case("quantize then multiply, on the solution's own codes and scale")
def end_to_end(ctx):
    w = tie_free(ctx.randn(4, 8, scale=0.02))
    x = ctx.randn(3, 8)
    t, scale, y = native_layer(ctx.solution, ctx.fw, w, x)
    _, _, want_y = native_layer(ctx.oracle, ctx.oracle_fw, w, x)
    check.allclose(
        y, want_y, atol=1e-5, dtype=True, label="ternary_matmul(x, *ternary_quantize(w))"
    )
    check.allclose(y, x.astype(np.float64) @ dense(t, scale).T, atol=1e-5, label="layer output")


@case("one outlier does not set the scale")
def outlier(ctx):
    w = ctx.randn(4, 8, scale=0.05)
    w[1, 3] = 9.0
    w[3, 0] = -30.0
    check_quantize(ctx, tie_free(w))


@case("matrices of one sign, and weights far beyond the scale")
def one_sign(ctx):
    w = np.abs(ctx.randn(4, 8)) + 0.01
    check_quantize(ctx, tie_free(w))
    check_quantize(ctx, tie_free(-w))
    # Heavy tails: many weights round to +-2 and beyond before the clamp.
    check_quantize(ctx, tie_free(ctx.randn(4, 8) ** 3))


@case("a square matrix, a single row, a single column, and a 1 x 1 matrix")
def shapes(ctx):
    check_quantize(ctx, tie_free(ctx.randn(8, 8, scale=2.0)))
    check_quantize(ctx, tie_free(ctx.randn(1, 8)))
    check_quantize(ctx, tie_free(ctx.randn(8, 1, scale=1e-3)))
    t, scale = check_quantize(ctx, np.array([[-3.0]], dtype=np.float32))
    check.that(
        int(t[0, 0]) == -1 and float(scale) == 3.0,
        "a 1 x 1 matrix must quantize to the sign of its weight and the scale |w|",
    )
    t8 = np.sign(ctx.randn(8, 8)).astype(np.int8)
    t8[ctx.rng.random((8, 8)) < 0.4] = 0
    check_matmul(ctx, ctx.randn(8), t8, np.float32(1.5))
    check_matmul(ctx, ctx.randn(8, 8), t8, np.float32(2e-3))


@case("border weights among random ones, on a scale that is a power of two")
def border_among_others(ctx):
    # Magnitudes from {0, 1/2, 1, 3/2, 2, 3} with mean exactly 1: every element exact.
    values = np.array([0.0, 0.5, 0.5, 1.0, 1.5, 1.5, 2.0, 1.0] * 4, dtype=np.float32)
    signs = np.where(ctx.rng.random(32) < 0.5, -1.0, 1.0).astype(np.float32)
    w = (ctx.rng.permutation(values) * signs).reshape(4, 8)
    for factor in (1.0, 2.0**-7, 2.0**5):
        t, scale = check_quantize(ctx, w * np.float32(factor))
        on_border = np.abs(w) == 0.5
        check.that(
            float(scale) == factor and not t[on_border].any(),
            "a weight of magnitude exactly scale / 2 must get code 0 (round half to even)",
            f"scale = {scale}, codes of the border weights = {t[on_border].tolist()}",
        )


@case("the layer equals x @ (scale * t).T for random weights, inputs, and magnitudes")
def layer_property(ctx):
    shapes = [((4, 8), (3, 8)), ((8, 8), (8,)), ((4, 8), (2, 3, 8))]

    def prop(rng):
        w_shape, x_shape = shapes[int(rng.integers(len(shapes)))]
        magnitude = float(10.0 ** rng.uniform(-3, 2))
        w = ctx.randn(*w_shape, scale=magnitude)
        w[rng.random(w_shape) < 0.2] = 0.0
        w = tie_free(w)
        x = ctx.randn(*x_shape, scale=float(10.0 ** rng.uniform(-1, 1)))
        t, scale, y = native_layer(ctx.solution, ctx.fw, w, x)
        check.shape(t, w_shape, label="t")
        check.shape(scale, (), label="scale")
        check.shape(y, (*x_shape[:-1], w_shape[0]), label="ternary_matmul")
        check.that(
            t.dtype == np.int8 and bool(np.isin(t, (-1, 0, 1)).all()),
            "t must be int8 with values -1, 0, and 1 only",
            f"dtype {t.dtype}, values present: {np.unique(t).tolist()}",
        )
        check.allclose(scale, absmean(w), label="scale")
        # Nearest of the three grid points: within half a scale, except beyond the ends.
        error = np.abs(dense(t, scale) - w)
        inside = np.abs(w) <= 1.5 * float(scale)
        check.that(
            bool((error[inside] <= 0.5 * float(scale) * 1.001).all()),
            "a weight within 1.5 scales of zero is more than half a scale from its grid point",
            f"scale = {scale}, largest such error = {error[inside].max():.6g}",
        )
        want = x.astype(np.float64) @ dense(t, scale).T
        check.allclose(y, want, rtol=1e-4, atol=1e-5 * magnitude * 10, label="layer output")

    check.forall(prop, trials=8)
