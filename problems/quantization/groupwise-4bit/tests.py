"""Tests for groupwise-4bit.

Codes are compared exactly, so no input may sit near a rounding tie: frameworks could
then round (w - bias) / scale to different sides because of a one-ulp difference. Every
input goes through `tie_free`, which moves the offending elements onto a grid point of
their own group without changing the group's minimum or maximum.

JAX compiles every new combination of shape and group size (about 0.15 s each), so the
cases share a handful of layouts instead of drawing fresh shapes.
"""

import numpy as np

from learnai.judge import case, check

MARGIN = 0.05


def group_stats(w, bits, group_size):
    """(groups, lo, scale) of the statement, in float32 like a solution computes them."""
    out_dim, in_dim = w.shape
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.min(axis=-1, keepdims=True)
    hi = groups.max(axis=-1, keepdims=True)
    levels = np.float32(2**bits - 1)
    scale = np.where(hi > lo, (hi - lo) / levels, np.float32(1.0)).astype(np.float32)
    return groups, lo, hi, scale


def steps(w, bits, group_size):
    """(w - bias) / scale in float64: the number each code is rounded from."""
    groups, lo, _, scale = group_stats(w, bits, group_size)
    return (groups.astype(np.float64) - lo) / scale


def near_tie(t):
    return np.abs(t - np.floor(t) - 0.5) < MARGIN


def tie_free(w, bits, group_size):
    """A copy of w in which no element is within MARGIN of a rounding tie."""
    w = np.array(w, dtype=np.float32)
    groups, lo, hi, scale = group_stats(w, bits, group_size)
    t = steps(w, bits, group_size)
    snapped = np.clip(lo + np.round(t) * scale.astype(np.float64), lo, hi).astype(np.float32)
    fixed = np.where(near_tie(t), snapped, groups).reshape(w.shape)
    _, lo2, hi2, _ = group_stats(fixed, bits, group_size)
    ok = np.array_equal(lo, lo2) and np.array_equal(hi, hi2)
    if not ok or near_tie(steps(fixed, bits, group_size)).any():
        raise AssertionError("could not make the input tie-free")
    return fixed


def dequantize(codes, scale, bias, group_size):
    """codes * scale + bias in float64, each group with its own scale and bias."""
    return codes.astype(np.float64) * np.repeat(scale, group_size, axis=1) + np.repeat(
        bias, group_size, axis=1
    )


def check_quantize(ctx, w, bits, group_size):
    """Compare with the oracle, then check dtype and value range on the solution's output."""
    w = tie_free(w, bits, group_size)
    codes, scale, bias = ctx.check("quantize_groups", w, bits, group_size)
    check.that(
        codes.dtype == np.uint8,
        "codes must be a uint8 tensor",
        f"got dtype {codes.dtype}",
    )
    return w, codes, scale, bias


def blocks(ctx, out_dim, n_groups, group_size, *, spread=(0.05, 2.0), offset=1.0):
    """A matrix whose groups have ranges and centers of their own."""
    width = ctx.uniform(out_dim, n_groups, 1, low=spread[0], high=spread[1])
    center = ctx.uniform(out_dim, n_groups, 1, low=-offset, high=offset)
    w = ctx.randn(out_dim, n_groups, group_size) * width + center
    return w.reshape(out_dim, n_groups * group_size)


WORKED = np.array(
    [
        [0.0, 0.33, -0.5, 1.0, 2.0, 2.2, 2.9, 2.6],
        [-1.0, -1.0, -1.0, -1.0, 0.5, -0.25, 0.1, 0.3],
    ],
    dtype=np.float32,
)


@case("the worked example: two rows, two groups of four, 4 bits", public=True)
def worked_example(ctx):
    codes, scale, bias = ctx.call("quantize_groups", WORKED, 4, 4)
    check.equal(
        codes,
        np.array([[5, 8, 0, 15, 0, 3, 15, 10], [0, 0, 0, 0, 15, 0, 7, 11]]),
        label="codes",
    )
    check.allclose(scale, np.array([[0.1, 0.06], [1.0, 0.05]], np.float32), label="scale")
    check.allclose(bias, np.array([[-0.5, 2.0], [-1.0, -0.25]], np.float32), label="bias")
    check_quantize(ctx, WORKED, 4, 4)


# Rejects whole_row_range (groups of one row differ in range) and truncates.
@case("a (4, 64) matrix in groups of 16, each group with its own range", public=True)
def groups_with_own_ranges(ctx):
    check_quantize(ctx, blocks(ctx, 4, 4, 16), 4, 16)


# Rejects constant_group_nan.
@case("constant groups: zeros, a repeated value, and a constant matrix", public=True)
def constant_groups(ctx):
    w = ctx.randn(3, 24)
    w[0, 8:16] = 0.0
    w[1, 0:8] = 0.375
    w[2, 16:24] = -2.5
    _, codes, scale, bias = check_quantize(ctx, w, 4, 8)
    values = dequantize(codes, scale, bias, 8)
    check.finite(scale, label="scale")
    check.that(
        np.array_equal(values[0, 8:16], w[0, 8:16]) and np.array_equal(values[1, 0:8], w[1, 0:8]),
        "a constant group must be recovered exactly by codes * scale + bias",
        f"codes = {codes}\nscale = {scale}\nbias = {bias}",
    )
    check_quantize(ctx, np.full((3, 24), 0.7, dtype=np.float32), 4, 8)
    check_quantize(ctx, np.zeros((3, 24), dtype=np.float32), 3, 8)


# Rejects assumes_four_bits and divides_by_code_count.
@case("other code widths: 2, 3, 6, and 8 bits", public=True)
def other_bit_widths(ctx):
    for bits in (2, 3, 6, 8):
        _, codes, _, _ = check_quantize(ctx, blocks(ctx, 4, 4, 16), bits, 16)
        check.that(
            int(codes.max()) == 2**bits - 1,
            f"the maximum of a group must get the largest code, 2**bits - 1, at {bits} bits",
            f"largest code returned: {int(codes.max())}",
        )


@case("one group per row (group_size equal to in), and group_size 1", public=True)
def extreme_group_sizes(ctx):
    check_quantize(ctx, blocks(ctx, 4, 1, 64), 4, 64)
    # With group_size 1 every group is constant: codes 0, scale 1, bias equal to w.
    check_quantize(ctx, ctx.randn(3, 6), 4, 1)


# Rejects groups_along_output_axis where its reshape is valid: out is a multiple of group_size.
@case("a square matrix whose row count is a multiple of group_size")
def square_matrix(ctx):
    check_quantize(ctx, blocks(ctx, 8, 2, 4), 4, 4)
    check_quantize(ctx, blocks(ctx, 8, 2, 4), 3, 4)


@case("groups far from zero: all-positive and all-negative groups keep their own bias")
def offset_groups(ctx):
    w = blocks(ctx, 4, 4, 16, spread=(0.01, 0.2), offset=40.0)
    check_quantize(ctx, w, 4, 16)
    check_quantize(ctx, np.abs(blocks(ctx, 4, 4, 16)) + 3.0, 5, 16)
    check_quantize(ctx, -np.abs(blocks(ctx, 4, 4, 16)) - 0.5, 4, 16)


@case("one outlier in a row only costs its own group")
def outlier(ctx):
    w = ctx.randn(4, 64, scale=0.05)
    w[1, 40] = 9.0
    w[3, 63] = -30.0
    check_quantize(ctx, w, 4, 16)


@case("small weights in groups of 64 at 4 bits, the MLX default")
def mlx_default(ctx):
    check_quantize(ctx, ctx.randn(4, 64, scale=0.04), 4, 64)


@case("every weight is recovered within half a step of its own group")
def half_step_error(ctx):
    # (out, groups per row, group_size); the last layout is a single row.
    shapes = [(4, 4, 16), (3, 3, 8), (8, 2, 4), (1, 4, 4)]

    def prop(rng):
        out_dim, n_groups, group_size = shapes[int(rng.integers(len(shapes)))]
        bits = int(rng.integers(2, 9))
        offset = float(10.0 ** rng.uniform(-1, 1.5))
        w = tie_free(blocks(ctx, out_dim, n_groups, group_size, offset=offset), bits, group_size)
        codes, scale, bias = ctx.call("quantize_groups", w, bits, group_size)
        check.shape(codes, w.shape, label="codes")
        check.shape(scale, (out_dim, n_groups), label="scale")
        check.shape(bias, (out_dim, n_groups), label="bias")
        check.that(
            codes.dtype.kind in "iu" and codes.min() >= 0 and codes.max() <= 2**bits - 1,
            "codes must be integers in [0, 2**bits - 1]",
            f"bits = {bits}, dtype {codes.dtype}, range [{codes.min()}, {codes.max()}]",
        )
        check.finite(scale, label="scale")
        check.that(bool((scale > 0).all()), "every scale must be positive")
        error = np.abs(dequantize(codes, scale, bias, group_size) - w)
        bound = 0.5 * np.repeat(scale, group_size, axis=1).astype(np.float64) * 1.001 + 1e-5
        worst = np.unravel_index(np.argmax(error - bound), error.shape)
        check.that(
            bool((error <= bound).all()),
            "codes * scale + bias is more than half a step away from w",
            f"bits = {bits}, group_size = {group_size}, element {tuple(int(i) for i in worst)}: "
            f"|error| = {error[worst]:.6g}, half a step = {bound[worst]:.6g}",
        )
        ctx.check("quantize_groups", w, bits, group_size)

    check.forall(prop, trials=10)
