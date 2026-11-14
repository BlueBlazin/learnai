"""Tests for quantized-matmul.

Most cases compare with the oracle. The property case compares with the definition
instead: dequantize in float64, then multiply.

JAX compiles every new combination of shapes and group size (about 0.15 s each), so the
cases share a few layouts.
"""

import numpy as np

from whetstone.judge import case, check

RTOL, ATOL = 1e-4, 5e-4


def layer(ctx, out_dim, in_dim, group_size, *, bits=4, signed_scale=False):
    """Random codes, scales, and biases of a group-quantized (out_dim, in_dim) matrix."""
    n_groups = in_dim // group_size
    codes = ctx.rng.integers(0, 2**bits, size=(out_dim, in_dim)).astype(np.uint8)
    scale = ctx.uniform(out_dim, n_groups, low=0.02, high=3.0 / (2**bits - 1))
    if signed_scale:
        scale *= ctx.rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=scale.shape)
    bias = ctx.randn(out_dim, n_groups)
    return codes, scale, bias


def dequantize(codes, scale, bias, group_size):
    """The float64 matrix codes * scale + bias, each group with its own scale and bias."""
    return codes.astype(np.float64) * np.repeat(scale, group_size, axis=1) + np.repeat(
        bias, group_size, axis=1
    )


def check_qmatmul(ctx, x, codes, scale, bias, group_size):
    return ctx.check("qmatmul", x, codes, scale, bias, group_size, rtol=RTOL, atol=ATOL)


WORKED_CODES = np.array([[5, 8, 0, 15, 0, 3, 15, 10], [0, 0, 0, 0, 15, 0, 7, 11]], dtype=np.uint8)
WORKED_SCALE = np.array([[0.1, 0.06], [1.0, 0.05]], dtype=np.float32)
WORKED_BIAS = np.array([[-0.5, 2.0], [-1.0, -0.25]], dtype=np.float32)


@case("the worked example: two outputs, two groups of four inputs", public=True)
def worked_example(ctx):
    x = np.array([1.0, 2.0, -1.0, 0.5, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)
    y = ctx.call("qmatmul", x, WORKED_CODES, WORKED_SCALE, WORKED_BIAS, 4)
    check.allclose(y, np.array([4.08, -2.95], np.float32), atol=ATOL, label="qmatmul")
    check_qmatmul(ctx, x, WORKED_CODES, WORKED_SCALE, WORKED_BIAS, 4)


# Rejects drops_bias, strided_groups, one_scale_per_row, and bias_sum_times_input_sum.
@case("a batch of 3 inputs through a (5, 64) matrix in groups of 16", public=True)
def random_batch(ctx):
    check_qmatmul(ctx, ctx.randn(3, 64), *layer(ctx, 5, 64, 16), 16)


# One-hot inputs read out single columns of the dequantized matrix, so a misaligned
# group or a mispaired bias shows up at the column where it happens.
@case("one-hot inputs read out the columns of the dequantized matrix", public=True)
def one_hot_inputs(ctx):
    x = np.eye(8, dtype=np.float32)
    y = check_qmatmul(ctx, x, WORKED_CODES, WORKED_SCALE, WORKED_BIAS, 4)
    want = dequantize(WORKED_CODES, WORKED_SCALE, WORKED_BIAS, 4).T
    check.allclose(y, want.astype(np.float32), atol=ATOL, label="qmatmul(identity)")


# bias_sum_times_input_sum, strided_groups, and one_scale_per_row are right here.
@case("one group per row: group_size equal to in", public=True)
def one_group_per_row(ctx):
    check_qmatmul(ctx, ctx.randn(3, 64), *layer(ctx, 5, 64, 64), 64)


# Rejects bias_times_group_size on its home ground: after the all-ones input, where it
# is right, come inputs whose group sums differ from group_size.
@case("an input of all ones, then inputs with other group sums", public=True)
def group_sums(ctx):
    codes, scale, bias = layer(ctx, 5, 64, 16)
    x = np.ones((3, 64), dtype=np.float32)
    check_qmatmul(ctx, x, codes, scale, bias, 16)
    x[1, 16:32] = 0.0
    x[2] = np.where(np.arange(64) % 2 == 0, 1.0, -1.0)  # every group sums to 0
    check_qmatmul(ctx, x, codes, scale, bias, 16)


@case("batch axes: x of shape (2, 3, in) and of shape (in,)")
def batch_axes(ctx):
    codes, scale, bias = layer(ctx, 5, 64, 16)
    check_qmatmul(ctx, ctx.randn(2, 3, 64), codes, scale, bias, 16)
    check_qmatmul(ctx, ctx.randn(64), codes, scale, bias, 16)


@case("8-bit codes up to 255 and scales of both signs")
def eight_bit_codes(ctx):
    codes, scale, bias = layer(ctx, 5, 64, 16, bits=8, signed_scale=True)
    codes[0, :4] = 255
    check_qmatmul(ctx, ctx.randn(3, 64), codes, scale, bias, 16)


# Every axis has length 16 or 4, so a contraction over the wrong axis still has a valid shape.
@case("square everything: 16 inputs, 16 outputs, a batch of 16, and groups of 4")
def square_everything(ctx):
    check_qmatmul(ctx, ctx.randn(16, 16), *layer(ctx, 16, 16, 4), 4)
    check_qmatmul(ctx, ctx.randn(4, 16), *layer(ctx, 4, 16, 4), 4)


@case("group_size 1: every weight has its own scale and bias")
def group_size_one(ctx):
    check_qmatmul(ctx, ctx.randn(4, 16), *layer(ctx, 4, 16, 1), 1)


@case("matches dequantize-then-multiply for random group sizes on non-square matrices")
def matches_definition(ctx):
    def prop(rng):
        group_size = int(rng.choice([2, 3, 12, 48]))
        bits = int(rng.integers(2, 9))
        codes, scale, bias = layer(ctx, 5, 48, group_size, bits=bits, signed_scale=True)
        x = ctx.randn(2, 48, scale=float(10.0 ** rng.uniform(-1, 0.5)))
        y = ctx.call("qmatmul", x, codes, scale, bias, group_size)
        check.shape(y, (2, 5), label="qmatmul")
        want = x.astype(np.float64) @ dequantize(codes, scale, bias, group_size).T
        check.allclose(y, want, rtol=1e-3, atol=1e-3, label="qmatmul versus x @ dequantized.T")

    check.forall(prop, trials=8)
