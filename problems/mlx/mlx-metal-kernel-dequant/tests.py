"""Tests for mlx-metal-kernel-dequant.

A Metal kernel runs only on the Apple GPU, so every case is marked `mlx-gpu`.

The inputs are made here, in NumPy: random codes, packed into uint32 words lowest bits
first (the layout of `mx.quantize` and of MLX checkpoints), and random scales and
biases rounded to the dtype under test. The expected matrix is
`code * scale + bias` in float64 from exactly those values, so the oracle is neither
the reference solution nor `mx.dequantize` (which the worked example is checked
against as well, once, to tie the format to MLX's).

Each call of the solution is checked three ways: the lazy graph behind the result
(exactly one `CustomKernel`, otherwise only operations that rearrange data; the inputs
are evaluated first), its shape and dtype, and its values. The oracle never puts the
expected matrix on the GPU, so a kernel that leaves rows unwritten cannot find the
answer in recycled memory.

The last case times the solution against the reference kernel on a 4096 x 4096 layer.
"""

import io
import re

import mlx.core as mx
import numpy as np

from learnai.judge import case, check

FN = "metal_dequantize"
# The tolerances stated in Constraints, by the dtype of the scales.
TOL = {
    mx.float32: {"rtol": 1e-5, "atol": 1e-6},
    mx.float16: {"rtol": 2e-3, "atol": 1e-3},
    mx.bfloat16: {"rtol": 1e-2, "atol": 4e-3},
}
REARRANGE = {
    "Reshape",
    "Flatten",
    "Unflatten",
    "Squeeze",
    "ExpandDims",
    "Transpose",
    "Slice",
    "Broadcast",
    "Contiguous",
}


def graph_ops(array):
    """Names of the operations still pending behind `array` (empty: evaluated)."""
    dot = io.StringIO()
    mx.export_to_dot(dot, array)
    return re.findall(r'label ="(\w+)"', dot.getvalue())


def pack(codes, bits):
    """(out, in) codes -> (out, in * bits / 32) uint32 words, lowest bits first."""
    n_out, n_in = codes.shape
    per_word = 32 // bits
    grouped = codes.astype(np.uint64).reshape(n_out, n_in // per_word, per_word)
    shifts = np.arange(per_word, dtype=np.uint64) * np.uint64(bits)
    return (grouped << shifts).sum(axis=-1).astype(np.uint32)


def layer(rng, n_out, n_in, group_size, bits, dtype=mx.float32, codes=None):
    """A random quantized layer: (words, scales, biases) as evaluated MLX arrays, and
    the float64 matrix they stand for."""
    if codes is None:
        codes = rng.integers(0, 1 << bits, (n_out, n_in))
    groups = n_in // group_size
    # Weights of order 0.1, as in a real layer: bias is the low end of each group.
    scales = rng.uniform(0.2, 1.0, (n_out, groups)) / (1 << bits)
    scales = scales * rng.choice([-1.0, 1.0], (n_out, groups))
    biases = rng.uniform(-0.5, 0.5, (n_out, groups))
    scales = mx.array(scales.astype(np.float32)).astype(dtype)
    biases = mx.array(biases.astype(np.float32)).astype(dtype)
    words = mx.array(pack(codes, bits))
    mx.eval(words, scales, biases)
    s = np.repeat(np.array(scales.astype(mx.float32)).astype(np.float64), group_size, axis=1)
    b = np.repeat(np.array(biases.astype(mx.float32)).astype(np.float64), group_size, axis=1)
    return (words, scales, biases), codes.astype(np.float64) * s + b


def run(ctx, arrays, want, group_size, bits, label):
    """Call the solution on evaluated MLX arrays; check graph, shape, dtype, values."""
    words, scales, biases = arrays
    mx.eval(words, scales, biases)
    mx.synchronize()
    mx.clear_cache()
    out = ctx.solution.metal_dequantize(words, scales, biases, group_size, bits)
    check.that(isinstance(out, mx.array), f"{FN}: returned a {type(out).__name__}, not an mx.array")
    ops = graph_ops(out)
    detail = f"operations pending behind the result: {ops}"
    check.that(
        ops.count("CustomKernel") == 1,
        f"{label}: the result's graph holds {ops.count('CustomKernel')} custom kernels; "
        "it must be the lazy output of exactly one kernel built with mx.fast.metal_kernel",
        detail,
    )
    other = sorted(set(ops) - REARRANGE - {"CustomKernel"})
    check.that(
        not other,
        f"{label}: the result's graph contains {', '.join(other)}; besides the kernel, "
        "only operations that rearrange data are allowed",
        detail,
    )
    check.that(
        out.dtype == scales.dtype,
        f"{label}: got dtype {out.dtype}, which is not the expected dtype",
        f"expected {scales.dtype}, the dtype of the scales",
    )
    got = np.array(out.astype(mx.float32)).astype(np.float64)
    check.shape(got, want.shape, label=label)
    check.allclose(got, want, **TOL[scales.dtype], label=label)
    return got


def sweep(ctx, n_out, n_in, group_size, bits, dtype=mx.float32, *, public=False):
    arrays, want = layer(ctx.rng, n_out, n_in, group_size, bits, dtype)
    kind = str(dtype).rsplit(".", 1)[-1]
    # A hidden case's message is shown to the learner, so its label names no input.
    label = FN
    if public:
        label = f"{FN}({n_out} x {n_in}, bits={bits}, group_size={group_size}, {kind})"
    return run(ctx, arrays, want, group_size, bits, label)


@case("the worked example: 2 rows of 64 weights, 4 bits, groups of 32", public=True,
      requires="mlx-gpu")  # fmt: skip
def worked_example(ctx):
    words = mx.array(
        np.array([[0x76543210, 0xFEDCBA98] * 4, [0x00000000, 0xFFFFFFFF] * 4], dtype=np.uint32)
    )
    scales = mx.array([[0.5, -0.25], [1.0, 0.125]])
    biases = mx.array([[1.0, 2.0], [-3.0, 0.5]])
    codes = np.stack([np.tile(np.arange(16), 4), np.tile(np.repeat([0, 15], 8), 4)])
    s = np.repeat(np.array(scales), 32, axis=1)
    b = np.repeat(np.array(biases), 32, axis=1)
    want = codes * s + b
    got = run(ctx, (words, scales, biases), want, 32, 4, "metal_dequantize(example)")
    # The entries the statement works out.
    check.allclose(
        [got[0, 5], got[0, 37], got[1, 7], got[1, 63]],
        [3.5, 0.75, -3.0, 2.375],
        label="metal_dequantize(example) at [0, 5], [0, 37], [1, 7], [1, 63]",
    )
    builtin = np.array(mx.dequantize(words, scales, biases, 32, 4))
    if not np.array_equal(builtin, want.astype(np.float32)):
        raise RuntimeError("authoring error: mx.dequantize disagrees with the example")


# Rejects grid_one_row and group_ignores_row (rows other than the first), and
# high_bits_first.
@case("a 48 x 256 layer at 4 bits in groups of 64", public=True, requires="mlx-gpu")
def four_bits(ctx):
    sweep(ctx, 48, 256, 64, 4, public=True)


# Rejects mask_four_bits; signed_shift fails on every random layer.
@case("8 bits in groups of 32, and 2 bits in groups of 128", public=True, requires="mlx-gpu")
def other_widths(ctx):
    sweep(ctx, 24, 128, 32, 8, public=True)
    sweep(ctx, 24, 256, 128, 2, public=True)


# Rejects float32_output.
@case("float16 and bfloat16 scales: the result has the dtype of the scales", public=True,
      requires="mlx-gpu")  # fmt: skip
def half_precision(ctx):
    sweep(ctx, 16, 128, 64, 8, mx.float16, public=True)
    sweep(ctx, 16, 128, 64, 8, mx.bfloat16, public=True)
    sweep(ctx, 16, 128, 64, 4, mx.float16, public=True)


# Hidden cases.


@case("every combination of 2, 4, 8 bits with groups of 32, 64, 128", requires="mlx-gpu")
def all_combinations(ctx):
    for bits in (2, 4, 8):
        for group_size in (32, 64, 128):
            sweep(ctx, 5, 384, group_size, bits)


@case("one row, one group, and a tall layer with a single group per row", requires="mlx-gpu")
def extreme_shapes(ctx):
    sweep(ctx, 1, 64, 64, 4)
    sweep(ctx, 1, 2048, 64, 8)
    sweep(ctx, 700, 32, 32, 8)
    sweep(ctx, 3, 128, 128, 2, mx.float16)


@case("the largest and smallest codes: all ones, all zeros, only the top bit",
      requires="mlx-gpu")  # fmt: skip
def extreme_codes(ctx):
    for bits in (2, 4, 8):
        top = (1 << bits) - 1
        codes = np.empty((4, 128), dtype=np.int64)
        codes[0] = top
        codes[1] = 0
        codes[2] = 1 << (bits - 1)
        codes[3] = np.where(np.arange(128) % 2 == 0, top, 0)
        arrays, want = layer(ctx.rng, 4, 128, 64, bits, codes=codes)
        run(ctx, arrays, want, 64, bits, FN)


# Rejects strided_flat_index.
@case("inputs that are strided views: every second row, and rows in reverse",
      requires="mlx-gpu")  # fmt: skip
def strided_inputs(ctx):
    (words, scales, biases), want = layer(ctx.rng, 12, 256, 64, 4)
    run(ctx, (words[::2], scales[::2], biases[::2]), want[::2], 64, 4,
        FN)  # fmt: skip
    run(ctx, (words[::-1], scales[::-1], biases[::-1]), want[::-1], 64, 4,
        FN)  # fmt: skip
    run(ctx, (words[3:9], scales[3:9], biases[3:9]), want[3:9], 64, 4,
        FN)  # fmt: skip


@case("the shape of a real layer: 512 x 2560 at 8 bits with bfloat16 scales",
      requires="mlx-gpu")  # fmt: skip
def real_layer_shape(ctx):
    # model.layers.*.self_attn.k_proj of the 8-bit Granite 4.2 3B checkpoint.
    sweep(ctx, 512, 2560, 64, 8, mx.bfloat16)


# Rejects one_thread_per_row (about 14x) and python_row_loop (about 60x).
@case("a 4096 x 4096 layer at 4 bits: within 3x of the reference kernel", requires="mlx-gpu",
      time_limit_s=30)  # fmt: skip
def speed(ctx):
    rng = ctx.rng
    n = 4096
    codes = rng.integers(0, 16, (n, n))
    words = pack(codes, 4)
    scales = rng.uniform(0.01, 0.05, (n, n // 64)).astype(np.float32)
    biases = rng.uniform(-0.5, 0.5, (n, n // 64)).astype(np.float32)
    got = ctx.call(FN, words, scales, biases, 64, 4)
    rows = [0, 1, 2047, 4095]
    want = codes[rows] * np.repeat(scales[rows], 64, axis=1) + np.repeat(biases[rows], 64, axis=1)
    check.shape(got, (n, n), label=FN)
    check.allclose(got[rows], want, rtol=1e-5, atol=1e-6, label=FN)
    ctx.check_speed(FN, words, scales, biases, 64, 4, max_ratio=3.0)
