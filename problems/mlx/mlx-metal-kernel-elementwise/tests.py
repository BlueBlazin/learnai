"""Tests for mlx-metal-kernel-elementwise.

A Metal kernel runs only on the Apple GPU, so every case is marked `mlx-gpu`.

Each call of the solution is checked three ways:

1. the lazy result, before it is evaluated: `mx.export_to_dot` lists the operations
   pending behind it. There must be exactly one `CustomKernel` (what a kernel built by
   `mx.fast.metal_kernel` appears as) and nothing else except operations that only
   rearrange data. The inputs are evaluated first, so everything pending was added by
   the solution;
2. its shape and dtype;
3. its values, against `a / (1 + exp(-a)) * b` computed here in float64 NumPy from the
   very values the solution received. The oracle never touches MLX, so no buffer on
   the GPU ever holds the expected answer: a kernel that leaves part of its output
   unwritten cannot pass by finding it in recycled memory.
"""

import io
import re

import mlx.core as mx
import numpy as np

from learnai.judge import case, check

FN = "fused_silu_mul"
# The tolerances stated in Constraints.
TOL = {mx.float32: {"rtol": 1e-4, "atol": 1e-6}, mx.float16: {"rtol": 5e-3, "atol": 2e-3}}
# Operations that move data without computing on it.
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


def exact(a, b):
    """silu(a) * b in float64, from evaluated MLX arrays."""
    a64 = np.array(a).astype(np.float64)
    b64 = np.array(b).astype(np.float64)
    with np.errstate(over="ignore"):
        return a64 / (1.0 + np.exp(-a64)) * b64


def run(ctx, a, b, label, *, graph=True):
    """Call the solution on two MLX arrays and check graph, shape, dtype, and values."""
    if graph:
        mx.eval(a, b)
    mx.synchronize()
    mx.clear_cache()
    out = ctx.solution.fused_silu_mul(a, b)
    check.that(isinstance(out, mx.array), f"{FN}: returned a {type(out).__name__}, not an mx.array")
    if graph:
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
        out.dtype == a.dtype,
        f"{label}: got dtype {out.dtype}, expected dtype {a.dtype}",
        f"expected {a.dtype}, the dtype of the inputs",
    )
    want = exact(a, b)
    got = np.array(out)
    check.shape(got, want.shape, label=label)
    check.allclose(got.astype(np.float64), want, **TOL[a.dtype], label=label)
    return got


def pair(ctx, *shape, dtype=mx.float32, scale=2.0):
    a = mx.array(ctx.randn(*shape, scale=scale)).astype(dtype)
    b = mx.array(ctx.uniform(*shape, low=-3.0, high=3.0)).astype(dtype)
    # Keep b away from zero, so that an element that was never written is wrong.
    b = mx.where(mx.abs(b) < 0.25, mx.array(1.5, dtype=dtype), b)
    return a, b


@case("the worked example: six elements", public=True, requires="mlx-gpu")
def worked_example(ctx):
    a = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    b = mx.array([1.0, 2.0, 3.0, 4.0, 0.5, -1.0])
    got = run(ctx, a, b, "fused_silu_mul(example)")
    check.allclose(
        got,
        [-0.238406, -0.537883, 0.0, 2.924234, 0.880797, -2.857722],
        atol=1e-5,
        label="fused_silu_mul(example)",
    )


# Rejects grid_first_dimension (48 of 3072 elements computed) and float32_only.
@case("a (64, 48) matrix, in float32 and in float16", public=True, requires="mlx-gpu")
def matrix_both_dtypes(ctx):
    run(ctx, *pair(ctx, 64, 48), "fused_silu_mul(float32 matrix)")
    run(ctx, *pair(ctx, 64, 48, dtype=mx.float16), "fused_silu_mul(float16 matrix)")


# Rejects flat_index_strided; the reversed view rejects strided_elem_to_loc.
@case("strided views: transposed, stepped, reversed, broadcast", public=True, requires="mlx-gpu")
def strided_views(ctx):
    a, b = pair(ctx, 24, 36)
    run(ctx, a.T, b.T, "fused_silu_mul(transposed)")
    run(ctx, a[::2, 1::3], b[::2, 1::3], "fused_silu_mul(every second row, every third column)")
    run(ctx, a[::-1], b[::-1], "fused_silu_mul(rows reversed)")
    run(ctx, a, mx.broadcast_to(b[:1], a.shape), "fused_silu_mul(b broadcast from one row)")


# Rejects sigmoid_overflow.
@case("large magnitudes: up to 100 in float32, up to 20 in float16", public=True,
      requires="mlx-gpu")  # fmt: skip
def large_magnitudes(ctx):
    a = mx.array([-1e4, -100.0, -90.0, -30.0, -12.0, 0.0, 12.0, 30.0, 90.0, 100.0, 1e4])
    b = mx.array(np.linspace(-2.0, 3.0, 11, dtype=np.float32))
    run(ctx, a, b, "fused_silu_mul(float32, |a| up to 1e4)")
    a16 = mx.array([-20.0, -12.0, -11.0, -3.0, 0.0, 3.0, 11.0, 12.0, 20.0]).astype(mx.float16)
    b16 = mx.array(np.linspace(-2.0, 2.0, 9, dtype=np.float32)).astype(mx.float16)
    run(ctx, a16, b16, "fused_silu_mul(float16, |a| up to 20)")


# Hidden cases. Their labels say nothing about the inputs: a hidden case's message is shown.


@case("three and four dimensions, sizes that are not multiples of the threadgroup",
      requires="mlx-gpu")  # fmt: skip
def more_dimensions(ctx):
    run(ctx, *pair(ctx, 7, 129, 3), "fused_silu_mul")
    run(ctx, *pair(ctx, 2, 3, 5, 7), "fused_silu_mul")
    run(ctx, *pair(ctx, 1, 1, 301), "fused_silu_mul")
    run(ctx, *pair(ctx, 513, 1, dtype=mx.float16), "fused_silu_mul")


# Rejects scalar_unhandled.
@case("0-d arrays, a single element, and arrays with no elements", requires="mlx-gpu")
def degenerate_shapes(ctx):
    run(ctx, mx.array(1.5), mx.array(-2.0), "fused_silu_mul")
    run(
        ctx,
        mx.array(-0.75, dtype=mx.float16),
        mx.array(3.0, dtype=mx.float16),
        "fused_silu_mul",
    )
    run(ctx, mx.array([2.5]), mx.array([4.0]), "fused_silu_mul")
    run(ctx, mx.zeros((0, 5)), mx.zeros((0, 5)), "fused_silu_mul")
    run(ctx, mx.zeros((0,), dtype=mx.float16), mx.zeros((0,), dtype=mx.float16),
        "fused_silu_mul")  # fmt: skip


@case("float16 views: a transposed matrix and a permuted 3-D array", requires="mlx-gpu")
def half_precision_views(ctx):
    a, b = pair(ctx, 40, 24, dtype=mx.float16)
    run(ctx, a.T, b.T, "fused_silu_mul")
    a, b = pair(ctx, 4, 5, 6, dtype=mx.float16)
    run(ctx, a.transpose(2, 0, 1), b.transpose(2, 0, 1), "fused_silu_mul")


@case("views with negative steps on other axes, and a permuted 3-D array", requires="mlx-gpu")
def reversed_views(ctx):
    a, b = pair(ctx, 6, 10, 14)
    run(ctx, a[:, ::-1], b[:, ::-1], "fused_silu_mul")
    run(ctx, a[::-2, :, ::-3], b[::-2, :, ::-3], "fused_silu_mul")
    run(ctx, a.transpose(1, 2, 0), b.transpose(1, 2, 0), "fused_silu_mul")
    run(ctx, a[2:5, 3:9, 1:], b[2:5, 3:9, 1:], "fused_silu_mul")


@case("a million elements: 1024 x 1031", requires="mlx-gpu")
def many_elements(ctx):
    run(ctx, *pair(ctx, 1024, 1031), "fused_silu_mul")


@case("inputs that are still lazy, and two calls in a row with different values",
      requires="mlx-gpu")  # fmt: skip
def lazy_inputs(ctx):
    a, b = pair(ctx, 33, 17)
    # The inputs are graphs here; the kernel joins them and nothing is evaluated early.
    run(ctx, a * 0.5 + 1.0, b - 0.25, "fused_silu_mul", graph=False)
    for _ in range(3):
        run(ctx, *pair(ctx, 33, 17), "fused_silu_mul")
