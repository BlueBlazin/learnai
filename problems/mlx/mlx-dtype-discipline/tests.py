"""Tests for mlx-dtype-discipline.

The numbers are compared with a float64 NumPy implementation of the block, with a
tolerance sized for float16 arithmetic. The dtype of the result is checked on every
call. Two cases, marked `mlx-gpu`, measure peak memory with `mlx_memory`: with hidden
activations of 8 MiB in float16, the reference peaks at 30 MiB and a block whose
hidden activations are float32 at 64 MiB or more, whatever the dtype of its result.
"""

import math

import mlx.core as mx
import numpy as np

from whetstone.judge import case, check
from whetstone.kit import mlx_memory

FN = "forward_f16"
GELU_C = math.sqrt(2.0 / math.pi)
#: Peak memory allowed above the inputs, for hidden activations of 8 MiB in float16.
LIMIT_MIB = 45.0


def oracle(params, x):
    """The block in float64, from the float16 inputs."""
    gain, w_in, w_out = (params[name].astype(np.float64) for name in ("gain", "w_in", "w_out"))
    x = x.astype(np.float64)
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5)
    h = (x / rms * gain) @ w_in.T
    h = 0.5 * h * (1.0 + np.tanh(GELU_C * (h + 0.044715 * h**3)))
    return x + h @ w_out.T


def make_params(ctx, d, hidden, gain_spread=0.2):
    """float16 parameters scaled so that hidden activations stay of order 1."""
    return {
        "gain": (1.0 + ctx.randn(d, scale=gain_spread)).astype(np.float16),
        "w_in": ctx.randn(hidden, d, scale=1.0 / math.sqrt(d)).astype(np.float16),
        "w_out": ctx.randn(d, hidden, scale=1.0 / math.sqrt(hidden)).astype(np.float16),
    }


def to_mlx(params, x):
    params = {name: mx.array(value) for name, value in params.items()}
    x = mx.array(x)
    if x.dtype != mx.float16 or any(v.dtype != mx.float16 for v in params.values()):
        raise RuntimeError("authoring error: the inputs are not float16")
    mx.eval(params, x)
    return params, x


def check_dtype(out):
    check.that(isinstance(out, mx.array), f"{FN}: must return an mx.array")
    check.that(
        out.dtype == mx.float16,
        f"{FN}: got dtype {str(out.dtype).rsplit('.', 1)[-1]}, which is not the expected dtype",
        "the result must be float16; something in the graph promoted it "
        "(an mx.array constant, mx.zeros, mx.arange, ... are float32 unless given a dtype)",
    )


def run(ctx, params, x):
    """Call the solution on float16 inputs; check dtype, then values."""
    out = ctx.solution.forward_f16(*to_mlx(params, x))
    check_dtype(out)
    got = np.array(out).astype(np.float32)
    check.allclose(got, oracle(params, x).astype(np.float32), rtol=1e-2, atol=2e-2, label=FN)
    return got


def run_memory(ctx, n, d, hidden):
    """Peak memory of one call above its (evaluated) inputs, and the dtype of the result."""
    params, x = to_mlx(make_params(ctx, d, hidden), ctx.randn(n, d).astype(np.float16))
    with mlx_memory() as mem:
        out = ctx.solution.forward_f16(params, x)
        mx.eval(out)  # the work must happen inside the block
    check_dtype(out)
    check.finite(np.array(out).astype(np.float32), label=FN)
    check.memory(mem, at_most_mib=LIMIT_MIB)


@case("the worked example: two rows of width 2, one of them zero", public=True)
def worked_example(ctx):
    params = {
        "gain": np.array([1.0, 0.5], dtype=np.float16),
        "w_in": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float16),
        "w_out": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]], dtype=np.float16),
    }
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float16)
    got = run(ctx, params, x)
    if not np.allclose(got, [[3.6804, 3.1012], [0.0, 0.0]], atol=5e-3):
        raise RuntimeError("authoring error: the worked example changed")


@case("a random block: 16 rows of width 32, hidden width 64", public=True)
def random_block(ctx):
    run(ctx, make_params(ctx, 32, 64), ctx.randn(16, 32).astype(np.float16))


@case("rows of zeros (padding) come out as zeros, not NaN", public=True)
def zero_rows(ctx):
    # Rejects eps_underflows: with an epsilon that rounds to 0 in float16 the zero rows
    # are 0 / sqrt(0).
    x = ctx.randn(6, 16).astype(np.float16)
    x[[1, 4]] = 0.0
    run(ctx, make_params(ctx, 16, 32), x)


# Rejects final_astype and computes_in_float32 (66 and 68 MiB; the reference: 30 MiB).
@case(
    "peak memory: 4096 rows of width 256 through a hidden width of 1024, under 45 MiB",
    public=True,
    requires="mlx-gpu",
)
def peak_memory(ctx):
    run_memory(ctx, 4096, 256, 1024)


@case("rows on very different scales in one batch")
def mixed_scales(ctx):
    # Rejects global_rms: each row must be divided by its own RMS.
    x = ctx.randn(8, 32)
    x *= np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0], dtype=np.float32)[:, None]
    run(ctx, make_params(ctx, 32, 64), x.astype(np.float16))


@case("shapes: a single row, a hidden width below the input width, a wide block")
def shapes(ctx):
    for n, d, hidden in [(1, 8, 16), (5, 24, 12), (64, 64, 128)]:
        run(ctx, make_params(ctx, d, hidden), ctx.randn(n, d).astype(np.float16))


@case("gains far from one, some negative, and an all-zero batch")
def gains(ctx):
    params = make_params(ctx, 16, 48)
    params["gain"] = np.array([-2.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 3.0] * 2, dtype=np.float16)
    run(ctx, params, ctx.randn(10, 16).astype(np.float16))
    run(ctx, params, np.zeros((3, 16), dtype=np.float16))


@case("peak memory: 2048 rows of width 512 through a hidden width of 2048", requires="mlx-gpu")
def peak_memory_wide(ctx):
    run_memory(ctx, 2048, 512, 2048)
