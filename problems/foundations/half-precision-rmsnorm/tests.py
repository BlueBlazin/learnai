"""Tests for half-precision-rmsnorm.

The harness converts floating arrays to float32, so these tests build float16 tensors
themselves and call the solution directly (`run` below). Three things are checked on
every call:

1. The result is a float16 tensor of the input's shape.
2. Its values match the exact result, computed here in float64 from the float16 inputs
   and rounded to float16, within the float16 tolerance of the statement. (The oracle
   reference cannot serve as the expected value: it is itself a float16 computation
   with float16 rounding errors, which is what the tolerance is for.)
3. No floating-point tensor wider than float16 was computed on the way. How that is
   observed depends on the framework:
   - torch: the inputs are instances of a `torch.Tensor` subclass whose
     `__torch_function__` sees the result of every torch operation applied to them.
   - JAX: the solution is traced with `jax.make_jaxpr`, and the dtypes of the
     intermediate values are read from the jaxpr. `jnp.mean` of a float16 array widens to
     float32 internally by itself, so a wide value is
     tolerated when it comes from the reduction pattern (`convert_element_type`,
     `reduce_sum`, `broadcast_in_dim`, `div`, reshapes) and reported when any other
     operation, such as a multiplication or a square root, produces it. A solution
     that cannot be traced (Python control flow on values) is judged on 1 and 2 only.
   - MLX: the lazy graph of the result is exported with `mx.export_to_dot` before it
     is evaluated; any dtype conversion appears in it as an `AsType` node. A solution
     that evaluates its result itself hides its graph and is judged on 1 and 2 only.
   This is a check that the statement's rule was followed, not a sandbox.
"""

import os
import tempfile

import numpy as np

from learnai.judge import case, check

RTOL = 5e-3  # about five float16 ulps (one ulp is between 4.9e-4 and 9.8e-4, relative)
ATOL = 4e-6
F16_MAX = 65504.0
NAME = "rms_norm_f16"

# In JAX, a float wider than float16 may only come out of these primitives: together
# they are what jnp.mean does on its own to a float16 input.
JAX_REDUCTION_PRIMITIVES = {
    "convert_element_type",
    "reduce_sum",
    "broadcast_in_dim",
    "div",
    "reshape",
    "squeeze",
    "expand_dims",
}


def f16(values):
    """A float16 array; values beyond the float16 range are clipped to it."""
    return np.clip(np.asarray(values, dtype=np.float64), -F16_MAX, F16_MAX).astype(np.float16)


def exact(x, weight, eps):
    """The exact result for float16 inputs, rounded to float16."""
    x64, w64 = x.astype(np.float64), weight.astype(np.float64)
    rms = np.sqrt(np.mean(np.square(x64), axis=-1, keepdims=True) + eps)
    return (x64 / rms * w64).astype(np.float16)


def _torch_call(ctx, x, weight, eps):
    import torch

    wide = []

    class Watched(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            out = super().__torch_function__(func, types, args, kwargs)
            for value in out if isinstance(out, (tuple, list)) else (out,):
                if (
                    isinstance(value, torch.Tensor)
                    and value.is_floating_point()
                    and value.dtype != torch.float16
                ):
                    name = getattr(func, "__name__", str(func))
                    wide.append(
                        f"`{name}` produced a {str(value.dtype).removeprefix('torch.')} tensor"
                    )
            return out

    xt = ctx.fw.tensor(x, dtype=np.float16).as_subclass(Watched)
    wt = ctx.fw.tensor(weight, dtype=np.float16).as_subclass(Watched)
    out = ctx.solution.rms_norm_f16(xt, wt, eps)
    if isinstance(out, torch.Tensor):
        out = out.as_subclass(torch.Tensor)
    return out, wide


def _jax_wide_values(jaxpr, found):
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            aval = var.aval
            dtype = np.dtype(aval.dtype)
            if (
                dtype.kind == "f"
                and dtype.itemsize > 2
                and len(aval.shape) >= 1
                and eqn.primitive.name not in JAX_REDUCTION_PRIMITIVES
            ):
                found.append(f"`{eqn.primitive.name}` produced a {dtype.name} array")
        for param in eqn.params.values():
            inner = getattr(param, "jaxpr", param)
            if hasattr(inner, "eqns"):
                _jax_wide_values(inner, found)


def _jax_call(ctx, x, weight, eps):
    import jax

    xj = ctx.fw.tensor(x, dtype=np.float16)
    wj = ctx.fw.tensor(weight, dtype=np.float16)
    out = ctx.solution.rms_norm_f16(xj, wj, eps)
    wide = []
    try:
        closed = jax.make_jaxpr(lambda a, b: ctx.solution.rms_norm_f16(a, b, eps))(xj, wj)
    except Exception:  # noqa: BLE001  not traceable: judged on dtype and values only
        return out, wide
    _jax_wide_values(closed.jaxpr, wide)
    return out, wide


def _mlx_call(ctx, x, weight, eps):
    import mlx.core as mx

    xm = ctx.fw.tensor(x, dtype=np.float16)
    wm = ctx.fw.tensor(weight, dtype=np.float16)
    out = ctx.solution.rms_norm_f16(xm, wm, eps)
    wide = []
    if isinstance(out, mx.array):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "graph.dot")
            mx.export_to_dot(path, out)  # the graph of the still-lazy result
            with open(path, encoding="utf-8") as handle:
                conversions = handle.read().count('"AsType"')
        if conversions:
            wide.append(f"the computation contains {conversions} dtype conversions (`AsType`)")
    return out, wide


def run(ctx, x, weight, eps=1e-5, *, label=NAME):
    """Call the solution on float16 tensors and check dtype, values, and intermediates."""
    x, weight = f16(x), f16(weight)
    call = {"torch": _torch_call, "jax": _jax_call, "mlx": _mlx_call}[ctx.framework]
    out, wide = call(ctx, x, weight, eps)
    check.that(
        ctx.fw.is_tensor(out),
        f"{label}: returned a {type(out).__name__}, which is not a tensor of your framework",
    )
    got = np.asarray(ctx.fw.numpy(out))
    check.that(
        got.dtype == np.float16,
        f"{label}: got dtype {got.dtype}, expected dtype float16",
        "the result must be a float16 tensor",
    )
    check.shape(got, x.shape, label=label)
    check.allclose(
        got.astype(np.float32),
        exact(x, weight, eps).astype(np.float32),
        rtol=RTOL,
        atol=ATOL,
        label=label,
    )
    check.that(
        not wide,
        f"{label}: a floating-point tensor wider than float16 was computed ({wide[0]})"
        if wide
        else "",
        "every floating-point tensor on the way must be float16:\n  " + "\n  ".join(wide[:10]),
    )
    return got


def gain(ctx, d):
    """A non-trivial gain vector: both signs, magnitudes between 0.5 and 2."""
    sign = np.where(ctx.rng.random(d) < 0.25, -1.0, 1.0)
    return f16(ctx.uniform(d, low=0.5, high=2.0) * sign)


def rows(ctx, scales, d):
    """One row of d normal values per scale."""
    scales = np.asarray(scales, dtype=np.float64)[:, None]
    return f16(ctx.rng.standard_normal((len(scales), d)) * scales)


@case("the worked example: [3, 4], [30000, 40000], [3e-4, 4e-4], and zeros", public=True)
def worked_example(ctx):
    weight = [1.0, 0.5]
    got = run(ctx, [3.0, 4.0], weight)
    want = np.array([0.848528, 0.565685], dtype=np.float32)
    check.allclose(got.astype(np.float32), want, rtol=RTOL, atol=ATOL, label=NAME)
    big = run(ctx, [30000.0, 40000.0], weight, label="rms_norm_f16([30000, 40000])")
    check.allclose(big.astype(np.float32), want, rtol=RTOL, atol=ATOL, label=NAME)
    small = run(ctx, [3e-4, 4e-4], weight, label="rms_norm_f16([3e-4, 4e-4])")
    want_small = np.array([0.094259, 0.062864], dtype=np.float32)
    check.allclose(small.astype(np.float32), want_small, rtol=RTOL, atol=ATOL, label=NAME)
    run(ctx, [0.0, 0.0], weight, label="rms_norm_f16([0, 0])")


@case("a random batch of ordinary activations, shape (4, 8)", public=True)
def random_batch(ctx):
    run(ctx, rows(ctx, [1.0, 1.0, 3.0, 0.3], 8), gain(ctx, 8))


# Rejects naive_square.
@case("large activations: rows of size 100, 1000, 5000, and 20000", public=True)
def large_activations(ctx):
    run(ctx, rows(ctx, [100.0, 1000.0, 5000.0, 20000.0], 8), gain(ctx, 8))


# Rejects eps_not_rescaled (first call) and clamps_scale_at_one (second call: with
# eps = 1e-6 the squares of 2e-4, 3e-4, and 4e-4 round to a multiple of 6e-8 that is off
# by a third, which moves the result by 1.3% to 2%; measured, 2.6 to 3.9 times the
# tolerance). naive_square fails both.
@case("small activations: sizes 1e-1 to 1e-4, and vectors of +-3e-4 with eps = 1e-6", public=True)
def small_activations(ctx):
    weight = gain(ctx, 8)
    run(ctx, rows(ctx, [1e-1, 1e-2, 1e-3, 1e-4], 8), weight)
    signs = np.where(ctx.rng.random((4, 8)) < 0.5, -1.0, 1.0)
    x = signs * np.array([[3e-4], [2e-4], [4e-4], [1e-3]])
    run(ctx, x, weight, 1e-6, label="rms_norm_f16 with eps = 1e-6")


# Rejects upcasts_to_float32: the values are easy, only the intermediates can fail.
@case("float16 throughout: no wider tensor is computed on the way", public=True)
def stays_in_float16(ctx):
    run(ctx, rows(ctx, [1.0, 2.0], 16).reshape(2, 1, 16).repeat(5, axis=1), gain(ctx, 16))


# Rejects rescales_by_mean: the mean of |x| is about 1/512 of the largest entry.
@case("one massive activation in a vector of d = 512", public=True)
def one_massive_activation(ctx):
    x = np.zeros((4, 512))
    x[0, 7], x[1, 100], x[2, 511], x[3, 0] = 50000.0, -900.0, 3.0, 0.02
    x[1, 200:204] = [1.0, -2.0, 0.5, 4.0]
    run(ctx, x, gain(ctx, 512))


# Rejects no_zero_guard.
@case("all-zero vectors come out as zeros")
def zero_vectors(ctx):
    x = rows(ctx, [1.0, 0.0, 300.0, 0.0], 8)
    got = run(ctx, x, gain(ctx, 8))
    check.that(bool((got[[1, 3]] == 0).all()), "rms_norm_f16: a zero vector gave nonzero output")
    run(ctx, np.zeros((2, 5, 16)), gain(ctx, 16))


# Rejects eps_over_scale_squared and no_zero_guard: eps / max|x|^2 is beyond float16.
@case("tiny activations, down to subnormals, with eps from 1e-6 to 1e-2")
def tiny_activations(ctx):
    x = rows(ctx, [1e-4, 1e-5, 1e-6, 2e-7], 8)
    weight = gain(ctx, 8)
    for eps in (1e-6, 1e-5, 1e-2):
        run(ctx, x, weight, eps)


@case("the largest float16 values: rows holding 65504 and -65504")
def largest_values(ctx):
    x = rows(ctx, [1.0, 2e4, 6e4, 6e4], 8)
    x[0, 3], x[1, 0], x[2, :] = F16_MAX, -F16_MAX, F16_MAX
    run(ctx, x, gain(ctx, 8))
    run(ctx, np.full((2, 5, 16), -F16_MAX), gain(ctx, 16))


@case("other values of eps on activations of every size")
def eps_values(ctx):
    x = rows(ctx, [3e-3, 3e-2, 0.3, 3000.0], 8)
    weight = gain(ctx, 8)
    for eps in (1e-6, 1e-3, 1e-2):
        run(ctx, x, weight, eps)


@case("rows on very different scales in one tensor, each normalized on its own")
def mixed_scales(ctx):
    scales = 10.0 ** ctx.rng.uniform(-6.5, 4.5, size=10)
    x = rows(ctx, scales, 16).reshape(2, 5, 16)
    run(ctx, x, gain(ctx, 16))


@case("a single vector, d = 1, a 4-D tensor, and d = 512")
def shapes(ctx):
    run(ctx, rows(ctx, [700.0], 8)[0], gain(ctx, 8))
    run(ctx, f16([[4e4], [-3.0], [0.0], [1e-3], [5e-6]]), f16([1.5]))
    run(ctx, f16(ctx.rng.standard_normal((2, 3, 4, 8)) * 2000.0), gain(ctx, 8))
    run(ctx, rows(ctx, [1e-3, 1.0, 400.0, 2e4], 512), gain(ctx, 512))


@case("random scales, shapes, gains, and eps")
def random_sweep(ctx):
    for shape in ((4, 8), (2, 5, 16), (2, 3, 4, 8), (4, 512)):
        n = int(np.prod(shape[:-1]))
        scales = 10.0 ** ctx.rng.uniform(-7, 4.7, size=n)
        x = rows(ctx, scales, shape[-1]).reshape(shape)
        run(ctx, x, gain(ctx, shape[-1]), float(10.0 ** ctx.rng.uniform(-6, -2)))
