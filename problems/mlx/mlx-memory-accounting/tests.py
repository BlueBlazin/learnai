"""Tests for mlx-memory-accounting.

`predicted_bytes` is compared exactly with sizes computed here. `measure` is checked in
two ways:

- everywhere: it returns the evaluated result of `fn` (compared with NumPy) and a dict
  of the specified form;
- on the Apple GPU (`requires="mlx-gpu"`): its three byte counts are compared with the
  sizes of the arrays each fixture allocates, within 10 percent plus 1 MiB. The
  fixtures use arrays of 8 to 32 MiB, so sizes dominate allocator overhead, and MLX's
  allocator is deterministic for a fixed graph: the reference reports the predicted
  sizes to within a few kilobytes on every run.

Before every measured call the test does unrelated work: it allocates, evaluates, and
frees a 64 MiB array. That leaves a high-water mark of 64 MiB and 64 MiB of freed
buffers in MLX's cache, which a correct `measure` must not attribute to `fn`.
"""

import io
import math

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from learnai.judge import case, check

MIB = 2**20
KEYS = {"result", "peak_bytes", "active_bytes", "cache_bytes"}
SIDE = 2048  # a (2048, 2048) float32 array is 16 MiB


def nbytes(*shapes, itemsize=4):
    return sum(math.prod(shape) for shape in shapes) * itemsize


def pending_ops(tree):
    """Number of operations not yet computed behind the arrays in `tree` (0: evaluated)."""
    arrays = [leaf for _, leaf in tree_flatten([tree]) if isinstance(leaf, mx.array)]
    if not arrays:
        return 0
    dot = io.StringIO()
    mx.export_to_dot(dot, *arrays)
    return dot.getvalue().count("shape=rectangle")


def earlier_work():
    """Unrelated work before a measurement: a 64 MiB array, evaluated and freed."""
    scratch = mx.full((4096, 4096), 2.0) * 3.0
    mx.eval(scratch)
    del scratch
    mx.synchronize()  # the mess is settled: its buffers are in the cache, not in flight


def big_input(ctx):
    """A 16 MiB float32 array, evaluated, and its NumPy twin."""
    values = ctx.randn(SIDE, SIDE)
    x = mx.array(values)
    mx.eval(x)
    return x, values


def call(ctx, fn, *args):
    """Run the solution's measure after some earlier work; check the form of the report."""
    earlier_work()
    report = ctx.solution.measure(fn, *args)
    check.that(isinstance(report, dict), "measure: must return a dict")
    check.that(
        set(report) == KEYS,
        f"measure: got the keys {sorted(map(str, report))}, expected {sorted(KEYS)}",
        f"expected {sorted(KEYS)}, got {sorted(map(str, report))}",
    )
    for name in ("peak_bytes", "active_bytes", "cache_bytes"):
        check.that(
            type(report[name]) is int,
            f"measure: {name} must be a Python int (a number of bytes)",
            f"got {type(report[name]).__name__}",
        )
    return report


def check_bytes(ctx, report, *, peak, active, cache):
    """Each byte count within 10 percent plus 1 MiB of the size the fixture implies."""
    for name, expected in (("peak_bytes", peak), ("active_bytes", active), ("cache_bytes", cache)):
        got = report[name]
        ctx.metric(name.replace("bytes", "mib"), round(got / MIB, 2), "MiB")
        slack = 0.10 * expected + MIB
        check.that(
            abs(got - expected) <= slack,
            f"measure: {name} is {got / MIB:.2f} MiB, which is outside the expected range",
            f"expected {expected / MIB:.2f} MiB, give or take {slack / MIB:.2f} MiB",
        )


def double(a):
    return a * 2.0


def keeps_half(a):
    """Makes a full-size temporary, evaluates it, and returns something half its size."""
    temporary = a * 2.0
    mx.eval(temporary)
    return temporary[: SIDE // 2] + 1.0


@case("predicted_bytes: the sizes of the worked example", public=True)
def predicted_example(ctx):
    for shapes, dtype, expected in [
        ([(2048, 2048)], mx.float32, 16_777_216),
        ([(2048, 2048)], mx.float16, 8_388_608),
        ([(4096, 1024), (1024,)], mx.float16, 8_390_656),
        ([(1, 8, 4096, 64)] * 2, mx.bfloat16, 8_388_608),
    ]:
        got = ctx.solution.predicted_bytes(shapes, dtype)
        check.that(type(got) is int, "predicted_bytes: must return a Python int")
        check.equal(got, expected, label="predicted_bytes")


@case(
    "measure: doubling a 16 MiB array, then a function that frees a temporary",
    public=True,
    requires="mlx-gpu",
)
def measure_example(ctx):
    x, values = big_input(ctx)
    full = nbytes((SIDE, SIDE))

    report = call(ctx, double, x)
    check.allclose(np.array(report["result"]), values * 2.0, label="result")
    check_bytes(ctx, report, peak=full, active=full, cache=0)

    # Rejects cache_counted_as_active: the freed temporary is in the cache, not in use.
    report = call(ctx, keeps_half, x)
    check.allclose(np.array(report["result"]), values[: SIDE // 2] * 2.0 + 1.0, label="result")
    check_bytes(ctx, report, peak=full + full // 2, active=full // 2, cache=full)


@case("measure returns the result of fn, already evaluated", public=True)
def result_is_evaluated(ctx):
    # Rejects lazy_result without measuring memory: the returned result must have no
    # pending operations.
    values = ctx.randn(64, 64)
    x = mx.array(values)
    report = call(ctx, lambda a, b: (a + b, {"sum": (a * b).sum()}), x, 2.0)
    pending = pending_ops(report["result"])
    first, rest = report["result"]
    check.allclose(np.array(first), values + 2.0, label="result[0]")
    check.allclose(np.array(rest["sum"]), (values * 2.0).sum(), rtol=1e-3, label="result[1]['sum']")
    check.that(
        pending == 0,
        f"measure: the result of fn is not evaluated ({pending} operations still pending)",
        "fn returns a lazy graph; evaluate it inside the measurement, or nothing is measured",
    )


@case("a function whose result is a scalar: everything it allocated is freed", requires="mlx-gpu")
def scalar_result(ctx):
    x, values = big_input(ctx)
    report = call(ctx, lambda a: (a * 2.0).sum(), x)
    expected = float((values.astype(np.float64) * 2.0).sum())
    check.allclose(np.array(report["result"]), np.float32(expected), rtol=1e-3, atol=1.0)
    full = nbytes((SIDE, SIDE))
    check_bytes(ctx, report, peak=full, active=0, cache=full)


@case("predicted_bytes: scalars, empty arrays, an empty list, one-byte dtypes")
def predicted_edges(ctx):
    for shapes, dtype, expected in [
        ([()], mx.float32, 4),
        ([(), (), (3,)], mx.float16, 10),
        ([(5, 0, 7)], mx.float32, 0),
        ([], mx.float32, 0),
        ([(1000, 1000)], mx.bool_, 1_000_000),
        ([(3000, 1000), (7,)], mx.int8, 3_000_007),
        ([(2, 3, 4)], mx.uint32, 96),
        ([(2, 3, 4)], mx.int64, 192),
    ]:
        got = ctx.solution.predicted_bytes(shapes, dtype)
        check.that(type(got) is int, "predicted_bytes: must return a Python int")
        check.equal(got, expected, label="predicted_bytes")


@case("a lazy argument: computing it is not charged to fn", requires="mlx-gpu")
def lazy_argument(ctx):
    # Rejects arguments_not_evaluated: the 16 MiB argument would count as fn's memory.
    lazy = mx.full((SIDE, SIDE), 3.0) * 1.5
    report = call(ctx, lambda a: a[: SIDE // 2] + 1.0, lazy)
    check.allclose(np.array(report["result"]), np.full((SIDE // 2, SIDE), 5.5, np.float32))
    half = nbytes((SIDE // 2, SIDE))
    check_bytes(ctx, report, peak=half, active=half, cache=0)


@case("half precision, a tuple of results, and a Python int argument", requires="mlx-gpu")
def other_results(ctx):
    report = call(ctx, lambda rows: mx.full((rows, 1024), 1.5, dtype=mx.float16), 4096)
    check.allclose(np.array(report["result"]).astype(np.float32), np.full((4096, 1024), 1.5))
    size = nbytes((4096, 1024), itemsize=2)
    check_bytes(ctx, report, peak=size, active=size, cache=0)

    x, values = big_input(ctx)
    report = call(ctx, lambda a: (a * 2.0, a + 1.0), x)
    check.allclose(np.array(report["result"][0]), values * 2.0, label="result[0]")
    check.allclose(np.array(report["result"][1]), values + 1.0, label="result[1]")
    both = nbytes((SIDE, SIDE), (SIDE, SIDE))
    check_bytes(ctx, report, peak=both, active=both, cache=0)


@case("two measurements in a row: the second does not see the first", requires="mlx-gpu")
def back_to_back(ctx):
    # Rejects peak_not_reset and cache_not_cleared even without the test's own earlier
    # work: the first measurement is the earlier work of the second.
    x, values = big_input(ctx)
    full = nbytes((SIDE, SIDE))
    first = ctx.solution.measure(lambda a: mx.concatenate([a, a], axis=0).sum(), x)
    del first
    report = ctx.solution.measure(double, x)
    check.allclose(np.array(report["result"]), values * 2.0, label="result")
    check_bytes(ctx, report, peak=full, active=full, cache=0)
