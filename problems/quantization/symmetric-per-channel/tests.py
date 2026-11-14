"""Tests for symmetric-per-channel.

Codes are compared exactly, so no input may sit near a rounding tie: two frameworks
could then round w / scale to different sides because of a one-ulp difference in the
scale. Every weight matrix goes through `tie_free`, which keeps w / scale at least 0.05
away from a half-integer, as the statement promises.
"""

import numpy as np

from whetstone.judge import case, check

MARGIN = 0.05


def qmax_of(bits):
    return 2 ** (bits - 1) - 1


def near_tie(t):
    """True where t (float64) is within MARGIN of a half-integer."""
    return np.abs(t - np.floor(t) - 0.5) < MARGIN


def row_scales(w, bits):
    """The statement's scale per row, as a float32 column."""
    max_abs = np.abs(w).max(axis=1, keepdims=True)
    scale = np.where(max_abs > 0, max_abs / np.float32(qmax_of(bits)), np.float32(1.0))
    return scale.astype(np.float32)


def tie_free(w, *all_bits):
    """Move every element whose code would be near a tie onto the nearest grid point,
    for each of the given bit widths. The largest magnitude of a row is never moved
    (its code is exactly +-qmax), so the scales stay what they were."""
    w = np.array(w, dtype=np.float32)
    for _ in range(20):
        clean = True
        for bits in all_bits:
            scale = row_scales(w, bits).astype(np.float64)
            t = w.astype(np.float64) / scale
            bad = near_tie(t)
            if bad.any():
                clean = False
                w[bad] = (np.round(t[bad]) * np.broadcast_to(scale, w.shape)[bad]).astype(
                    np.float32
                )
        if clean:
            return w
    raise AssertionError("could not make the input tie-free")


def native_round_trip(module, fw, w, bits):
    """quantize_weight -> dequantize on framework tensors, so that the codes and scales
    reach dequantize exactly as quantize_weight returned them. Returns numpy arrays."""
    q, scale = module.quantize_weight(fw.tensor(w), bits)
    return fw.numpy(q), fw.numpy(scale), fw.numpy(module.dequantize(q, scale))


def check_all(ctx, w, bits):
    """Both functions against the oracle, then chained on the solution's own outputs."""
    out, _ = w.shape
    q, scale = ctx.call("quantize_weight", w, bits)
    want_q, want_scale = ctx.expected("quantize_weight", w, bits)
    check.shape(scale, (out,), label="scale")
    check.allclose(scale, want_scale, dtype=True, label="scale")
    check.shape(q, w.shape, label="codes")
    check.that(
        q.dtype == np.int8, "quantize_weight must return int8 codes", f"got dtype {q.dtype}"
    )
    check.equal(q, want_q, label="codes")

    ctx.check("dequantize", want_q, want_scale)

    _, _, values = native_round_trip(ctx.solution, ctx.fw, w, bits)
    _, _, want_values = native_round_trip(ctx.oracle, ctx.oracle_fw, w, bits)
    check.allclose(values, want_values, dtype=True, label="dequantize(*quantize_weight(w, bits))")
    return q, scale, values


@case("the worked example: three rows, at 8 bits and at 4 bits", public=True)
def worked_example(ctx):
    w = np.array(
        [[0.4, -1.0, 0.25, 0.8], [0.02, 0.01, -0.03, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    q, scale = ctx.call("quantize_weight", w, 8)
    check.allclose(scale, np.array([1.0 / 127, 0.03 / 127, 1.0], np.float32), label="scale")
    check.equal(q, np.array([[51, -127, 32, 102], [85, 42, -127, 0], [0, 0, 0, 0]]), label="codes")
    q, scale = ctx.call("quantize_weight", w, 4)
    check.allclose(scale, np.array([1.0 / 7, 0.03 / 7, 1.0], np.float32), label="scale")
    check.equal(q, np.array([[3, -7, 2, 6], [5, 2, -7, 0], [0, 0, 0, 0]]), label="codes")
    check_all(ctx, w, 8)
    check_all(ctx, w, 4)


@case("a random weight matrix, shape (4, 8), 8 bits", public=True)
def random_matrix(ctx):
    check_all(ctx, tie_free(ctx.randn(4, 8, scale=0.1), 8), 8)


# Rejects per_tensor_scale: the small rows would collapse to a few codes.
@case("rows on very different scales each get their own grid", public=True)
def rows_on_different_scales(ctx):
    magnitudes = np.array([[1e-3], [0.05], [1.0], [40.0], [2e-2]], dtype=np.float32)
    w = tie_free(ctx.randn(5, 6) * magnitudes, 8)
    q, _, _ = check_all(ctx, w, 8)
    largest = np.abs(q.astype(np.int64)).max(axis=1)
    check.that(
        bool(np.all(largest == 127)),
        "every row must use its full code range: its largest magnitude maps to +-127",
        f"largest |code| per row = {largest.tolist()}",
    )


# Rejects per_column_scale and dequantize_broadcast, which only fit a square matrix.
@case("a square matrix: scales belong to rows, not columns", public=True)
def square_matrix(ctx):
    magnitudes = np.array([[0.01], [0.3], [1.0], [5.0], [0.07], [20.0]], dtype=np.float32)
    w = tie_free(ctx.randn(6, 6) * magnitudes, 8, 5)
    check_all(ctx, w, 8)
    check_all(ctx, w, 5)


# Rejects zero_row_divides and zero_row_epsilon.
@case("all-zero rows get scale 1 and codes 0, with no NaN", public=True)
def zero_rows(ctx):
    w = tie_free(ctx.randn(4, 5, scale=0.5), 8, 3)
    w[1] = 0.0
    w[3] = 0.0
    for bits in (8, 3):
        q, scale, values = check_all(ctx, w, bits)
        check.finite(scale, label="scale")
        check.that(
            float(scale[1]) == 1.0 and float(scale[3]) == 1.0,
            "an all-zero row must get scale 1",
            f"scale = {scale}",
        )
        check.that(
            not q[[1, 3]].any() and not values[[1, 3]].any(),
            "an all-zero row must quantize to codes 0 and dequantize to 0.0",
            f"codes = {q[[1, 3]]}\nvalues = {values[[1, 3]]}",
        )
    check_all(ctx, np.zeros((2, 3), dtype=np.float32), 8)


# Rejects ignores_bits.
@case("every bit width from 2 to 8")
def every_bit_width(ctx):
    w = tie_free(ctx.randn(3, 16, scale=0.2), *range(2, 9))
    for bits in range(2, 9):
        q, _, _ = check_all(ctx, w, bits)
        check.that(
            int(np.abs(q.astype(np.int64)).max()) == qmax_of(bits),
            "the largest magnitude of a row must map to the code +-(2**(bits - 1) - 1)",
            f"bits = {bits}, largest |code| = {int(np.abs(q.astype(np.int64)).max())}",
        )


# Rejects asymmetric_range (the row maximum is positive) and signed_max (it is negative).
@case("the extreme weight of a row is positive in some rows, negative in others")
def sign_of_the_extreme(ctx):
    w = ctx.uniform(6, 7, low=-0.5, high=0.5)
    w[0, 2], w[1, 5], w[2, 0] = 0.9, -0.9, 0.75
    w[3] = -np.abs(w[3]) - 0.1  # every weight negative
    w[4] = np.abs(w[4]) + 0.1  # every weight positive
    w[5, 1], w[5, 4] = 0.6, -0.6  # the extreme magnitude occurs with both signs
    w = tie_free(w, 8, 4)
    for bits in (8, 4):
        q, _, _ = check_all(ctx, w, bits)
        qmax = qmax_of(bits)
        check.that(
            int(q[0, 2]) == qmax and int(q[1, 5]) == -qmax,
            "the largest magnitude of a row must map to +qmax or -qmax, according to its sign",
            f"bits = {bits}: got codes {int(q[0, 2])} and {int(q[1, 5])}",
        )
        check.that(
            int(q.astype(np.int64).min()) >= -qmax,
            "codes must stay in [-qmax, qmax]: the code -2**(bits - 1) is not used",
            f"bits = {bits}: smallest code = {int(q.min())}",
        )


@case("a single row, a single column, and a 1 x 1 matrix")
def degenerate_shapes(ctx):
    check_all(ctx, tie_free(ctx.randn(1, 9), 8), 8)
    check_all(ctx, tie_free(ctx.randn(7, 1, scale=3.0), 6), 6)
    check_all(ctx, np.array([[-2.5]], dtype=np.float32), 8)


@case("dequantize: every code from -127 to 127, with one scale per row")
def dequantize_all_codes(ctx):
    codes = np.arange(-127, 128, dtype=np.int8)
    q = np.stack([codes, codes[::-1], np.roll(codes, 40)])
    ctx.check("dequantize", q, np.array([0.5, 1e-3, 12.0], dtype=np.float32))
    square = np.stack([np.roll(codes[:5], k) for k in range(5)])
    ctx.check("dequantize", square, np.array([1.0, 0.1, 7.0, 3e-3, 250.0], dtype=np.float32))


@case("every weight comes back within half a step of its row")
def half_step_error(ctx):
    def prop(rng):
        bits = int(rng.integers(2, 9))
        magnitudes = (10.0 ** rng.uniform(-3, 2, size=(4, 1))).astype(np.float32)
        w = tie_free(ctx.randn(4, 12) * magnitudes, bits)
        w[int(rng.integers(0, 4))] = 0.0
        _, scale, values = native_round_trip(ctx.solution, ctx.fw, w, bits)
        check.shape(scale, (4,), label="scale")
        check.shape(values, w.shape, label="dequantize(*quantize_weight(w, bits))")
        check.that(bool(np.all(scale > 0)), "every scale must be positive", f"scale = {scale}")
        error = np.abs(values.astype(np.float64) - w)
        limit = 0.5 * scale.astype(np.float64)[:, None] * 1.001
        check.that(
            bool(np.all(error <= limit)),
            "the round-trip error exceeds half a quantization step of the row",
            f"bits = {bits}\nmax error per row = {error.max(axis=1)}\nscale per row = {scale}",
        )

    check.forall(prop, trials=6)
